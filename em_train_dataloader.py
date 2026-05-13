"""
EarthMind UNet3D Training — Original logic + GC pre-materialization + patch limit.

Usage:
  python em_train_dataloader.py --max-patches-per-ic 500
  python em_train_dataloader.py --max-patches-per-ic 500 --num-workers 0
  python em_train_dataloader.py   # all patches, same as original
"""

import os
import gc
import time
import argparse
import resource
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet3DConditionModel, LCMScheduler

# =============================================================================
# CLI args
# =============================================================================
parser = argparse.ArgumentParser(description="EarthMind UNet3D training — fast GC + patch limit")
parser.add_argument("--max-patches-per-ic", type=int, default=0,
                    help="Max patches per IC (0 = use all). E.g. 1000")
parser.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers for AORC loading (default: 0 = main process)")
parser.add_argument("--prefetch-factor", type=int, default=2,
                    help="DataLoader prefetch factor (only used if num-workers > 0)")
args = parser.parse_args()

MAX_PATCHES = args.max_patches_per_ic
NUM_WORKERS = args.num_workers
PREFETCH_FACTOR = args.prefetch_factor

# =============================================================================
# Timestamp helper
# =============================================================================
def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# =============================================================================
# Paths / bookkeeping
# =============================================================================
done_file = "/media/airlab/ROCSTOR/earthmind_highres/models_unet3d_fast/done_ics.txt"
model_dir = "/media/airlab/ROCSTOR/earthmind_highres/models_unet3d_fast"
os.makedirs(model_dir, exist_ok=True)

done_ics = set()
if os.path.exists(done_file):
    with open(done_file, "r") as f:
        for line in f:
            s = line.strip()
            if s:
                done_ics.add(s)
print(f"[{ts()}] already done ICs: {len(done_ics)}")

global_best_info = os.path.join(model_dir, "global_best.txt")

# =============================================================================
# Memory logging
# =============================================================================
def log_mem(prefix: str):
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    msg = f"[{ts()}] [MEM] {prefix} | CPU maxrss={rss_gb:.2f} GB"
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024.0**3)
        reserv = torch.cuda.memory_reserved() / (1024.0**3)
        msg += f" | GPU alloc={alloc:.2f} GB reserved={reserv:.2f} GB"
    print(msg)

# =============================================================================
# Chunking knobs
# =============================================================================
AORC_CHUNKS = {"time": 24, "latitude": 256, "longitude": 256}
GC_CHUNKS   = {"time": 1, "prediction_timedelta": 12, "latitude": 256, "longitude": 256}

# =============================================================================
# Dataset loading
# =============================================================================
ds_aorc = xr.open_mfdataset(
    "/media/airlab/ROCSTOR/earthmind_highres/noaa_aorc_usa/noaa_aorc_usa_????_day_????????.nc",
    combine="by_coords",
    chunks=AORC_CHUNKS,
)
print(f"[{ts()}] AORC loaded (lazy)")
print(ds_aorc)

ds_gc = xr.open_zarr(
    "/home/airlab/Documents/airlab/earthmind_highres/graphcast_training_2020_2021_new.zarr",
    chunks=GC_CHUNKS,
).sel(time=slice("2021-03-01", "2021-12-31"))  # Same as original
print(f"[{ts()}] GraphCast loaded (lazy)")
print(ds_gc)

gc_min = xr.open_zarr("/home/airlab/Documents/airlab/earthmind_highres/graphcast_training_2020_2021_min.zarr").compute()
gc_max = xr.open_zarr("/home/airlab/Documents/airlab/earthmind_highres/graphcast_training_2020_2021_max.zarr").compute()

aorc_min = xr.open_dataset("/home/airlab/Documents/airlab/earthmind_highres/usa_aorc_min.nc").compute()
aorc_max = xr.open_dataset("/home/airlab/Documents/airlab/earthmind_highres/usa_aorc_max.nc").compute()

ds_svf = xr.open_dataset("/home/airlab/Documents/airlab/earthmind_highres/SKY_VIEW_FACTOR_1k_c230606.nc")
ds_topo = xr.open_dataset("/home/airlab/Documents/airlab/earthmind_highres/elevation.nc")

# =============================================================================
# 1) Expand level variables
# =============================================================================
levels = [850, 700, 500]
level_vars = [v for v, da in ds_gc.data_vars.items() if "level" in da.dims]

out = ds_gc.copy()
for v in level_vars:
    for lev in levels:
        out[f"{v}_level_{lev}"] = ds_gc[v].sel(level=lev).drop_vars("level")
out = out.drop_vars(level_vars)
print(f"[{ts()}] Level variables expanded")
print(out)

# =============================================================================
# 2) Normalize GraphCast
# =============================================================================
def minmax_norm(da, vmin, vmax, eps=1e-12):
    denom = (vmax - vmin)
    denom = xr.where(np.abs(denom) < eps, np.nan, denom)
    return (da - vmin) / denom

def log_minmax_norm_precip(da, vmin, vmax, eps=1e-12):
    da0 = da.clip(min=0)
    vmin0 = vmin.clip(min=0)
    vmax0 = vmax.clip(min=0)
    da_l   = xr.ufuncs.log1p(da0)
    vmin_l = xr.ufuncs.log1p(vmin0)
    vmax_l = xr.ufuncs.log1p(vmax0)
    denom = (vmax_l - vmin_l)
    denom = xr.where(np.abs(denom) < eps, np.nan, denom)
    return (da_l - vmin_l) / denom

precip_gc = {"total_precipitation_6hr"}

out_norm = out.copy()
for v in out.data_vars:
    if v not in gc_min.data_vars or v not in gc_max.data_vars:
        continue
    vmin = gc_min[v]
    vmax = gc_max[v]
    if v in precip_gc:
        out_norm[v] = log_minmax_norm_precip(out[v], vmin, vmax)
    else:
        out_norm[v] = minmax_norm(out[v], vmin, vmax)

out_norm = out_norm.map(lambda da: da.clip(0, 1) if isinstance(da, xr.DataArray) else da)

# =============================================================================
# 3) Lon to [-180,180], rename, interp to AORC grid
# =============================================================================
out_ll = out_norm.assign_coords(lon=(((out_norm.lon + 180) % 360) - 180)).sortby("lon")
out_ll = out_ll.rename({"lat": "latitude", "lon": "longitude"})

out_norm_interp = out_ll.interp(
    latitude=ds_aorc.latitude,
    longitude=ds_aorc.longitude,
    method="linear",
)
print(f"[{ts()}] GraphCast interpolated to AORC grid (lazy)")
print(out_norm_interp)

# =============================================================================
# 4) Interpolate topo + svf to AORC grid and attach
# =============================================================================
ds_topo = ds_topo.rename({"lat": "latitude", "lon": "longitude"})
ds_svf  = ds_svf.rename({"lat": "latitude", "lon": "longitude"})

ds_topo_interp = ds_topo.interp(
    latitude=ds_aorc.latitude,
    longitude=ds_aorc.longitude,
    method="linear",
).fillna(0)

ds_svf_interp = ds_svf.interp(
    latitude=ds_aorc.latitude,
    longitude=ds_aorc.longitude,
    method="linear",
)

out_norm_interp["topo"] = ds_topo_interp["norm_elevation"]
out_norm_interp["svf"]  = ds_svf_interp["SKY_VIEW_FACTOR"]

# =============================================================================
# 5) Normalize AORC
# =============================================================================
precip_aorc = {"APCP_surface"}

ds_aorc_norm = ds_aorc.copy()
for v in ds_aorc.data_vars:
    if v not in aorc_min.data_vars or v not in aorc_max.data_vars:
        continue
    vmin = aorc_min[v]
    vmax = aorc_max[v]
    if v in precip_aorc:
        ds_aorc_norm[v] = log_minmax_norm_precip(ds_aorc[v], vmin, vmax)
    else:
        ds_aorc_norm[v] = minmax_norm(ds_aorc[v], vmin, vmax)

ds_aorc_norm = ds_aorc_norm.map(lambda da: da.clip(0, 1) if isinstance(da, xr.DataArray) else da)
print(f"[{ts()}] AORC normalized (lazy)")
print(ds_aorc_norm)

# =============================================================================
# 6) Solar zenith (NOAA method) - patch-level
# =============================================================================
def cos_sza_noaa_3d(valid_time_np, lat_deg_np, lon_deg_np):
    dt = valid_time_np.astype("datetime64[ns]")
    doy = (dt.astype("datetime64[D]") - dt.astype("datetime64[Y]")).astype(np.int64) + 1
    sec_of_day = (dt - dt.astype("datetime64[D]")).astype("timedelta64[s]").astype(np.int64)
    frac_hour = sec_of_day / 3600.0

    lat = np.deg2rad(lat_deg_np)
    gamma = 2.0 * np.pi / 365.0 * (doy - 1 + (frac_hour - 12.0) / 24.0)

    eot = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )

    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )

    tst = frac_hour * 60.0 + eot + 4.0 * lon_deg_np
    ha = np.deg2rad(tst / 4.0 - 180.0)

    cos_zen = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    return np.clip(cos_zen, 0.0, 1.0).astype("float32")

def cos_sza_patch(valid_time_1d, lat_1d, lon_1d):
    vt = valid_time_1d.astype("datetime64[ns]")
    lat2d, lon2d = np.meshgrid(lat_1d.astype(np.float32), lon_1d.astype(np.float32), indexing="ij")
    vt3d = vt[:, None, None]
    return cos_sza_noaa_3d(vt3d, lat2d[None, :, :], lon2d[None, :, :])

# =============================================================================
# 7) Define feature lists
# =============================================================================
vars_keep_3d_no_sza = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
    "total_precipitation_6hr",
    "geopotential_level_850",
    "geopotential_level_700",
    "geopotential_level_500",
    "specific_humidity_level_850",
    "specific_humidity_level_700",
    "specific_humidity_level_500",
    "temperature_level_850",
    "temperature_level_700",
    "temperature_level_500",
    "vertical_velocity_level_850",
    "vertical_velocity_level_700",
    "vertical_velocity_level_500",
]
vars_keep_2d = ["topo", "svf"]

aorc_vars_keep = [
    "APCP_surface",
    "DLWRF_surface",
    "DSWRF_surface",
    "PRES_surface",
    "SPFH_2maboveground",
    "TMP_2maboveground",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
]

# =============================================================================
# 8) Build model + scheduler + optimizer
# =============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[{ts()}] device: {device}")

patch = 64
Tcond = 12
Ttarget = 67
B = 1

cond_channels = 20   # 17 (3d) + 2 (2d) + 1 (cos_sza)
target_channels = 8  # AORC has 8 vars
cross_attention_dim = 1

model = UNet3DConditionModel(
    sample_size=None,
    in_channels=target_channels + cond_channels,
    out_channels=target_channels,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 512),
    down_block_types=("DownBlock3D", "DownBlock3D", "DownBlock3D", "DownBlock3D"),
    up_block_types=("UpBlock3D", "UpBlock3D", "UpBlock3D", "UpBlock3D"),
    norm_num_groups=8,
    cross_attention_dim=cross_attention_dim,
    attention_head_dim=8,
).to(device)

noise_scheduler = LCMScheduler(num_train_timesteps=1000)

loss_fn = nn.MSELoss()
opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

encoder_hidden_states = torch.zeros(1, 1, cross_attention_dim, device=device)

# =============================================================================
# 9) Training loop setup + resume
# =============================================================================
model.train()

times = out_norm_interp.time.values
print(f"[{ts()}] num IC times: {len(times)}  first: {times[0]}  last: {times[-1]}")

step = 0
global_best_loss = float("inf")

# Resume global best if exists
if os.path.exists(global_best_info):
    best_ckpt_path = None
    with open(global_best_info, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("best_loss_global="):
                global_best_loss = float(line.split("=", 1)[1])
            if line.startswith("checkpoint="):
                best_ckpt_path = line.split("=", 1)[1]

    if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        if "best_loss_global" in ckpt:
            global_best_loss = float(ckpt["best_loss_global"])
        print(f"[{ts()}] LOADED GLOBAL BEST: {global_best_loss} | from: {best_ckpt_path}")
    else:
        print(f"[{ts()}] global_best.txt exists but checkpoint path missing/not found")
else:
    print(f"[{ts()}] No global best found; starting fresh")

# ---- load precomputed common patches ----
common_patch_file = os.path.join(model_dir, f"common_valid_patches_patch{patch}.npy")
if not os.path.exists(common_patch_file):
    raise FileNotFoundError(f"Missing common patch file: {common_patch_file}")
valid_patches = np.load(common_patch_file, allow_pickle=True).tolist()
print(f"[{ts()}] Loaded common valid patches: {len(valid_patches)} from: {common_patch_file}")

# Compute bounding box for GC pre-materialization
def compute_bbox(patches, ps):
    ii = [p[0] for p in patches]
    jj = [p[1] for p in patches]
    return min(ii), max(ii) + ps, min(jj), max(jj) + ps

BBOX_I_MIN, BBOX_I_MAX, BBOX_J_MIN, BBOX_J_MAX = compute_bbox(valid_patches, patch)
bbox_h = BBOX_I_MAX - BBOX_I_MIN
bbox_w = BBOX_J_MAX - BBOX_J_MIN
est_gc_gb = (len(vars_keep_3d_no_sza) * Tcond + len(vars_keep_2d)) * bbox_h * bbox_w * 4 / (1024**3)
print(f"[{ts()}] GC bbox: i=[{BBOX_I_MIN},{BBOX_I_MAX}) j=[{BBOX_J_MIN},{BBOX_J_MAX}) -> {bbox_h}x{bbox_w} | est ~{est_gc_gb:.1f} GB")

log_mem("startup")
print(f"[{ts()}] Config: max_patches_per_ic={MAX_PATCHES if MAX_PATCHES > 0 else 'all'}, num_workers={NUM_WORKERS}")

# =============================================================================
# GC pre-materialization function
# =============================================================================
def prematerialize_gc_bbox(gc_ic_lazy, ic_str, i_min, i_max, j_min, j_max):
    """Load GraphCast bbox into numpy arrays ONCE per IC. ~40s, ~18 GB."""
    t0 = time.time()
    gc_bbox = gc_ic_lazy.isel(latitude=slice(i_min, i_max), longitude=slice(j_min, j_max))
    lat_bbox = gc_bbox.latitude.values
    lon_bbox = gc_bbox.longitude.values

    gc_3d_all = {}
    for v in vars_keep_3d_no_sza:
        print(f"[{ts()}]   GC: {v} ...", end="", flush=True)
        t1 = time.time()
        arr = gc_bbox[v].transpose("prediction_timedelta", "latitude", "longitude").values
        gc_3d_all[v] = arr.astype(np.float32)
        del arr
        print(f" {time.time()-t1:.1f}s")

    gc_2d_all = {}
    for v in vars_keep_2d:
        print(f"[{ts()}]   GC: {v} ...", end="", flush=True)
        t1 = time.time()
        arr = gc_bbox[v].transpose("latitude", "longitude").values
        gc_2d_all[v] = arr.astype(np.float32)
        del arr
        print(f" {time.time()-t1:.1f}s")

    print(f"[{ts()}]   GC pre-materialized in {time.time()-t0:.1f}s")
    return gc_3d_all, gc_2d_all, lat_bbox, lon_bbox

# =============================================================================
# 10) Training loop: ICs then patches (SAME STRUCTURE AS ORIGINAL)
# =============================================================================
for time_ic in times:
    ic_str = str(np.datetime64(time_ic, "ns"))

    if ic_str in done_ics:
        print(f"\n[{ts()}] === SKIP (already done) IC: {ic_str} ===")
        continue

    print(f"\n[{ts()}] === IC: {ic_str} ===")

    # Select GraphCast IC (lazy) — SAME AS ORIGINAL
    gc_ic = out_norm_interp.sel(time=time_ic)

    # Select AORC target window (lazy) — SAME AS ORIGINAL
    ds_aorc_win = ds_aorc_norm.sel(time=slice(time_ic, time_ic + np.timedelta64(66, "h")))
    if ds_aorc_win.sizes["time"] != Ttarget:
        print(f"[{ts()}] skip (AORC window length) {ds_aorc_win.sizes['time']}")
        continue

    if len(valid_patches) == 0:
        print(f"[{ts()}] skip IC (no valid patches)")
        continue

    # Precompute valid_time_1d — SAME AS ORIGINAL
    valid_time_1d = (np.datetime64(time_ic, "ns") + gc_ic["prediction_timedelta"].values).astype("datetime64[ns]")

    # Deterministic shuffle — SAME AS ORIGINAL
    seed = (np.uint64(np.datetime64(time_ic, "ns").astype("int64")) ^ np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    rng = np.random.default_rng(int(seed))
    patches_shuffled = valid_patches.copy()
    rng.shuffle(patches_shuffled)

    # Don't truncate upfront — iterate full list, count only successful (non-NaN) trains
    print(f"[{ts()}] IC {ic_str}: target {MAX_PATCHES if MAX_PATCHES > 0 else 'all'} non-NaN patches from {len(patches_shuffled)} total")

    # *** PRE-MATERIALIZE GraphCast ONCE per IC ***
    print(f"[{ts()}] Pre-materializing GraphCast bbox...")
    gc_3d_all, gc_2d_all, lat_bbox, lon_bbox = prematerialize_gc_bbox(
        gc_ic, ic_str, BBOX_I_MIN, BBOX_I_MAX, BBOX_J_MIN, BBOX_J_MAX,
    )
    log_mem(f"after GC prematerialize IC {ic_str}")

    t_ic_start = time.time()
    t_step_start = time.time()
    trained_this_ic = 0  # count only successful (non-NaN) training steps

    # *** PATCH LOOP — SAME STRUCTURE AS ORIGINAL ***
    for (i0, j0) in patches_shuffled:
        # ----------------------------
        # Build X patch (GraphCast cond) — from pre-materialized numpy (FAST)
        # ----------------------------
        # Convert to bbox-local coordinates
        gi0 = i0 - BBOX_I_MIN
        gj0 = j0 - BBOX_J_MIN
        gi1 = gi0 + patch
        gj1 = gj0 + patch

        lat_1d = lat_bbox[gi0:gi1]
        lon_1d = lon_bbox[gj0:gj1]

        # Compute cos_sza for patch only — SAME AS ORIGINAL
        cos_sza_np = cos_sza_patch(valid_time_1d, lat_1d, lon_1d).astype(np.float32)

        # Load 3D vars from pre-materialized numpy — SAME DATA, just faster
        X_list = []
        for v in vars_keep_3d_no_sza:
            arr = gc_3d_all[v][:, gi0:gi1, gj0:gj1]  # (Tcond, P, P) — numpy slice, microseconds
            X_list.append(arr)

        # 2D vars expanded across time — SAME AS ORIGINAL
        for v in vars_keep_2d:
            arr2 = gc_2d_all[v][gi0:gi1, gj0:gj1]  # (P, P)
            arr3 = np.broadcast_to(arr2[None, :, :], (Tcond, patch, patch))
            X_list.append(arr3)

        # Append cos_sza — SAME AS ORIGINAL
        X_list.append(cos_sza_np)

        # Stack — SAME AS ORIGINAL
        X_np = np.stack(X_list, axis=0)  # (20, 12, P, P)

        # ----------------------------
        # Build Y patch (AORC target) — SAME AS ORIGINAL (lazy dask load)
        # ----------------------------
        aorc_patch = ds_aorc_win.isel(latitude=slice(i0, i0 + patch), longitude=slice(j0, j0 + patch))

        Y_list = []
        for v in aorc_vars_keep:
            arr = aorc_patch[v].transpose("time", "latitude", "longitude").data
            arr = np.asarray(arr).astype(np.float32)  # (Ttarget, P, P)
            Y_list.append(arr)
        Y_np = np.stack(Y_list, axis=0)  # (8, 67, P, P)

        # Safety NaN check — SAME AS ORIGINAL
        if np.isnan(X_np).any() or np.isnan(Y_np).any():
            print(f"[{ts()}] UNEXPECTED NaN | IC {ic_str} | patch {i0} {j0}")
            del X_np, Y_np, X_list, Y_list, cos_sza_np
            gc.collect()
            continue

        # ----------------------------
        # Torch tensors — SAME AS ORIGINAL
        # ----------------------------
        x = torch.from_numpy(X_np)[None].to(device, non_blocking=True)  # (1,20,12,P,P)
        y = torch.from_numpy(Y_np)[None].to(device, non_blocking=True)  # (1,8,67,P,P)

        # ----------------------------
        # One diffusion training step — SAME AS ORIGINAL
        # ----------------------------
        noise = torch.randn_like(y)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
        noisy_y = noise_scheduler.add_noise(y, noise, timesteps)

        x_match = F.interpolate(x, size=(Ttarget, patch, patch), mode="trilinear", align_corners=False)
        net_input = torch.cat([noisy_y, x_match], dim=1)

        noise_pred = model(
            sample=net_input,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        loss_val = loss_fn(noise_pred, noise)

        opt.zero_grad(set_to_none=True)
        loss_val.backward()
        opt.step()

        loss_float = float(loss_val.detach().item())

        # ---------- global best ---------- SAME AS ORIGINAL
        if loss_float < global_best_loss:
            global_best_loss = loss_float
            ic_safe = ic_str.replace(":", "").replace("-", "").replace(".000000000", "")
            if "T" in ic_safe and len(ic_safe) > 15:
                ic_safe = ic_safe[:15]

            global_best_path = os.path.join(model_dir, f"unet3d_global_best_ic_{ic_safe}.pt")
            torch.save(
                {
                    "ic": ic_str,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "best_loss_global": global_best_loss,
                },
                global_best_path,
            )
            with open(global_best_info, "w") as f:
                f.write(f"best_loss_global={global_best_loss}\n")
                f.write(f"ic={ic_str}\n")
                f.write(f"checkpoint={global_best_path}\n")

            print(
                f"[{ts()}] NEW GLOBAL BEST: {global_best_loss} | saved: {global_best_path}"
                f" | IC {ic_str} | patch {i0} {j0}"
            )

        step += 1
        trained_this_ic += 1
        if step % 10 == 0:
            t_now = time.time()
            elapsed = t_now - t_step_start
            sps = 10.0 / elapsed
            print(
                f"[{ts()}] step {step} | IC {str(time_ic)} | patch {i0} {j0}"
                f" | loss {loss_float:.6f} | {elapsed:.1f}s/10steps ({sps:.2f} steps/s)"
                f" | trained {trained_this_ic}/{MAX_PATCHES if MAX_PATCHES > 0 else len(patches_shuffled)}"
            )
            log_mem(f"step {step}")
            t_step_start = t_now

        # Cleanup — SAME AS ORIGINAL
        del X_np, Y_np, X_list, Y_list, cos_sza_np
        del x, y, noise, noisy_y, x_match, net_input, noise_pred, loss_val
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # *** Stop after reaching target non-NaN patches ***
        if MAX_PATCHES > 0 and trained_this_ic >= MAX_PATCHES:
            print(f"[{ts()}] Reached {MAX_PATCHES} trained patches for IC {ic_str}, moving to next IC")
            break

    # *** INNER LOOP FINISHED → Free GC data, mark IC done, move to next IC ***
    del gc_3d_all, gc_2d_all, lat_bbox, lon_bbox
    gc.collect()

    ic_elapsed = time.time() - t_ic_start
    with open(done_file, "a") as f:
        f.write(ic_str + "\n")
    done_ics.add(ic_str)
    print(f"[{ts()}] DONE IC: {ic_str} | trained {trained_this_ic} non-NaN patches in {ic_elapsed:.0f}s ({ic_elapsed/60:.1f} min) | best loss: {global_best_loss}")
    log_mem(f"done IC {ic_str}")