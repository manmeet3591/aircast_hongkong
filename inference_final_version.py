#!/usr/bin/env python3
"""
Earthmind Inference Script — Final 
=========================================================

Usage example:
  python earthmind_inference_final.py \
    --graphcast_zarr /path/to/graphcast.zarr \
    --gc_min_zarr /path/to/gc_min.zarr \
    --gc_max_zarr /path/to/gc_max.zarr \
    --topo_nc /path/to/elevation.nc \
    --svf_nc /path/to/sky_view_factor.nc \
    --aorc_min_nc /path/to/aorc_min.nc \
    --aorc_max_nc /path/to/aorc_max.nc \
    --checkpoint /path/to/model.pt \
    --start_date 2021-07-04 \
    --end_date 2021-12-31 \
    --num_inits 10 \
    --lat_min 25.0 --lat_max 50.0 \
    --lon_min -125.0 --lon_max -65.0 \
    --patch 256 --stride 128 \
    --steps 4 8 25 50 \
    --out_dir /path/to/output \
    --out_name conus_inference \
    --overwrite
"""
from __future__ import annotations

import argparse
import contextlib
import os
import re
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import xarray as xr
import dask
import zarr
from tqdm import tqdm

import torch
import torch.nn.functional as F
from diffusers import LCMScheduler, UNet3DConditionModel
from diffusers.models.attention_processor import AttnProcessor


# =============================================================================
# Constants — architectural constraints fixed at training time
# =============================================================================

TCOND   = 12   # number of GraphCast prediction_timedelta steps (asserted at runtime)
TTARGET = 67   # number of hourly AORC output timesteps (asserted at runtime)

TARGET_CHANNELS = 8    # number of AORC output variables
COND_CHANNELS   = 20   # 17 (3D GC vars) + 2 (topo, svf) + 1 (cos_sza)


# =============================================================================
# Variable lists — must match training exactly
# =============================================================================

VARS_3D = [
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
]  # len = 17

VARS_2D = ["topo", "svf"]  # len = 2
# + 1 cos_sza → total COND_CHANNELS = 20

AORC_VARS = [
    "APCP_surface",
    "DLWRF_surface",
    "DSWRF_surface",
    "PRES_surface",
    "SPFH_2maboveground",
    "TMP_2maboveground",
    "UGRD_10maboveground",
    "VGRD_10maboveground",
]  # len = 8

PRECIP_GC_VARS   = {"total_precipitation_6hr"}
PRECIP_AORC_VARS = {"APCP_surface"}
PRESSURE_LEVELS  = [850, 700, 500]


# =============================================================================
# Logging helpers
# =============================================================================

def ts() -> str:
    """Return current timestamp string for inline use."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    tqdm.write(f"[{ts()}] {msg}")


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


# =============================================================================
# Coordinate helpers
# =============================================================================

def _standardize_lat_lon_coords(ds: xr.Dataset) -> Tuple[str, str]:
    for lat_name, lon_name in (("lat", "lon"), ("latitude", "longitude")):
        if lat_name in ds.coords and lon_name in ds.coords:
            return lat_name, lon_name
    raise KeyError(
        f"Could not find lat/lon coords in dataset. Found coords: {list(ds.coords)}"
    )


def _maybe_convert_lon_360_to_180(lon_1d: np.ndarray) -> np.ndarray:
    lon = lon_1d.astype(np.float32)
    if np.nanmax(lon) > 180.0:
        lon = (((lon + 180.0) % 360.0) - 180.0).astype(np.float32)
    return lon


# =============================================================================
# Zarr v3 compressor
# =============================================================================

def make_zarr_v3_compressors() -> Optional[tuple]:
    try:
        from zarr.codecs import BloscCodec
        return (BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle"),)
    except Exception:
        return None


# =============================================================================
# Normalization
# =============================================================================

def minmax_norm(
    da: xr.DataArray, vmin: xr.DataArray, vmax: xr.DataArray, eps: float = 1e-12
) -> xr.DataArray:
    denom = xr.where(abs(vmax - vmin) < eps, np.nan, vmax - vmin)
    return (da - vmin) / denom


def log_minmax_norm_precip(
    da: xr.DataArray, vmin: xr.DataArray, vmax: xr.DataArray, eps: float = 1e-12
) -> xr.DataArray:
    da0, vmin0, vmax0 = da.clip(min=0), vmin.clip(min=0), vmax.clip(min=0)
    da_l   = xr.ufuncs.log1p(da0)
    vmin_l = xr.ufuncs.log1p(vmin0)
    vmax_l = xr.ufuncs.log1p(vmax0)
    denom  = xr.where(abs(vmax_l - vmin_l) < eps, np.nan, vmax_l - vmin_l)
    return (da_l - vmin_l) / denom


def inverse_scale(
    y_norm: np.ndarray,
    aorc_min: Dict[str, float],
    aorc_max: Dict[str, float],
) -> np.ndarray:
    """
    Inverse scale from [0,1] normalized space back to physical units.
    Shape: (B, C, T, P, P) → (B, C, T, P, P)
    Precip uses expm1 (inverse of log1p); all others are linear.
    """
    y_norm = np.clip(y_norm, 0.0, 1.0)  # match training's post-normalization [0,1] clip
    y_phys = np.empty_like(y_norm, dtype=np.float32)
    for k, v in enumerate(AORC_VARS):
        vmin, vmax = aorc_min[v], aorc_max[v]
        if v in PRECIP_AORC_VARS:
            vmin_l = np.log1p(max(vmin, 0.0))
            vmax_l = np.log1p(max(vmax, 0.0))
            y_phys[:, k] = np.clip(
                np.expm1(y_norm[:, k] * (vmax_l - vmin_l) + vmin_l), 0, None
            )
        else:
            y_phys[:, k] = y_norm[:, k] * (vmax - vmin) + vmin
    return y_phys


# =============================================================================
# Solar zenith angle
# =============================================================================

def cos_sza_noaa(
    valid_time_np: np.ndarray,
    lat_deg_np: np.ndarray,
    lon_deg_np: np.ndarray,
) -> np.ndarray:
    """
    NOAA solar geometry. Inputs are broadcast-compatible numpy arrays.
    Returns cos(SZA) clipped to [0, 1], float32.
    """
    dt      = valid_time_np.astype("datetime64[ns]")
    doy     = (dt.astype("datetime64[D]") - dt.astype("datetime64[Y]")).astype(np.int64) + 1
    sec     = (dt - dt.astype("datetime64[D]")).astype("timedelta64[s]").astype(np.int64)
    frac_hr = sec / 3600.0

    lat   = np.deg2rad(lat_deg_np)
    gamma = 2.0 * np.pi / 365.0 * (doy - 1 + (frac_hr - 12.0) / 24.0)

    eot = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)   - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2*gamma) - 0.040849 * np.sin(2*gamma)
    )

    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)   + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2*gamma) + 0.000907 * np.sin(2*gamma)
        - 0.002697 * np.cos(3*gamma) + 0.001480 * np.sin(3*gamma)
    )

    tst = frac_hr * 60.0 + eot + 4.0 * lon_deg_np
    ha  = np.deg2rad(tst / 4.0 - 180.0)

    cos_zen = (
        np.sin(lat) * np.sin(decl)
        + np.cos(lat) * np.cos(decl) * np.cos(ha)
    )
    return np.clip(cos_zen, 0.0, 1.0).astype(np.float32)


def cos_sza_for_tile(
    valid_time_1d: np.ndarray,  # shape (Tcond,) datetime64[ns] — matches training
    lat_1d: np.ndarray,         # shape (P,) float32
    lon_1d: np.ndarray,         # shape (P,) float32
) -> np.ndarray:
    """
    Compute cos(SZA) for a spatial tile over Tcond lead times.
    Returns shape (Tcond, P, P) float32 — identical to training's cos_sza_patch().
    valid_time_1d is always (Tcond,) — no extra batch dimension.
    """
    lat2d, lon2d = np.meshgrid(
        lat_1d.astype(np.float32),
        lon_1d.astype(np.float32),
        indexing="ij",
    )  # both (P, P)

    vt3d  = valid_time_1d[:, None, None]  # (Tcond, 1, 1)
    lat3d = lat2d[None, :, :]             # (1, P, P)
    lon3d = lon2d[None, :, :]             # (1, P, P)

    return cos_sza_noaa(vt3d, lat3d, lon3d)  # (Tcond, P, P)


# =============================================================================
# GraphCast — pressure level expansion & normalization
# =============================================================================

def expand_pressure_levels(ds: xr.Dataset) -> xr.Dataset:
    level_vars = [v for v, da in ds.data_vars.items() if "level" in da.dims]
    out = ds.copy()
    for v in level_vars:
        for lev in PRESSURE_LEVELS:
            out[f"{v}_level_{lev}"] = ds[v].sel(level=lev).drop_vars("level")
    return out.drop_vars(level_vars)


def normalize_graphcast(
    ds: xr.Dataset, gc_min: xr.Dataset, gc_max: xr.Dataset
) -> xr.Dataset:
    out = ds.copy()
    for v in ds.data_vars:
        if v not in gc_min.data_vars or v not in gc_max.data_vars:
            continue
        if v in PRECIP_GC_VARS:
            out[v] = log_minmax_norm_precip(ds[v], gc_min[v], gc_max[v])
        else:
            out[v] = minmax_norm(ds[v], gc_min[v], gc_max[v])
    return out.map(lambda da: da.clip(0, 1) if isinstance(da, xr.DataArray) else da)


# =============================================================================
# GraphCast data loaders
# =============================================================================

def open_and_prepare_graphcast(
    graphcast_zarr: str,
    gc_min_zarr: str,
    gc_max_zarr: str,
    start_date: str,
    end_date: str,
) -> xr.Dataset:
    log("Opening local GraphCast Zarr…")
    ds_raw = xr.open_zarr(graphcast_zarr).sel(time=slice(start_date, end_date))

    # Keep only the first TCOND=12 lead times — matches training exactly
    ds_raw = ds_raw.isel(prediction_timedelta=slice(0, TCOND))
    log(f"Sliced to first {TCOND} prediction_timedelta steps (6hr → 72hr).")

    log("Opening min/max Zarr (computing into memory)…")
    gc_min = xr.open_zarr(gc_min_zarr, consolidated=False).compute()
    gc_max = xr.open_zarr(gc_max_zarr, consolidated=False).compute()

    log("Expanding pressure levels…")
    ds = expand_pressure_levels(ds_raw)

    log("Normalizing GraphCast…")
    ds_norm = normalize_graphcast(ds, gc_min, gc_max)

    g_lat, g_lon = _standardize_lat_lon_coords(ds_norm)
    ds_norm = ds_norm.rename({g_lat: "latitude", g_lon: "longitude"})
    ds_norm = ds_norm.assign_coords(
        longitude=_maybe_convert_lon_360_to_180(ds_norm["longitude"].values)
    ).sortby("longitude")

    return ds_norm


# ------------------------------------------------------------------
# GCS GraphCast helpers (format A / B / C)
# ------------------------------------------------------------------

def _find_zarr_stores_in_dir(year_dir: Path) -> List[Tuple[Path, str]]:
    results: List[Tuple[Path, str]] = []
    per_ic_pattern = re.compile(r"^\d{8}_\d{2}hr_\d{2}_preds$")
    per_ic_dirs = [
        d for d in sorted(year_dir.iterdir())
        if d.is_dir() and per_ic_pattern.match(d.name)
    ]
    if per_ic_dirs:
        for ic_dir in per_ic_dirs:
            pz = ic_dir / "predictions.zarr"
            if pz.is_dir():
                results.append((pz, "C"))
        return results
    for zarr_path in sorted(year_dir.glob("**/*.zarr")):
        if not zarr_path.is_dir():
            continue
        name = zarr_path.name
        if "_6_hours.zarr" in name:
            results.append((zarr_path, "A"))
        elif "_12_hours.zarr" in name:
            results.append((zarr_path, "B"))
    return results


def _norm_prediction_timedelta(ds: xr.Dataset) -> xr.Dataset:
    if "prediction_timedelta" not in ds.coords:
        return ds
    pd_val = ds["prediction_timedelta"].values
    if np.issubdtype(pd_val.dtype, np.integer):
        td_ns = pd_val.astype("int64") * np.int64(3_600_000_000_000)
        ds = ds.assign_coords(
            prediction_timedelta=xr.DataArray(
                td_ns.astype("timedelta64[ns]"), dims=["prediction_timedelta"]
            )
        )
    return ds


def _open_fmt_AB(zarr_path: Path, start_date: str, end_date: str) -> xr.Dataset:
    # decode_timedelta=True: suppress FutureWarning about timedelta decoding
    ds = xr.open_zarr(
        str(zarr_path), consolidated=True, decode_timedelta=True
    ).sel(time=slice(start_date, end_date))
    return _norm_prediction_timedelta(ds)


def _open_fmt_C(zarr_path: Path) -> xr.Dataset:
    ic_dir_name = zarr_path.parent.name
    m = re.match(r"(\d{4})(\d{2})(\d{2})_(\d{2})hr", ic_dir_name)
    if not m:
        raise ValueError(f"Cannot parse IC datetime from folder: {ic_dir_name}")
    ic_str  = f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}:00"
    ic_time = np.datetime64(ic_str, "ns")

    # decode_timedelta=True: suppress FutureWarning about timedelta decoding
    ds_raw   = xr.open_zarr(str(zarr_path), consolidated=True, decode_timedelta=True)
    lead_hrs = ds_raw["time"].values.astype("int64")
    td_ns    = lead_hrs * np.int64(3_600_000_000_000)

    ds = ds_raw.rename({"time": "prediction_timedelta"})
    ds = ds.assign_coords(
        prediction_timedelta=xr.DataArray(
            td_ns.astype("timedelta64[ns]"), dims=["prediction_timedelta"]
        )
    )
    ds = ds.expand_dims({"time": np.array([ic_time], dtype="datetime64[ns]")})
    for drop_var in ("init_time", "datetime", "template_success"):
        ds = ds.drop_vars(drop_var, errors="ignore")
    return ds


def open_gcs_graphcast(
    gcs_root: str,
    gc_min_zarr: str,
    gc_max_zarr: str,
    start_date: str,
    end_date: str,
) -> xr.Dataset:
    gcs_root_p = Path(gcs_root)
    start_dt   = np.datetime64(start_date, "D")
    end_dt     = np.datetime64(end_date,   "D")

    datasets: List[xr.Dataset] = []

    for year_dir in sorted(gcs_root_p.iterdir()):
        if not year_dir.is_dir():
            continue
        m = re.search(r"(\d{4})_to_(\d{4}|present)", year_dir.name)
        if not m:
            continue
        yr_start  = int(m.group(1))
        yr_end    = 2099 if m.group(2) == "present" else int(m.group(2))
        dir_start = np.datetime64(f"{yr_start}-01-01", "D")
        dir_end   = np.datetime64(f"{yr_end}-12-31",   "D")

        if dir_end < start_dt or dir_start > end_dt:
            continue

        for zarr_path, fmt in _find_zarr_stores_in_dir(year_dir):
            try:
                if fmt == "C":
                    ic_dir_name = zarr_path.parent.name
                    mm = re.match(r"(\d{4})(\d{2})(\d{2})_(\d{2})hr", ic_dir_name)
                    if not mm:
                        continue
                    ic_day = np.datetime64(
                        f"{mm.group(1)}-{mm.group(2)}-{mm.group(3)}", "D"
                    )
                    if ic_day < start_dt or ic_day > end_dt:
                        continue
                    ds = _open_fmt_C(zarr_path)
                else:
                    ds = _open_fmt_AB(zarr_path, start_date, end_date)
                    if ds["time"].size == 0:
                        continue
                datasets.append(ds)
            except Exception as e:
                log(f"  [WARN] skipping {zarr_path}: {e}")

    if not datasets:
        raise RuntimeError(
            f"No GCS GraphCast data found for {start_date}→{end_date} in {gcs_root}"
        )

    log(f"  Concatenating {len(datasets)} GCS zarr stores…")
    ds_all = xr.concat(datasets, dim="time").sortby("time")

    # Keep only the first TCOND=12 lead times — matches training exactly.
    # GCS data often contains 40 steps (10-day forecast); the model was
    # trained on 12 steps (6hr → 72hr) only.
    ds_all = ds_all.isel(prediction_timedelta=slice(0, TCOND))
    log(f"  Sliced to first {TCOND} prediction_timedelta steps (6hr → 72hr).")

    g_lat, g_lon = _standardize_lat_lon_coords(ds_all)
    ds_all = ds_all.rename({g_lat: "latitude", g_lon: "longitude"})
    ds_all = ds_all.assign_coords(
        longitude=_maybe_convert_lon_360_to_180(ds_all["longitude"].values)
    ).sortby("longitude")

    log("  Expanding pressure levels (GCS)…")
    ds_all = expand_pressure_levels(ds_all)

    log("  Opening min/max for normalisation (GCS)…")
    gc_min = xr.open_zarr(gc_min_zarr, consolidated=False).compute()
    gc_max = xr.open_zarr(gc_max_zarr, consolidated=False).compute()

    log("  Normalizing GraphCast (GCS)…")
    ds_all = normalize_graphcast(ds_all, gc_min, gc_max)

    return ds_all


# =============================================================================
# Static grids (topo + svf)
# =============================================================================

def load_static_grids(
    topo_nc: str, svf_nc: str
) -> Tuple[xr.Dataset, xr.Dataset]:
    topo0 = xr.open_dataset(topo_nc)
    svf0  = xr.open_dataset(svf_nc)

    t_lat, t_lon = _standardize_lat_lon_coords(topo0)
    s_lat, s_lon = _standardize_lat_lon_coords(svf0)

    topo = topo0.rename({t_lat: "latitude", t_lon: "longitude"})
    svf  = svf0.rename({s_lat: "latitude", s_lon: "longitude"})

    topo = topo.assign_coords(
        longitude=_maybe_convert_lon_360_to_180(topo["longitude"].values)
    ).sortby("longitude")
    svf = svf.assign_coords(
        longitude=_maybe_convert_lon_360_to_180(svf["longitude"].values)
    ).sortby("longitude")

    return topo, svf


def subset_static_to_bbox(
    topo: xr.Dataset, svf: xr.Dataset,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
) -> Tuple[np.ndarray, np.ndarray, xr.DataArray, xr.DataArray]:
    topo_sub = topo.sel(
        latitude=slice(lat_min, lat_max),
        longitude=slice(lon_min, lon_max),
    )
    svf_sub = svf.sel(
        latitude=slice(lat_min, lat_max),
        longitude=slice(lon_min, lon_max),
    )

    lat = topo_sub["latitude"].values.astype(np.float32)
    lon = topo_sub["longitude"].values.astype(np.float32)

    if "norm_elevation" not in topo_sub:
        raise KeyError(
            f"topo file missing 'norm_elevation'. Available vars: {list(topo_sub.data_vars)}"
        )
    if "SKY_VIEW_FACTOR" not in svf_sub:
        raise KeyError(
            f"svf file missing 'SKY_VIEW_FACTOR'. Available vars: {list(svf_sub.data_vars)}"
        )

    topo_arr = topo_sub["norm_elevation"].astype(np.float32)
    svf_arr  = svf_sub["SKY_VIEW_FACTOR"].astype(np.float32)
    return lat, lon, topo_arr, svf_arr


# =============================================================================
# Bounding-box coverage check
# =============================================================================

def check_bbox_coverage(
    ds_gc: xr.Dataset,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
) -> None:
    """
    Assert that the requested bounding box is fully covered by the
    GraphCast grid. Fails early with a clear message rather than
    silently producing NaN-filled output at domain edges.
    """
    gc_lat_min = float(ds_gc["latitude"].min())
    gc_lat_max = float(ds_gc["latitude"].max())
    gc_lon_min = float(ds_gc["longitude"].min())
    gc_lon_max = float(ds_gc["longitude"].max())

    log(
        f"GraphCast grid extent: "
        f"lat=[{gc_lat_min:.3f}, {gc_lat_max:.3f}]  "
        f"lon=[{gc_lon_min:.3f}, {gc_lon_max:.3f}]"
    )
    log(
        f"Requested bbox:        "
        f"lat=[{lat_min:.3f}, {lat_max:.3f}]  "
        f"lon=[{lon_min:.3f}, {lon_max:.3f}]"
    )

    errors = []
    if lat_min < gc_lat_min:
        errors.append(f"  lat_min {lat_min} < GC grid lat_min {gc_lat_min:.3f}")
    if lat_max > gc_lat_max:
        errors.append(f"  lat_max {lat_max} > GC grid lat_max {gc_lat_max:.3f}")
    if lon_min < gc_lon_min:
        errors.append(f"  lon_min {lon_min} < GC grid lon_min {gc_lon_min:.3f}")
    if lon_max > gc_lon_max:
        errors.append(f"  lon_max {lon_max} > GC grid lon_max {gc_lon_max:.3f}")

    if errors:
        raise ValueError(
            "Requested bounding box exceeds GraphCast domain:\n"
            + "\n".join(errors)
            + "\nAdjust --lat_min/lat_max/lon_min/lon_max accordingly."
        )

    log("Bounding box coverage check passed.")


# =============================================================================
# Tiling
# =============================================================================

def iter_tiles(
    nlat: int, nlon: int, patch: int, stride: int
) -> Iterable[Tuple[int, int, int, int]]:
    if patch > nlat or patch > nlon:
        raise ValueError(
            f"patch={patch} is larger than the target grid ({nlat} x {nlon}). "
            f"Reduce --patch or expand your bounding box."
        )

    i_starts = list(range(0, nlat - patch + 1, stride))
    j_starts = list(range(0, nlon - patch + 1, stride))

    if i_starts[-1] != nlat - patch:
        i_starts.append(nlat - patch)
    if j_starts[-1] != nlon - patch:
        j_starts.append(nlon - patch)

    for i0 in i_starts:
        for j0 in j_starts:
            yield i0, i0 + patch, j0, j0 + patch


# =============================================================================
# Overlap blend weights
# =============================================================================

def make_blend_weights_2d(patch: int, overlap: int) -> np.ndarray:
    """
    Cosine-taper 2D weight mask for one tile.
    Interior pixels → 1.0; within `overlap` pixels of any edge → cosine ramp.
    When two adjacent tiles' masks overlap they sum to ~1.0 → seamless blend.
    """
    if overlap <= 0:
        return np.ones((patch, patch), dtype=np.float32)

    w     = np.ones(patch, dtype=np.float32)
    t     = np.linspace(0.0, 1.0, overlap + 2, dtype=np.float32)[1:-1]
    taper = (0.5 * (1.0 - np.cos(np.pi * t))).astype(np.float32)
    w[:overlap]  = taper
    w[-overlap:] = taper[::-1]

    return np.outer(w, w).astype(np.float32)


# =============================================================================
# Model construction & loading
# =============================================================================

def create_unet(device: str) -> UNet3DConditionModel:
    return UNet3DConditionModel(
        sample_size=None,
        in_channels=TARGET_CHANNELS + COND_CHANNELS,   # 8 + 20 = 28
        out_channels=TARGET_CHANNELS,                   # 8
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=("DownBlock3D",) * 4,
        up_block_types=("UpBlock3D",)   * 4,
        norm_num_groups=8,
        cross_attention_dim=1,
        attention_head_dim=8,
    ).to(device)


def load_model(
    checkpoint_path: str,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> torch.nn.Module:
    ckpt = Path(checkpoint_path).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    log(f"Loading checkpoint: {ckpt}")
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    if "model_state_dict" not in payload:
        raise KeyError(
            f"'model_state_dict' key missing from checkpoint. "
            f"Found keys: {list(payload.keys())}"
        )

    model     = create_unet("cpu")
    raw_state = payload["model_state_dict"]
    state     = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    model.set_attn_processor(AttnProcessor())

    if use_amp and device == "cuda":
        model = model.to(dtype=amp_dtype)

    total = sum(p.numel() for p in model.parameters())
    log(f"Model ready on {device}. Total parameters: {total:,}")
    return model


# =============================================================================
# Inference (diffusion reverse process)
# =============================================================================

def infer_batch(
    X_np: np.ndarray,
    model: torch.nn.Module,
    scheduler: LCMScheduler,
    device: str,
    ttarget: int,
    patch: int,
    num_steps: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> np.ndarray:
    """
    Run diffusion denoising for one tile.

    Parameters
    ----------
    X_np : (B, C, Tcond, P, P)  conditioning input, float32
    Returns
    -------
    np.ndarray  (B, C, Ttarget, P, P)  predicted output in [0,1] normalized space
    """
    B = X_np.shape[0]
    x = torch.from_numpy(X_np).to(device=device, dtype=torch.float32)

    # Interpolate conditioning from Tcond → Ttarget along time axis
    x_match = F.interpolate(
        x, size=(ttarget, patch, patch), mode="trilinear", align_corners=False
    )

    pred   = torch.randn(
        (B, TARGET_CHANNELS, ttarget, patch, patch),
        device=device, dtype=torch.float32,
    )
    cad    = int(model.config.cross_attention_dim if hasattr(model, "config") else 1)
    enc_hs = torch.zeros((B, 1, cad), device=device, dtype=torch.float32)

    scheduler.set_timesteps(num_steps, device=device)

    t_denoise_start = time.time()
    log(f"  [denoise] starting {num_steps} denoising steps at {ts()}")

    with torch.inference_mode():
        for step_i, t in enumerate(
            tqdm(
                scheduler.timesteps,
                desc=f"  [{ts()}] denoise",
                leave=False,
                unit="step",
            ),
            start=1,
        ):
            net_in = torch.cat([pred, x_match], dim=1)

            sdp_ctx = contextlib.nullcontext()
            if device == "cuda":
                sdp_ctx = torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_mem_efficient=False, enable_math=True
                )

            with sdp_ctx:
                if use_amp and device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        out = model(
                            sample=net_in, timestep=t, encoder_hidden_states=enc_hs
                        )
                else:
                    out = model(
                        sample=net_in, timestep=t, encoder_hidden_states=enc_hs
                    )

            residual = out.sample if hasattr(out, "sample") else out[0]
            pred     = scheduler.step(residual.float(), t, pred).prev_sample

            # Log every 25% of denoising steps
            if step_i % max(1, num_steps // 4) == 0 or step_i == num_steps:
                elapsed = time.time() - t_denoise_start
                log(
                    f"  [denoise] step {step_i}/{num_steps}  "
                    f"elapsed={elapsed:.1f}s  ts={ts()}"
                )

    total_denoise = time.time() - t_denoise_start
    log(f"  [denoise] completed {num_steps} steps in {total_denoise:.2f}s  ({ts()})")

    return pred.detach().cpu().numpy().astype(np.float32)


# =============================================================================
# Zarr output store
# =============================================================================

def init_output_store(
    out_zarr: str,
    time_ic: np.ndarray,
    lead_hours: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    chunk_patch: int,
    overwrite: bool,
) -> None:
    if overwrite and os.path.exists(out_zarr):
        log(f"Removing existing output: {out_zarr}")
        shutil.rmtree(out_zarr)

    root        = zarr.open_group(out_zarr, mode="w")
    compressors = make_zarr_v3_compressors()
    _NS_PER_S   = np.int64(1_000_000_000)

    t_ic_s  = time_ic.astype("datetime64[ns]").astype("int64") // _NS_PER_S
    arr_tic = root.create_array("time_ic", data=t_ic_s, overwrite=True)
    arr_tic.attrs["units"]    = "seconds since 1970-01-01 00:00:00"
    arr_tic.attrs["calendar"] = "proleptic_gregorian"

    root.create_array("lead_time",  data=lead_hours.astype(np.int32), overwrite=True)
    root.create_array("latitude",   data=lat.astype(np.float32),      overwrite=True)
    root.create_array("longitude",  data=lon.astype(np.float32),      overwrite=True)

    arr_vt = root.create_array(
        "valid_time",
        shape=(time_ic.size, lead_hours.size),
        chunks=(1, lead_hours.size),
        dtype="i8",
        compressors=compressors,
        overwrite=True,
    )
    arr_vt.attrs["units"]    = "seconds since 1970-01-01 00:00:00"
    arr_vt.attrs["calendar"] = "proleptic_gregorian"

    shape  = (time_ic.size, lead_hours.size, lat.size, lon.size)
    chunks = (1, lead_hours.size, chunk_patch, chunk_patch)

    for v in AORC_VARS:
        root.create_array(
            v, shape=shape, chunks=chunks, dtype="f4",
            compressors=compressors, overwrite=True,
        )

    log(f"Initialized output Zarr v3: {out_zarr}  shape={shape}  chunks={chunks}")


def write_valid_time(
    root: zarr.Group, ic_idx: int, valid_time: np.ndarray
) -> None:
    root["valid_time"][ic_idx, :] = (
        valid_time.astype("datetime64[ns]").astype("int64")
        // np.int64(1_000_000_000)
    )


# =============================================================================
# Output summary
# =============================================================================

def _print_zarr_summary(out_zarr: str) -> None:
    root = zarr.open_group(out_zarr, mode="r")
    sep  = "─" * 72

    tqdm.write(f"\n{sep}")
    tqdm.write(f"  Output Zarr Summary: {out_zarr}")
    tqdm.write(sep)

    coord_keys = ["time_ic", "lead_time", "latitude", "longitude", "valid_time"]
    tqdm.write("  Coordinates:")
    for key in coord_keys:
        if key not in root:
            continue
        arr   = root[key]
        units = dict(arr.attrs).get("units", "")
        tqdm.write(
            f"    {key:<20s}  shape={str(arr.shape):<20s}  dtype={str(arr.dtype):<8s}"
            + (f"  units='{units}'" if units else "")
        )

    tqdm.write("  Data variables:")
    total_bytes = 0
    for v in AORC_VARS:
        if v not in root:
            continue
        arr    = root[v]
        nbytes = arr.nbytes
        total_bytes += nbytes
        tqdm.write(
            f"    {v:<28s}  shape={str(arr.shape):<28s}  "
            f"dtype={str(arr.dtype):<8s}  size={nbytes / 1e6:>8.2f} MB"
        )

    tqdm.write(f"  {'─'*68}")
    tqdm.write(
        f"  Total uncompressed data: {total_bytes / 1e9:.3f} GB  |  vars={len(AORC_VARS)}"
    )

    first_var = AORC_VARS[0]
    if first_var in root and root[first_var].shape[0] > 0:
        sample = root[first_var][0]
        finite = sample[np.isfinite(sample)]
        if finite.size > 0:
            tqdm.write(
                f"  Sample check [{first_var}, IC=0]:  "
                f"min={finite.min():.4f}  max={finite.max():.4f}  "
                f"mean={finite.mean():.4f}  "
                f"nan%={100*(1 - finite.size / sample.size):.2f}%"
            )
    tqdm.write(sep + "\n")


# =============================================================================
# Main
# =============================================================================

def run(args: argparse.Namespace) -> None:

    run_start = time.time()
    log(f"{'='*72}")
    log(f"Earthmind inference started  |  {ts()}")
    log(f"{'='*72}")

    # ------------------------------------------------------------------ device
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp   = device == "cuda"
    amp_dtype = torch.bfloat16
    if use_amp and not torch.cuda.is_bf16_supported():
        amp_dtype = torch.float16
    log(f"Device={device}  AMP={use_amp}  dtype={amp_dtype if use_amp else 'fp32'}")

    ensure_dir(args.out_dir)

    # -------------------------------------------------------- static grids
    log(f"[{ts()}] Loading static grids (topo + svf)…")
    topo, svf = load_static_grids(args.topo_nc, args.svf_nc)
    lat, lon, topo_arr, svf_arr = subset_static_to_bbox(
        topo, svf,
        args.lat_min, args.lat_max,
        args.lon_min, args.lon_max,
    )
    nlat, nlon = lat.size, lon.size
    log(
        f"Target grid: nlat={nlat} nlon={nlon}  "
        f"lat=[{lat.min():.3f}, {lat.max():.3f}]  "
        f"lon=[{lon.min():.3f}, {lon.max():.3f}]"
    )

    # -------------------------------------------------------- GraphCast
    log(f"[{ts()}] Loading GraphCast data…")
    if args.gcs_zarr_root:
        log(f"Using GCS-style GraphCast data from: {args.gcs_zarr_root}")
        ds_gc = open_gcs_graphcast(
            args.gcs_zarr_root, args.gc_min_zarr, args.gc_max_zarr,
            args.start_date, args.end_date,
        )
    else:
        ds_gc = open_and_prepare_graphcast(
            args.graphcast_zarr, args.gc_min_zarr, args.gc_max_zarr,
            args.start_date, args.end_date,
        )
    log(f"[{ts()}] GraphCast data loaded.")

    # ------------------------------------------------ print input structure
    print("\n" + "=" * 72)
    print("  GraphCast input dataset structure (after normalization & coord fix):")
    print("=" * 72)
    print(ds_gc)
    print("=" * 72 + "\n")

    # ----------------------------------------- bbox coverage check
    check_bbox_coverage(ds_gc, args.lat_min, args.lat_max, args.lon_min, args.lon_max)

    # ---------------------------------------------------- time axis
    all_times = np.sort(ds_gc["time"].values.astype("datetime64[ns]"))
    if all_times.size == 0:
        raise RuntimeError(
            "No initialization times found in the requested date range "
            f"({args.start_date} → {args.end_date}). Check your GraphCast data."
        )

    chosen = np.sort(all_times[: args.num_inits])
    log(
        f"Init times (ICs) selected: {chosen.size}  "
        f"first={chosen[0]}  last={chosen[-1]}"
    )

    # ------------------------------------------------ assert Tcond == 12
    cond_offsets = ds_gc["prediction_timedelta"].values.astype("timedelta64[ns]")
    tcond        = cond_offsets.size
    assert tcond == TCOND, (
        f"GraphCast zarr has {tcond} prediction_timedelta steps "
        f"but model was trained with TCOND={TCOND}. "
        f"Ensure you are using the correct GraphCast data."
    )
    log(f"Tcond assertion passed: prediction_timedelta steps = {tcond}")

    # ------------------------------------------------ assert ttarget == 67
    assert args.ttarget == TTARGET, (
        f"--ttarget={args.ttarget} but model was trained with TTARGET={TTARGET}. "
        f"Use --ttarget {TTARGET}."
    )
    log(f"Ttarget assertion passed: output lead hours = {args.ttarget}")

    lead_hours = np.arange(args.ttarget, dtype=np.int32)
    lead_td    = lead_hours.astype("timedelta64[h]")

    # ------------------------------------------------ AORC inverse scale
    aorc_min_ds = xr.open_dataset(args.aorc_min_nc)
    aorc_max_ds = xr.open_dataset(args.aorc_max_nc)
    aorc_min    = {v: float(aorc_min_ds[v].values) for v in AORC_VARS}
    aorc_max    = {v: float(aorc_max_ds[v].values) for v in AORC_VARS}

    # ------------------------------------------------ model + scheduler
    log(f"[{ts()}] Loading model checkpoint…")
    model     = load_model(args.checkpoint, device, use_amp, amp_dtype)
    scheduler = LCMScheduler(num_train_timesteps=1000)
    log(f"[{ts()}] Model and scheduler ready.")

    # ------------------------------------------------ tiling
    tiles = list(iter_tiles(nlat, nlon, args.patch, args.stride))
    log(f"Tiling: patch={args.patch}  stride={args.stride}  total_tiles={len(tiles)}")

    overlap   = args.patch - args.stride
    blend_w2d = make_blend_weights_2d(args.patch, max(overlap, 0))
    log(
        f"Blend weights: overlap={overlap}px  taper=cosine  "
        f"w_min={blend_w2d.min():.4f}  w_max={blend_w2d.max():.4f}"
    )

    step_list = [int(s) for s in args.steps]
    log(f"Diffusion step configs to run: {step_list}")

    # ==============================================================
    # Outer loop: diffusion step configs
    # ==============================================================
    for steps in tqdm(step_list, desc="diffusion-steps", unit="cfg"):

        cfg_start = time.time()
        log(f"\n{'─'*72}")
        log(f"[{ts()}] Starting diffusion config: steps={steps}")
        log(f"{'─'*72}")

        out_zarr = str(Path(args.out_dir) / f"{args.out_name}_steps{steps:02d}.zarr")

        init_output_store(
            out_zarr=out_zarr,
            time_ic=chosen,
            lead_hours=lead_hours,
            lat=lat,
            lon=lon,
            chunk_patch=args.patch,
            overwrite=args.overwrite,
        )

        root = zarr.open_group(out_zarr, mode="a")

        ic_bar = tqdm(
            enumerate(chosen), total=chosen.size,
            desc="init-times", unit="IC",
        )

        # ============================================================
        # IC loop
        # ============================================================
        for ic_idx, t0 in ic_bar:
            ic_start = time.time()
            ic_bar.set_postfix({"t0": str(t0)[:16], "wall": ts()})
            log(
                f"\n[{ts()}] [steps={steps}] ── IC {ic_idx+1}/{chosen.size}: {t0} ──"
            )

            valid_time_1d = (t0 + lead_td).astype("datetime64[ns]")
            write_valid_time(root, ic_idx, valid_time_1d)

            # cos_sza valid times: shape (Tcond,) — matches training exactly
            vt_cond_1d = (t0 + cond_offsets).astype("datetime64[ns]")

            # Accumulate blended predictions into zarr; weight map in RAM
            accum_weight = np.zeros((nlat, nlon), dtype=np.float32)

            # Zero out this IC's zarr slices before accumulation
            zero_chunk = np.zeros(
                (args.ttarget, args.patch, args.patch), dtype=np.float32
            )
            for (i0, i1, j0, j1) in tiles:
                for v in AORC_VARS:
                    root[v][ic_idx, :, i0:i1, j0:j1] = zero_chunk
            log(f"  [{ts()}] Zeroed output zarr for IC {ic_idx+1}")

            # ========================================================
            # Tile loop
            # ========================================================
            tile_bar = tqdm(
                enumerate(tiles, start=1), total=len(tiles),
                desc="  tiles", unit="tile", leave=False,
            )

            for tile_i, (i0, i1, j0, j1) in tile_bar:
                tile_start = time.time()
                tile_bar.set_postfix({
                    "i": f"{i0}:{i1}",
                    "j": f"{j0}:{j1}",
                    "wall": ts(),
                })

                lat_tile = lat[i0:i1]   # (P,)
                lon_tile = lon[j0:j1]   # (P,)

                # --------------------------------------------------
                # Build input X exactly as in training:
                #   each channel → (Tcond, P, P)
                #   stack → (C=20, Tcond, P, P) → add batch → (1, 20, Tcond, P, P)
                # --------------------------------------------------
                X_list: List[np.ndarray] = []

                # 3D GraphCast vars: 17 channels, each (Tcond, P, P)
                with dask.config.set(scheduler="threads"):
                    ds_tile = (
                        ds_gc[VARS_3D]
                        .sel(time=[t0])
                        .interp(
                            latitude=("latitude",   lat_tile),
                            longitude=("longitude", lon_tile),
                            method="linear",
                        )
                        .transpose(
                            "time", "prediction_timedelta", "latitude", "longitude"
                        )
                        .load()
                    )

                for v in VARS_3D:
                    arr = ds_tile[v].values[0].astype(np.float32)  # (Tcond, P, P)
                    if v in PRECIP_GC_VARS:
                        arr = np.clip(arr, 0.0, None)
                    X_list.append(arr)

                # topo: broadcast (P,P) → (Tcond, P, P)
                topo_tile = (
                    topo_arr
                    .isel(latitude=slice(i0, i1), longitude=slice(j0, j1))
                    .values.astype(np.float32)
                )
                X_list.append(
                    np.broadcast_to(
                        topo_tile[None, :, :], (TCOND, args.patch, args.patch)
                    ).copy()
                )

                # svf: broadcast (P,P) → (Tcond, P, P)
                svf_tile = (
                    svf_arr
                    .isel(latitude=slice(i0, i1), longitude=slice(j0, j1))
                    .values.astype(np.float32)
                )
                X_list.append(
                    np.broadcast_to(
                        svf_tile[None, :, :], (TCOND, args.patch, args.patch)
                    ).copy()
                )

                # cos_sza: (Tcond, P, P) — vt_cond_1d is (Tcond,) no batch dim
                cos_sza = cos_sza_for_tile(vt_cond_1d, lat_tile, lon_tile)
                X_list.append(cos_sza)

                # Stack → (C=20, Tcond, P, P) → add batch → (1, 20, Tcond, P, P)
                X_np = np.stack(X_list, axis=0)[None].astype(np.float32)
                assert X_np.shape[1] == COND_CHANNELS, (
                    f"X channel count mismatch: got {X_np.shape[1]}, "
                    f"expected {COND_CHANNELS}"
                )

                # --------------------------------------------------
                # Diffusion reverse process (denoising)
                # --------------------------------------------------
                y_norm = infer_batch(
                    X_np=X_np,
                    model=model,
                    scheduler=scheduler,
                    device=device,
                    ttarget=args.ttarget,
                    patch=args.patch,
                    num_steps=steps,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                )  # (1, 8, Ttarget, P, P)

                # --------------------------------------------------
                # Inverse scale → physical units, then blend.
                # Blending is in physical space (after inverse scaling)
                # because precip uses expm1 (nonlinear); blending in
                # log-space then inverting would give wrong results.
                # --------------------------------------------------
                y_phys = inverse_scale(y_norm, aorc_min, aorc_max)[0]
                # y_phys: (8, Ttarget, P, P)

                for k, v in enumerate(AORC_VARS):
                    current = root[v][ic_idx, :, i0:i1, j0:j1]  # (T, P, P)
                    root[v][ic_idx, :, i0:i1, j0:j1] = (
                        current + y_phys[k] * blend_w2d[np.newaxis, :, :]
                    )
                accum_weight[i0:i1, j0:j1] += blend_w2d

                tile_elapsed = time.time() - tile_start
                if (tile_i % args.log_every_tiles) == 0:
                    log(
                        f"  [{ts()}] tile {tile_i}/{len(tiles)}  "
                        f"(i={i0}:{i1}, j={j0}:{j1})  "
                        f"tile_time={tile_elapsed:.2f}s"
                    )

            # ---- normalize by accumulated blend weights  ----
            accum_weight = np.maximum(accum_weight, 1e-12)
            strip_size   = 256
            for row_start in range(0, nlat, strip_size):
                row_end  = min(row_start + strip_size, nlat)
                w_strip  = accum_weight[row_start:row_end, :]        # (strip, nlon)
                w_strip4 = w_strip[np.newaxis, :, :]                 # (1, strip, nlon)
                for v in AORC_VARS:
                    chunk = root[v][ic_idx, :, row_start:row_end, :] # (T, strip, nlon)
                    root[v][ic_idx, :, row_start:row_end, :] = (
                        chunk / w_strip4
                    ).astype(np.float32)

            del accum_weight

            ic_elapsed = time.time() - ic_start
            log(
                f"  [{ts()}] IC {ic_idx+1}/{chosen.size} complete  "
                f"elapsed={ic_elapsed:.1f}s  ({ic_elapsed/60:.1f} min)"
            )

        zarr.consolidate_metadata(out_zarr)

        cfg_elapsed = time.time() - cfg_start
        log(
            f"\n[{ts()}] [DONE] steps={steps}  output={out_zarr}  "
            f"total_time={cfg_elapsed:.1f}s  ({cfg_elapsed/60:.1f} min)"
        )
        _print_zarr_summary(out_zarr)

    total_elapsed = time.time() - run_start
    log(f"\n{'='*72}")
    log(
        f"All diffusion configs complete.  "
        f"Total wall time: {total_elapsed:.1f}s  ({total_elapsed/60:.1f} min)"
    )
    log(f"{'='*72}\n")


# =============================================================================
# Argument parser
# =============================================================================

def parse_args() -> argparse.Namespace:

    class _Fmt(
        argparse.RawDescriptionHelpFormatter,
        argparse.ArgumentDefaultsHelpFormatter,
    ):
        pass

    p = argparse.ArgumentParser(
        description=(
            "Earthmind Downscaling Inference\n"
            "-------------------------------------------------------\n"
            "Authors  : Somnath Luitel & Manmeet Singh\n"
            "Lab      : AI Research Lab, Earth, Environment and\n"
            "           Atmospheric Sciences, WKU, Bowling Green, KY, USA\n"
        ),
        formatter_class=_Fmt,
    )

    # GraphCast source — mutually exclusive
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--graphcast_zarr",
        default=None,
        help="Path to a single local GraphCast zarr file.",
    )
    src.add_argument(
        "--gcs_zarr_root",
        default=None,
        help=(
            "Root directory of GCS-style GraphCast data "
            "(subdirs named YYYY_to_YYYY or YYYY_to_present)."
        ),
    )

    # Normalization files
    p.add_argument("--gc_min_zarr", required=True, help="GraphCast min-values zarr.")
    p.add_argument("--gc_max_zarr", required=True, help="GraphCast max-values zarr.")
    p.add_argument("--aorc_min_nc", required=True, help="AORC min-values .nc (for inverse scaling).")
    p.add_argument("--aorc_max_nc", required=True, help="AORC max-values .nc (for inverse scaling).")

    # Static grids
    p.add_argument("--topo_nc", required=True, help="Topography .nc file (var: norm_elevation).")
    p.add_argument("--svf_nc",  required=True, help="Sky-view-factor .nc file (var: SKY_VIEW_FACTOR).")

    # Model
    p.add_argument("--checkpoint", required=True, help="Path to trained .pt checkpoint file.")

    # Time range
    p.add_argument("--start_date", required=True, help="Start date e.g. 2021-07-04.")
    p.add_argument("--end_date",   required=True, help="End date   e.g. 2021-12-31.")
    p.add_argument(
        "--num_inits", type=int, default=3,
        help=(
            "Number of GraphCast initialization times (ICs) to run. "
            "Each IC is one timestamp on the GraphCast 'time' axis "
            "(e.g. 2021-07-04T00:00, 2021-07-04T06:00, …). "
            "The first --num_inits chronological ICs are used."
        ),
    )

    # Bounding box — must be within GraphCast domain; coverage is checked at runtime
    p.add_argument("--lat_min", type=float, required=True, help="Minimum latitude  of target domain.")
    p.add_argument("--lat_max", type=float, required=True, help="Maximum latitude  of target domain.")
    p.add_argument("--lon_min", type=float, required=True, help="Minimum longitude of target domain.")
    p.add_argument("--lon_max", type=float, required=True, help="Maximum longitude of target domain.")

    # Tiling & model config
    p.add_argument("--patch",  type=int, default=256, help="Tile (patch) size in pixels.")
    p.add_argument(
        "--stride", type=int, default=128,
        help="Stride between tiles. overlap = patch - stride. Default gives 128px overlap.",
    )
    p.add_argument(
        "--ttarget", type=int, default=TTARGET,
        help=f"Number of hourly output lead times. Must be {TTARGET} (architectural constraint).",
    )
    p.add_argument(
        "--steps", nargs="+", default=["25"],
        help=(
            "Diffusion denoising steps to run. "
            "One output zarr produced per value. "
            "Paper runs: 4 8 25 50.  Default: 25."
        ),
    )

    # Output
    p.add_argument("--out_dir",         required=True,              help="Output directory.")
    p.add_argument("--out_name",        default="conus_inference",  help="Base name for output zarr files.")
    p.add_argument("--overwrite",       action="store_true",        help="Overwrite existing output.")
    p.add_argument("--log_every_tiles", type=int, default=25,       help="Log progress every N tiles.")

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())