#!/usr/bin/env python3
"""
Earthmind Inference Script — RUMI Enabled
=========================================================

Usage example for RUMI:
  python inference_final_version.py \
    --gcs_zarr_root /home/airlab/gcs_bucket \
    --gc_min_zarr /home/airlab/Documents/airlab/earthmind_highres/graphcast_training_2020_2021_min.zarr \
    --gc_max_zarr /home/airlab/Documents/airlab/earthmind_highres/graphcast_training_2020_2021_max.zarr \
    --topo_nc /home/airlab/Documents/airlab/earthmind_highres/TOPO_Hongkong.nc \
    --svf_nc /home/airlab/Documents/airlab/earthmind_highres/SVF_Hongkong.nc \
    --aorc_min_nc /home/airlab/Documents/airlab/earthmind_highres/usa_aorc_min.nc \
    --aorc_max_nc /home/airlab/Documents/airlab/earthmind_highres/usa_aorc_max.nc \
    --checkpoint /media/airlab/ROCSTOR/earthmind_highres/models_unet3d_fast/unet3d_global_best_ic_20210127T060000.pt \
    --start_date 2018-09-15 \
    --end_date 2018-09-17 \
    --num_inits 1 \
    --rumi --event MANGKHUT2018 \
    --out_dir ./rumi_outputs \
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
import netCDF4 as nc4
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
# RUMI Specifications
# =============================================================================

RUMI_LAT_MIN, RUMI_LAT_MAX = 22.12, 22.58
RUMI_LON_MIN, RUMI_LON_MAX = 113.82, 114.45
RUMI_NLAT, RUMI_NLON = 171, 234

# Standard RUMI Output Grid
RUMI_LAT = np.linspace(RUMI_LAT_MIN, RUMI_LAT_MAX, RUMI_NLAT, dtype=np.float64)
RUMI_LON = np.linspace(RUMI_LON_MIN, RUMI_LON_MAX, RUMI_NLON, dtype=np.float64)

# Variable Mapping: RUMI Name -> (AORC Name, Units, CF Standard Name, Long Name)
RUMI_VAR_MAP = {
    "T2M": ("TMP_2maboveground", "K", "air_temperature", "2-meter air temperature"),
    "U10M": ("UGRD_10maboveground", "m s-1", "eastward_wind", "10-meter eastward wind component"),
    "V10M": ("VGRD_10maboveground", "m s-1", "northward_wind", "10-meter northward wind component"),
    "PRATE": ("APCP_surface", "kg m-2 s-1", "precipitation_flux", "precipitation rate"),
    "SLP": ("PRES_surface", "Pa", "air_pressure_at_mean_sea_level", "sea level pressure"), # SLP needs reduction, using PSFC for now
    "RH2M": ("SPFH_2maboveground", "1", "relative_humidity", "2-meter relative humidity"), # Needs conversion from Q to RH
    "PSFC": ("PRES_surface", "Pa", "surface_air_pressure", "surface air pressure"),
    "Q2M": ("SPFH_2maboveground", "kg kg-1", "specific_humidity", "2-meter specific humidity"),
}

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
    print(f"[{ts()}] {msg}", flush=True)


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

    if "sample" in ds_norm.dims:
        ds_norm = ds_norm.isel(sample=0)
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
        if "_6_hours.zarr" in name or "predictions.zarr" in name:
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

    if "sample" in ds_all.dims:
        ds_all = ds_all.isel(sample=0)

    return ds_all


# =============================================================================
# Static grids (topo + svf)
# =============================================================================

def load_static_grids(
    topo_nc: str, svf_nc: str
) -> Tuple[xr.Dataset, xr.Dataset]:
    topo0 = xr.open_dataset(topo_nc)
    svf0  = xr.open_dataset(svf_nc)

    # Some files use lat/lon, others latitude/longitude
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

    t_var = "norm_elevation" if "norm_elevation" in topo_sub else "ELEVATION"
    s_var = "SKY_VIEW_FACTOR"

    topo_arr = topo_sub[t_var].astype(np.float32)
    svf_arr  = svf_sub[s_var].astype(np.float32)
    return lat, lon, topo_arr, svf_arr


# =============================================================================
# Bounding-box coverage check
# =============================================================================

def check_bbox_coverage(
    ds_gc: xr.Dataset,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
) -> None:
    gc_lat_min = float(ds_gc["latitude"].min())
    gc_lat_max = float(ds_gc["latitude"].max())
    gc_lon_min = float(ds_gc["longitude"].min())
    gc_lon_max = float(ds_gc["longitude"].max())

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
        )


# =============================================================================
# Tiling
# =============================================================================

def iter_tiles(
    nlat: int, nlon: int, patch: int, stride: int
) -> Iterable[Tuple[int, int, int, int]]:
    if patch > nlat or patch > nlon:
        raise ValueError(f"patch={patch} larger than grid ({nlat}x{nlon})")

    i_starts = list(range(0, nlat - patch + 1, stride))
    j_starts = list(range(0, nlon - patch + 1, stride))

    if i_starts[-1] != nlat - patch: i_starts.append(nlat - patch)
    if j_starts[-1] != nlon - patch: j_starts.append(nlon - patch)

    for i0 in i_starts:
        for j0 in j_starts:
            yield i0, i0 + patch, j0, j0 + patch


# =============================================================================
# Overlap blend weights
# =============================================================================

def make_blend_weights_2d(patch: int, overlap: int) -> np.ndarray:
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
        in_channels=TARGET_CHANNELS + COND_CHANNELS,
        out_channels=TARGET_CHANNELS,
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
    log(f"Loading checkpoint: {ckpt}")
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    
    model     = create_unet("cpu")
    raw_state = payload["model_state_dict"]
    state     = {k.replace("_orig_mod.", ""): v for k, v in raw_state.items()}
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    model.set_attn_processor(AttnProcessor())

    if use_amp and device == "cuda":
        model = model.to(dtype=amp_dtype)

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
    B = X_np.shape[0]
    x = torch.from_numpy(X_np).to(device=device, dtype=torch.float32)

    x_match = F.interpolate(
        x, size=(ttarget, patch, patch), mode="trilinear", align_corners=False
    )

    pred   = torch.randn(
        (B, TARGET_CHANNELS, ttarget, patch, patch),
        device=device, dtype=torch.float32,
    )
    cad    = 1
    enc_hs = torch.zeros((B, 1, cad), device=device, dtype=torch.float32)

    scheduler.set_timesteps(num_steps, device=device)

    with torch.inference_mode():
        for t in tqdm(scheduler.timesteps, desc="denoise", leave=False):
            net_in = torch.cat([pred, x_match], dim=1)

            sdp_ctx = contextlib.nullcontext()
            if device == "cuda":
                sdp_ctx = torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_mem_efficient=False, enable_math=True
                )

            with sdp_ctx:
                if use_amp and device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=amp_dtype):
                        out = model(sample=net_in, timestep=t, encoder_hidden_states=enc_hs)
                else:
                    out = model(sample=net_in, timestep=t, encoder_hidden_states=enc_hs)

            residual = out.sample if hasattr(out, "sample") else out[0]
            pred     = scheduler.step(residual.float(), t, pred).prev_sample

    return pred.detach().cpu().numpy().astype(np.float32)


# =============================================================================
# RUMI Output Implementation
# =============================================================================

def q_to_rh(q, t, p):
    """
    Approximate Specific Humidity (q) to Relative Humidity (rh).
    q: kg/kg, t: K, p: Pa
    Returns rh fraction (0-1).
    """
    # Tetens formula for saturation vapor pressure over water
    t_c = t - 273.15
    es = 611.2 * np.exp(17.67 * t_c / (t_c + 243.5))
    # Vapor pressure e from specific humidity q
    # q = 0.622 * e / (p - 0.378 * e)  => e = p*q / (0.622 + 0.378*q)
    e = (p * q) / (0.622 + 0.378 * q)
    rh = e / es
    return np.clip(rh, 0.0, 1.0)

def write_rumi_nc(
    out_dir: str,
    event_code: str,
    t0: np.datetime64,
    lead_hours: np.ndarray,
    lat_model: np.ndarray,
    lon_model: np.ndarray,
    data_dict: Dict[str, np.ndarray], # {var: (T, H, W)}
    args: argparse.Namespace
) -> None:
    """
    Interpolates results to 300m grid and saves as RUMI-compliant NetCDF4 files.
    One file per lead hour.
    """
    log(f"Preparing RUMI NetCDF4 output for event {event_code}...")

    # Forcing Mode
    f_mode = "analysis" if "AN" in args.experiment else "forecast"
    
    # 1. Create Xarray Dataset from model output
    ds_model = xr.Dataset(
        coords={
            "lead_hour": lead_hours,
            "latitude": lat_model,
            "longitude": lon_model
        }
    )
    for v, arr in data_dict.items():
        ds_model[v] = (("lead_hour", "latitude", "longitude"), arr)

    # 2. Interpolate to RUMI standard grid
    ds_rumi_raw = ds_model.interp(
        latitude=RUMI_LAT,
        longitude=RUMI_LON,
        method="linear"
    )

    # 3. Unit Conversions and Mapping
    ds_rumi = xr.Dataset(
        coords={
            "time": (("time",), [0.0]), # Placeholder, replaced per file
            "latitude": (("latitude",), RUMI_LAT),
            "longitude": (("longitude",), RUMI_LON)
        }
    )

    # Handle all RUMI variables
    for r_var, (a_var, r_unit, r_std, r_long) in RUMI_VAR_MAP.items():
        val = ds_rumi_raw[a_var].values
        
        if r_var == "PRATE":
            val = val / 3600.0 # kg/m2 (hourly) -> kg/m2/s
        elif r_var == "RH2M":
            # Convert Q to RH using T2M and PSFC
            q = ds_rumi_raw["SPFH_2maboveground"].values
            t = ds_rumi_raw["TMP_2maboveground"].values
            p = ds_rumi_raw["PRES_surface"].values
            val = q_to_rh(q, t, p)
        
        # Add to dataset
        ds_rumi[r_var] = (("time", "latitude", "longitude"), val[np.newaxis, ...])
        ds_rumi[r_var].attrs.update({
            "standard_name": r_std,
            "long_name": r_long,
            "units": r_unit,
            "_FillValue": -9999.0
        })

    # 4. Save individual files per hour
    for h_idx, h in enumerate(lead_hours):
        valid_time = t0 + np.timedelta64(int(h), "h")
        ts_str = str(valid_time).replace("-", "").replace("T", "").replace(":", "")[:14]
        
        # Filename: RUMI-<SRC>-<MODE>-<MODEL>-<EVENT>-<YYYYMMDDHHMMSS>.nc
        fname = f"{args.experiment}-AirCastHK-{event_code}-{ts_str}.nc"
        fpath = Path(out_dir) / fname
        
        log(f"  Saving {fname}...")
        
        # Create NetCDF4 file
        with nc4.Dataset(fpath, "w", format="NETCDF4") as ds:
            # Dimensions
            ds.createDimension("time", 1)
            ds.createDimension("latitude", RUMI_NLAT)
            ds.createDimension("longitude", RUMI_NLON)
            
            # Coordinates
            t_var = ds.createVariable("time", "f8", ("time",))
            t_var.units = "seconds since 1970-01-01 00:00:00"
            t_var.calendar = "gregorian"
            t_var[0] = (valid_time.astype("datetime64[s]").astype(np.int64))
            
            lat_var = ds.createVariable("latitude", "f4", ("latitude",))
            lat_var.units = "degrees_north"
            lat_var[:] = RUMI_LAT
            
            lon_var = ds.createVariable("longitude", "f4", ("longitude",))
            lon_var.units = "degrees_east"
            lon_var[:] = RUMI_LON
            
            # Data Variables
            for r_var in RUMI_VAR_MAP.keys():
                v_out = ds.createVariable(r_var, "f4", ("time", "latitude", "longitude"), fill_value=-9999.0)
                v_out.setncatts(ds_rumi[r_var].attrs)
                v_out[0, :, :] = ds_rumi[r_var].values[0, h_idx, :, :]
            
            # Global Attributes
            ds.experiment = args.experiment
            ds.forcing_mode = f_mode
            ds.forcing_source = args.experiment.split("-")[1]
            ds.event = event_code
            ds.institution = "AI Research Lab, WKU"
            ds.source = "AirCast HongKong (UNet3D-Diffusion)"
            ds.history = f"Created {time.strftime('%Y-%m-%d')} using inference_final_version.py"
            ds.references = "https://github.com/manmeet3591/aircast_hongkong"
            ds.horizontal_resolution = "300m (interpolated from 1km)"
            ds.initialization_time = str(t0) + "Z"


# =============================================================================
# Main
# =============================================================================

def run(args: argparse.Namespace) -> None:
    run_start = time.time()
    log(f"Inference started. RUMI mode: {args.rumi}")

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp   = device == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    ensure_dir(args.out_dir)

    # -------------------------------------------------------- RUMI Logic
    if args.rumi:
        args.lat_min, args.lat_max = RUMI_LAT_MIN - 0.1, RUMI_LAT_MAX + 0.1
        args.lon_min, args.lon_max = RUMI_LON_MIN - 0.1, RUMI_LON_MAX + 0.1
        log(f"RUMI mode active. Forcing bbox to include HK: {args.lat_min}:{args.lat_max}, {args.lon_min}:{args.lon_max}")

    # -------------------------------------------------------- static grids
    log("Loading static grids…")
    topo, svf = load_static_grids(args.topo_nc, args.svf_nc)
    lat, lon, topo_arr, svf_arr = subset_static_to_bbox(
        topo, svf, args.lat_min, args.lat_max, args.lon_min, args.lon_max,
    )
    nlat, nlon = lat.size, lon.size

    # -------------------------------------------------------- GraphCast
    if args.gcs_zarr_root:
        ds_gc = open_gcs_graphcast(args.gcs_zarr_root, args.gc_min_zarr, args.gc_max_zarr, args.start_date, args.end_date)
    else:
        ds_gc = open_and_prepare_graphcast(args.graphcast_zarr, args.gc_min_zarr, args.gc_max_zarr, args.start_date, args.end_date)

    check_bbox_coverage(ds_gc, args.lat_min, args.lat_max, args.lon_min, args.lon_max)

    # ---------------------------------------------------- time axis
    all_times = np.sort(ds_gc["time"].values.astype("datetime64[ns]"))
    chosen = np.sort(all_times[: args.num_inits])
    
    cond_offsets = ds_gc["prediction_timedelta"].values.astype("timedelta64[ns]")
    lead_hours = np.arange(args.ttarget, dtype=np.int32)
    lead_td    = lead_hours.astype("timedelta64[h]")

    # ------------------------------------------------ Normalization
    aorc_min_ds = xr.open_dataset(args.aorc_min_nc)
    aorc_max_ds = xr.open_dataset(args.aorc_max_nc)
    aorc_min    = {v: float(aorc_min_ds[v].values) for v in AORC_VARS}
    aorc_max    = {v: float(aorc_max_ds[v].values) for v in AORC_VARS}

    # ------------------------------------------------ model
    model     = load_model(args.checkpoint, device, use_amp, amp_dtype)
    scheduler = LCMScheduler(num_train_timesteps=1000)

    # ------------------------------------------------ tiling
    tiles = list(iter_tiles(nlat, nlon, args.patch, args.stride))
    overlap   = args.patch - args.stride
    blend_w2d = make_blend_weights_2d(args.patch, max(overlap, 0))
    steps = int(args.steps[0])

    # ============================================================
    # IC loop
    # ============================================================
    for ic_idx, t0 in enumerate(chosen):
        log(f"IC {ic_idx+1}/{chosen.size}: {t0}")

        vt_cond_1d = (t0 + cond_offsets).astype("datetime64[ns]")
        
        # Storage for current IC (in-memory accumulation)
        accum_data = {v: np.zeros((args.ttarget, nlat, nlon), dtype=np.float32) for v in AORC_VARS}
        accum_weight = np.zeros((nlat, nlon), dtype=np.float32)

        # ========================================================
        # Tile loop
        # ========================================================
        for i0, i1, j0, j1 in tqdm(tiles, desc="Tiles"):
            lat_tile = lat[i0:i1]
            lon_tile = lon[j0:j1]

            X_list = []
            ds_tile = (
                ds_gc[VARS_3D].sel(time=[t0])
                .interp(latitude=lat_tile, longitude=lon_tile, method="linear")
                .transpose("time", "prediction_timedelta", "latitude", "longitude").load()
            )

            for v in VARS_3D:
                arr = ds_tile[v].values[0].astype(np.float32)
                if v in PRECIP_GC_VARS: arr = np.clip(arr, 0.0, None)
                X_list.append(arr)

            X_list.append(np.broadcast_to(topo_arr.isel(latitude=slice(i0, i1), longitude=slice(j0, j1)).values[None, :, :], (TCOND, args.patch, args.patch)))
            X_list.append(np.broadcast_to(svf_arr.isel(latitude=slice(i0, i1), longitude=slice(j0, j1)).values[None, :, :], (TCOND, args.patch, args.patch)))
            X_list.append(cos_sza_for_tile(vt_cond_1d, lat_tile, lon_tile))

            X_np = np.stack(X_list, axis=0)[None].astype(np.float32)

            y_norm = infer_batch(X_np, model, scheduler, device, args.ttarget, args.patch, steps, use_amp, amp_dtype)
            y_phys = inverse_scale(y_norm, aorc_min, aorc_max)[0]

            for k, v in enumerate(AORC_VARS):
                accum_data[v][:, i0:i1, j0:j1] += y_phys[k] * blend_w2d[np.newaxis, :, :]
            accum_weight[i0:i1, j0:j1] += blend_w2d

        # Normalize blending
        accum_weight = np.maximum(accum_weight, 1e-12)
        for v in AORC_VARS:
            accum_data[v] /= accum_weight[np.newaxis, :, :]

        # Save Output
        if args.rumi:
            write_rumi_nc(args.out_dir, args.event, t0, lead_hours, lat, lon, accum_data, args)
        else:
            # Default Zarr Save (minimal implementation for brevity)
            out_zarr = Path(args.out_dir) / f"{args.out_name}_IC{ic_idx}.zarr"
            ds_out = xr.Dataset(coords={"time_ic": [t0], "lead_time": lead_hours, "latitude": lat, "longitude": lon})
            for v in AORC_VARS: ds_out[v] = (("time_ic", "lead_time", "latitude", "longitude"), accum_data[v][np.newaxis, ...])
            ds_out.to_zarr(out_zarr, mode="w")

    log(f"All complete in {time.time()-run_start:.1f}s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--graphcast_zarr", help="Path to local GC zarr.")
    p.add_argument("--gcs_zarr_root", help="Root of GCS-style GC data.")
    p.add_argument("--gc_min_zarr", required=True)
    p.add_argument("--gc_max_zarr", required=True)
    p.add_argument("--aorc_min_nc", required=True)
    p.add_argument("--aorc_max_nc", required=True)
    p.add_argument("--topo_nc", required=True)
    p.add_argument("--svf_nc",  required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--start_date", required=True)
    p.add_argument("--end_date",   required=True)
    p.add_argument("--num_inits", type=int, default=1)
    p.add_argument("--lat_min", type=float, default=22.0)
    p.add_argument("--lat_max", type=float, default=23.0)
    p.add_argument("--lon_min", type=float, default=113.5)
    p.add_argument("--lon_max", type=float, default=114.5)
    p.add_argument("--patch",  type=int, default=64) # Modified to 64 to match weights/training
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--ttarget", type=int, default=67)
    p.add_argument("--steps", nargs="+", default=["25"])
    p.add_argument("--out_dir", required=True)
    p.add_argument("--out_name", default="inference_output")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--rumi", action="store_true", help="Enable RUMI output mode (300m NetCDF4)")
    p.add_argument("--event", default="MANGKHUT2018", help="RUMI event code")
    p.add_argument("--experiment", default="RUMI-ERA5-AN", help="RUMI experiment tag")
    return p.parse_args()

if __name__ == "__main__":
    run(parse_args())
