# AirCast HongKong

AirCast HongKong is a 3D UNet-based model designed for high-resolution weather downscaling. This repository contains the training and inference code, along with the Apptainer definition file to set up the execution environment.

## Overview

The model uses a 3D UNet architecture to downscale GraphCast weather predictions to AORC-scale resolution. It incorporates static features such as topography and sky-view factor (SVF) and uses a diffusion-based denoising process for inference.

## Environment Setup (Apptainer)

To ensure consistency, it is recommended to run the model using Apptainer (formerly Singularity).

### Building the Image

Use the provided `earthmind_highres.def` to build the Apptainer image:

```bash
apptainer build earthmind_highres.sif earthmind_highres.def
```

### Running the Container

You can run scripts inside the container using:

```bash
apptainer exec --nv earthmind_highres.sif python3 <script_name>.py [args]
```

## Training

The training script `em_train_dataloader.py` handles the training process with a custom dataloader that materializes GraphCast data for efficiency.

### Usage

```bash
python3 em_train_dataloader.py --max-patches-per-ic 500 --num-workers 4
```

- `--max-patches-per-ic`: Limits the number of patches processed per initialization time (IC).
- `--num-workers`: Number of workers for data loading.

## Inference

The inference script `inference_final_version.py` performs the downscaling using a trained checkpoint.

### Weights

The recommended weights for this model are:
- `unet3d_global_best_ic_20210127T060000.pt` (Used for surface pressure)
- `unet3d_global_best_ic_20210204T120000.pt`

### Running Inference

Example command for running inference:

```bash
python3 inference_final_version.py \
    --graphcast_zarr /path/to/graphcast.zarr \
    --gc_min_zarr /path/to/gc_min.zarr \
    --gc_max_zarr /path/to/gc_max.zarr \
    --topo_nc /path/to/elevation.nc \
    --svf_nc /path/to/sky_view_factor.nc \
    --aorc_min_nc /path/to/aorc_min.nc \
    --aorc_max_nc /path/to/aorc_max.nc \
    --checkpoint /path/to/unet3d_global_best_ic_20210127T060000.pt \
    --start_date 2021-07-04 \
    --end_date 2021-07-05 \
    --num_inits 1 \
    --lat_min 25.0 --lat_max 50.0 \
    --lon_min -125.0 --lon_max -65.0 \
    --patch 256 --stride 128 \
    --steps 25 \
    --out_dir ./inference_outputs \
    --out_name aircast_hk_inference \
    --overwrite
```
