# RUMI Phase 1: Hong Kong Output Specifications

This document summarizes the requirements for the Regional Urban Model Intercomparison (RUMI) Phase 1.

## 1. Experimental Cases
- **MANGKHUT2018**: Typhoon Mangkhut (Sept 16, 2018)
- **HRAIN2023**: Black Rainstorm (Sept 7–8, 2023)
- **HRAIN2025**: Black Rainstorm (Aug 2–5, 2025)
- **HEAT2022**: Heatwave (July 21–29, 2022)
- **HEAT2024**: Heatwave (August 24–28, 2024)

## 2. Standard Output Grid
- **Type**: Regular Latitude-Longitude
- **Resolution**: 9.7 arc-seconds (~300m / 0.002694°)
- **Latitude**: 22.12°N to 22.58°N
- **Longitude**: 113.82°E to 114.45°E
- **Dimensions**: 234 x 171 (Lon x Lat)

## 3. Mandatory Variable Mapping
| RUMI Name | Description | Model (AORC) Equivalent | Units |
| :--- | :--- | :--- | :--- |
| **T2M** | 2m Temperature | TMP_2maboveground | K |
| **U10M** | 10m Zonal Wind | UGRD_10maboveground | m s-1 |
| **V10M** | 10m Meridional Wind | VGRD_10maboveground | m s-1 |
| **PRATE** | Precipitation Rate | APCP_surface (converted) | kg m-2 s-1 |
| **SLP** | Sea Level Pressure | PRES_surface (reduced) | Pa |
| **RH2M** | 2m Relative Humidity | SPFH (converted) | 1 (fraction) |
| **PSFC** | Surface Pressure | PRES_surface | Pa |
| **Q2M** | 2m Specific Humidity | SPFH_2maboveground | kg kg-1 |

## 4. File Naming Convention
`RUMI-<SOURCE>-<MODE>-<MODEL>-<EVENT>-<YYYYMMDDHHMMSS>.nc`
- **SOURCE**: e.g., ERA5, GFS
- **MODE**: AN (Analysis) or FC (Forecast)
- **MODEL**: e.g., AirCastHK

## 5. Mandatory Global Attributes
- `experiment`: e.g., "RUMI-ERA5-AN"
- `forcing_mode`: "analysis" or "forecast"
- `event`: e.g., "MANGKHUT2018"
- `physics_schemes`: Microphysics, PBL, Land Surface, etc.
