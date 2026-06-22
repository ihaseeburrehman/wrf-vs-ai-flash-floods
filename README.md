# WRF vs. AI Weather Models for Flash Flood Forecasting

**Associated publication:**  
Rehman, H., & Teferle, F. N. R. (2026). *Physics vs. AI in Weather Prediction: Evaluating GraphCast, AIFS, and FuXi against an Observation-Corrected WRF Model for Flash Floods.* Geoscientific Model Development (under review).

## Overview

This repository contains the Python scripts used to produce all results in the above paper. The study compares a 3D-VAR data-assimilating WRF configuration against three AI weather models (GraphCast, AIFS Single 1.0, FuXi) over three high-impact flood events in Luxembourg and the Greater Region (2016, 2018, 2021).

## Repository Structure

```
01_data_download/          ERA5 and NOAA observation download scripts
02_observation_preprocessing/  GNSS ZTD and conventional obs pre-processing
03_wrf_rapid_cycle/        WRF 6-hourly Rapid Update Cycle execution scripts
04_ai_model_extraction/    Variable extraction from AIFS, FuXi, GraphCast outputs
05_radar_processing/       RADFLOOD21 radar aggregation and spatial comparison
06_verification_statistics/ Categorical and continuous score computation
07_figures_maps/           Figures and maps used in the paper
```

## Workflow

```
ERA5 + NOAA obs + GNSS ZTD
        │
        ▼
01_data_download  ──►  02_observation_preprocessing
                                │
                                ▼
                       03_wrf_rapid_cycle  ──►  04_ai_model_extraction
                                │                        │
                                └────────────┬───────────┘
                                             ▼
                                  06_verification_statistics
                                             │
                                  ┌──────────┴──────────┐
                                  ▼                      ▼
                           05_radar_processing    07_figures_maps
```

## Case Studies

| Event | Period simulated | Flood type |
|-------|-----------------|------------|
| July 2016 | 10 Jul – 6 Aug 2016 | Flash flood, NE Luxembourg |
| June 2018 | 20 May – 20 Jun 2018 | Mullerthal flash flood |
| July 2021 | 20 Jun – 20 Jul 2021 | Catastrophic European flood |

## Models

| Model | Version | Source |
|-------|---------|--------|
| WRF/WRFDA | 4.5 | https://github.com/wrf-model/WRF |
| GraphCast | — | https://github.com/google-deepmind/graphcast |
| AIFS | Single 1.0 | https://huggingface.co/ecmwf/aifs-single-1.0 |
| FuXi | — | https://github.com/ecmwf-lab/ai-models-fuxi |

## Data Sources

- **ERA5 reanalysis**: Copernicus CDS (https://cds.climate.copernicus.eu)
- **GNSS ZTD**: Nevada Geodetic Laboratory (http://geodesy.unr.edu)
- **Conventional obs**: NOAA Integrated Surface Database (https://www.ncei.noaa.gov)
- **AgriMeteo stations**: https://www.agrimeteo.lu
- **RADFLOOD21 radar**: RMI Belgium, Zenodo DOI: https://doi.org/10.5281/zenodo.7740059

## Dependencies

```
python >= 3.9
numpy, pandas, xarray, netCDF4
scipy, scikit-learn
matplotlib, cartopy
wrf-python
eccodes / cfgrib (for GRIB files)
cdsapi (for ERA5 download)
```

Install with:
```bash
conda env create -f environment.yml   # (see environment.yml)
```

## WRF Configuration

The WRF domain (d01, 12 km, 120×120 grid points, 33 levels, model top 50 hPa) was configured using:
- **Microphysics**: Thompson et al.
- **PBL**: YSU
- **Cumulus**: Grell–Freitas
- **LSM**: Noah
- **Data assimilation**: WRFDA 3D-VAR, CV5 background error covariance

WRF namelist files and WRFDA configuration files (ob.ascii, be.dat) are stored on the ECMWF HPC. Contact the corresponding author for access.

## Citation

If you use these scripts, please cite the associated paper (DOI to be updated upon acceptance).

## License

MIT License — see [LICENSE](LICENSE).

## Contact

**Haseeb ur Rehman**  
PhD Researcher, University of Luxembourg  
haseeb.rehman@uni.lu
