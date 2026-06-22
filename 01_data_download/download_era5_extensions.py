#!/usr/bin/env python3
"""
ERA5 download for the *extension* dates so met_em can cover the obs windows
already on HPC.

  2018: need 2018-06-10 -> 2018-06-20  (extends past existing 2018-06-10)
  2016: need 2016-07-31 -> 2016-08-09  (extends past existing 2016-07-31)

Submits all 4 requests (2 PL + 2 SL) in parallel threads.
Each writes to <out>.partial then renames atomically.
"""
import os, sys, threading, time
from datetime import datetime
import cdsapi

MET = "/Users/haseeb.rehman/Documents/gis4wrf/datasets/met"

PL_VARIABLES = [
    "temperature", "u_component_of_wind", "v_component_of_wind",
    "vertical_velocity", "specific_humidity", "relative_humidity",
    "geopotential", "divergence", "vorticity", "potential_vorticity",
    "fraction_of_cloud_cover", "specific_cloud_ice_water_content",
    "specific_cloud_liquid_water_content", "specific_rain_water_content",
    "specific_snow_water_content", "ozone_mass_mixing_ratio",
]
PL_LEVELS = [
    "7","10","20","30","50","70","100","125","150","175","200","225",
    "250","300","350","400","450","500","550","600","650","700","750",
    "775","800","825","850","875","900","925","950","975","1000",
]
SL_VARIABLES = [
    "10m_u_component_of_wind","10m_v_component_of_wind","2m_temperature",
    "2m_dewpoint_temperature","mean_sea_level_pressure","surface_pressure",
    "sea_ice_cover","land_sea_mask","snow_density","snow_depth",
    "skin_temperature","soil_type","sea_surface_temperature",
    "soil_temperature_level_1","soil_temperature_level_2",
    "soil_temperature_level_3","soil_temperature_level_4",
    "significant_height_of_combined_wind_waves_and_swell",
    "volumetric_soil_water_layer_1","volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3","volumetric_soil_water_layer_4",
    "total_precipitation",
]
TIMES = ["00:00","06:00","12:00","18:00"]
GRID  = [0.25, 0.25]

JOBS = [
    {
        "label": "2018 PL ext (Jun 10-20)",
        "dataset": "reanalysis-era5-pressure-levels",
        "date": "2018-06-10/2018-06-20",
        "out": f"{MET}/20180510_20180610_ECMWF/pressure_level/2018_june_10_to_20_pressure_level",
        "request": {
            "product_type": "reanalysis", "format": "grib",
            "variable": PL_VARIABLES, "pressure_level": PL_LEVELS,
            "time": TIMES, "grid": GRID,
        },
    },
    {
        "label": "2018 SL ext (Jun 10-20)",
        "dataset": "reanalysis-era5-single-levels",
        "date": "2018-06-10/2018-06-20",
        "out": f"{MET}/20180510_20180610_ECMWF/single_level/single_level_ext.grib",
        "request": {
            "product_type": "reanalysis", "format": "grib",
            "variable": SL_VARIABLES, "time": TIMES, "grid": GRID,
        },
    },
    {
        "label": "2016 PL ext (Jul 31-Aug 9)",
        "dataset": "reanalysis-era5-pressure-levels",
        "date": "2016-07-31/2016-08-09",
        "out": f"{MET}/20160701_20160731_ECMWF/pressure_level/2016_jul_31_to_aug_09_pressure_level",
        "request": {
            "product_type": "reanalysis", "format": "grib",
            "variable": PL_VARIABLES, "pressure_level": PL_LEVELS,
            "time": TIMES, "grid": GRID,
        },
    },
    {
        "label": "2016 SL ext (Jul 31-Aug 9)",
        "dataset": "reanalysis-era5-single-levels",
        "date": "2016-07-31/2016-08-09",
        "out": f"{MET}/20160701_20160731_ECMWF/single_level/single_level_ext.grib",
        "request": {
            "product_type": "reanalysis", "format": "grib",
            "variable": SL_VARIABLES, "time": TIMES, "grid": GRID,
        },
    },
]

PRINT_LOCK = threading.Lock()
def log(m):
    with PRINT_LOCK:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {m}", flush=True)

def fetch(job):
    label, ds, out = job["label"], job["dataset"], job["out"]
    partial = out + ".partial"
    if os.path.exists(out) and os.path.getsize(out) > 0:
        log(f"{label}: SKIP exists")
        return True
    os.makedirs(os.path.dirname(out), exist_ok=True)
    client = cdsapi.Client(quiet=True)
    req = dict(job["request"], date=job["date"])
    try:
        log(f"{label}: submitting CDS request ({job['date']})")
        client.retrieve(ds, req, partial)
        os.replace(partial, out)
        sz = os.path.getsize(out) / 1e9
        log(f"{label}: DONE -> {out} ({sz:.2f} GB)")
        return True
    except Exception as e:
        log(f"{label}: FAIL {e}")
        return False

def main():
    log(f"launching {len(JOBS)} parallel CDS requests")
    threads, results = [], {}
    def runner(j):
        results[j["label"]] = fetch(j)
    for j in JOBS:
        t = threading.Thread(target=runner, args=(j,), name=j["label"])
        t.start(); threads.append(t)
        time.sleep(2)
    for t in threads:
        t.join()
    log("--- SUMMARY ---")
    n_ok = sum(1 for ok in results.values() if ok)
    for label, ok in results.items():
        log(f"  [{'OK' if ok else 'FAIL'}] {label}")
    log(f"{n_ok}/{len(JOBS)} succeeded")
    sys.exit(0 if n_ok == len(JOBS) else 1)

if __name__ == "__main__":
    main()
