#!/usr/bin/env python3
"""
ERA5 download for 2018 and 2016 events.

Mirrors the existing 2021 layout under datasets/met/:
  - Pressure-level: 16 vars x 33 levels, 6-hourly (00/06/12/18 UTC),
    chunked ~10 days per file, GRIB format, global 0.25 deg.
  - Single-level: 23 vars, 6-hourly, one GRIB per event.

Events:
  - 2018: 2018-05-10 -> 2018-06-10  (split PL into 3 chunks)
  - 2016: 2016-07-01 -> 2016-07-31  (split PL into 3 chunks)
          -> also deletes the obsolete 2016 single-level folder first.
"""

import os
import shutil
import sys
import time
from datetime import datetime

import cdsapi

MET_ROOT = "/Users/haseeb.rehman/Documents/gis4wrf/datasets/met"

# Match 2021 exactly
PL_VARIABLES = [
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
    "relative_humidity",
    "geopotential",
    "divergence",
    "vorticity",
    "potential_vorticity",
    "fraction_of_cloud_cover",
    "specific_cloud_ice_water_content",
    "specific_cloud_liquid_water_content",
    "specific_rain_water_content",
    "specific_snow_water_content",
    "ozone_mass_mixing_ratio",
]

PL_LEVELS = [
    "1", "2", "3", "5", "7", "10", "20", "30", "50", "70",
    "100", "125", "150", "175", "200", "225", "250", "300", "350", "400",
    "450", "500", "550", "600", "650", "700", "750", "775", "800", "825",
    "850", "875", "900", "925", "950", "975", "1000",
]
# Restrict to the 33 levels actually present in the 2021 files
PL_LEVELS = [
    "7", "10", "20", "30", "50", "70",
    "100", "125", "150", "175", "200", "225", "250", "300", "350", "400",
    "450", "500", "550", "600", "650", "700", "750", "775", "800", "825",
    "850", "875", "900", "925", "950", "975", "1000",
]

SL_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
    "mean_sea_level_pressure",
    "surface_pressure",
    "sea_ice_cover",
    "land_sea_mask",
    "snow_density",
    "snow_depth",
    "skin_temperature",
    "soil_type",
    "sea_surface_temperature",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "significant_height_of_combined_wind_waves_and_swell",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "total_precipitation",
]

TIMES = ["00:00", "06:00", "12:00", "18:00"]
GRID = [0.25, 0.25]


def date_range_str(y1, m1, d1, y2, m2, d2):
    """CDS date-range string: 'YYYY-MM-DD/YYYY-MM-DD' (inclusive)."""
    start = f"{y1:04d}-{m1:02d}-{d1:02d}"
    end = f"{y2:04d}-{m2:02d}-{d2:02d}"
    n_days = (datetime(y2, m2, d2) - datetime(y1, m1, d1)).days + 1
    return f"{start}/{end}", n_days


def submit_pl(client, date_str, n_days, out_path):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"  PL  SKIP (exists, {os.path.getsize(out_path)/1e9:.2f} GB): {out_path}")
        return
    print(f"  PL  {out_path}  ({date_str}, {n_days} days)")
    client.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "format": "grib",
            "variable": PL_VARIABLES,
            "pressure_level": PL_LEVELS,
            "date": date_str,
            "time": TIMES,
            "grid": GRID,
        },
        out_path,
    )


def submit_sl(client, date_str, n_days, out_path):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"  SL  SKIP (exists, {os.path.getsize(out_path)/1e9:.2f} GB): {out_path}")
        return
    print(f"  SL  {out_path}  ({date_str}, {n_days} days)")
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "grib",
            "variable": SL_VARIABLES,
            "date": date_str,
            "time": TIMES,
            "grid": GRID,
        },
        out_path,
    )


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_one(label, fn):
    print(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] START {label}")
    t0 = time.time()
    try:
        fn()
        dt = time.time() - t0
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] DONE  {label}  ({dt/60:.1f} min)")
        return True
    except Exception as e:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] FAIL  {label}: {e}")
        return False


def main():
    client = cdsapi.Client()

    # ----- 2016: remove the obsolete single-level folder -----
    old_2016 = os.path.join(MET_ROOT, "20160710_20160810_ECMWF")
    if os.path.isdir(old_2016):
        print(f"Removing obsolete folder: {old_2016}")
        shutil.rmtree(old_2016)

    # ----- 2018 paths -----
    ev2018 = os.path.join(MET_ROOT, "20180510_20180610_ECMWF")
    ev2018_pl = os.path.join(ev2018, "pressure_level")
    ev2018_sl = os.path.join(ev2018, "single_level")
    ensure_dir(ev2018_pl)
    ensure_dir(ev2018_sl)

    # ----- 2016 paths -----
    ev2016 = os.path.join(MET_ROOT, "20160701_20160731_ECMWF")
    ev2016_pl = os.path.join(ev2016, "pressure_level")
    ev2016_sl = os.path.join(ev2016, "single_level")
    ensure_dir(ev2016_pl)
    ensure_dir(ev2016_sl)

    # ----- 2018 chunks (10-day PL chunks; 32-day full-range SL) -----
    chunks_2018 = [
        ("2018_may_10_to_20_pressure_level",  date_range_str(2018, 5, 10, 2018, 5, 20)),
        ("2018_may_21_to_31_pressure_level",  date_range_str(2018, 5, 21, 2018, 5, 31)),
        ("2018_june_01_to_10_pressure_level", date_range_str(2018, 6, 1,  2018, 6, 10)),
    ]
    sl_2018 = date_range_str(2018, 5, 10, 2018, 6, 10)

    # ----- 2016 chunks -----
    chunks_2016 = [
        ("2016_july_01_to_10_pressure_level", date_range_str(2016, 7, 1,  2016, 7, 10)),
        ("2016_july_11_to_20_pressure_level", date_range_str(2016, 7, 11, 2016, 7, 20)),
        ("2016_july_21_to_31_pressure_level", date_range_str(2016, 7, 21, 2016, 7, 31)),
    ]
    sl_2016 = date_range_str(2016, 7, 1, 2016, 7, 31)

    jobs = []

    # 2018 single-level first (smaller, validates auth quickly)
    ds, nd = sl_2018
    jobs.append((
        "2018 single-level",
        lambda ds=ds, nd=nd: submit_sl(client, ds, nd, os.path.join(ev2018_sl, "single_level.grib")),
    ))
    # 2016 single-level
    ds, nd = sl_2016
    jobs.append((
        "2016 single-level",
        lambda ds=ds, nd=nd: submit_sl(client, ds, nd, os.path.join(ev2016_sl, "single_level.grib")),
    ))
    # 2018 PL chunks
    for name, (ds, nd) in chunks_2018:
        out = os.path.join(ev2018_pl, name)
        jobs.append((f"2018 PL {name}", lambda ds=ds, nd=nd, o=out: submit_pl(client, ds, nd, o)))
    # 2016 PL chunks
    for name, (ds, nd) in chunks_2016:
        out = os.path.join(ev2016_pl, name)
        jobs.append((f"2016 PL {name}", lambda ds=ds, nd=nd, o=out: submit_pl(client, ds, nd, o)))

    results = []
    for label, fn in jobs:
        ok = run_one(label, fn)
        results.append((label, ok))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label, ok in results:
        print(f"  [{'OK' if ok else 'FAIL'}] {label}")
    n_ok = sum(1 for _, ok in results if ok)
    print(f"\n{n_ok}/{len(results)} jobs succeeded")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
