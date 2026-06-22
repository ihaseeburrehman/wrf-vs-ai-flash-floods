#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export the radar/obs/WRF/AI time series (10-17 Jul 2021) as CSVs for the
pgfplots figure in the paper (Model_vs_RADAR_vs_observed/station_*_ai.csv)."""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from osgeo import gdal
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree

gdal.UseExceptions()

from radar_wrf_ai_precip_comparison import (
    RADAR_BASE, WRF_DIR, GC_DIR, FUXI_DIR, AIFS_DIR)

OUT_DIR = ("/Users/haseeb.rehman/Documents/Phd_thesis/Research_papers/"
           "WRF_vs_AI_v1/Model_vs_RADAR_vs_observed")

OB = ("/Users/haseeb.rehman/Documents/Misc/Data_Datasets/"
      "Stations_and_Observations/Luxembourg_stations_for_validation/2021_Event")
OBS_GEN = os.path.join(OB, "stations_6hr_cumulative.xlsx")
OBS_OTH = os.path.join(OB, "Stations_other_than_lux",
                       "station_weather_data_june_july_2021_6hr.xlsx")

STATIONS = {
    "Ettelbruck":  {"coords": (49.85172, 6.09754), "path": OBS_GEN,  "sheet": "Ettelbruck"},
    "Spangdahlem": {"coords": (49.973,   6.693),   "path": OBS_OTH, "sheet": "Spangdahlem ab"},
    "Liege":       {"coords": (50.637,   5.443),   "path": OBS_OTH, "sheet": "Liege"},
    "Vatry":       {"coords": (48.776,   4.184),   "path": OBS_OTH, "sheet": "Vatry"},
}

TS_START = datetime(2021, 7, 10, 0)
TS_END   = datetime(2021, 7, 17, 18)

names  = list(STATIONS.keys())
lats   = [STATIONS[s]["coords"][0] for s in names]
lons   = [STATIONS[s]["coords"][1] for s in names]

# ── observations ────────────────────────────────────────────────────────────
obs_series = {}
for s, info in STATIONS.items():
    df = pd.read_excel(info["path"], sheet_name=info["sheet"])
    df.columns = [str(c).strip() for c in df.columns]
    dt_col = next(c for c in df.columns if "utc" in c.lower() or "date" in c.lower())
    dt = pd.to_datetime(df[dt_col], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    if dt.isna().all():
        dt = pd.to_datetime(df[dt_col], errors="coerce")
    pr_col = next(c for c in df.columns if "precip" in c.lower() and "mm" in c.lower())
    obs_series[s] = pd.Series(pd.to_numeric(df[pr_col], errors="coerce").values,
                              index=dt).dropna()

# ── loop over 6h steps ──────────────────────────────────────────────────────
rows = {s: [] for s in names}
t = TS_START
while t <= TS_END:
    days = (t - TS_START).total_seconds() / 86400.0

    # RADAR: sum 6 x 1h tiffs
    accum = gt = wkt = None
    for k in range(6):
        ts_sub = t - timedelta(hours=5 - k)
        path = os.path.join(RADAR_BASE, f"{ts_sub:%Y}", f"{ts_sub:%m}",
                            f"{ts_sub:%d}", "accum1h", "tif",
                            f"{ts_sub:%Y%m%d%H}0000.radclim.accum1h.tif")
        if not os.path.exists(path):
            continue
        ds = gdal.Open(path)
        arr = ds.GetRasterBand(1).ReadAsArray().astype(float)
        nd = ds.GetRasterBand(1).GetNoDataValue()
        if nd is not None:
            arr = np.where(arr == nd, 0.0, arr)
        arr = np.where(arr < 0, 0.0, arr)
        if accum is None:
            accum, gt, wkt = arr.copy(), ds.GetGeoTransform(), ds.GetProjection()
        else:
            accum += arr
    radar_vals = [np.nan] * len(names)
    if accum is not None:
        ny, nx = accum.shape
        X = gt[0] + (np.arange(nx) + 0.5) * gt[1]
        Y = gt[3] + (np.arange(ny) + 0.5) * gt[5]
        X2, Y2 = np.meshgrid(X, Y)
        tr = Transformer.from_crs(CRS.from_wkt(wkt), "EPSG:4326", always_xy=True)
        lon_g, lat_g = tr.transform(X2, Y2)
        tree = cKDTree(np.column_stack([lat_g.ravel(), lon_g.ravel()]))
        _, idx = tree.query(list(zip(lats, lons)))
        radar_vals = [float(accum.ravel()[i] / 100.0) for i in idx]

    # WRF (6h forecast file valid at t; RAIN* = accumulation since init 6h ago)
    wrf_vals = [np.nan] * len(names)
    fpath = os.path.join(WRF_DIR, "wrfout_d01_" + t.strftime("%Y-%m-%d_%H_%M_%S"))
    if os.path.exists(fpath):
        ds = gdal.Open(fpath)
        subs = ds.GetSubDatasets()
        nd = {v: next((s[0] for s in subs if s[0].rsplit(":", 1)[-1] == v), None)
              for v in ("RAINNC", "RAINC", "RAINSH", "XLAT", "XLONG")}
        if None not in nd.values():
            cum = (gdal.Open(nd["RAINC"]).ReadAsArray()
                 + gdal.Open(nd["RAINNC"]).ReadAsArray()
                 + gdal.Open(nd["RAINSH"]).ReadAsArray())
            la = np.squeeze(gdal.Open(nd["XLAT"]).ReadAsArray())
            lo = np.squeeze(gdal.Open(nd["XLONG"]).ReadAsArray())
            cum = np.where(np.isfinite(np.squeeze(cum)), np.squeeze(cum), 0.0)
            tree = cKDTree(np.column_stack([la.ravel(), lo.ravel()]))
            _, idx = tree.query(list(zip(lats, lons)))
            wrf_vals = [float(cum.ravel()[i]) for i in idx]

    # AI models
    def ai_vals(model_dir):
        vals = [np.nan] * len(names)
        init = t - timedelta(hours=6)
        path = os.path.join(model_dir, f"forecast_{init:%Y%m%dT%H}.nc")
        if not os.path.exists(path):
            return vals
        with xr.open_dataset(path) as ds:
            pr = (ds["total_precipitation_6hr"] if "total_precipitation_6hr" in ds
                  else ds["tp"]).squeeze().values
            if 0.0001 < np.nanmax(pr) < 5.0:
                pr = pr * 1000.0
            la = (ds["lat"] if "lat" in ds else ds["latitude"]).values
            lo = (ds["lon"] if "lon" in ds else ds["longitude"]).values
        if pr.ndim == 1:
            lo_w = np.where(lo > 180, lo - 360, lo)
            la_f, lo_f = la, lo_w
        else:
            lo2, la2 = np.meshgrid(lo, la)
            la_f, lo_f = la2.ravel(), lo2.ravel()
        tree = cKDTree(np.column_stack([la_f, lo_f]))
        _, idx = tree.query(list(zip(lats, lons)))
        return [float(np.where(np.isfinite(pr.ravel()), pr.ravel(), 0.0)[i]) for i in idx]

    gc_v, fx_v, af_v = ai_vals(GC_DIR), ai_vals(FUXI_DIR), ai_vals(AIFS_DIR)

    for i, s in enumerate(names):
        ob = obs_series[s].get(t, np.nan)
        if isinstance(ob, pd.Series):
            ob = float(ob.iloc[0])
        rows[s].append({"Days": round(days, 4), "Obs": ob, "RADAR": radar_vals[i],
                        "WRF": wrf_vals[i], "GC": gc_v[i],
                        "FuXi": fx_v[i], "AIFS": af_v[i]})
    print(f"{t:%Y-%m-%d %H} done", flush=True)
    t += timedelta(hours=6)

os.makedirs(OUT_DIR, exist_ok=True)
for s in names:
    df = pd.DataFrame(rows[s]).round(3)
    out = os.path.join(OUT_DIR, f"station_{s}_ai.csv")
    df.to_csv(out, index=False, na_rep="nan")
    print(f"Saved {out}  ({len(df)} rows)")
