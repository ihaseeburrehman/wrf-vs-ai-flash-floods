#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild matched forecast-observation pairs using the corrected, independent
observation set (obs_rv1_<event>.csv). Includes WRF Before DA and After DA.

Output: merged_pairs_rv1.csv
"""

import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_wrf_gc_fuxi_aifs as C

GC_XLSX = "/Users/haseeb.rehman/Documents/Misc/AI_Models/GraphCast/graphcast_all_variables.xlsx"
FUXI_CSV = "/Users/haseeb.rehman/Documents/Misc/AI_Models/FuXi/fuxi_all_variables.csv"
AIFS_CSV = "/Users/haseeb.rehman/Documents/Misc/AI_Models/AIFS/aifs_all_variables.csv"
WRF_HPC = "/Users/haseeb.rehman/Documents/Misc/From_HPC_and_WRF/WRF_from_HPC/4th_year"
OUT = os.path.join(HERE, "merged_pairs_rv1.csv")


def load_model_csv(path, year, pcol, tcol):
    df = pd.read_csv(path)
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    df = df[df["Event"].astype(str) == year]
    if df.empty:
        return pd.DataFrame()
    df["Station"] = df["Station"].map(lambda s: C.MODEL_ALIAS.get(s, s))
    return pd.DataFrame({
        "Station": df["Station"].values,
        "UTC_Datetime": pd.to_datetime(df["Valid_Time"], format="%Y%m%dT%H", errors="coerce"),
        pcol: pd.to_numeric(df["Precip_mm"], errors="coerce").values,
        tcol: pd.to_numeric(df["T2m_C"], errors="coerce").values,
    }).dropna(subset=["UTC_Datetime"])


frames = []
for year in ["2016", "2018", "2021"]:
    print(f"── {year} ──", flush=True)
    obs = pd.read_csv(os.path.join(HERE, f"obs_rv1_{year}.csv"),
                      parse_dates=["UTC_Datetime"])
    gc = C.load_model_xlsx(GC_XLSX, year, out_precip="GC_Precip", out_t2m="GC_T2m")
    fx = load_model_csv(FUXI_CSV, year, "FuXi_Precip", "FuXi_T2m")
    af = load_model_csv(AIFS_CSV, year, "AIFS_Precip", "AIFS_T2m")

    common = sorted(set(obs["Station"]) & set(gc["Station"]) & set(fx["Station"])
                    & set(af["Station"]) & set(C.STATION_COORDS))
    latlon = [C.STATION_COORDS[s] for s in common]
    print(f"  common stations: {len(common)}")

    wrf_a = C.extract_wrf(f"{WRF_HPC}/{year}_ERA5_cv5/After_DA", common, latlon)
    wrf_b = C.extract_wrf(f"{WRF_HPC}/{year}_ERA5_cv5/Before_DA", common, latlon)
    wrf_b = wrf_b.rename(columns={"WRF_Precip": "WRFB_Precip", "WRF_T2m": "WRFB_T2m"})

    keep = lambda d, cols: d[d["Station"].isin(common)][["Station", "UTC_Datetime"] + cols]
    m = (keep(wrf_a, ["WRF_Precip", "WRF_T2m"])
         .merge(keep(wrf_b, ["WRFB_Precip", "WRFB_T2m"]), on=["Station", "UTC_Datetime"], how="left")
         .merge(keep(obs, ["Obs_Precip", "Obs_T2m"]), on=["Station", "UTC_Datetime"])
         .merge(keep(gc, ["GC_Precip", "GC_T2m"]), on=["Station", "UTC_Datetime"])
         .merge(keep(fx, ["FuXi_Precip", "FuXi_T2m"]), on=["Station", "UTC_Datetime"])
         .merge(keep(af, ["AIFS_Precip", "AIFS_T2m"]), on=["Station", "UTC_Datetime"]))
    m["Event"] = year
    print(f"  merged rows={len(m)}  stations={m['Station'].nunique()}")
    frames.append(m)

pooled = pd.concat(frames, ignore_index=True)
pooled.to_csv(OUT, index=False)
print(f"\nSaved -> {OUT}")
print(f"total rows={len(pooled)}")
print("\nprecip-valid pairs per event:")
print(pooled.dropna(subset=["Obs_Precip", "WRF_Precip", "GC_Precip", "FuXi_Precip", "AIFS_Precip"])
      .groupby("Event").size().to_string())
