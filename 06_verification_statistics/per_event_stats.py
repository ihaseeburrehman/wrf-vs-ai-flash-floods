#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-event + pooled verification stats for WRF (After DA), GraphCast, FuXi, AIFS.
Reuses the loaders of compare_wrf_gc_fuxi_aifs.py; prints console + LaTeX table rows."""

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_wrf_gc_fuxi_aifs as C

GC_XLSX  = "/Users/haseeb.rehman/Documents/Misc/AI_Models/GraphCast/graphcast_all_variables.xlsx"
FUXI_CSV = "/Users/haseeb.rehman/Documents/Misc/AI_Models/FuXi/fuxi_all_variables.csv"
AIFS_CSV = "/Users/haseeb.rehman/Documents/Misc/AI_Models/AIFS/aifs_all_variables.csv"


def load_model_csv(csv_path, year, out_precip, out_t2m):
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    df = df[df["Event"].astype(str) == year]
    if df.empty:
        return pd.DataFrame()
    df["Station"] = df["Station"].map(lambda s: C.MODEL_ALIAS.get(s, s))
    out = pd.DataFrame({
        "Station":      df["Station"].values,
        "UTC_Datetime": pd.to_datetime(df["Valid_Time"], format="%Y%m%dT%H", errors="coerce"),
        out_precip:     pd.to_numeric(df["Precip_mm"], errors="coerce").values,
        out_t2m:        pd.to_numeric(df["T2m_C"],     errors="coerce").values,
    }).dropna(subset=["UTC_Datetime"])
    return out


merged_by_event = {}
for year, cfg in C.EVENTS.items():
    print(f"── Event {year} ──", flush=True)
    obs = C.load_obs(cfg)
    gc  = C.load_model_xlsx(GC_XLSX, year, out_precip="GC_Precip", out_t2m="GC_T2m")
    fx  = load_model_csv(FUXI_CSV, year, "FuXi_Precip", "FuXi_T2m")
    af  = load_model_csv(AIFS_CSV, year, "AIFS_Precip", "AIFS_T2m")
    common = sorted(set(obs["Station"]) & set(gc["Station"]) & set(fx["Station"])
                    & set(af["Station"]) & set(C.STATION_COORDS))
    latlon = [C.STATION_COORDS[s] for s in common]
    wrf = C.extract_wrf(cfg["wrf_dir"], common, latlon)
    m = (wrf[["Station","UTC_Datetime","WRF_Precip","WRF_T2m"]]
         .merge(obs[obs["Station"].isin(common)][["Station","UTC_Datetime","Obs_Precip","Obs_T2m"]],
                on=["Station","UTC_Datetime"])
         .merge(gc[gc["Station"].isin(common)][["Station","UTC_Datetime","GC_Precip","GC_T2m"]],
                on=["Station","UTC_Datetime"])
         .merge(fx[fx["Station"].isin(common)][["Station","UTC_Datetime","FuXi_Precip","FuXi_T2m"]],
                on=["Station","UTC_Datetime"])
         .merge(af[af["Station"].isin(common)][["Station","UTC_Datetime","AIFS_Precip","AIFS_T2m"]],
                on=["Station","UTC_Datetime"])
         .dropna())
    print(f"  rows={len(m)} stations={m['Station'].nunique()}")
    merged_by_event[year] = m

pooled = pd.concat(merged_by_event.values(), ignore_index=True)
blocks = list(merged_by_event.items()) + [("All", pooled)]
SYS = [("WRF", "WRF (After DA)"), ("GC", "GraphCast"), ("FuXi", "FuXi"), ("AIFS", "AIFS")]

print("\n================ PRECIPITATION ================")
print(f"{'Event':<6}{'System':<16}{'POD':>7}{'FAR':>7}{'CSI':>7}{'ETS':>7}{'Bias':>8}{'N':>7}")
latex_p = []
for ev, m in blocks:
    for key, name in SYS:
        s = C.calc_metrics(m[f"{key}_Precip"], m["Obs_Precip"], is_precip=True)
        print(f"{ev:<6}{name:<16}{s['POD']:7.3f}{s['FAR']:7.3f}{s['CSI']:7.3f}{s['ETS']:7.3f}{s['Bias']:8.3f}{s['N']:7d}")
        latex_p.append((ev, name, s))
    print()

print("================ TEMPERATURE ================")
print(f"{'Event':<6}{'System':<16}{'RMSE':>7}{'MAE':>7}{'Corr':>7}{'Bias':>8}")
latex_t = []
for ev, m in blocks:
    for key, name in SYS:
        s = C.calc_metrics(m[f"{key}_T2m"], m["Obs_T2m"])
        print(f"{ev:<6}{name:<16}{s['RMSE']:7.3f}{s['MAE']:7.3f}{s['Corr']:7.3f}{s['Bias']:8.3f}")
        latex_t.append((ev, name, s))
    print()

print("\n================ LATEX ROWS ================")
for ev, m in blocks:
    label = ev if ev != "All" else "All events"
    n = len(m)
    print(f"\\midrule\n\\multicolumn{{10}}{{l}}{{\\textbf{{{label}}} ($N={n}$, {m['Station'].nunique()} stations)}}\\\\")
    for key, name in SYS:
        p = C.calc_metrics(m[f"{key}_Precip"], m["Obs_Precip"], is_precip=True)
        t = C.calc_metrics(m[f"{key}_T2m"], m["Obs_T2m"])
        print(f"{name} & {p['POD']:.3f} & {p['FAR']:.3f} & {p['CSI']:.3f} & {p['ETS']:.3f} & {p['Bias']:.3f} & "
              f"{t['RMSE']:.3f} & {t['MAE']:.3f} & {t['Corr']:.3f} & {t['Bias']:.3f} \\\\")
