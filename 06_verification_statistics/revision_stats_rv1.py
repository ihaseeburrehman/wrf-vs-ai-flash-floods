#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Revision analyses for GMD referee comments (RC1):
1. Categorical precip scores at thresholds 1/5/10/20 mm per 6 h (pooled + per event)
2. 95% bootstrap confidence intervals (1000 resamples) for pooled scores at 1 mm
   and for temperature RMSE/MAE
3. Event-only verification windows (flood days only)
Also caches the merged matched-pair dataframe to CSV for figure work.
"""

import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_wrf_gc_fuxi_aifs as C

GC_XLSX  = "/Users/haseeb.rehman/Documents/Misc/AI_Models/GraphCast/graphcast_all_variables.xlsx"
FUXI_CSV = "/Users/haseeb.rehman/Documents/Misc/AI_Models/FuXi/fuxi_all_variables.csv"
AIFS_CSV = "/Users/haseeb.rehman/Documents/Misc/AI_Models/AIFS/aifs_all_variables.csv"
CACHE    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merged_pairs_cache.csv")
OUT      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "revision_stats_rv1_output.txt")

# Event (flood) day windows, inclusive
EVENT_WINDOWS = {
    "2016": ("2016-07-21", "2016-07-24"),   # 21-23 July + trailing 6h
    "2018": ("2018-05-31", "2018-06-03"),   # 31 May - 2 June
    "2021": ("2021-07-13", "2021-07-16"),   # 14-15 July, incl. onset 13th
}

THRESHOLDS = [1.0, 5.0, 10.0, 20.0]
NBOOT = 1000
RNG = np.random.default_rng(42)


def load_model_csv(csv_path, year, out_precip, out_t2m):
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
    df = df[df["Event"].astype(str) == year]
    if df.empty:
        return pd.DataFrame()
    df["Station"] = df["Station"].map(lambda s: C.MODEL_ALIAS.get(s, s))
    return pd.DataFrame({
        "Station":      df["Station"].values,
        "UTC_Datetime": pd.to_datetime(df["Valid_Time"], format="%Y%m%dT%H", errors="coerce"),
        out_precip:     pd.to_numeric(df["Precip_mm"], errors="coerce").values,
        out_t2m:        pd.to_numeric(df["T2m_C"],     errors="coerce").values,
    }).dropna(subset=["UTC_Datetime"])


def cat_scores(p, o, thr):
    p = np.asarray(p, float); o = np.asarray(o, float)
    ok = np.isfinite(p) & np.isfinite(o); p, o = p[ok], o[ok]
    n = p.size
    pb = p >= thr; ob = o >= thr
    H = int((pb & ob).sum()); M = int((~pb & ob).sum()); F = int((pb & ~ob).sum())
    pod = H/(H+M) if H+M else np.nan
    far = F/(H+F) if H+F else np.nan
    csi = H/(H+M+F) if H+M+F else np.nan
    Hr  = (H+M)*(H+F)/n if n else 0
    ets = (H-Hr)/(H+M+F-Hr) if (H+M+F-Hr) > 0 else np.nan
    return dict(POD=pod, FAR=far, CSI=csi, ETS=ets, N=n, Hits=H, Miss=M, FA=F,
                ObsEvents=H+M)


def boot_ci(func, *cols, n=NBOOT):
    """95% CI by resampling matched pairs."""
    N = len(cols[0])
    idx = np.arange(N)
    vals = []
    for _ in range(n):
        s = RNG.choice(idx, size=N, replace=True)
        vals.append(func(*[c[s] for c in cols]))
    vals = np.asarray(vals, float)
    return np.nanpercentile(vals, 2.5), np.nanpercentile(vals, 97.5)


# ── Build or load merged dataset ────────────────────────────────────────────
if os.path.exists(CACHE):
    print("Loading cached merged pairs ...")
    pooled = pd.read_csv(CACHE, parse_dates=["UTC_Datetime"])
else:
    frames = []
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
        m["Event"] = year
        print(f"  rows={len(m)} stations={m['Station'].nunique()}")
        frames.append(m)
    pooled = pd.concat(frames, ignore_index=True)
    pooled.to_csv(CACHE, index=False)
    print(f"Cached -> {CACHE}")

SYS = [("WRF", "WRF (After DA)"), ("GC", "GraphCast"), ("FuXi", "FuXi"), ("AIFS", "AIFS")]
lines = []
def emit(s=""):
    print(s); lines.append(s)

# ── 1. Threshold sensitivity (pooled + per event) ───────────────────────────
emit("=" * 76)
emit("1. CATEGORICAL SCORES BY THRESHOLD (mm per 6 h)")
emit("=" * 76)
blocks = [(y, pooled[pooled["Event"] == y]) for y in ["2016", "2018", "2021"]] + [("All", pooled)]
for thr in THRESHOLDS:
    emit(f"\n--- Threshold {thr:g} mm ---")
    emit(f"{'Event':<6}{'System':<16}{'POD':>7}{'FAR':>7}{'CSI':>7}{'ETS':>7}{'ObsEv':>7}{'N':>7}")
    for ev, m in blocks:
        for key, name in SYS:
            s = cat_scores(m[f"{key}_Precip"].values, m["Obs_Precip"].values, thr)
            emit(f"{ev:<6}{name:<16}{s['POD']:7.3f}{s['FAR']:7.3f}{s['CSI']:7.3f}"
                 f"{s['ETS']:7.3f}{s['ObsEvents']:7d}{s['N']:7d}")

# ── 2. Bootstrap 95% CIs (pooled, 1 mm) ─────────────────────────────────────
emit("\n" + "=" * 76)
emit(f"2. POOLED 95% BOOTSTRAP CI ({NBOOT} resamples), threshold 1 mm")
emit("=" * 76)
o_p = pooled["Obs_Precip"].values
o_t = pooled["Obs_T2m"].values
emit(f"{'System':<16}{'Metric':<6}{'Value':>8}{'CI_lo':>8}{'CI_hi':>8}")
for key, name in SYS:
    p_p = pooled[f"{key}_Precip"].values
    p_t = pooled[f"{key}_T2m"].values
    for metric in ["POD", "FAR", "CSI", "ETS"]:
        val = cat_scores(p_p, o_p, 1.0)[metric]
        lo, hi = boot_ci(lambda a, b, m=metric: cat_scores(a, b, 1.0)[m], p_p, o_p)
        emit(f"{name:<16}{metric:<6}{val:8.3f}{lo:8.3f}{hi:8.3f}")
    for metric, fn in [("RMSE", lambda a, b: float(np.sqrt(np.mean((a-b)**2)))),
                       ("MAE",  lambda a, b: float(np.mean(np.abs(a-b))))]:
        val = fn(p_t, o_t)
        lo, hi = boot_ci(fn, p_t, o_t)
        emit(f"{name:<16}{metric:<6}{val:8.3f}{lo:8.3f}{hi:8.3f}")
    emit("")

# ── 3. Event-only windows ───────────────────────────────────────────────────
emit("=" * 76)
emit("3. EVENT-ONLY VERIFICATION (flood days), threshold 1 mm + temperature")
emit("=" * 76)
ev_frames = []
for year, (d0, d1) in EVENT_WINDOWS.items():
    m = pooled[(pooled["Event"] == year)
               & (pooled["UTC_Datetime"] >= d0)
               & (pooled["UTC_Datetime"] < pd.Timestamp(d1) + pd.Timedelta(hours=6))]
    ev_frames.append(m)
    emit(f"\n--- {year} event window {d0} .. {d1} (rows={len(m)}) ---")
    emit(f"{'System':<16}{'POD':>7}{'FAR':>7}{'CSI':>7}{'ETS':>7}"
         f"{'P_Bias':>8}{'T_RMSE':>8}{'T_MAE':>7}{'T_Bias':>8}")
    for key, name in SYS:
        s = cat_scores(m[f"{key}_Precip"].values, m["Obs_Precip"].values, 1.0)
        dt = m[f"{key}_T2m"].values - m["Obs_T2m"].values
        dp = m[f"{key}_Precip"].values - m["Obs_Precip"].values
        emit(f"{name:<16}{s['POD']:7.3f}{s['FAR']:7.3f}{s['CSI']:7.3f}{s['ETS']:7.3f}"
             f"{np.mean(dp):8.3f}{np.sqrt(np.mean(dt**2)):8.3f}"
             f"{np.mean(np.abs(dt)):7.3f}{np.mean(dt):8.3f}")

allev = pd.concat(ev_frames, ignore_index=True)
emit(f"\n--- All event windows pooled (rows={len(allev)}) ---")
emit(f"{'System':<16}{'POD':>7}{'FAR':>7}{'CSI':>7}{'ETS':>7}"
     f"{'P_Bias':>8}{'T_RMSE':>8}{'T_MAE':>7}{'T_Bias':>8}")
for key, name in SYS:
    s = cat_scores(allev[f"{key}_Precip"].values, allev["Obs_Precip"].values, 1.0)
    dt = allev[f"{key}_T2m"].values - allev["Obs_T2m"].values
    dp = allev[f"{key}_Precip"].values - allev["Obs_Precip"].values
    emit(f"{name:<16}{s['POD']:7.3f}{s['FAR']:7.3f}{s['CSI']:7.3f}{s['ETS']:7.3f}"
         f"{np.mean(dp):8.3f}{np.sqrt(np.mean(dt**2)):8.3f}"
         f"{np.mean(np.abs(dt)):7.3f}{np.mean(dt):8.3f}")

# Event-window higher thresholds (pooled events)
emit("\n--- All event windows pooled: higher thresholds ---")
for thr in THRESHOLDS:
    emit(f"\nThreshold {thr:g} mm")
    emit(f"{'System':<16}{'POD':>7}{'FAR':>7}{'CSI':>7}{'ETS':>7}{'ObsEv':>7}")
    for key, name in SYS:
        s = cat_scores(allev[f"{key}_Precip"].values, allev["Obs_Precip"].values, thr)
        emit(f"{name:<16}{s['POD']:7.3f}{s['FAR']:7.3f}{s['CSI']:7.3f}{s['ETS']:7.3f}{s['ObsEvents']:7d}")

with open(OUT, "w") as f:
    f.write("\n".join(lines))
print(f"\nSaved -> {OUT}")
