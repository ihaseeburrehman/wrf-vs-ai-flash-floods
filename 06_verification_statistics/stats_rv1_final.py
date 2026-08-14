#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full verification statistics for GMD revision 1, using the corrected and
independent observation set (merged_pairs_rv1.csv).

Sections
  1. Data-assimilation impact (WRF Before DA vs After DA)
  2. Pooled model comparison (precipitation + temperature)
  3. Per-event breakdown
  4. Threshold sensitivity (1/5/10/20 mm per 6 h)
  5. Bootstrap 95% confidence intervals
  6. Event-only (flood-day) verification
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "merged_pairs_rv1.csv")
OUT = os.path.join(HERE, "stats_rv1_final_output.txt")

THRESHOLDS = [1.0, 5.0, 10.0, 20.0]
NBOOT = 1000
RNG = np.random.default_rng(42)
EVENT_WINDOWS = {
    "2016": ("2016-07-21", "2016-07-24"),
    "2018": ("2018-05-31", "2018-06-03"),
    "2021": ("2021-07-13", "2021-07-16"),
}

d = pd.read_csv(CACHE, parse_dates=["UTC_Datetime"])
d["Event"] = d["Event"].astype(str)

lines = []
def emit(s=""):
    print(s); lines.append(s)


def cat(p, o, thr):
    p = np.asarray(p, float); o = np.asarray(o, float)
    k = np.isfinite(p) & np.isfinite(o); p, o = p[k], o[k]
    N = p.size
    pb, ob = p >= thr, o >= thr
    H = int((pb & ob).sum()); M = int((~pb & ob).sum()); F = int((pb & ~ob).sum())
    Hr = (H + M) * (H + F) / N if N else 0
    return dict(POD=H / (H + M) if H + M else np.nan,
                FAR=F / (H + F) if H + F else np.nan,
                CSI=H / (H + M + F) if H + M + F else np.nan,
                ETS=(H - Hr) / (H + M + F - Hr) if (H + M + F - Hr) > 0 else np.nan,
                Bias=float(np.mean(p - o)), N=N, ObsEv=H + M)


def cont(p, o):
    p = np.asarray(p, float); o = np.asarray(o, float)
    k = np.isfinite(p) & np.isfinite(o); p, o = p[k], o[k]
    if p.size == 0:
        return dict(RMSE=np.nan, MAE=np.nan, Corr=np.nan, Bias=np.nan, N=0)
    return dict(RMSE=float(np.sqrt(np.mean((p - o) ** 2))),
                MAE=float(np.mean(np.abs(p - o))),
                Corr=float(np.corrcoef(p, o)[0, 1]) if p.std() > 0 and o.std() > 0 else np.nan,
                Bias=float(np.mean(p - o)), N=p.size)


def boot(fn, *cols):
    N = len(cols[0]); idx = np.arange(N); v = []
    for _ in range(NBOOT):
        s = RNG.choice(idx, N, replace=True)
        v.append(fn(*[c[s] for c in cols]))
    v = np.asarray(v, float)
    return np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)


SYS = [("WRF", "WRF (After DA)"), ("GC", "GraphCast"),
       ("FuXi", "FuXi"), ("AIFS", "AIFS")]

emit("=" * 78)
emit("GMD REVISION 1 -- VERIFICATION ON CORRECTED, INDEPENDENT OBSERVATIONS")
emit(f"pairs={len(d)}  stations/event=" +
     str(d.groupby("Event").Station.nunique().to_dict()))
emit("=" * 78)

# ── 1. DA impact ────────────────────────────────────────────────────────────
emit("\n1. DATA ASSIMILATION IMPACT (WRF)")
sub = d.dropna(subset=["WRFB_Precip"])
emit(f"   pairs with both configurations: {len(sub)}")
emit(f"\n{'Config':14}{'POD':>7}{'FAR':>7}{'CSI':>7}{'ETS':>7}{'Bias':>8}")
for key, lbl in [("WRFB", "Before DA"), ("WRF", "After DA")]:
    s = cat(sub[key + "_Precip"], sub["Obs_Precip"], 1.0)
    emit(f"{lbl:14}{s['POD']:7.3f}{s['FAR']:7.3f}{s['CSI']:7.3f}{s['ETS']:7.3f}{s['Bias']:8.3f}")
emit(f"\n{'Config':14}{'RMSE':>7}{'MAE':>7}{'Corr':>7}{'Bias':>8}")
for key, lbl in [("WRFB", "Before DA"), ("WRF", "After DA")]:
    s = cont(sub[key + "_T2m"], sub["Obs_T2m"])
    emit(f"{lbl:14}{s['RMSE']:7.3f}{s['MAE']:7.3f}{s['Corr']:7.3f}{s['Bias']:8.3f}")

# ── 2. Pooled comparison ────────────────────────────────────────────────────
emit("\n\n2. POOLED MODEL COMPARISON")
emit(f"\nPrecipitation (1 mm)\n{'System':16}{'POD':>7}{'FAR':>7}{'CSI':>7}{'ETS':>7}{'Bias':>8}{'N':>7}")
for k, nm in SYS:
    s = cat(d[k + "_Precip"], d["Obs_Precip"], 1.0)
    emit(f"{nm:16}{s['POD']:7.3f}{s['FAR']:7.3f}{s['CSI']:7.3f}{s['ETS']:7.3f}{s['Bias']:8.3f}{s['N']:7d}")
emit(f"\n2 m temperature\n{'System':16}{'RMSE':>7}{'MAE':>7}{'Corr':>7}{'Bias':>8}{'N':>7}")
for k, nm in SYS:
    s = cont(d[k + "_T2m"], d["Obs_T2m"])
    emit(f"{nm:16}{s['RMSE']:7.3f}{s['MAE']:7.3f}{s['Corr']:7.3f}{s['Bias']:8.3f}{s['N']:7d}")

# ── 3. Per event ────────────────────────────────────────────────────────────
emit("\n\n3. PER-EVENT BREAKDOWN")
for ev in ["2016", "2018", "2021"]:
    m = d[d.Event == ev]
    emit(f"\n--- {ev} (N={len(m)}, {m.Station.nunique()} stations) ---")
    emit(f"{'System':16}{'POD':>7}{'FAR':>7}{'CSI':>7}{'ETS':>7}{'PBias':>8}"
         f"{'RMSE':>7}{'MAE':>7}{'r':>7}{'TBias':>8}")
    for k, nm in SYS:
        c = cat(m[k + "_Precip"], m["Obs_Precip"], 1.0)
        t = cont(m[k + "_T2m"], m["Obs_T2m"])
        emit(f"{nm:16}{c['POD']:7.3f}{c['FAR']:7.3f}{c['CSI']:7.3f}{c['ETS']:7.3f}"
             f"{c['Bias']:8.3f}{t['RMSE']:7.3f}{t['MAE']:7.3f}{t['Corr']:7.3f}{t['Bias']:8.3f}")

# ── 4. Thresholds ───────────────────────────────────────────────────────────
emit("\n\n4. THRESHOLD SENSITIVITY (pooled)")
for thr in THRESHOLDS:
    emit(f"\n--- {thr:g} mm ---")
    emit(f"{'System':16}{'POD':>7}{'FAR':>7}{'CSI':>7}{'ETS':>7}{'ObsEv':>7}")
    for k, nm in SYS:
        s = cat(d[k + "_Precip"], d["Obs_Precip"], thr)
        emit(f"{nm:16}{s['POD']:7.3f}{s['FAR']:7.3f}{s['CSI']:7.3f}{s['ETS']:7.3f}{s['ObsEv']:7d}")

# ── 5. Bootstrap CIs ────────────────────────────────────────────────────────
emit(f"\n\n5. BOOTSTRAP 95% CI ({NBOOT} resamples, 1 mm)")
dd = d.dropna(subset=["Obs_Precip", "Obs_T2m"])
op, ot = dd["Obs_Precip"].values, dd["Obs_T2m"].values
emit(f"{'System':16}{'Metric':7}{'Value':>8}{'CI_lo':>8}{'CI_hi':>8}")
for k, nm in SYS:
    pp, pt = dd[k + "_Precip"].values, dd[k + "_T2m"].values
    for met in ["POD", "FAR", "CSI", "ETS"]:
        val = cat(pp, op, 1.0)[met]
        lo, hi = boot(lambda a, b, m=met: cat(a, b, 1.0)[m], pp, op)
        emit(f"{nm:16}{met:7}{val:8.3f}{lo:8.3f}{hi:8.3f}")
    for met, fn in [("RMSE", lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))),
                    ("MAE", lambda a, b: float(np.mean(np.abs(a - b))))]:
        val = fn(pt, ot); lo, hi = boot(fn, pt, ot)
        emit(f"{nm:16}{met:7}{val:8.3f}{lo:8.3f}{hi:8.3f}")
    emit("")

# ── 6. Event-only ───────────────────────────────────────────────────────────
emit("\n6. EVENT-ONLY (flood days)")
evf = []
for y, (a, b) in EVENT_WINDOWS.items():
    m = d[(d.Event == y) & (d.UTC_Datetime >= a) &
          (d.UTC_Datetime < pd.Timestamp(b) + pd.Timedelta(hours=6))]
    evf.append(m)
    emit(f"\n--- {y} {a}..{b} (N={len(m)}) ---")
    emit(f"{'System':16}{'POD':>7}{'FAR':>7}{'CSI':>7}{'ETS':>7}{'PBias':>8}{'RMSE':>7}{'MAE':>7}")
    for k, nm in SYS:
        c = cat(m[k + "_Precip"], m["Obs_Precip"], 1.0)
        t = cont(m[k + "_T2m"], m["Obs_T2m"])
        emit(f"{nm:16}{c['POD']:7.3f}{c['FAR']:7.3f}{c['CSI']:7.3f}{c['ETS']:7.3f}"
             f"{c['Bias']:8.3f}{t['RMSE']:7.3f}{t['MAE']:7.3f}")
ae = pd.concat(evf, ignore_index=True)
emit(f"\n--- all events pooled (N={len(ae)}) ---")
emit(f"{'System':16}{'POD':>7}{'FAR':>7}{'CSI':>7}{'ETS':>7}{'PBias':>8}{'RMSE':>7}{'MAE':>7}")
for k, nm in SYS:
    c = cat(ae[k + "_Precip"], ae["Obs_Precip"], 1.0)
    t = cont(ae[k + "_T2m"], ae["Obs_T2m"])
    emit(f"{nm:16}{c['POD']:7.3f}{c['FAR']:7.3f}{c['CSI']:7.3f}{c['ETS']:7.3f}"
         f"{c['Bias']:8.3f}{t['RMSE']:7.3f}{t['MAE']:7.3f}")
emit("\n--- event windows, CSI by threshold ---")
emit(f"{'System':16}" + "".join(f"{t:g}mm".rjust(9) for t in THRESHOLDS))
for k, nm in SYS:
    emit(f"{nm:16}" + "".join(f"{cat(ae[k+'_Precip'],ae['Obs_Precip'],t)['CSI']:9.3f}"
                              for t in THRESHOLDS))

with open(OUT, "w") as f:
    f.write("\n".join(lines))
print(f"\nSaved -> {OUT}")
