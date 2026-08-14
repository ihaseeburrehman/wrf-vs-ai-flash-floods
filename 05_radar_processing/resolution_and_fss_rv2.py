#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spatial verification for GMD revision 2 (referee RC2).

Two analyses, both restricted to the July 2021 flood peak, which is the only
period for which an open weather-radar composite (RADFLOOD21) exists:

  A. Fractions Skill Score (FSS) of the four 12 km forecasting systems against
     the radar composite, as a function of neighbourhood scale and threshold
     (RC2 comment 6). Adds spatial correlation as a secondary measure.

  B. Resolution effect: the same metrics for the WRF 12 / 4 / 1.33 km nest
     (RC2 comment 3), evaluated over the innermost domain footprint so that all
     three resolutions are scored on identical geography.

Everything is regridded to a common 0.02 deg lat/lon mesh before scoring, so
differences reflect the forecast fields rather than the native grids.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from netCDF4 import Dataset
from scipy.ndimage import uniform_filter
from scipy.interpolate import griddata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import radar_wrf_ai_precip_comparison as R

NEST = ("/Users/haseeb.rehman/Documents/Misc/From_HPC_and_WRF/WRF_Local_machine/"
        "4th_year/2021_ERA5_local_machine_3_domains/After_DA")
OUT = os.path.join(HERE, "resolution_fss_rv2_output.txt")

TIMES = [datetime(2021, 7, 14, 18), datetime(2021, 7, 15, 0), datetime(2021, 7, 15, 6)]
THRESHOLDS = [1.0, 5.0, 10.0, 20.0]
SCALES_KM = [12, 25, 50, 100]          # neighbourhood widths
GRID_DEG = 0.02                        # ~2.2 km common mesh

lines = []
def emit(s=""):
    print(s, flush=True)
    lines.append(s)


# Two scoring domains.
#   WIDE  = the radar/Fig. 9 extent, used to compare the four forecasting
#           systems. The AI models are on a 0.25 deg grid, so a domain of this
#           size is required for their fields to be resolved at all.
#   INNER = the 1.33 km domain footprint, used only for the WRF nest
#           comparison, so that all three resolutions are scored on identical
#           geography.
WIDE_LAT, WIDE_LON = (47.5, 51.5), (4.0, 8.5)
INNER_LAT, INNER_LON = (49.21, 50.12), (5.57, 6.97)
KM_PER_CELL = GRID_DEG * 111.0          # ~2.2 km


def make_grid(latr, lonr):
    gy = np.arange(latr[0], latr[1] + 1e-9, GRID_DEG)
    gx = np.arange(lonr[0], lonr[1] + 1e-9, GRID_DEG)
    return np.meshgrid(gx, gy)

GX_W, GY_W = make_grid(WIDE_LAT, WIDE_LON)
GX_I, GY_I = make_grid(INNER_LAT, INNER_LON)


def regrid(vals, lat, lon, GX=None, GY=None):
    """Nearest-neighbour regrid of a 2-D field onto the common mesh."""
    if vals is None or lat is None or lon is None:
        return None
    lat = np.asarray(lat); lon = np.asarray(lon); vals = np.asarray(vals, float)
    if lat.ndim == 1:
        lon2, lat2 = np.meshgrid(lon, lat)
    else:
        lat2, lon2 = lat, lon
    pts = np.column_stack([lat2.ravel(), lon2.ravel()])
    return griddata(pts, vals.ravel(), (GY, GX), method="nearest")


def native_cells(lat, lon, latr, lonr):
    """How many native grid points of a field fall inside a scoring domain."""
    lat = np.asarray(lat); lon = np.asarray(lon)
    if lat.ndim == 1:
        lon2, lat2 = np.meshgrid(lon, lat)
    else:
        lat2, lon2 = lat, lon
    m = ((lat2 >= latr[0]) & (lat2 <= latr[1]) &
         (lon2 >= lonr[0]) & (lon2 <= lonr[1]))
    return int(m.sum())


def fss(fc, ob, thr, scale_cells):
    """Fractions Skill Score at one threshold and neighbourhood width."""
    fb = (fc >= thr).astype(float)
    ob_ = (ob >= thr).astype(float)
    if ob_.sum() == 0 and fb.sum() == 0:
        return np.nan
    n = max(1, int(round(scale_cells)))
    ff = uniform_filter(fb, size=n, mode="constant", cval=0.0)
    of = uniform_filter(ob_, size=n, mode="constant", cval=0.0)
    num = np.mean((ff - of) ** 2)
    den = np.mean(ff ** 2) + np.mean(of ** 2)
    return 1.0 - num / den if den > 0 else np.nan


def spatial_corr(fc, ob):
    a, b = fc.ravel(), ob.ravel()
    k = np.isfinite(a) & np.isfinite(b)
    if k.sum() < 10 or a[k].std() == 0 or b[k].std() == 0:
        return np.nan
    return float(np.corrcoef(a[k], b[k])[0, 1])


# ── WRF nest reader (6-hour accumulation; RAINNC resets each cycle) ─────────
def nest_6h(domain, t):
    pat = os.path.join(NEST, "**", f"wrfout_d{domain}_{t:%Y-%m-%d_%H}_00_00")
    fs = sorted(glob.glob(pat, recursive=True))
    if not fs:
        return None, None, None
    nc = Dataset(fs[0])
    try:
        rn = np.squeeze(nc.variables["RAINNC"][:]).astype(float)
        rc = np.squeeze(nc.variables["RAINC"][:]).astype(float)
        rs = (np.squeeze(nc.variables["RAINSH"][:]).astype(float)
              if "RAINSH" in nc.variables else np.zeros_like(rn))
        lat = np.squeeze(nc.variables["XLAT"][:])
        lon = np.squeeze(nc.variables["XLONG"][:])
    finally:
        nc.close()
    return rn + rc + rs, lat, lon


def main():
    emit("=" * 78)
    emit("GMD REVISION 2 -- SPATIAL VERIFICATION AGAINST RADAR")
    emit("July 2021 flood peak only (RADFLOOD21 is the only open radar composite)")
    emit(f"common mesh {GRID_DEG} deg (~{KM_PER_CELL:.1f} km) over the 1.33 km domain footprint")
    emit("=" * 78)

    panels = []           # WIDE domain: four forecasting systems
    inner = []            # INNER domain: WRF nest only
    for t in TIMES:
        emit(f"\n--- loading {t:%Y-%m-%d %H:%M} UTC ---")
        rad, gt, wkt, _ = R.aggregate_radar_6h(t)
        if rad is None:
            emit("   radar missing, skipping"); continue
        rlat, rlon = R.radar_grid_to_latlon(gt, wkt, rad.shape)

        pw = {"t": t, "RADAR": regrid(rad, rlat, rlon, GX_W, GY_W)}
        for key, fn in [("WRF (After DA) 12 km", R.wrf_6h), ("GraphCast", R.gc_6h),
                        ("FuXi", R.fuxi_6h), ("AIFS", R.aifs_6h)]:
            v, la, lo = fn(t)
            pw[key] = regrid(v, la, lo, GX_W, GY_W)
            if t == TIMES[0] and v is not None:
                emit(f"   native cells of {key} inside WIDE domain: "
                     f"{native_cells(la, lo, WIDE_LAT, WIDE_LON)}")
        panels.append(pw)

        pi = {"t": t, "RADAR": regrid(rad, rlat, rlon, GX_I, GY_I)}
        for dom, lbl in [("01", "WRF nest 12 km"), ("02", "WRF nest 4 km"),
                         ("03", "WRF nest 1.33 km")]:
            v, la, lo = nest_6h(dom, t)
            pi[lbl] = regrid(v, la, lo, GX_I, GY_I)
            if t == TIMES[0] and v is not None:
                emit(f"   native cells of {lbl} inside INNER domain: "
                     f"{native_cells(la, lo, INNER_LAT, INNER_LON)}")
        inner.append(pi)

    if not panels:
        emit("no panels built"); return

    def score_block(title, systems, pnl):
        emit("\n" + "=" * 78)
        emit(title)
        emit("=" * 78)
        for thr in THRESHOLDS:
            emit(f"\n--- FSS at {thr:g} mm/6h (mean over {len(pnl)} panels) ---")
            emit(f"{'System':22}" + "".join(f"{s:>9}km" for s in SCALES_KM) + f"{'corr':>9}")
            for s in systems:
                if not any(p.get(s) is not None for p in pnl):
                    continue
                row = ""
                for sc in SCALES_KM:
                    cells = sc / KM_PER_CELL
                    vals = [fss(p[s], p["RADAR"], thr, cells)
                            for p in pnl if p.get(s) is not None]
                    row += f"{np.nanmean(vals):11.3f}" if len(vals) else f"{'-':>11}"
                cs = [spatial_corr(p[s], p["RADAR"]) for p in pnl if p.get(s) is not None]
                emit(f"{s:22}{row}{np.nanmean(cs):9.3f}")

    score_block("A. FOUR FORECASTING SYSTEMS (RC2 comment 6)  -- WIDE domain "
                f"{WIDE_LAT[0]}-{WIDE_LAT[1]}N, {WIDE_LON[0]}-{WIDE_LON[1]}E",
                ["WRF (After DA) 12 km", "GraphCast", "FuXi", "AIFS"], panels)
    score_block("B. RESOLUTION EFFECT: WRF 12 / 4 / 1.33 km NEST (RC2 comment 3) "
                f"-- INNER domain {INNER_LAT[0]}-{INNER_LAT[1]}N, {INNER_LON[0]}-{INNER_LON[1]}E",
                ["WRF nest 12 km", "WRF nest 4 km", "WRF nest 1.33 km"], inner)

    # domain-mean accumulation, a simple amplitude check
    emit("\n" + "=" * 78)
    emit("C. DOMAIN-MEAN AND 99th-PERCENTILE 6-h ACCUMULATION (mm)")
    emit("=" * 78)
    emit("WIDE domain:")
    emit(f"{'System':22}{'mean':>10}{'p99':>10}{'max':>10}")
    for s in ["RADAR", "WRF (After DA) 12 km", "GraphCast", "FuXi", "AIFS"]:
        vals = [p[s] for p in panels if p.get(s) is not None]
        if not vals: continue
        a = np.concatenate([v.ravel() for v in vals])
        emit(f"{s:22}{np.nanmean(a):10.2f}{np.nanpercentile(a,99):10.2f}{np.nanmax(a):10.2f}")
    emit("\nINNER domain (1.33 km footprint):")
    emit(f"{'System':22}{'mean':>10}{'p99':>10}{'max':>10}")
    for s in ["RADAR", "WRF nest 12 km", "WRF nest 4 km", "WRF nest 1.33 km"]:
        vals = [p[s] for p in inner if p.get(s) is not None]
        if not vals:
            continue
        a = np.concatenate([v.ravel() for v in vals])
        emit(f"{s:22}{np.nanmean(a):10.2f}{np.nanpercentile(a,99):10.2f}{np.nanmax(a):10.2f}")

    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    emit(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
