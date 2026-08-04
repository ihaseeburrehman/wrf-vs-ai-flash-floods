#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Revision figure (RC1 comment 5): reproduce the July 2021 peak radar-vs-model panel
figure, overlaying station-gauge 6-hour totals as filled circles on the same colour
scale so station observations, radar, and model forecasts can be compared directly.

Outputs: output/radar_wrf_ai_2021_peak_stations.png
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import radar_wrf_ai_precip_comparison as R
import compare_wrf_gc_fuxi_aifs as C

OUT = os.path.join(R.OUTPUT_DIR, "radar_wrf_ai_2021_peak_stations.png")
CACHE = os.path.join(HERE, "merged_pairs_cache.csv")
WRF_BEFORE_DIR = ("/Users/haseeb.rehman/Documents/Misc/From_HPC_and_WRF/WRF_from_HPC/"
                  "4th_year/2021_ERA5_cv5/Before_DA")


def wrf_before_6h(target_time):
    """6-hour WRF accumulation from the Before-DA (no assimilation) run."""
    saved = R.WRF_DIR
    try:
        R.WRF_DIR = WRF_BEFORE_DIR
        return R.wrf_6h(target_time)
    finally:
        R.WRF_DIR = saved

# Stations whose 2021 records are zero for the entire window (missing reported as zero)
BAD = {"Dusseldorf", "Frankfurt main", "Kassel calden", "Mirecourt", "Vatry", "Ernage"}


def load_station_obs():
    df = pd.read_csv(CACHE, parse_dates=["UTC_Datetime"])
    df["Event"] = df["Event"].astype(str)
    df = df[df["Event"] == "2021"]
    df["lat"] = df["Station"].map(lambda s: C.STATION_COORDS.get(s, (np.nan, np.nan))[0])
    df["lon"] = df["Station"].map(lambda s: C.STATION_COORDS.get(s, (np.nan, np.nan))[1])
    return df.dropna(subset=["lat", "lon"])


def main():
    os.makedirs(R.OUTPUT_DIR, exist_ok=True)
    obs = load_station_obs()

    panels = []
    for t in R.TIMESTAMPS:
        print(f"\n=== {t:%Y-%m-%d %H:%M UTC} ===")
        rad_accum, gt, wkt, files = R.aggregate_radar_6h(t)
        r_lat = r_lon = None
        if rad_accum is not None:
            r_lat, r_lon = R.radar_grid_to_latlon(gt, wkt, rad_accum.shape)
        wrf_d, wrf_lat, wrf_lon = R.wrf_6h(t)
        wrfb_d, wrfb_lat, wrfb_lon = wrf_before_6h(t)
        gc_d, gc_lat, gc_lon = R.gc_6h(t)
        fuxi_d, fuxi_lat, fuxi_lon = R.fuxi_6h(t)
        aifs_d, aifs_lat, aifs_lon = R.aifs_6h(t)
        panels.append({
            "t": t,
            "RAD": (rad_accum, r_lat, r_lon),
            "WRFB": (wrfb_d, wrfb_lat, wrfb_lon),
            "WRF": (wrf_d, wrf_lat, wrf_lon),
            "GC":  (gc_d, gc_lat, gc_lon),
            "FuXi": (fuxi_d, fuxi_lat, fuxi_lon),
            "AIFS": (aifs_d, aifs_lat, aifs_lon),
        })

    all_vals = [np.nanmax(p[k][0]) for p in panels
                for k in ("RAD", "WRFB", "WRF", "GC", "FuXi", "AIFS") if p[k][0] is not None]
    vmax = float(np.ceil(max(all_vals) / 10.0) * 10.0) if all_vals else 50.0
    levels = np.arange(0, vmax + 0.001, max(2.0, vmax / 20))
    cmap = R.make_colormap()
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    proj = ccrs.PlateCarree()
    n_cols, n_rows = len(panels), 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 4.4 * n_rows),
                             subplot_kw={"projection": proj})
    row_labels = ["RADAR", "WRF (Before DA)", "WRF (After DA)", "GraphCast", "FuXi", "AIFS"]
    row_keys = ["RAD", "WRFB", "WRF", "GC", "FuXi", "AIFS"]
    last_cf = None

    for col, p in enumerate(panels):
        # station observations valid at this panel time
        o = obs[obs["UTC_Datetime"] == p["t"]]
        o_ok = o[~o["Station"].isin(BAD)]
        o_bad = o[o["Station"].isin(BAD)]

        for row, (key, label) in enumerate(zip(row_keys, row_labels)):
            ax = axes[row, col]
            arr, lat, lon = p[key]
            ax.set_extent([R.LON_MIN, R.LON_MAX, R.LAT_MIN, R.LAT_MAX], crs=proj)

            if arr is not None and lat is not None and lon is not None:
                cf = ax.contourf(lon, lat, arr, levels=levels, cmap=cmap, norm=norm,
                                 transform=proj, extend="max")
                last_cf = cf
            else:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="grey", fontsize=11)

            # --- station gauge overlay, same colour scale ---
            if len(o_ok):
                ax.scatter(o_ok["lon"], o_ok["lat"], c=o_ok["Obs_Precip"],
                           cmap=cmap, norm=norm, transform=proj,
                           s=95, marker="o", edgecolors="black", linewidths=1.0,
                           zorder=6)
            if len(o_bad):
                ax.scatter(o_bad["lon"], o_bad["lat"], transform=proj,
                           s=95, marker="X", facecolors="none",
                           edgecolors="black", linewidths=1.2, zorder=6)

            ax.add_feature(cfeature.BORDERS, linewidth=0.7, edgecolor="black")
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black")
            ax.add_feature(cfeature.RIVERS, linewidth=0.3, edgecolor="#5b8db8", alpha=0.6)

            gl = ax.gridlines(draw_labels=True, alpha=0.25, linewidth=0.4)
            gl.top_labels = False
            gl.right_labels = False
            if col > 0:
                gl.left_labels = False
            if row < n_rows - 1:
                gl.bottom_labels = False
            gl.xlabel_style = {"size": 8}
            gl.ylabel_style = {"size": 8}

            for name, lon_c, lat_c in [("BELGIUM", 4.9, 50.6), ("FRANCE", 5.2, 48.6),
                                       ("GERMANY", 7.8, 50.3), ("LUXEMBOURG", 6.13, 49.78)]:
                ax.text(lon_c, lat_c, name, transform=proj, color="#444444",
                        fontsize=7, fontweight="bold", ha="center", va="center", alpha=0.7)

            if row == 0:
                ax.set_title(p["t"].strftime("%Y-%m-%d  %H:%M UTC"), fontsize=12, pad=6)
            if col == 0:
                ax.text(-0.17, 0.5, label, transform=ax.transAxes, rotation=90,
                        va="center", ha="center", fontsize=12, fontweight="bold")

    plt.subplots_adjust(left=0.06, right=0.91, top=0.93, bottom=0.07,
                        wspace=0.05, hspace=0.08)
    if last_cf is not None:
        cbar_ax = fig.add_axes([0.925, 0.10, 0.012, 0.80])
        cbar = fig.colorbar(last_cf, cax=cbar_ax, orientation="vertical", extend="max")
        cbar.set_label("6-hour precipitation (mm)", fontsize=11)
        cbar.ax.tick_params(labelsize=9)

    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#cccccc",
               markeredgecolor="black", markersize=10, label="Station gauge (same colour scale)"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="none",
               markeredgecolor="black", markersize=10, label="Station reporting zero for entire window"),
    ], loc="lower center", ncol=2, frameon=False, fontsize=11,
        bbox_to_anchor=(0.5, 0.045))

    plt.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
