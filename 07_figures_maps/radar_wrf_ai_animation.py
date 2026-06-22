#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Animated 5-panel comparison of 6-hour precipitation pattern.

Each frame: RADAR | WRF (After DA) | GraphCast | FuXi | AIFS
Cadence: 6 hours
Window: 2021-07-10 06 UTC -> 2021-07-17 18 UTC
Output: ~/Desktop/For_Animation/4th_year/Miscs/radar_wrf_ai_2021_animation.mp4

Requires ffmpeg in PATH.
"""

import os
import sys
import shutil
from datetime import datetime, timedelta

if sys.platform == "darwin":
    homebrew_bin = "/opt/homebrew/bin"
    if homebrew_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = homebrew_bin + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
from matplotlib.animation import FuncAnimation, FFMpegWriter

import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from radar_wrf_ai_precip_comparison import (
    aggregate_radar_6h, wrf_6h, gc_6h, fuxi_6h, aifs_6h, radar_grid_to_latlon, make_colormap,
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TS_START = datetime(2021, 7, 10, 6)
TS_END   = datetime(2021, 7, 17, 18)
DT_HOURS = 6
FPS      = 2

OUTPUT_DIR  = "/Users/haseeb.rehman/Python scripts/output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "radar_wrf_ai_2021_animation.mp4")

ROW_LABELS  = ["RADAR", "WRF (After DA)", "GraphCast", "FuXi", "AIFS"]
ROW_KEYS    = ["RAD", "WRF", "GC", "FuXi", "AIFS"]

COUNTRY_LABELS = [
    ("BELGIUM",    4.9, 50.6),
    ("FRANCE",     5.2, 48.6),
    ("GERMANY",    7.8, 50.3),
    ("LUXEMBOURG", 6.13, 49.78),
]


def build_timestamps():
    out, t = [], TS_START
    while t <= TS_END:
        out.append(t)
        t += timedelta(hours=DT_HOURS)
    return out


def gather_all_frames(timestamps):
    frames = []
    for t in timestamps:
        print(f"  {t:%Y-%m-%d %H UTC}")
        rad, gt, wkt, files = aggregate_radar_6h(t)
        r_lat = r_lon = None
        if rad is not None:
            r_lat, r_lon = radar_grid_to_latlon(gt, wkt, rad.shape)
            print(f"      RADAR {len(files)} files  max={np.nanmax(rad):6.1f}")
        else:
            print("      RADAR  no data")

        w, w_lat, w_lon = wrf_6h(t)
        g, g_lat, g_lon = gc_6h(t)
        f, f_lat, f_lon = fuxi_6h(t)
        a, a_lat, a_lon = aifs_6h(t)

        for name, arr in zip(["WRF", "GC", "FuXi", "AIFS"], [w, g, f, a]):
            if arr is not None:
                print(f"      {name:4s}                 max={np.nanmax(arr):6.1f}")
            else:
                print(f"      {name:4s}   no data")

        frames.append({
            "t":   t,
            "RAD": (rad, r_lat, r_lon),
            "WRF": (w,    w_lat, w_lon),
            "GC":  (g,    g_lat, g_lon),
            "FuXi":(f,    f_lat, f_lon),
            "AIFS":(a,    a_lat, a_lon),
        })
    return frames


def shared_colour_scale(frames):
    all_max = []
    for fr in frames:
        for key in ROW_KEYS:
            arr = fr[key][0]
            if arr is not None:
                m = float(np.nanmax(arr))
                if np.isfinite(m):
                    all_max.append(m)
    vmax = float(np.ceil(max(all_max) / 10.0) * 10.0) if all_max else 50.0
    levels = np.arange(0, vmax + 0.001, max(2.0, vmax / 20))
    return vmax, levels


def make_animation(frames, vmax, levels):
    cmap = make_colormap()
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    proj = ccrs.PlateCarree()

    fig, axes2d = plt.subplots(2, 3, figsize=(13.8, 9.6),
                               subplot_kw={"projection": proj})
    axes = axes2d.ravel()[:5]          # 5 data panels
    axes2d.ravel()[5].set_visible(False)  # unused 6th cell

    for ax, label in zip(axes, ROW_LABELS):
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.7, edgecolor="black")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="black")
        ax.add_feature(cfeature.RIVERS,    linewidth=0.3, edgecolor="#5b8db8", alpha=0.6)
        gl = ax.gridlines(draw_labels=True, alpha=0.25, linewidth=0.4)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}
        ax.set_title(label, fontsize=12, fontweight="bold", pad=8)

        for name, lon_c, lat_c in COUNTRY_LABELS:
            ax.text(lon_c, lat_c, name, transform=proj,
                    color="#444444", fontsize=7, fontweight="bold",
                    ha="center", va="center", alpha=0.7)

    plt.subplots_adjust(left=0.05, right=0.92, top=0.90, bottom=0.05,
                        wspace=0.08, hspace=0.16)
    cbar_ax = fig.add_axes([0.935, 0.12, 0.014, 0.72])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax, extend="max")
    cbar.set_label("6-hour precipitation (mm)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    time_text = fig.text(0.5, 0.955, "", ha="center", va="bottom",
                         fontsize=13, fontweight="bold")

    contour_handles = [None] * 5
    no_data_text    = [None] * 5

    def draw_frame(i):
        fr = frames[i]
        time_text.set_text(f"Valid:  {fr['t']:%Y-%m-%d  %H:%M UTC}  (6-h accumulation)")

        for ax_idx, key in enumerate(ROW_KEYS):
            ax = axes[ax_idx]
            if contour_handles[ax_idx] is not None:
                for coll in contour_handles[ax_idx].collections:
                    coll.remove()
                contour_handles[ax_idx] = None
            if no_data_text[ax_idx] is not None:
                no_data_text[ax_idx].remove()
                no_data_text[ax_idx] = None

            arr, lat, lon = fr[key]
            if arr is None or lat is None or lon is None:
                no_data_text[ax_idx] = ax.text(
                    0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="grey", fontsize=11)
                continue
            cf = ax.contourf(lon, lat, arr, levels=levels, cmap=cmap,
                             norm=norm, transform=proj, extend="max")
            contour_handles[ax_idx] = cf

        return [time_text]

    anim = FuncAnimation(fig, draw_frame, frames=len(frames),
                         interval=1000 / FPS, blit=False, repeat=False)
    return fig, anim


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found in PATH.")
        print("Install it with:  brew install ffmpeg")
        sys.exit(1)

    timestamps = build_timestamps()
    frames = gather_all_frames(timestamps)
    vmax, levels = shared_colour_scale(frames)

    fig, anim = make_animation(frames, vmax, levels)

    writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=2500,
                          extra_args=["-pix_fmt", "yuv420p", "-preset", "medium", "-tune", "stillimage"])
    anim.save(OUTPUT_FILE, writer=writer, dpi=160)
    plt.close(fig)
    print(f"\nSaved -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
