#!/usr/bin/env python3
"""
Study-area map for the WRF-vs-AI -> LISFLOOD-FP paper (v1).
Distinct look from the WRF-LISFLOOD paper: terrain+hillshade DEM, the 5 validation
gauges (discharge vs water-level markers), and an overlay of the AI models' ~0.25 deg
grid to make the resolution-mismatch point. Plain matplotlib in EPSG:4326 (no cartopy).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from matplotlib.colors import LightSource
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from pyproj import Transformer

BASE = "/Users/haseeb.rehman/Documents/Misc/Lisflood_Simulations/Lisflood_Alzette_river_basin"
DEM = f"{BASE}/sub_basins/10m/ready_for_simulation/Alzette_sub_basin_10m_bridge_burn.asc"
RIVER = f"{BASE}/sub_basins/5m/sub_basin_complete/pre_processing/alzette_river.shp"
STREAMS = f"{BASE}/sub_basins/5m/sub_basin_complete/pre_processing/streams_alzette_basin.shp"
FULL_BASIN = f"{BASE}/Alzette_basin_cleaned.shp"
COUNTRIES = "/Users/haseeb.rehman/Documents/gis4wrf/projects/2021_07_Luxembourg/Countries_near_Lux.shp"
LUX = "/Users/haseeb.rehman/Documents/gis4wrf/projects/2021_07_Luxembourg/Luxembourg_Regions.shp"
OUT = "/Users/haseeb.rehman/Documents/Phd_thesis/Research_papers/WRF_vs_AI_LISFLOOD_v1/figures/figure1.png"

STATIONS = {
    "Walferdange": (77256, 81571, "water_level"),
    "Steinsel":    (77432, 82659, "discharge"),
    "Pfaffenthal": (77409, 76226, "discharge"),
    "Livange":     (76151, 65753, "discharge"),
    "Hesperange":  (78623, 72404, "discharge"),
}
AI_RES = 0.25  # deg, GraphCast/FuXi/AIFS native grid

os.makedirs(os.path.dirname(OUT), exist_ok=True)

# --- DEM -> EPSG:4326 ---
with rasterio.open(DEM) as src:
    scrs = src.crs if src.crs else "EPSG:2169"
    tr, w, h = calculate_default_transform(scrs, "EPSG:4326", src.width, src.height, *src.bounds)
    dst = np.full((h, w), -9999, dtype=np.float32)
    reproject(src.read(1), dst, src_transform=src.transform, src_crs=scrs,
              dst_transform=tr, dst_crs="EPSG:4326",
              resampling=Resampling.bilinear, src_nodata=-9999, dst_nodata=-9999)
dem = np.ma.masked_equal(dst, -9999)
b = rasterio.transform.array_bounds(h, w, tr)
ext = [b[0], b[2], b[1], b[3]]  # W,E,S,N
vmin, vmax = float(dem.min()), float(dem.max())

fig, ax = plt.subplots(figsize=(11, 10))
# terrain + hillshade
ax.imshow(dem, extent=ext, origin="upper", cmap="terrain", vmin=vmin, vmax=vmax, zorder=1)
ls = LightSource(azdeg=315, altdeg=45)
hs = ls.hillshade(dem.filled(vmin), vert_exag=0.0008, dx=10, dy=10)
ax.imshow(np.ma.masked_array(hs, dem.mask), extent=ext, origin="upper",
          cmap="gray", alpha=0.35, zorder=1.1)

# rivers
try:
    gpd.read_file(STREAMS).to_crs(4326).plot(ax=ax, color="#1f6fff", linewidth=0.5, alpha=0.85, zorder=2)
except Exception as e:
    print("streams:", e)
try:
    gpd.read_file(RIVER).to_crs(4326).plot(ax=ax, color="#0040d0", linewidth=1.3, alpha=0.95, zorder=2.2)
except Exception as e:
    print("river:", e)

# --- AI 0.25 deg grid overlay ---
glon = np.arange(np.floor(ext[0] / AI_RES) * AI_RES, ext[1] + AI_RES, AI_RES)
glat = np.arange(np.floor(ext[2] / AI_RES) * AI_RES, ext[3] + AI_RES, AI_RES)
for gl in glon:
    ax.axvline(gl, color="0.25", lw=1.1, ls=(0, (5, 3)), alpha=0.8, zorder=3)
for gl in glat:
    ax.axhline(gl, color="0.25", lw=1.1, ls=(0, (5, 3)), alpha=0.8, zorder=3)
ncell = 0
for i in range(len(glon) - 1):
    for j in range(len(glat) - 1):
        if glon[i] < ext[1] and glon[i + 1] > ext[0] and glat[j] < ext[3] and glat[j + 1] > ext[2]:
            ax.add_patch(Rectangle((glon[i], glat[j]), AI_RES, AI_RES, facecolor="orange",
                                   alpha=0.12, edgecolor="none", zorder=2.6))
            ncell += 1
ax.text(0.98, 0.02, f"AI native grid: {AI_RES}°  (≈{ncell} cells over the domain)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=9, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.4", alpha=0.85), zorder=6)

# --- gauges ---
t2169 = Transformer.from_crs(2169, 4326, always_xy=True)
for name, (x, y, typ) in STATIONS.items():
    lon, lat = t2169.transform(x, y)
    mk, col = ("s", "#d62728") if typ == "water_level" else ("^", "#ffd400")
    ax.scatter(lon, lat, marker=mk, s=90, c=col, edgecolors="black", linewidths=0.8, zorder=5)
    ax.annotate(name, (lon, lat), textcoords="offset points", xytext=(6, 4),
                fontsize=8, fontweight="bold", color="black", zorder=6)

ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
ax.set_aspect(1.0 / np.cos(np.radians((ext[2] + ext[3]) / 2)))
ax.set_xlabel("Longitude (°E)", fontsize=13); ax.set_ylabel("Latitude (°N)", fontsize=13)
ax.tick_params(labelsize=11)

cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap="terrain"),
                  ax=ax, fraction=0.035, pad=0.02)
cb.set_label("Elevation (m a.s.l.)", fontsize=11)

leg = [Line2D([0], [0], color="#0040d0", lw=1.3, label="Alzette river / streams"),
       Line2D([0], [0], marker="^", color="#ffd400", mec="black", ls="None", ms=9, label="Discharge gauge"),
       Line2D([0], [0], marker="s", color="#d62728", mec="black", ls="None", ms=9, label="Water-level gauge"),
       Line2D([0], [0], color="0.25", lw=1.1, ls=(0, (5, 3)), label=f"AI {AI_RES}° grid")]
ax.legend(handles=leg, loc="upper left", fontsize=8, framealpha=0.92)

# north arrow
ax.annotate("", xy=(0.06, 0.93), xytext=(0.06, 0.86), xycoords="axes fraction",
            arrowprops=dict(facecolor="black", width=2, headwidth=8))
ax.text(0.06, 0.935, "N", transform=ax.transAxes, ha="center", fontsize=11, fontweight="bold")

# --- inset: Luxembourg context ---
axin = ax.inset_axes([0.66, 0.04, 0.32, 0.34])
try:
    gpd.read_file(COUNTRIES).to_crs(4326).plot(ax=axin, facecolor="none", edgecolor="0.5", lw=0.5)
    gpd.read_file(LUX).to_crs(4326).dissolve().plot(ax=axin, facecolor="#f0f0f0", edgecolor="black", lw=0.7)
    fb = gpd.read_file(FULL_BASIN)
    fb = fb.set_crs(2169) if fb.crs is None else fb
    fb.to_crs(4326).plot(ax=axin, facecolor="#9ecae1", edgecolor="none", alpha=0.7)
    cx, cy = (ext[0] + ext[1]) / 2, (ext[2] + ext[3]) / 2
    axin.add_patch(Rectangle((ext[0], ext[2]), ext[1] - ext[0], ext[3] - ext[2],
                             facecolor="red", edgecolor="darkred", alpha=0.8))
    lb = gpd.read_file(LUX).to_crs(4326).total_bounds
    axin.set_xlim(lb[0] - 0.15, lb[2] + 0.15); axin.set_ylim(lb[1] - 0.1, lb[3] + 0.1)
except Exception as e:
    print("inset:", e)
axin.set_xticks([]); axin.set_yticks([]); axin.set_aspect(1.0 / np.cos(np.radians(49.8)))
for s in axin.spines.values():
    s.set_edgecolor("black"); s.set_linewidth(0.8)

plt.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
print("SAVED", OUT, "| AI cells over domain:", ncell, "| elev", round(vmin), round(vmax))
