"""
Assimilation observation-network map -- refreshed aesthetic for the
"WRF vs GraphCast" paper. Cool light-grey land / pale-blue ocean and marker
colours drawn from the paper's muted result-figure palette (distinct from the
warm cream/steel-blue styling of the other paper).

Output: <paper>/figures/Stations_for_assimilation.png
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import patheffects
from matplotlib.lines import Line2D

import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

warnings.filterwarnings("ignore")

BASE   = "/Users/haseeb.rehman/Documents/gis4wrf/projects/2021_07_Luxembourg"
SYNOP  = f"{BASE}/SYNOP_data.shp"
TEMP   = f"{BASE}/TEMP_data.shp"
TAMDAR = f"{BASE}/TAMDAR_data.shp"
GPSZD  = f"{BASE}/GPSZD_data.shp"
OUT    = "/Users/haseeb.rehman/Documents/Research_papers/WRF_vs_GraphCast_v1/figures/Stations_for_assimilation.png"

# ---- refreshed palette
LAND = "#ECEFF1"; OCEAN = "#DCE8F0"; BORDER = "#9AA3AB"; COAST = "#6B7884"; LABELC = "#5A6570"

PLATE = ccrs.PlateCarree()

PROJ = ccrs.LambertConformal(central_longitude=6.0, central_latitude=49.63,
                             standard_parallels=(40, 58))
fig = plt.figure(figsize=(7.2, 6.0))
ax  = plt.axes(projection=PROJ)
ax.set_extent([-2.5, 16.0, 43.5, 56.5], crs=PLATE)

ax.set_facecolor(OCEAN)
ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor=OCEAN, edgecolor="none", zorder=0)
ax.add_feature(cfeature.LAND.with_scale("50m"),  facecolor=LAND,  edgecolor="none", zorder=1)
ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor=OCEAN, edgecolor="none", zorder=2)
ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.8, edgecolor=BORDER, zorder=3)
ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.8, edgecolor=COAST, zorder=3)

_pe_txt = [patheffects.withStroke(linewidth=2.0, foreground="white")]

def _load(path):
    return gpd.read_file(path).to_crs(4326)

# marker palette consistent with the paper's bar charts
obs_cfg = [
    # (shapefile, marker, face, edge, size, alpha, label, zorder)
    (SYNOP,  "o", "#4C72B0", "white", 15, 0.80, "SYNOP",    5),
    (GPSZD,  "s", "#DD8452", "white", 24, 0.90, "GNSS ZTD", 6),
    (TEMP,   "^", "#55A868", "white", 44, 0.95, "TEMP",     7),
    (TAMDAR, "D", "#8E6F8E", "white", 30, 0.95, "TAMDAR",   8),
]
for path, mk, fc, ec, sz, al, lbl, zo in obs_cfg:
    gdf = _load(path)
    ax.scatter(gdf.geometry.x, gdf.geometry.y, transform=PLATE,
               marker=mk, s=sz, facecolor=fc, edgecolor=ec,
               linewidth=0.5, alpha=al, zorder=zo)

country_labels = {
    "FRANCE": (2.5, 46.5), "GERMANY": (10.5, 51.5), "BELGIUM": (4.3, 50.7),
    "NETHERLANDS": (5.3, 52.5), "SWITZERLAND": (8.2, 47.0), "AUSTRIA": (14.0, 47.5),
    "DENMARK": (10.0, 55.5), "CZECH REP.": (15.5, 50.0), "LUX.": (6.1, 49.7),
}
for name, (lon, lat) in country_labels.items():
    ax.text(lon, lat, name, transform=PLATE, fontsize=6.5, fontweight="bold",
            color=LABELC, ha="center", va="center", zorder=9, path_effects=_pe_txt)

legend_handles = [
    Line2D([], [], marker=mk, color="none", markerfacecolor=fc,
           markeredgecolor=ec, markeredgewidth=0.6,
           markersize=np.sqrt(sz) * 0.9, label=lbl)
    for (_, mk, fc, ec, sz, _, lbl, _) in obs_cfg
]
leg = ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
                facecolor="white", edgecolor="0.7", framealpha=0.95,
                borderpad=0.6, handlelength=1.4, fancybox=True)
leg.set_zorder(20)

gl = ax.gridlines(draw_labels=True, alpha=0.4, linewidth=0.4,
                  linestyle=(0, (1, 3)), color="grey",
                  x_inline=False, y_inline=False)
gl.top_labels = False
gl.right_labels = False
gl.rotate_labels = False
gl.xlocator = mticker.FixedLocator(range(0, 17, 4))
gl.ylocator = mticker.FixedLocator(range(44, 57, 2))
gl.xlabel_style = {"size": 8}
gl.ylabel_style = {"size": 8}

plt.savefig(OUT, dpi=400, bbox_inches="tight", pad_inches=0.05)
print(f"Saved -> {OUT}")
