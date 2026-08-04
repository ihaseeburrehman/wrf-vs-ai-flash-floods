"""
Study-area figure v3 (professor's layout):
Main panel  : zoom on Luxembourg + Greater Region with subtle SRTM hillshade,
              real rivers (merged_rivers.shp) labelled along their courses,
              all validation stations, and the four time-series stations named.
Inset       : Europe overview with the WRF d01 domain box and the main-panel extent.
Minimal legend (stations only); everything else explained in the caption.

Output: <paper>/figures/study_area_map.png
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.colors import LightSource

import geopandas as gpd
import rasterio
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely.geometry as sgeom

warnings.filterwarnings("ignore")

BASE   = "/Users/haseeb.rehman/Documents/gis4wrf/projects/2021_07_Luxembourg"
GR     = f"{BASE}/Greater_Region.shp"
RIVERS = f"{BASE}/merged_rivers/merged_rivers.shp"
DEM    = "/Users/haseeb.rehman/Documents/SRTM_DEM_for_study_area/DEM_SRTM_30m.tif"
OUT    = ("/Users/haseeb.rehman/Documents/Phd_thesis/Research_papers/"
          "WRF_vs_AI_v1/figures/study_area_map_rv1.png")

PLATE = ccrs.PlateCarree()
PROJ  = ccrs.LambertConformal(central_longitude=6.0, central_latitude=49.63,
                              standard_parallels=(40, 58))   # match assimilation map
EXT   = [2.75, 9.40, 47.52, 51.86]          # main panel extent

BORDER="#6E6E6E"; GRC="#C0603A"; STATION="#C44E52"; TSCOL="#5B2C6F"
RIVERC="#3A6FA8"; LABELC="#5A6570"
_pe  = [patheffects.withStroke(linewidth=2.2, foreground="white")]
_per = [patheffects.withStroke(linewidth=2.6, foreground="white")]

# ── data ─────────────────────────────────────────────────────────────────────
gr  = gpd.read_file(GR).to_crs(epsg=4326)
from shapely.geometry import Polygon, MultiPolygon
_u = gr.unary_union.buffer(0.015).buffer(-0.015)   # close slivers between admin units
if isinstance(_u, MultiPolygon):
    _u = max(_u.geoms, key=lambda g: g.area)
gr_outer = gpd.GeoDataFrame(geometry=[Polygon(_u.exterior)], crs=gr.crs)
riv = gpd.read_file(RIVERS).to_crs(epsg=4326)
riv = gpd.clip(riv, sgeom.box(EXT[0], EXT[2], EXT[1], EXT[3]))

import cartopy.io.shapereader as shpreader
_ne = gpd.read_file(shpreader.natural_earth(resolution="10m", category="cultural",
                                            name="admin_0_countries"))
lux = _ne[_ne["NAME"].str.contains("Luxem", case=False, na=False)].to_crs(epsg=4326)

with rasterio.open(DEM) as src:
    f = 16                                                   # decimation factor
    z = src.read(1, out_shape=(src.height // f, src.width // f)).astype(float)
    z = np.where(z < -100, 0.0, z)
    b = src.bounds
hs = LightSource(azdeg=315, altdeg=45).hillshade(z, vert_exag=12.0)

# ── figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(8.4, 6.8))
ax  = plt.axes(projection=PROJ)

# Largest axis-aligned (in Lambert) rectangle that stays inside the DEM footprint,
# so no corner shows outside the hillshade. Sample the DEM edges, project, inscribe.
_n = 200
_lon = np.linspace(b.left, b.right, _n)
_lat = np.linspace(b.bottom, b.top, _n)
def _xy(lons, lats):
    pts = PROJ.transform_points(PLATE, np.asarray(lons), np.asarray(lats))
    return pts[:, 0], pts[:, 1]
xl, _ = _xy(np.full(_n, b.left),  _lat)          # left edge
xr, _ = _xy(np.full(_n, b.right), _lat)          # right edge
_, yb = _xy(_lon, np.full(_n, b.bottom))         # bottom edge
_, yt = _xy(_lon, np.full(_n, b.top))            # top edge
xmin, xmax = xl.max(), xr.min()
ymin, ymax = yb.max(), yt.min()
mx = 0.01 * (xmax - xmin); my = 0.01 * (ymax - ymin)
ax.set_extent([xmin + mx, xmax - mx, ymin + my, ymax - my], crs=PROJ)

# ── elevation bands (RC1: add levels to the topography) ─────────────────────
_elev_levels = [0, 100, 200, 300, 400, 500, 600, 700]
_zlon = np.linspace(b.left, b.right, z.shape[1])
_zlat = np.linspace(b.top, b.bottom, z.shape[0])       # rasterio rows run N->S
_ZL, _ZT = np.meshgrid(_zlon, _zlat)
_terr = ax.contourf(_ZL, _ZT, z, levels=_elev_levels, cmap="terrain",
                    vmin=-250, vmax=750, extend="max", transform=PLATE,
                    alpha=0.72, zorder=0.9)
ax.contour(_ZL, _ZT, z, levels=_elev_levels[1:], colors="0.35",
           linewidths=0.25, alpha=0.55, transform=PLATE, zorder=1.2)

ax.imshow(hs, extent=[b.left, b.right, b.bottom, b.top], transform=PLATE,
          cmap="Greys_r", vmin=0.0, vmax=1.55, alpha=0.30, zorder=1.3)

ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#DCE8F0", zorder=1.5)
ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.6, edgecolor=BORDER, zorder=3)
ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.8, edgecolor=BORDER, zorder=3)

# Greater Region (outer boundary only) + Luxembourg
ax.add_geometries(gr_outer.geometry, crs=PLATE, facecolor="none", edgecolor=GRC,
                  linewidth=1.6, zorder=4)
ax.add_geometries(lux.geometry, crs=PLATE, facecolor=GRC, alpha=0.13,
                  edgecolor=GRC, linewidth=1.4, zorder=4)

# Rivers: real geometries, labelled along their courses
MAJOR = {"Moselle", "Meuse", "Rhine"}
for _, row in riv.iterrows():
    g = row.geometry
    if g is None or g.is_empty:
        continue
    lw = 1.5 if row["river_name"] in MAJOR else 0.95
    ax.add_geometries([g], crs=PLATE, facecolor="none", edgecolor=RIVERC,
                      linewidth=lw, zorder=3.6)

LABEL_AT = {  # river -> fraction along the (longest) line + label size
    "Moselle": (0.42, 8.5), "Meuse": (0.28, 8.5), "Rhine": (0.50, 8.5),
    "Sauer": (0.72, 7.5), "Our": (0.15, 7.5),
    "Marne": (0.50, 7.5),
}
for _, row in riv.iterrows():
    name = row["river_name"]
    if name not in LABEL_AT:
        continue
    frac, fs = LABEL_AT[name]
    g = row.geometry
    line = g if g.geom_type == "LineString" else max(g.geoms, key=lambda s: s.length)
    p  = line.interpolate(frac, normalized=True)
    p1 = line.interpolate(min(frac + 0.02, 1.0), normalized=True)
    p0 = line.interpolate(max(frac - 0.02, 0.0), normalized=True)
    ang = np.degrees(np.arctan2(p1.y - p0.y, p1.x - p0.x))
    if ang > 90:  ang -= 180
    if ang < -90: ang += 180
    ax.text(p.x, p.y, name, transform=PLATE, fontsize=fs, style="italic",
            color="#1F4E79", rotation=ang, rotation_mode="anchor",
            ha="center", va="bottom", zorder=8, path_effects=_per)

# Validation stations
met = [(50.12385,6.06622),(49.8031,6.44337),(49.85172,6.09754),(49.5122,5.9011),
       (49.491,6.349),(49.63265,6.23293),(49.7945,5.8202),(49.99314,6.10147),
       (49.76739,5.96748),(49.63353,6.0193),(49.85891,5.84868),(50.09686,5.96961),
       (49.68087,6.43541),(50.0093,5.8475),(49.79806,6.2773),(49.8741,6.2095),
       (49.91445,6.19508),(49.762,6.11179),(49.93595,5.98093),(50.9,3.117),
       (47.917,7.4),(49.973,6.693),(51.408,9.378),(48.776,4.184),(50.583,4.683),
       (51.289,6.767),(50.637,5.443),(48.325,6.07),(50.026,8.543),(51.199,2.862),
       (51.35,3.2),(51.115,9.286),(47.85,3.497),(47.59,7.53)]
ax.scatter([s[1] for s in met], [s[0] for s in met], transform=PLATE, marker="o",
           s=30, facecolor=STATION, edgecolor="white", linewidth=0.6, zorder=9)

# Time-series stations (Section 6.3)
ts = {"Ettelbruck":  (49.85172, 6.09754, (-0.62,  0.16)),
      "Spangdahlem": (49.973,   6.693,   ( 0.62,  0.16)),
      "Liège":  (50.637,   5.443,   ( 0.02,  0.20)),
      "Vatry":       (48.776,   4.184,   ( 0.02, -0.33))}
ax.scatter([v[1] for v in ts.values()], [v[0] for v in ts.values()], transform=PLATE,
           marker="s", s=66, facecolor=TSCOL, edgecolor="white", linewidth=0.9, zorder=10)
for name, (la, lo, (dx, dy)) in ts.items():
    ax.text(lo + dx, la + dy, name, transform=PLATE, fontsize=8.2, fontweight="bold",
            color=TSCOL, ha="center", va="center", zorder=10, path_effects=_per)

# Country labels
for c, (lo, la) in {"GERMANY": (8.45, 51.25), "FRANCE": (3.85, 48.10),
                    "BELGIUM": (4.15, 50.55), "LUXEMBOURG": (5.38, 49.33)}.items():
    ax.text(lo, la, c, transform=PLATE, fontsize=8, fontweight="bold", color=LABELC,
            ha="center", va="center", zorder=7, path_effects=_pe)

# Minimal legend: stations only
leg = ax.legend(handles=[
        Line2D([], [], marker="o", color="none", markerfacecolor=STATION,
               markeredgecolor="white", markersize=7, label="Validation stations"),
        Line2D([], [], marker="s", color="none", markerfacecolor=TSCOL,
               markeredgecolor="white", markersize=8, label="Time-series stations"),
    ], loc="upper left", fontsize=8.5, facecolor="white", edgecolor="0.7",
    framealpha=0.95, borderpad=0.55, fancybox=True)
leg.set_zorder(20)

gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False,
                  linewidth=0.4, color="grey", alpha=0.4, linestyle=(0, (1, 3)))
gl.top_labels = False; gl.right_labels = False; gl.rotate_labels = False
gl.xlocator = mticker.FixedLocator(np.arange(3, 10, 1))
gl.ylocator = mticker.FixedLocator(np.arange(48, 52, 1))
gl.xlabel_style = {"size": 8}; gl.ylabel_style = {"size": 8}

# ── inset: Europe overview with WRF d01 box ─────────────────────────────────
axi = fig.add_axes([0.655, 0.115, 0.27, 0.27], projection=PLATE)
axi.set_extent([-4.5, 16.5, 42.5, 56.5], crs=PLATE)
axi.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#DCE8F0", zorder=0)
axi.add_feature(cfeature.LAND.with_scale("50m"),  facecolor="#F2F2F0", zorder=1)
axi.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.35, edgecolor="#ADB5BC", zorder=2)
axi.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.4, edgecolor="#8A949E", zorder=2)
d = dict(lo0=-1.324371337890625, la0=44.60498809814453,
         lo1=13.7489013671875,  la1=54.220977783203125)
axi.plot([d['lo0'], d['lo1'], d['lo1'], d['lo0'], d['lo0']],
         [d['la0'], d['la0'], d['la1'], d['la1'], d['la0']],
         transform=PLATE, color="#2C5F8A", linewidth=1.4, zorder=5)
axi.text(d['lo1']-0.8, d['la1']-1.0, "d01", transform=PLATE, fontsize=7,
         fontweight="bold", color="#2C5F8A", ha="right", va="top", zorder=6,
         path_effects=_pe)
axi.plot([EXT[0], EXT[1], EXT[1], EXT[0], EXT[0]],
         [EXT[2], EXT[2], EXT[3], EXT[3], EXT[2]],
         transform=PLATE, color=GRC, linewidth=1.1, zorder=6)
for sp in axi.spines.values():
    sp.set_edgecolor("0.45"); sp.set_linewidth(0.8)

cax = ax.inset_axes([0.022, 0.045, 0.022, 0.30])
cax.set_facecolor("white")
cb = fig.colorbar(_terr, cax=cax, orientation="vertical", extend="max")
cb.set_label("Elevation (m a.s.l.)", fontsize=7, labelpad=2)
cb.ax.tick_params(labelsize=6, length=2, pad=1.5)
cb.outline.set_linewidth(0.5)
cb.ax.set_zorder(20)

plt.savefig(OUT, dpi=400, bbox_inches="tight", pad_inches=0.05)
print(f"Saved -> {OUT}")
