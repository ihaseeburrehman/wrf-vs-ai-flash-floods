#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-way 6-hour precipitation comparison for the 2021 Luxembourg flood
peak — Belgium RADAR vs WRF (After DA, ERA5-driven) vs AI Models (GraphCast, FuXi, AIFS).

Layout: one figure, 3 columns (timestamps) x 5 rows (datasets)

    cols : 2021-07-14 18 UTC, 2021-07-15 00 UTC, 2021-07-15 06 UTC
    rows : Belgium RADAR / WRF After_DA / GraphCast / FuXi / AIFS

Output:
    /Users/haseeb.rehman/Desktop/For_Animation/4th_year/Miscs/
        radar_wrf_ai_2021_peak.png
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm

import cartopy.crs as ccrs
import cartopy.feature as cfeature

import xarray as xr
from osgeo import gdal, osr
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree

gdal.UseExceptions()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TIMESTAMPS = [
    datetime(2021, 7, 14, 18),  # valid 18 UTC (6 h ending 13-18 UTC)
    datetime(2021, 7, 15,  0),  # valid 00 UTC (6 h ending 19-00 UTC)
    datetime(2021, 7, 15,  6),  # valid 06 UTC (6 h ending 01-06 UTC)
]

# Belgium radar
RADAR_BASE = "/Users/haseeb.rehman/Documents/Misc/Data_Datasets/Radar_and_Weather/Belgium_Radar_data_2021"

# WRF After_DA (ERA5-driven)
WRF_DIR = "/Users/haseeb.rehman/Documents/Misc/From_HPC_and_WRF/WRF_from_HPC/4th_year/2021_ERA5_cv5/After_DA"
WRF_PREFIX = "wrfout_d01_"

# AI Models Directories
GC_DIR = "/Users/haseeb.rehman/Documents/Misc/AI_Models/GraphCast/2021_Event/forecasts/rapid"
FUXI_DIR = "/Users/haseeb.rehman/Documents/Misc/AI_Models/FuXi/2021_Event/forecasts/rapid"
AIFS_DIR = "/Users/haseeb.rehman/Documents/Misc/AI_Models/AIFS/2021_Event/forecasts/rapid"

# Plot extent (Greater Region: Luxembourg + neighbours)
LON_MIN, LON_MAX = 4.0, 8.5
LAT_MIN, LAT_MAX = 47.5, 51.5

OUTPUT_DIR = "/Users/haseeb.rehman/Python scripts/output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "radar_wrf_ai_2021_peak.png")


# ---------------------------------------------------------------------------
# RADAR --------------------------------------------------------------------
# ---------------------------------------------------------------------------
def _radar_dir_for(ts):
    return os.path.join(RADAR_BASE,
                        f"{ts:%Y}", f"{ts:%m}", f"{ts:%d}", "accum1h", "tif")


def _read_geotiff(path):
    ds = gdal.Open(path)
    if ds is None:
        return None, None
    arr = ds.GetRasterBand(1).ReadAsArray().astype(float)
    nd = ds.GetRasterBand(1).GetNoDataValue()
    if nd is not None:
        arr = np.where(arr == nd, 0.0, arr)
    arr = np.where(arr < 0, 0.0, arr)
    arr /= 100.0
    gt = ds.GetGeoTransform()
    wkt = ds.GetProjection()
    return arr, (gt, wkt)


def aggregate_radar_6h(target_time):
    accum = None
    gt = None
    wkt = None
    used = []
    for k in range(6):
        t = target_time - timedelta(hours=5 - k)
        fname = f"{t:%Y%m%d%H}0000.radclim.accum1h.tif"
        path = os.path.join(_radar_dir_for(t), fname)
        if not os.path.exists(path):
            continue
        arr, meta = _read_geotiff(path)
        if arr is None:
            continue
        if accum is None:
            accum = arr.copy()
            gt, wkt = meta
        else:
            accum += arr
        used.append(os.path.basename(path))
    if accum is None:
        return None, None, None, []
    return accum, gt, wkt, used


def radar_grid_to_latlon(gt, wkt, shape):
    ny, nx = shape
    cols = np.arange(nx) + 0.5
    rows = np.arange(ny) + 0.5
    ox, dx, _, oy, _, dy = gt
    X = ox + cols * dx
    Y = oy + rows * dy
    X2, Y2 = np.meshgrid(X, Y)
    t = Transformer.from_crs(CRS.from_wkt(wkt), "EPSG:4326", always_xy=True)
    lon, lat = t.transform(X2, Y2)
    return lat, lon


# ---------------------------------------------------------------------------
# WRF ----------------------------------------------------------------------
# ---------------------------------------------------------------------------
def _wrf_path(ts):
    return os.path.join(WRF_DIR, WRF_PREFIX + ts.strftime("%Y-%m-%d_%H_%M_%S"))


def _wrf_cum(path):
    ds = gdal.Open(path)
    if ds is None:
        return None, None, None
    subs = ds.GetSubDatasets()

    def find(var):
        for s in subs:
            if s[0].rsplit(":", 1)[-1] == var:
                return s[0]
        return None
    needed = {v: find(v) for v in ("RAINNC", "RAINC", "RAINSH", "XLAT", "XLONG")}
    if None in needed.values():
        return None, None, None
    cum = (gdal.Open(needed["RAINC"]).ReadAsArray()
         + gdal.Open(needed["RAINNC"]).ReadAsArray()
         + gdal.Open(needed["RAINSH"]).ReadAsArray())
    lat = gdal.Open(needed["XLAT"]).ReadAsArray()
    lon = gdal.Open(needed["XLONG"]).ReadAsArray()
    cum = np.squeeze(cum); lat = np.squeeze(lat); lon = np.squeeze(lon)
    cum = np.where(np.isfinite(cum), cum, 0.0)
    return cum, lat, lon


def wrf_6h(target_time):
    p, lat, lon = _wrf_cum(_wrf_path(target_time))
    if p is None:
        return None, None, None
    return np.clip(p, 0, None), lat, lon


# ---------------------------------------------------------------------------
# AI Models (GraphCast, FuXi, AIFS) ----------------------------------------
# ---------------------------------------------------------------------------
def ai_model_6h(target_time, model_dir):
    init = target_time - timedelta(hours=6)
    path = os.path.join(model_dir, f"forecast_{init:%Y%m%dT%H}.nc")
    if not os.path.exists(path):
        return None, None, None
    
    with xr.open_dataset(path) as ds:
        # Some models use different variable names, check for known ones
        if "total_precipitation_6hr" in ds:
            pr = ds["total_precipitation_6hr"].squeeze().values
        elif "tp" in ds:
            pr = ds["tp"].squeeze().values
        else:
            return None, None, None
            
        # Convert to mm if likely in meters
        if np.nanmax(pr) < 5.0 and np.nanmax(pr) > 0.0001:
            pr = pr * 1000.0
            
        if "lat" in ds:
            lats = ds["lat"].values
            lons = ds["lon"].values
        elif "latitude" in ds:
            lats = ds["latitude"].values
            lons = ds["longitude"].values
        else:
            return None, None, None
        
    pr = np.where(np.isfinite(pr), pr, 0.0)
    if pr.ndim == 1:
        mask = (lats >= LAT_MIN - 1.0) & (lats <= LAT_MAX + 1.0) & (lons >= LON_MIN - 1.0) & (lons <= LON_MAX + 1.0)
        p_1d, lat_1d, lon_1d = pr[mask], lats[mask], lons[mask]
        if len(p_1d) == 0:
            return None, None, None
        from scipy.interpolate import griddata
        grid_lat = np.linspace(LAT_MIN - 0.5, LAT_MAX + 0.5, 100)
        grid_lon = np.linspace(LON_MIN - 0.5, LON_MAX + 0.5, 100)
        lon2d, lat2d = np.meshgrid(grid_lon, grid_lat)
        pr_2d = griddata((lon_1d, lat_1d), p_1d, (lon2d, lat2d), method='linear', fill_value=0.0)
        return pr_2d, lat2d, lon2d
    else:
        la_m = (lats >= LAT_MIN - 0.5) & (lats <= LAT_MAX + 0.5)
        lo_m = (lons >= LON_MIN - 0.5) & (lons <= LON_MAX + 0.5)
        pr = pr[np.ix_(la_m, lo_m)]
        lat_sub, lon_sub = lats[la_m], lons[lo_m]
        lon2d, lat2d = np.meshgrid(lon_sub, lat_sub)
        return pr, lat2d, lon2d

def gc_6h(target_time):
    return ai_model_6h(target_time, GC_DIR)

def fuxi_6h(target_time):
    return ai_model_6h(target_time, FUXI_DIR)

def aifs_6h(target_time):
    return ai_model_6h(target_time, AIFS_DIR)


# ---------------------------------------------------------------------------
# Plot ---------------------------------------------------------------------
# ---------------------------------------------------------------------------
def make_colormap():
    base = matplotlib.colormaps.get_cmap("YlGnBu")
    cols = [(1, 1, 1, 1)] + [base(i / 255) for i in range(20, 256)]
    return mcolors.LinearSegmentedColormap.from_list("WhiteYlGnBu", cols, N=256)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    panels = []
    for t in TIMESTAMPS:
        print(f"\n=== {t:%Y-%m-%d %H:%M UTC} ===")
        rad_accum, gt, wkt, files = aggregate_radar_6h(t)
        r_lat = r_lon = None
        if rad_accum is not None:
            r_lat, r_lon = radar_grid_to_latlon(gt, wkt, rad_accum.shape)
            print(f"  Radar: max = {np.nanmax(rad_accum):.1f} mm")

        wrf_d, wrf_lat, wrf_lon = wrf_6h(t)
        gc_d, gc_lat, gc_lon = gc_6h(t)
        fuxi_d, fuxi_lat, fuxi_lon = fuxi_6h(t)
        aifs_d, aifs_lat, aifs_lon = aifs_6h(t)

        for name, arr in zip(["WRF", "GC", "FuXi", "AIFS"], [wrf_d, gc_d, fuxi_d, aifs_d]):
            if arr is not None:
                print(f"  {name}: max = {np.nanmax(arr):.1f} mm")
            else:
                print(f"  {name}: NO data")

        panels.append({
            "t":   t,
            "RAD": (rad_accum, r_lat, r_lon),
            "WRF": (wrf_d,    wrf_lat, wrf_lon),
            "GC":  (gc_d,     gc_lat,  gc_lon),
            "FuXi":(fuxi_d,   fuxi_lat, fuxi_lon),
            "AIFS":(aifs_d,   aifs_lat, aifs_lon),
        })

    all_vals = []
    for p in panels:
        for key in ("RAD", "WRF", "GC", "FuXi", "AIFS"):
            arr = p[key][0]
            if arr is not None:
                all_vals.append(np.nanmax(arr))
    vmax = float(np.ceil(max(all_vals) / 10.0) * 10.0) if all_vals else 50.0
    levels = np.arange(0, vmax + 0.001, max(2.0, vmax / 20))
    cmap = make_colormap()
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    proj = ccrs.PlateCarree()
    n_cols = len(panels)
    n_rows = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 4.4 * n_rows),
                             subplot_kw={"projection": proj})

    row_labels = ["RADAR", "WRF (After DA)", "GraphCast", "FuXi", "AIFS"]
    row_keys   = ["RAD", "WRF", "GC", "FuXi", "AIFS"]
    last_cf = None

    for col, p in enumerate(panels):
        for row, (key, label) in enumerate(zip(row_keys, row_labels)):
            ax = axes[row, col]
            arr, lat, lon = p[key]
            ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)

            if arr is not None and lat is not None and lon is not None:
                cf = ax.contourf(lon, lat, arr, levels=levels, cmap=cmap, norm=norm,
                                 transform=proj, extend="max")
                last_cf = cf
            else:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="grey", fontsize=11)

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

            country_labels = [
                ("BELGIUM", 4.9, 50.6),
                ("FRANCE", 5.2, 48.6),
                ("GERMANY", 7.8, 50.3),
                ("LUXEMBOURG", 6.13, 49.78)
            ]
            for name, lon_c, lat_c in country_labels:
                ax.text(lon_c, lat_c, name, transform=proj,
                        color="#444444", fontsize=7, fontweight="bold",
                        ha="center", va="center", alpha=0.7)

            if row == 0:
                ax.set_title(p["t"].strftime("%Y-%m-%d  %H:%M UTC"),
                             fontsize=12, pad=6)
            if col == 0:
                ax.text(-0.17, 0.5, label, transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=12,
                        fontweight="bold")

    plt.subplots_adjust(left=0.06, right=0.91, top=0.93, bottom=0.07,
                        wspace=0.05, hspace=0.08)
    if last_cf is not None:
        cbar_ax = fig.add_axes([0.925, 0.10, 0.012, 0.80])
        cbar = fig.colorbar(last_cf, cax=cbar_ax, orientation="vertical",
                            extend="max")
        cbar.set_label("6-hour precipitation (mm)", fontsize=11)
        cbar.ax.tick_params(labelsize=9)

    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved -> {OUTPUT_FILE}")

    output_ts_file = os.path.join(OUTPUT_DIR, "radar_wrf_ai_ts_comparison.png")
    generate_timeseries_plot(output_ts_file)


def generate_timeseries_plot(output_ts_file):
    print("\n=== Generating Time Series at Validation Stations ===")
    ob_dir = ("/Users/haseeb.rehman/Documents/Misc/Data_Datasets/"
              "Stations_and_Observations/Luxembourg_stations_for_validation")
    obs_gen_path = os.path.join(ob_dir, "2021_Event", "stations_6hr_cumulative.xlsx")
    obs_oth_path = os.path.join(ob_dir, "2021_Event", "Stations_other_than_lux",
                                "station_weather_data_june_july_2021_6hr.xlsx")

    candidates = {
        "Luxembourg": [
            ("Ettelbruck",    (49.85172, 6.09754), obs_gen_path),
            ("Echternach",    (49.8031,  6.44337), obs_gen_path),
            ("Mamer",         (49.63353, 6.0193),  obs_gen_path),
            ("Useldange",     (49.76739, 5.96748), obs_gen_path),
        ],
        "Germany": [
            ("SAARBRUCKEN",   (49.215, 7.110),    obs_oth_path),
            ("Spangdahlem ab",(49.973, 6.693),    obs_oth_path),
            ("Trier",         (49.7472, 6.6583),  obs_oth_path),
        ],
        "Belgium": [
            ("Liege",         (50.637, 5.443),    obs_oth_path),
            ("Charleroi",     (50.459, 4.453),    obs_oth_path),
            ("ERNAGE",        (50.585, 4.683),    obs_oth_path),
            ("KOKSIJDE",      (51.090, 2.652),    obs_oth_path),
        ],
        "France": [
            ("TILLE",         (49.4544, 2.1128),  obs_oth_path),
            ("Vatry",         (48.776, 4.184),    obs_oth_path),
            ("Metz",          (49.0717, 6.1311),  obs_oth_path),
        ],
    }

    def pick_station(country, cand_list):
        for name, coords, path in cand_list:
            if not os.path.exists(path): continue
            try: sheets = pd.ExcelFile(path).sheet_names
            except Exception: continue
            matched = next((s for s in sheets if s.strip().lower() == name.strip().lower()), None)
            if matched: return matched, coords, path
        return None, None, None

    ts_stations = {}
    for country, cand in candidates.items():
        name, coords, path = pick_station(country, cand)
        if name: ts_stations[name] = {"coords": coords, "country": country, "path": path}

    if not ts_stations:
        print("  ! No stations resolved — aborting time series plot")
        return

    obs_data = {}
    for s, info in ts_stations.items():
        df = pd.read_excel(info["path"], sheet_name=s)
        df.columns = [str(c).strip() for c in df.columns]
        dt_col = next((c for c in df.columns if c.lower().replace(" ", "") in
                       ("utc_datetime", "utcdatetime", "datetime", "date")), None)
        if not dt_col: continue
        dt = pd.to_datetime(df[dt_col], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
        if dt.isna().all(): dt = pd.to_datetime(df[dt_col], errors="coerce")
        df[dt_col] = dt
        df = df.dropna(subset=[dt_col])
        pr_col = next((c for c in df.columns if "precip" in c.lower() and "mm" in c.lower()), None)
        if not pr_col: continue
        df = df.rename(columns={dt_col: "UTC_Datetime", pr_col: "Obs_Precip"})
        obs_data[s] = df.set_index("UTC_Datetime")["Obs_Precip"]

    ts_start = datetime(2021, 7, 10, 0)
    ts_end   = datetime(2021, 7, 17, 18)
    ts_times = []
    t = ts_start
    while t <= ts_end:
        ts_times.append(t)
        t += timedelta(hours=6)
    
    station_names = list(ts_stations.keys())
    station_lats = [ts_stations[s]["coords"][0] for s in station_names]
    station_lons = [ts_stations[s]["coords"][1] for s in station_names]
    
    ts_results = {s: {"times": [], "Obs": [], "Radar": [], "WRF": [], "GC": [], "FuXi": [], "AIFS": []} for s in station_names}
    
    for ts in ts_times:
        # RADAR
        accum = None
        gt, wkt = None, None
        for k in range(6):
            t_sub = ts - timedelta(hours=5 - k)
            fname = f"{t_sub:%Y%m%d%H}0000.radclim.accum1h.tif"
            path = os.path.join(RADAR_BASE, f"{t_sub:%Y}", f"{t_sub:%m}", f"{t_sub:%d}", "accum1h", "tif", fname)
            if os.path.exists(path):
                ds = gdal.Open(path)
                arr = ds.GetRasterBand(1).ReadAsArray().astype(float)
                nd = ds.GetRasterBand(1).GetNoDataValue()
                if nd is not None: arr = np.where(arr == nd, 0.0, arr)
                arr = np.where(arr < 0, 0.0, arr)
                if accum is None:
                    accum = arr.copy(); gt = ds.GetGeoTransform(); wkt = ds.GetProjection()
                else: accum += arr
        
        radar_vals = [np.nan] * len(station_names)
        if accum is not None:
            ny, nx = accum.shape
            cols = np.arange(nx) + 0.5; rows = np.arange(ny) + 0.5
            ox, dx, _, oy, _, dy = gt
            X = ox + cols * dx; Y = oy + rows * dy
            X2, Y2 = np.meshgrid(X, Y)
            t_proj = Transformer.from_crs(CRS.from_wkt(wkt), "EPSG:4326", always_xy=True)
            lon_grid, lat_grid = t_proj.transform(X2, Y2)
            tree = cKDTree(np.column_stack([lat_grid.ravel(), lon_grid.ravel()]))
            _, indices = tree.query(list(zip(station_lats, station_lons)))
            radar_vals = [float(accum[divmod(idx, nx)] / 100.0) for idx in indices]
            
        # WRF
        wrf_vals = [np.nan] * len(station_names)
        fpath = os.path.join(WRF_DIR, "wrfout_d01_" + ts.strftime("%Y-%m-%d_%H_%M_%S"))
        if os.path.exists(fpath):
            ds = gdal.Open(fpath)
            subs = ds.GetSubDatasets()
            needed = {v: next((s[0] for s in subs if s[0].rsplit(":", 1)[-1] == v), None) for v in ("RAINNC", "RAINC", "RAINSH", "XLAT", "XLONG")}
            if None not in needed.values():
                cum = (gdal.Open(needed["RAINC"]).ReadAsArray() + gdal.Open(needed["RAINNC"]).ReadAsArray() + gdal.Open(needed["RAINSH"]).ReadAsArray())
                lat = np.squeeze(gdal.Open(needed["XLAT"]).ReadAsArray())
                lon = np.squeeze(gdal.Open(needed["XLONG"]).ReadAsArray())
                cum = np.where(np.isfinite(np.squeeze(cum)), np.squeeze(cum), 0.0)
                tree = cKDTree(np.column_stack([lat.ravel(), lon.ravel()]))
                _, indices = tree.query(list(zip(station_lats, station_lons)))
                wrf_vals = [float(cum.ravel()[idx]) for idx in indices]

        # AI Models
        def extract_ai(model_dir):
            vals = [np.nan] * len(station_names)
            init = ts - timedelta(hours=6)
            path = os.path.join(model_dir, f"forecast_{init:%Y%m%dT%H}.nc")
            if os.path.exists(path):
                ds = xr.open_dataset(path)
                pr = None
                if "total_precipitation_6hr" in ds:
                    pr = ds["total_precipitation_6hr"].squeeze().values
                elif "tp" in ds:
                    pr = ds["tp"].squeeze().values
                
                if pr is not None:
                    if np.nanmax(pr) < 5.0 and np.nanmax(pr) > 0.0001:
                        pr = pr * 1000.0
                    if "lat" in ds:
                        lats = ds["lat"].values
                        lons = ds["lon"].values
                    else:
                        lats = ds["latitude"].values
                        lons = ds["longitude"].values

                    if pr.ndim == 1:
                        lat_flat = lats
                        lon_flat = lons
                    else:
                        lon2d, lat2d = np.meshgrid(lons, lats)
                        lat_flat = lat2d.ravel()
                        lon_flat = lon2d.ravel()
                        
                    tree = cKDTree(np.column_stack([lat_flat, lon_flat]))
                    _, indices = tree.query(list(zip(station_lats, station_lons)))
                    vals = [float(pr.ravel()[idx]) for idx in indices]
                ds.close()
            return vals

        gc_vals = extract_ai(GC_DIR)
        fuxi_vals = extract_ai(FUXI_DIR)
        aifs_vals = extract_ai(AIFS_DIR)

        for i, s in enumerate(station_names):
            ts_results[s]["times"].append(ts)
            ts_results[s]["Radar"].append(radar_vals[i])
            ts_results[s]["WRF"].append(wrf_vals[i])
            ts_results[s]["GC"].append(gc_vals[i])
            ts_results[s]["FuXi"].append(fuxi_vals[i])
            ts_results[s]["AIFS"].append(aifs_vals[i])
            
            obs_val = np.nan
            if s in obs_data and ts in obs_data[s].index:
                v = obs_data[s].loc[ts]
                obs_val = float(v.iloc[0]) if isinstance(v, pd.Series) else float(v)
            ts_results[s]["Obs"].append(obs_val)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.patch.set_facecolor("#F7F9FC")
    
    colors = {
        "Obs":   "#722F37",
        "Radar": "#000000",
        "WRF":   "#2CA02C",
        "GC":    "#E06C2A",
        "FuXi":  "#2ECC71",
        "AIFS":  "#9B59B6"
    }
    markers = {
        "Obs": "o", "Radar": "s", "WRF": "^", "GC": "d", "FuXi": "v", "AIFS": "p"
    }
    labels = {
        "Obs": "Observation", "Radar": "RADAR", "WRF": "WRF (After DA)",
        "GC": "GraphCast", "FuXi": "FuXi", "AIFS": "AIFS"
    }
    
    import matplotlib.dates as mdates

    for idx, s in enumerate(station_names):
        if idx >= 4: break
        ax = axes[idx // 2, idx % 2]
        ax.set_facecolor("#F7F9FC")
        info = ts_stations[s]
        df = pd.DataFrame(ts_results[s])

        for col in ["Obs", "Radar", "WRF", "GC", "FuXi", "AIFS"]:
            sub = df[["times", col]].dropna()
            ax.plot(sub["times"], sub[col], color=colors[col], marker=markers[col],
                    markersize=3.5, linewidth=1.4, label=labels[col], alpha=0.95)

        ax.set_title(f"{info['country']} — {s}", fontsize=11, fontweight="bold", pad=6)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.set_ylabel("6-h precipitation (mm)", fontsize=9)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
        ax.tick_params(axis="x", which="major", labelsize=8, length=4)
        ax.tick_params(axis="x", which="minor", length=2)
        ax.tick_params(axis="y", which="major", labelsize=8)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        ax.spines["left"].set_color("#aaaaaa")
        ax.spines["bottom"].set_color("#aaaaaa")

    handles, labels_list = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_list, loc="lower center", bbox_to_anchor=(0.5, 0.01),
               ncol=6, frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(output_ts_file, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved -> {output_ts_file}")


if __name__ == "__main__":
    main()
