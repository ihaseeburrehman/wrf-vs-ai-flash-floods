#!/usr/bin/env python3
import os
import warnings
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

ROOT = "/Users/haseeb.rehman/Documents/Misc/From_HPC_and_WRF/WRF_from_HPC/4th_year"
OB = "/Users/haseeb.rehman/Documents/Misc/Data_Datasets/Stations_and_Observations/Luxembourg_stations_for_validation"
OUTPUT_DIR = "/Users/haseeb.rehman/Desktop/For_Animation/4th_year/Before_After_DA_Comparison"
GC_XLSX = "/Users/haseeb.rehman/Documents/Misc/GraphCast/graphcast_all_variables.xlsx"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRECIP_THRESHOLD = 1.0
HEALTHY = 40639268

GC_ALIAS = {
    "Spangdahlem": "Spangdahlem ab",
    "Kassel":      "Kassel calden",
    "Frankfurt":   "Frankfurt main",
}

STATION_COORDS = {
    "Briedfeld": (50.12385, 6.06622), "Echternach": (49.8031, 6.44337),
    "Ettelbruck": (49.85172, 6.09754), "Oberkorn": (49.5122, 5.9011),
    "Remerschen": (49.491, 6.349), "Findel": (49.63265, 6.23293),
    "Roodt": (49.7945, 5.8202), "Hosingen": (49.99314, 6.10147),
    "Useldange": (49.76739, 5.96748), "Mamer": (49.63353, 6.0193),
    "Arsdorf": (49.85891, 5.84868), "Asselborn": (50.09686, 5.96961),
    "Grevenmacher": (49.68087, 6.43541), "Schimpach": (50.0093, 5.8475),
    "Waldbillig": (49.79806, 6.2773), "Bettendorf": (49.8741, 6.2095),
    "Fouhren": (49.91445, 6.19508), "Beringen": (49.762, 6.11179),
    "Dahl": (49.93595, 5.98093),
    "Beitem": (50.9, 3.117), "Meyenheim": (47.917, 7.4),
    "Spangdahlem ab": (49.973, 6.693), "Kassel calden": (51.408, 9.378),
    "Vatry": (48.776, 4.184), "Ernage": (50.583, 4.683),
    "Dusseldorf": (51.289, 6.767), "Liege": (50.637, 5.443),
    "Mirecourt": (48.325, 6.07), "Frankfurt main": (50.026, 8.543),
    "Oostende": (51.199, 2.862), "Zeebrugge": (51.35, 3.2),
    "Fritzlar": (51.115, 9.286), "Branches": (47.85, 3.497),
    "Bale mulhouse": (47.59, 7.53),
}

EVENTS = {
    "2016": {"wrf": f"{ROOT}/2016_ERA5_cv5",
             "gen": f"{OB}/2016_Event/stations_6hr_cumulative.xlsx",
             "oth": f"{OB}/2016_Event/Stations_other_than_lux/station_weather_data_2016_6hr.xlsx"},
    "2018": {"wrf": f"{ROOT}/2018_ERA5_cv5",
             "gen": f"{OB}/2018_Event/stations_6hr_cumulative.xlsx",
             "oth": f"{OB}/2018_Event/Stations_other_than_lux/station_weather_data_2018_6hr.xlsx"},
    "2021": {"wrf": f"{ROOT}/2021_ERA5_cv5",
             "gen": f"{OB}/2021_Event/stations_6hr_cumulative.xlsx",
             "oth": f"{OB}/2021_Event/Stations_other_than_lux/station_weather_data_june_july_2021_6hr.xlsx"},
}

def parse_wrf_time(fname):
    base = os.path.basename(fname)
    if base.startswith("wrfvar_output_"):
        stamp = base.replace("wrfvar_output_", "")
        try:
            return datetime.strptime(stamp, "%Y%m%d%H")
        except ValueError:
            return None
    b = base.split("_")
    if len(b) < 6:
        return None
    try:
        return datetime.strptime(f"{b[2]}_{b[3]}_{b[4]}_{b[5]}", "%Y-%m-%d_%H_%M_%S")
    except (ValueError, IndexError):
        return None

def extract_wrf(wrf_dir, stations, latlon, label):
    rows = []
    idx = None
    files = sorted(glob(os.path.join(wrf_dir, "wrfout_d01_*"))
                   + glob(os.path.join(wrf_dir, "wrfvar_output_*")))
    for f in files:
        if os.path.getsize(f) != HEALTHY:
            continue
        t = parse_wrf_time(f)
        if t is None:
            continue
        try:
            nc = Dataset(f, "r")
            if any(v not in nc.variables for v in ("RAINNC", "RAINC", "T2", "XLAT", "XLONG")):
                nc.close()
                continue
            if idx is None:
                lat = np.squeeze(nc.variables["XLAT"][:].data)
                lon = np.squeeze(nc.variables["XLONG"][:].data)
                tree = cKDTree(np.column_stack([lat.ravel(), lon.ravel()]))
                _, idx = tree.query(np.array(latlon))
            rn = np.squeeze(nc.variables["RAINNC"][:].data).ravel()[idx]
            rc = np.squeeze(nc.variables["RAINC"][:].data).ravel()[idx]
            rs = (np.squeeze(nc.variables["RAINSH"][:].data).ravel()[idx]
                  if "RAINSH" in nc.variables else 0 * rn)
            t2 = np.squeeze(nc.variables["T2"][:].data).ravel()[idx]
            nc.close()
            
            vals_p = np.where(np.isnan(rc + rn + rs), 0.0, rc + rn + rs)
            vals_t = t2 - 273.15
            for s, vp, vt in zip(stations, vals_p, vals_t):
                rows.append({"Station": s, "UTC_Datetime": t, f"{label}_Precip": float(vp), f"{label}_T2m": float(vt)})
        except Exception:
            continue
    return pd.DataFrame(rows)

def load_obs(cfg, stations):
    rows = []
    for path in (cfg["gen"], cfg["oth"]):
        if not os.path.exists(path):
            continue
        xls = pd.ExcelFile(path)
        for sheet in xls.sheet_names:
            if sheet not in stations:
                continue
            df = pd.read_excel(xls, sheet_name=sheet)
            df.columns = [str(c).strip() for c in df.columns]
            dt_col = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None)
            pr_col = next((c for c in df.columns if "precip" in c.lower() and "mm" in c.lower()), None)
            t2_col = next((c for c in df.columns if "temp" in c.lower()), None)
            if dt_col is None:
                continue
            dt = pd.to_datetime(df[dt_col], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
            if dt.isna().all():
                dt = pd.to_datetime(df[dt_col], errors="coerce")
            df["UTC_Datetime"] = dt
            df = df.dropna(subset=["UTC_Datetime"])
            sub = df[["UTC_Datetime"]].copy()
            sub["Obs_Precip"] = pd.to_numeric(df[pr_col], errors="coerce") if pr_col else np.nan
            sub["Obs_T2m"] = pd.to_numeric(df[t2_col], errors="coerce") if t2_col else np.nan
            sub["Station"] = sheet
            rows.append(sub)
    return pd.concat(rows, ignore_index=True).dropna(subset=["UTC_Datetime"]) if rows else pd.DataFrame()

def load_gc(year):
    rows = []
    xls = pd.ExcelFile(GC_XLSX)
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet)
        df.columns = [str(c).strip() for c in df.columns]
        if "Valid Time" not in df.columns or "Event" not in df.columns:
            continue
        df = df[df["Event"].astype(str) == year]
        if df.empty:
            continue
        stn = GC_ALIAS.get(sheet, sheet)
        t = pd.to_datetime(df["Valid Time"], format="%Y%m%dT%H", errors="coerce")
        # we only need time and station to force an inner join mask
        rows.append(pd.DataFrame({
            "Station":      stn,
            "UTC_Datetime": t,
            "GC_Precip": 1.0 # dummy
        }).dropna(subset=["UTC_Datetime"]))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

def calc_precip_metrics(pred, obs, thr=1.0):
    p = np.asarray(pred, float); o = np.asarray(obs, float)
    m = np.isfinite(p) & np.isfinite(o)
    p, o = p[m], o[m]
    if p.size == 0:
        return {"POD": np.nan, "FAR": np.nan, "CSI": np.nan, "ETS": np.nan, "Bias": np.nan}
    pb, ob = p >= thr, o >= thr
    H = int((pb & ob).sum())
    M = int((~pb & ob).sum())
    F = int((pb & ~ob).sum())
    Z = int((~pb & ~ob).sum())
    N = H + M + F + Z
    
    pod = H / (H + M) if (H + M) > 0 else np.nan
    far = F / (H + F) if (H + F) > 0 else np.nan
    csi = H / (H + M + F) if (H + M + F) > 0 else np.nan
    H_rand = (H + M) * (H + F) / N if N > 0 else 0
    ets = (H - H_rand) / (H + M + F - H_rand) if (H + M + F - H_rand) > 0 else np.nan
    bias = float(np.mean(p - o))
    return {"POD": pod, "FAR": far, "CSI": csi, "ETS": ets, "Bias": bias}

def calc_temp_metrics(pred, obs):
    p = np.asarray(pred, float); o = np.asarray(obs, float)
    m = np.isfinite(p) & np.isfinite(o)
    p, o = p[m], o[m]
    if p.size == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "Corr": np.nan, "Bias": np.nan}
    rmse = float(np.sqrt(np.mean((p - o) ** 2)))
    mae = float(np.mean(np.abs(p - o)))
    bias = float(np.mean(p - o))
    if p.std() > 0 and o.std() > 0:
        corr = float(np.corrcoef(p, o)[0, 1])
    else:
        corr = np.nan
    return {"RMSE": rmse, "MAE": mae, "Corr": corr, "Bias": bias}

def plot_metrics(bef, aft, labels, title, out_path, colors=("#3A7EBF", "#E06C2A")):
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    fig.patch.set_facecolor("#F7F9FC")
    ax.set_facecolor("#F7F9FC")
    
    b1 = ax.bar(x - w / 2, bef, w, color=colors[0], edgecolor="white", zorder=3, label="Before DA")
    b2 = ax.bar(x + w / 2, aft, w, color=colors[1], edgecolor="white", zorder=3, label="After DA")
    
    ax.axhline(0, color="#555555", lw=0.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
    ax.grid(axis="y", ls="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
        
    for bars, vals in ((b1, bef), (b2, aft)):
        for bar, v in zip(bars, vals):
            if np.isnan(v):
                continue
            h = bar.get_height()
            ax.annotate(f"{v:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4 if h >= 0 else -11),
                        textcoords="offset points",
                        ha="center",
                        va="bottom" if h >= 0 else "top",
                        fontsize=9, fontweight="bold")
                        
    ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="#cccccc")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

def main():
    all_data = []
    
    for ev, cfg in EVENTS.items():
        gc = load_gc(ev)
        if gc.empty: continue
        gc_stns = set(gc["Station"].unique())
        
        obs_full = load_obs(cfg, set(STATION_COORDS.keys()))
        obs_stns = set(obs_full["Station"].unique())
        
        common_stns = sorted(gc_stns & obs_stns & set(STATION_COORDS.keys()))
        if not common_stns: continue
        
        latlon = [STATION_COORDS[s] for s in common_stns]
        
        bef = extract_wrf(f"{cfg['wrf']}/Before_DA", common_stns, latlon, "Before")
        aft = extract_wrf(f"{cfg['wrf']}/After_DA", common_stns, latlon, "After")
        
        # force exact same intersection as GraphCast by merging gc's subset
        m = bef.merge(aft, on=["Station", "UTC_Datetime"])\
               .merge(obs_full, on=["Station", "UTC_Datetime"])\
               .merge(gc, on=["Station", "UTC_Datetime"])
        m = m.dropna(subset=["Before_Precip", "After_Precip", "Before_T2m", "After_T2m"])
        all_data.append(m)

    if not all_data:
        print("No data matched.")
        return

    pooled = pd.concat(all_data, ignore_index=True)
    
    # Calculate pooled metrics
    bef_p_met = calc_precip_metrics(pooled["Before_Precip"], pooled["Obs_Precip"])
    aft_p_met = calc_precip_metrics(pooled["After_Precip"], pooled["Obs_Precip"])
    
    bef_t_met = calc_temp_metrics(pooled["Before_T2m"], pooled["Obs_T2m"])
    aft_t_met = calc_temp_metrics(pooled["After_T2m"], pooled["Obs_T2m"])
    
    print("\n| Metric | Before DA | After DA |")
    print("|--------|-----------|----------|")
    print("| **Precipitation** | | |")
    for k in ["POD", "FAR", "CSI", "ETS", "Bias"]:
        print(f"| {k} | {bef_p_met[k]:.3f} | {aft_p_met[k]:.3f} |")

    print("| **Temperature** | | |")
    for k in ["RMSE", "MAE", "Corr", "Bias"]:
        print(f"| {k} | {bef_t_met[k]:.3f} | {aft_t_met[k]:.3f} |")

    # Plots
    plot_metrics([bef_p_met[k] for k in ["POD", "FAR", "CSI", "ETS", "Bias"]],
                 [aft_p_met[k] for k in ["POD", "FAR", "CSI", "ETS", "Bias"]],
                 ["POD", "FAR", "CSI", "ETS", "Bias\n(mm)"],
                 f"Precipitation Metrics: Before DA vs After DA ({pooled['Station'].nunique()} stations)",
                 os.path.join(OUTPUT_DIR, "before_after_da_precip_stats.png"))
                 
    plot_metrics([bef_t_met[k] for k in ["RMSE", "MAE", "Corr", "Bias"]],
                 [aft_t_met[k] for k in ["RMSE", "MAE", "Corr", "Bias"]],
                 ["RMSE\n(°C)", "MAE\n(°C)", "Correlation\n(r)", "Bias\n(°C)"],
                 f"Temperature Metrics: Before DA vs After DA ({pooled['Station'].nunique()} stations)",
                 os.path.join(OUTPUT_DIR, "before_after_da_temp_stats.png"))

if __name__ == "__main__":
    main()
