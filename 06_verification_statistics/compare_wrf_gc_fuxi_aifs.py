#!/usr/bin/env python3
"""
WRF (ERA5 cv5, After DA) vs GraphCast vs FuXi vs AIFS — Precipitation & 2 m Temperature
Three events: 2016, 2018, 2021  |  Stations present in ALL datasets only.
"""

import os
import warnings
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

OUTPUT_DIR = "/Users/haseeb.rehman/Python scripts/output"

GC_XLSX   = "/Users/haseeb.rehman/Documents/Misc/AI_Models/GraphCast/graphcast_all_variables.xlsx"
FUXI_XLSX = "/Users/haseeb.rehman/Documents/Misc/AI_Models/FuXi/fuxi_all_variables.csv"
AIFS_XLSX = "/Users/haseeb.rehman/Documents/Misc/AI_Models/AIFS/aifs_all_variables.csv"
OB        = "/Users/haseeb.rehman/Documents/Misc/Data_Datasets/Stations_and_Observations/Luxembourg_stations_for_validation"
WRF_HPC   = "/Users/haseeb.rehman/Documents/Misc/From_HPC_and_WRF/WRF_from_HPC/4th_year"

EVENTS = {
    "2016": {
        "wrf_dir": f"{WRF_HPC}/2016_ERA5_cv5/After_DA",
        "obs_gen": f"{OB}/2016_Event/stations_6hr_cumulative.xlsx",
        "obs_oth": f"{OB}/2016_Event/Stations_other_than_lux/station_weather_data_2016_6hr.xlsx",
    },
    "2018": {
        "wrf_dir": f"{WRF_HPC}/2018_ERA5_cv5/After_DA",
        "obs_gen": f"{OB}/2018_Event/stations_6hr_cumulative.xlsx",
        "obs_oth": f"{OB}/2018_Event/Stations_other_than_lux/station_weather_data_2018_6hr.xlsx",
    },
    "2021": {
        "wrf_dir": f"{WRF_HPC}/2021_ERA5_cv5/After_DA",
        "obs_gen": f"{OB}/2021_Event/stations_6hr_cumulative.xlsx",
        "obs_oth": f"{OB}/2021_Event/Stations_other_than_lux/station_weather_data_june_july_2021_6hr.xlsx",
    },
}

# GC/FuXi sheet name → observation station name
MODEL_ALIAS = {
    "Spangdahlem": "Spangdahlem ab",
    "Kassel":      "Kassel calden",
    "Frankfurt":   "Frankfurt main",
}

STATION_COORDS = {
    "Briedfeld":    (50.12385,  6.06622),
    "Echternach":   (49.8031,   6.44337),
    "Ettelbruck":   (49.85172,  6.09754),
    "Oberkorn":     (49.5122,   5.9011),
    "Remerschen":   (49.491,    6.349),
    "Findel":       (49.63265,  6.23293),
    "Roodt":        (49.7945,   5.8202),
    "Hosingen":     (49.99314,  6.10147),
    "Useldange":    (49.76739,  5.96748),
    "Mamer":        (49.63353,  6.0193),
    "Arsdorf":      (49.85891,  5.84868),
    "Asselborn":    (50.09686,  5.96961),
    "Grevenmacher": (49.68087,  6.43541),
    "Schimpach":    (50.0093,   5.8475),
    "Waldbillig":   (49.79806,  6.2773),
    "Bettendorf":   (49.8741,   6.2095),
    "Fouhren":      (49.91445,  6.19508),
    "Beringen":     (49.762,    6.11179),
    "Dahl":         (49.93595,  5.98093),
    "Beitem":       (50.9,      3.117),
    "Meyenheim":    (47.917,    7.4),
    "Spangdahlem ab":(49.973,   6.693),
    "Kassel calden": (51.408,   9.378),
    "Vatry":        (48.776,    4.184),
    "Ernage":       (50.583,    4.683),
    "Dusseldorf":   (51.289,    6.767),
    "Liege":        (50.637,    5.443),
    "Mirecourt":    (48.325,    6.07),
    "Frankfurt main":(50.026,   8.543),
    "Oostende":     (51.199,    2.862),
    "Zeebrugge":    (51.35,     3.2),
    "Fritzlar":     (51.115,    9.286),
    "Branches":     (47.85,     3.497),
    "Bale mulhouse":(47.59,     7.53),
}

PRECIP_THR = 1.0
C_WRF, C_GC, C_FUXI, C_AIFS = "#3A7EBF", "#E06C2A", "#2ECC71", "#9B59B6"
HEALTHY = 40639268   # expected WRF file size in bytes

# ── Data loaders ─────────────────────────────────────────────────────────────

def parse_wrf_time(fname):
    b = os.path.basename(fname).split("_")
    if len(b) < 6:
        return None
    try:
        return datetime.strptime(f"{b[2]}_{b[3]}_{b[4]}_{b[5]}", "%Y-%m-%d_%H_%M_%S")
    except (ValueError, IndexError):
        return None

def extract_wrf(wrf_dir, stations, latlon):
    rows = []
    idx = grid_size = None
    all_files = sorted(f for f in glob(os.path.join(wrf_dir, "wrfout_d01_*"))
                       if not f.endswith(".xml"))
    for fpath in all_files:
        if os.path.getsize(fpath) != HEALTHY:
            continue
        t = parse_wrf_time(fpath)
        if t is None:
            continue
        try:
            nc = Dataset(fpath, "r")
            needed = ("RAINNC", "RAINC", "T2", "XLAT", "XLONG")
            if any(v not in nc.variables for v in needed):
                nc.close(); continue
            if idx is None:
                lat = np.squeeze(nc.variables["XLAT"][:].data)
                lon = np.squeeze(nc.variables["XLONG"][:].data)
                if lat.size == 0:
                    nc.close(); continue
                grid_size = lat.size
                tree = cKDTree(np.column_stack([lat.ravel(), lon.ravel()]))
                _, idx = tree.query(np.array(latlon))
            rn = np.squeeze(nc.variables["RAINNC"][:].data).ravel()
            if rn.size != grid_size:
                nc.close(); continue
            rn  = rn[idx]
            rc  = np.squeeze(nc.variables["RAINC"][:].data).ravel()[idx]
            rs  = (np.squeeze(nc.variables["RAINSH"][:].data).ravel()[idx]
                   if "RAINSH" in nc.variables else np.zeros_like(rn))
            t2  = np.squeeze(nc.variables["T2"][:].data).ravel()[idx]
            nc.close()
        except Exception:
            continue
        if np.any(t2 < 180.0) or np.any(t2 > 330.0):
            continue
        precip = np.where(np.isnan(rc + rn + rs), 0.0, rc + rn + rs)
        for i, stn in enumerate(stations):
            rows.append({"Station": stn, "UTC_Datetime": t,
                         "WRF_Precip": float(precip[i]),
                         "WRF_T2m":    float(t2[i] - 273.15)})
    return pd.DataFrame(rows)

def load_obs(year_cfg):
    rows = []
    for path_key in ("obs_gen", "obs_oth"):
        path = year_cfg[path_key]
        if not os.path.exists(path):
            continue
        xls = pd.ExcelFile(path)
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            df.columns = [str(c).strip() for c in df.columns]
            dt_col = next((c for c in df.columns
                           if any(k in c.lower() for k in ("utc", "time", "date"))), None)
            if dt_col is None:
                continue
            dt = pd.to_datetime(df[dt_col], errors="coerce")
            if dt.isna().all():
                dt = pd.to_datetime(df[dt_col], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
            pc = next((c for c in df.columns if "precip" in c.lower() and "mm" in c.lower()), None)
            tc = next((c for c in df.columns if "temp" in c.lower()), None)
            if pc is None and tc is None:
                continue
            rows.append(pd.DataFrame({
                "Station":      sheet,
                "UTC_Datetime": dt,
                "Obs_Precip":   pd.to_numeric(df[pc], errors="coerce") if pc else np.nan,
                "Obs_T2m":      pd.to_numeric(df[tc], errors="coerce") if tc else np.nan,
            }).dropna(subset=["UTC_Datetime"]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

def load_model_xlsx(xlsx_path, year, precip_col="Precip_mm", t2m_col="T2m_C",
                    out_precip="Model_Precip", out_t2m="Model_T2m"):
    """Generic loader for graphcast_all_variables.xlsx or fuxi_all_variables.csv."""
    rows = []
    if not os.path.exists(xlsx_path):
        print(f"  WARNING: {xlsx_path} not found")
        return pd.DataFrame()
    
    if xlsx_path.endswith('.csv'):
        df_all = pd.read_csv(xlsx_path)
        df_all.columns = [str(c).strip().replace(" ", "_") for c in df_all.columns]
        if "Valid_Time" not in df_all.columns or "Event" not in df_all.columns or "Station" not in df_all.columns:
            return pd.DataFrame()
        df_year = df_all[df_all["Event"].astype(str) == str(year)]
        for stn_name, df in df_year.groupby("Station"):
            stn = MODEL_ALIAS.get(stn_name, stn_name)
            t   = pd.to_datetime(df["Valid_Time"], format="%Y%m%dT%H", errors="coerce")
            p   = pd.to_numeric(df[precip_col], errors="coerce") if precip_col in df.columns else np.nan
            t2  = pd.to_numeric(df[t2m_col],   errors="coerce") if t2m_col   in df.columns else np.nan
            rows.append(pd.DataFrame({
                "Station":      stn,
                "UTC_Datetime": t,
                out_precip:     p.values if hasattr(p, "values") else p,
                out_t2m:        t2.values if hasattr(t2, "values") else t2,
            }).dropna(subset=["UTC_Datetime"]))
    else:
        xls = pd.ExcelFile(xlsx_path)
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]
            if "Valid_Time" not in df.columns or "Event" not in df.columns:
                continue
            df = df[df["Event"].astype(str) == str(year)]
            if df.empty:
                continue
            stn = MODEL_ALIAS.get(sheet, sheet)
            t   = pd.to_datetime(df["Valid_Time"], format="%Y%m%dT%H", errors="coerce")
            p   = pd.to_numeric(df[precip_col], errors="coerce") if precip_col in df.columns else np.nan
            t2  = pd.to_numeric(df[t2m_col],   errors="coerce") if t2m_col   in df.columns else np.nan
            rows.append(pd.DataFrame({
                "Station":      stn,
                "UTC_Datetime": t,
                out_precip:     p.values if hasattr(p, "values") else p,
                out_t2m:        t2.values if hasattr(t2, "values") else t2,
            }).dropna(subset=["UTC_Datetime"]))
            
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

# ── Metrics ──────────────────────────────────────────────────────────────────

def calc_metrics(pred, obs, is_precip=False):
    p = np.asarray(pred, float);  o = np.asarray(obs, float)
    ok = np.isfinite(p) & np.isfinite(o)
    p, o = p[ok], o[ok]
    n = int(p.size)
    if n == 0:
        return dict(N=0, RMSE=np.nan, MAE=np.nan, Bias=np.nan, Corr=np.nan,
                    POD=np.nan, FAR=np.nan, CSI=np.nan, ETS=np.nan)
    corr = float(np.corrcoef(p, o)[0, 1]) if p.std() > 0 and o.std() > 0 else np.nan
    out  = dict(N=n,
                RMSE=float(np.sqrt(np.mean((p - o) ** 2))),
                MAE =float(np.mean(np.abs(p - o))),
                Bias=float(np.mean(p - o)),
                Corr=corr,
                POD=np.nan, FAR=np.nan, CSI=np.nan, ETS=np.nan)
    if is_precip:
        pb = p >= PRECIP_THR;  ob = o >= PRECIP_THR
        H = int((pb & ob).sum());  M = int((~pb & ob).sum());  F = int((pb & ~ob).sum())
        out["POD"] = H / (H + M) if (H + M) else np.nan
        out["FAR"] = F / (H + F) if (H + F) else np.nan
        out["CSI"] = H / (H + M + F) if (H + M + F) else np.nan
        Hr = (H + M) * (H + F) / n if n > 0 else 0
        out["ETS"] = (H - Hr) / (H + M + F - Hr) if (H + M + F - Hr) > 0 else np.nan
    return out

# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_four_panel(wv, gv, fv, av, labels, title, out_path, unit=""):
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    fig.patch.set_facecolor("#F7F9FC")
    ax.set_facecolor("#F7F9FC")

    x = np.arange(len(labels))
    w = 0.19
    b1 = ax.bar(x - 1.5*w, wv, w, color=C_WRF,  edgecolor="white", lw=0.6, zorder=3, label="WRF (After DA)")
    b2 = ax.bar(x - 0.5*w, gv, w, color=C_GC,   edgecolor="white", lw=0.6, zorder=3, label="GraphCast")
    b3 = ax.bar(x + 0.5*w, fv, w, color=C_FUXI, edgecolor="white", lw=0.6, zorder=3, label="FuXi")
    b4 = ax.bar(x + 1.5*w, av, w, color=C_AIFS, edgecolor="white", lw=0.6, zorder=3, label="AIFS")

    ax.axhline(0, color="#555555", lw=0.8, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=15)
    ax.grid(axis="y", ls="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#aaaaaa")
    ax.spines["bottom"].set_color("#aaaaaa")

    for bars, vals in ((b1, wv), (b2, gv), (b3, fv), (b4, av)):
        for bar, v in zip(bars, vals):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            h = bar.get_height()
            ax.annotate(f"{v:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4 if h >= 0 else -11),
                        textcoords="offset points",
                        ha="center", va="bottom" if h >= 0 else "top",
                        fontsize=8.5, fontweight="bold", color="#222222")

    ax.legend(loc="best", fontsize=10, frameon=True, facecolor="white", edgecolor="#cccccc")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    pooled_precip = []
    pooled_temp   = []

    for year, cfg in EVENTS.items():
        print(f"\n── Event {year} ──")
        obs = load_obs(cfg)
        gc  = load_model_xlsx(GC_XLSX,   year, out_precip="GC_Precip",   out_t2m="GC_T2m")
        fx  = load_model_xlsx(FUXI_XLSX, year, out_precip="FuXi_Precip", out_t2m="FuXi_T2m")
        af  = load_model_xlsx(AIFS_XLSX, year, out_precip="AIFS_Precip", out_t2m="AIFS_T2m")

        if obs.empty or gc.empty or fx.empty or af.empty:
            print(f"  Skipping {year} — missing data")
            continue

        obs_stns  = set(obs["Station"].unique())
        gc_stns   = set(gc["Station"].unique())
        fx_stns   = set(fx["Station"].unique())
        af_stns   = set(af["Station"].unique())
        coord_stns = set(STATION_COORDS.keys())
        common = sorted(obs_stns & gc_stns & fx_stns & af_stns & coord_stns)
        if not common:
            print(f"  No common stations for {year}")
            continue
        print(f"  Common stations: {len(common)}")

        latlon = [STATION_COORDS[s] for s in common]
        wrf    = extract_wrf(cfg["wrf_dir"], common, latlon)
        if wrf.empty:
            print(f"  WRF data missing for {year}")
            continue

        # Merge all four datasets on Station + UTC_Datetime
        obs_f = obs[obs["Station"].isin(common)][["Station","UTC_Datetime","Obs_Precip","Obs_T2m"]]
        gc_f  = gc[gc["Station"].isin(common)][["Station","UTC_Datetime","GC_Precip","GC_T2m"]]
        fx_f  = fx[fx["Station"].isin(common)][["Station","UTC_Datetime","FuXi_Precip","FuXi_T2m"]]
        af_f  = af[af["Station"].isin(common)][["Station","UTC_Datetime","AIFS_Precip","AIFS_T2m"]]
        wrf_f = wrf[["Station","UTC_Datetime","WRF_Precip","WRF_T2m"]]

        merged = (wrf_f
                  .merge(obs_f, on=["Station","UTC_Datetime"])
                  .merge(gc_f,  on=["Station","UTC_Datetime"])
                  .merge(fx_f,  on=["Station","UTC_Datetime"])
                  .merge(af_f,  on=["Station","UTC_Datetime"])
                  .dropna())
        if merged.empty:
            print(f"  No merged rows for {year}")
            continue
        print(f"  Merged rows: {len(merged)}, stations: {merged['Station'].nunique()}")

        pooled_precip.append(merged[["Station","WRF_Precip","GC_Precip","FuXi_Precip","AIFS_Precip","Obs_Precip"]])
        pooled_temp.append(  merged[["Station","WRF_T2m",   "GC_T2m",   "FuXi_T2m",  "AIFS_T2m", "Obs_T2m"]])

    if not pooled_precip:
        print("\nNo data to plot.")
        return

    pp = pd.concat(pooled_precip, ignore_index=True)
    pt = pd.concat(pooled_temp,   ignore_index=True)

    # ── Metrics ──
    mwp = calc_metrics(pp["WRF_Precip"],  pp["Obs_Precip"], is_precip=True)
    mgp = calc_metrics(pp["GC_Precip"],   pp["Obs_Precip"], is_precip=True)
    mfp = calc_metrics(pp["FuXi_Precip"], pp["Obs_Precip"], is_precip=True)
    map_ = calc_metrics(pp["AIFS_Precip"], pp["Obs_Precip"], is_precip=True)

    mwt = calc_metrics(pt["WRF_T2m"],  pt["Obs_T2m"])
    mgt = calc_metrics(pt["GC_T2m"],   pt["Obs_T2m"])
    mft = calc_metrics(pt["FuXi_T2m"], pt["Obs_T2m"])
    mat = calc_metrics(pt["AIFS_T2m"], pt["Obs_T2m"])

    n_stn_p = pp["Station"].nunique()
    n_stn_t = pt["Station"].nunique()

    # ── Print table ──
    print(f"\n{'─'*62}")
    print(f"{'Metric':<12} {'WRF (After DA)':>15} {'GraphCast':>12} {'FuXi':>10} {'AIFS':>10}")
    print(f"{'─'*62}")
    print("  PRECIPITATION")
    for k in ["POD","FAR","CSI","ETS","Bias"]:
        unit = " mm" if k=="Bias" else ""
        print(f"  {k:<10} {mwp[k]:>15.3f} {mgp[k]:>12.3f} {mfp[k]:>10.3f} {map_[k]:>10.3f}{unit}")
    print("  TEMPERATURE (2m)")
    for k in ["RMSE","MAE","Corr","Bias"]:
        unit = " °C" if k!="Corr" else ""
        print(f"  {k:<10} {mwt[k]:>15.3f} {mgt[k]:>12.3f} {mft[k]:>10.3f} {mat[k]:>10.3f}{unit}")
    print(f"{'─'*62}")
    print(f"  N (precip samples): {int(mwp['N'])}   Stations: {n_stn_p}")
    print(f"  N (temp samples):   {int(mwt['N'])}   Stations: {n_stn_t}")

    # ── Plots ──
    labels_p = ["POD", "FAR", "CSI", "ETS", "Bias\n(mm)"]
    plot_four_panel(
        [mwp[k] for k in ["POD","FAR","CSI","ETS","Bias"]],
        [mgp[k] for k in ["POD","FAR","CSI","ETS","Bias"]],
        [mfp[k] for k in ["POD","FAR","CSI","ETS","Bias"]],
        [map_[k] for k in ["POD","FAR","CSI","ETS","Bias"]],
        labels_p,
        f"Precipitation: WRF vs GraphCast vs FuXi vs AIFS  ({n_stn_p} stations, all events pooled)",
        os.path.join(OUTPUT_DIR, "wrf_gc_fuxi_aifs_precip_stats.png"),
        unit="mm"
    )

    labels_t = ["RMSE\n(°C)", "MAE\n(°C)", "Corr\n(r)", "Bias\n(°C)"]
    plot_four_panel(
        [mwt[k] for k in ["RMSE","MAE","Corr","Bias"]],
        [mgt[k] for k in ["RMSE","MAE","Corr","Bias"]],
        [mft[k] for k in ["RMSE","MAE","Corr","Bias"]],
        [mat[k] for k in ["RMSE","MAE","Corr","Bias"]],
        labels_t,
        f"2 m Temperature: WRF vs GraphCast vs FuXi vs AIFS  ({n_stn_t} stations, all events pooled)",
        os.path.join(OUTPUT_DIR, "wrf_gc_fuxi_aifs_temp_stats.png"),
        unit="°C"
    )
    print("\nDone.")

if __name__ == "__main__":
    main()
