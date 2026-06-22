"""Minimal FuXi extraction: tp + t2m only, netCDF4 + precomputed indices."""
import glob, os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.spatial import cKDTree

BASE   = "/scratch/lux0804/fuxi_luxembourg"
OUTPUT = f"{BASE}/output/fuxi_all_variables.xlsx"

EVENTS = {
    "2016": f"{BASE}/output/2016_event/forecasts/rapid_full",
    "2018": f"{BASE}/output/2018_event/forecasts/rapid_full",
    "2021": f"{BASE}/output/2021_event/forecasts/rapid_full",
}

STATIONS = {
    "Briedfeld":    (50.12385, 6.06622), "Echternach":   (49.8031,  6.44337),
    "Ettelbruck":   (49.85172, 6.09754), "Oberkorn":     (49.5122,  5.9011),
    "Remerschen":   (49.491,   6.349),   "Findel":       (49.63265, 6.23293),
    "Roodt":        (49.7945,  5.8202),  "Hosingen":     (49.99314, 6.10147),
    "Useldange":    (49.76739, 5.96748), "Mamer":        (49.63353, 6.0193),
    "Arsdorf":      (49.85891, 5.84868), "Asselborn":    (50.09686, 5.96961),
    "Grevenmacher": (49.68087, 6.43541), "Schimpach":    (50.0093,  5.8475),
    "Waldbillig":   (49.79806, 6.2773),  "Bettendorf":   (49.8741,  6.2095),
    "Fouhren":      (49.91445, 6.19508), "Beringen":     (49.762,   6.11179),
    "Dahl":         (49.93595, 5.98093), "Beitem":       (50.9,     3.117),
    "Meyenheim":    (47.917,   7.4),     "Spangdahlem":  (49.973,   6.693),
    "Kassel":       (51.408,   9.378),   "Vatry":        (48.776,   4.184),
    "Ernage":       (50.583,   4.683),   "Dusseldorf":   (51.289,   6.767),
    "Liege":        (50.637,   5.443),   "Mirecourt":    (48.325,   6.07),
    "Frankfurt":    (50.026,   8.543),   "Oostende":     (51.199,   2.862),
}

stat_names = list(STATIONS.keys())

# ── Build indices once from first file ───────────────────────────────────────
first = sorted(glob.glob(f"{BASE}/output/2021_event/forecasts/rapid_full/forecast_*.nc"))[0]
with Dataset(first) as nc0:
    lats = nc0.variables["lat"][:]
    lons = nc0.variables["lon"][:]
NLAT, NLON = len(lats), len(lons)
lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
tree = cKDTree(np.column_stack([lat2d.ravel(), lon2d.ravel()]))
coords = np.array(list(STATIONS.values()))
_, flat = tree.query(coords)
li, lo = flat // NLON, flat % NLON   # lat/lon indices per station
print(f"Grid {NLAT}×{NLON} | {len(stat_names)} stations | indices ready\n", flush=True)

# ── Extract ──────────────────────────────────────────────────────────────────
all_rows = []
for event_name, folder in EVENTS.items():
    files = sorted(glob.glob(os.path.join(folder, "forecast_*.nc")))
    print(f"Event {event_name}: {len(files)} files", flush=True)
    for fpath in files:
        ts = os.path.basename(fpath).replace("forecast_","").replace(".nc","")
        try:
            with Dataset(fpath) as nc:
                tp  = np.array(nc.variables["total_precipitation_6hr"][0, 0])  # [721,1440]
                t2m = np.array(nc.variables["2m_temperature"][0, 0])           # [721,1440]
            for si, sname in enumerate(stat_names):
                all_rows.append({
                    "Event": event_name, "Valid_Time": ts, "Station": sname,
                    "Precip_mm": round(float(tp[li[si],  lo[si]]) * 1000.0, 4),
                    "T2m_C":     round(float(t2m[li[si], lo[si]]) - 273.15, 3),
                })
        except Exception as e:
            print(f"  Error {ts}: {e}", flush=True)
    print(f"  done.", flush=True)

print(f"\n{len(all_rows)} rows — writing Excel...", flush=True)
df = pd.DataFrame(all_rows)
with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
    for sname in stat_names:
        df[df["Station"]==sname].drop(columns="Station").reset_index(drop=True)\
            .to_excel(writer, sheet_name=sname[:31], index=False)
print(f"Done! {OUTPUT}", flush=True)
