"""AIFS extraction: tp + 2t at stations, netCDF4 + precomputed KD-tree indices.
AIFS output is N320 reduced Gaussian: 1D 'values' dim with latitude/longitude vars.
Saves CSV (built into xlsx locally, same as FuXi workflow)."""
import glob, os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
from scipy.spatial import cKDTree

BASE   = "/scratch/lux0804/aifs_luxembourg"
OUTPUT = f"{BASE}/output/aifs_all_variables.csv"

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

# ── Build indices once from first file (N320: 1D points) ────────────────────
first = sorted(glob.glob(f"{EVENTS['2021']}/forecast_*.nc"))[0]
with Dataset(first) as nc0:
    lats = np.array(nc0.variables["latitude"][:])
    lons = np.array(nc0.variables["longitude"][:])
lons_w = np.where(lons > 180.0, lons - 360.0, lons)   # 0..360 -> -180..180
tree = cKDTree(np.column_stack([lats, lons_w]))
_, idx = tree.query(np.array(list(STATIONS.values())))
print(f"Grid {lats.size} points | {len(stat_names)} stations | indices ready", flush=True)
for si, sname in enumerate(stat_names[:3]):
    print(f"  {sname}: requested {STATIONS[sname]}, got ({lats[idx[si]]:.3f},{lons_w[idx[si]]:.3f})", flush=True)

# ── Extract ──────────────────────────────────────────────────────────────────
all_rows = []
for event_name, folder in EVENTS.items():
    files = sorted(glob.glob(os.path.join(folder, "forecast_*.nc")))
    print(f"Event {event_name}: {len(files)} files", flush=True)
    for fpath in files:
        ts = os.path.basename(fpath).replace("forecast_", "").replace(".nc", "")
        try:
            with Dataset(fpath) as nc:
                tp = np.array(nc.variables["tp"][:])
                t2 = np.array(nc.variables["2t"][:])
            for si, sname in enumerate(stat_names):
                all_rows.append({
                    "Event": event_name, "Valid_Time": ts, "Station": sname,
                    "Precip_mm": round(float(tp[idx[si]]) * 1000.0, 4),
                    "T2m_C":     round(float(t2[idx[si]]) - 273.15, 3),
                })
        except Exception as e:
            print(f"  Error {ts}: {e}", flush=True)
    print("  done.", flush=True)

print(f"\n{len(all_rows)} rows — writing CSV...", flush=True)
pd.DataFrame(all_rows).to_csv(OUTPUT, index=False)
print(f"Done! {OUTPUT}", flush=True)
