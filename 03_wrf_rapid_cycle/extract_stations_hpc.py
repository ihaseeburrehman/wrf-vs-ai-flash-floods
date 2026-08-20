#!/usr/bin/env python3
"""Extract WRF station values on the cluster, writing CSVs.

Replicates Stations_data_process_from_wrf.py exactly, but reads the wrfout
files with scipy.io.netcdf_file instead of netCDF4/wrf-python, because neither
is available on the cluster. The wrfout files are NetCDF3 (CDF-2), so scipy
reads them natively.

The two wrf-python calls are reproduced explicitly:
    getvar("pressure") * 100  ->  P + PB                    [Pa]
    getvar("tk")             ->  (T + 300) * (p/1e5)^(2/7)  [K]
    rh(qv, p, tk)            ->  the wrf-python formulation below

Usage:  extract_stations_hpc.py <After_DA_dir> <Before_DA_dir> <out_dir>
Output: general_<label>.csv and ztd_<label>.csv per configuration.
"""

import os
import sys
import glob
import numpy as np
from scipy.io import netcdf_file
from scipy.spatial import cKDTree

# ── station lists (deduplicated, order preserved) ───────────────────────────
_GEN = [
    ("Briedfeld", 50.12385, 6.06622), ("Echternach", 49.8031, 6.44337),
    ("Ettelbruck", 49.85172, 6.09754), ("Oberkorn", 49.5122, 5.9011),
    ("Remerschen", 49.491, 6.349), ("Findel", 49.63265182, 6.23292867),
    ("Roodt", 49.7945, 5.8202), ("Hosingen", 49.99314, 6.10147),
    ("Useldange", 49.76739, 5.96748), ("Mamer", 49.63353, 6.0193),
    ("Arsdorf", 49.85891, 5.84868), ("Asselborn", 50.09685689, 5.96960753),
    ("Grevenmacher", 49.68087, 6.43541), ("Schimpach", 50.0093, 5.8475),
    ("Waldbillig", 49.79806, 6.2773), ("Bettendorf", 49.8741, 6.2095),
    ("Fouhren", 49.91445, 6.19508), ("Beringen", 49.762, 6.11179),
    ("Dahl", 49.93595, 5.98093), ("Beitem", 50.9, 3.117),
    ("Meyenheim", 47.917, 7.4), ("Spangdahlem ab", 49.973, 6.693),
    ("Kassel calden", 51.408, 9.378), ("Vatry", 48.776, 4.184),
    ("Ernage", 50.583, 4.683), ("Dusseldorf", 51.289, 6.767),
    ("Liege", 50.637, 5.443), ("Mirecourt", 48.325, 6.07),
    ("Frankfurt main", 50.026, 8.543), ("Oostende", 51.199, 2.862),
    ("Zeebrugge", 51.35, 3.2), ("Fritzlar", 51.115, 9.286),
    ("Branches", 47.85, 3.497), ("Bale mulhouse", 47.59, 7.53),
    ("Lesquin", 50.562, 3.089), ("Augsburg", 48.425, 10.932),
    ("Amberieu", 45.987, 5.328), ("Humain", 50.2, 5.25),
    ("Gueret_St_Laurent", 46.183, 1.95), ("Buckeburg", 52.279, 9.082),
    ("Saarbrucken", 49.215, 7.11), ("Fauville", 49.029, 1.22),
    ("Souche", 46.311, -0.402), ("Cochstedt", 51.855, 11.419),
    ("Koksijde", 51.09, 2.653), ("Laage", 53.918, 12.278),
    ("Schonefeld", 52.38, 13.523), ("Cap_Ferret", 44.633, -1.25),
    ("Merignac", 44.828, -0.716), ("Tille", 49.454, 2.113),
    ("Leipzig_Halle", 51.424, 12.236), ("Bourges", 47.058, 2.37),
    ("Wunstorf", 52.457, 9.427),
    ("Alencon_Valframbert", 48.45, 0.117), ("Champniers", 45.717, 0.217),
]
GENERAL_STATIONS, _seen = [], set()
for n, la, lo in _GEN:
    if n not in _seen:
        _seen.add(n); GENERAL_STATIONS.append((n, la, lo))

ZTD_STATIONS = [
    ("D596", 51.2, 8.524), ("KLEV", 51.768, 6.142), ("FFMJ", 50.091, 8.665),
    ("D624", 50.868, 7.056), ("NIKL", 51.141, 4.151), ("D402", 48.073, 8.528),
    ("LAIG", 47.842, 4.373), ("TRI2", 49.725, 6.618), ("CT58", 49.15, 3.044),
    ("BAT1", 50.637, 5.834), ("VIT2", 50.317, 6.085), ("MABO", 50.075, 5.739),
    ("DBMH", 48.604, 6.364), ("SMSP", 49.115, 4.581), ("REDU", 50.002, 5.145),
    ("D931", 49.314, 6.746),
]


def sq(a):
    """Drop the leading time dimension."""
    a = np.asarray(a, dtype=np.float64)
    return a[0] if a.ndim >= 3 and a.shape[0] == 1 else a


def wrf_rh(qv, p_pa, tk):
    """wrf-python's rh(), reproduced."""
    es = 6.112 * np.exp(17.67 * (tk - 273.15) / (tk - 29.65))       # hPa
    qvs = 0.622 * es / (0.01 * p_pa - (1.0 - 0.622) * es)
    r = qv / qvs
    return 100.0 * np.clip(r, 0.0, 1.0)


def compute_ztd(pres_pa, temp_k, qv, dz, p_top_hpa, h_top, lat):
    p_hpa = pres_pa / 100.0
    zhd_b = 1e-6 * np.sum((77.689 * p_hpa / temp_k) * dz, axis=0)
    zhd_t = 0.0022767 * p_top_hpa * (1 - 0.00266 * np.cos(2 * np.radians(lat))
                                     - 0.00000028 * h_top)
    e = qv * pres_pa / (0.622 + qv)
    n_wet = 22.1 * (e / temp_k) + 3.739e5 * (e / (temp_k * temp_k))
    zwd = 1e-6 * np.sum(n_wet * dz, axis=0)
    return zhd_t + zhd_b + zwd


def process(path, label, out_dir):
    files = sorted(glob.glob(os.path.join(path, "wrfout_d01_*")))
    if not files:
        print(f"[{label}] no wrfout files in {path}")
        return
    print(f"[{label}] {len(files)} files")

    gen_rows, ztd_rows, tree, shape = [], [], None, None
    gen_idx, ztd_idx = None, None

    for k, fn in enumerate(files, 1):
        try:
            nc = netcdf_file(fn, "r", mmap=False)
            v = nc.variables
            xlat, xlong = sq(v["XLAT"][:]), sq(v["XLONG"][:])
            if tree is None:                       # grid is fixed, build once
                shape = xlat.shape
                tree = cKDTree(np.column_stack([xlat.ravel(), xlong.ravel()]))
                gen_idx = [np.unravel_index(tree.query([la, lo])[1], shape)
                           for _, la, lo in GENERAL_STATIONS]
                ztd_idx = [np.unravel_index(tree.query([la, lo])[1], shape)
                           for _, la, lo in ZTD_STATIONS]

            t2, psfc = sq(v["T2"][:]), sq(v["PSFC"][:])
            rainnc, rainc = sq(v["RAINNC"][:]), sq(v["RAINC"][:])
            rainsh = sq(v["RAINSH"][:]) if "RAINSH" in v else np.zeros_like(rainnc)
            u10, v10 = sq(v["U10"][:]), sq(v["V10"][:])

            p3d, pb3d = sq(v["P"][:]), sq(v["PB"][:])
            qv3d, t3d = sq(v["QVAPOR"][:]), sq(v["T"][:])
            ph, phb = sq(v["PH"][:]), sq(v["PHB"][:])

            pres_pa = p3d + pb3d                                   # Pa
            tk = (t3d + 300.0) * (pres_pa / 100000.0) ** (2.0 / 7.0)
            rh2d = wrf_rh(qv3d[0], pres_pa[0], tk[0])

            # ZTD, matching the original (theta offset 290, as written there)
            geo_h = (ph + phb) / 9.80665
            dz = np.diff(geo_h, axis=0)
            theta_z = t3d + 290.0
            p_hpa = pres_pa / 100.0
            temp_z = theta_z * (p_hpa / 1000.0) ** (2.0 / 7.0)
            ztd2d = compute_ztd(pres_pa, temp_z, qv3d, dz,
                                pres_pa[-1] / 100.0, geo_h[-1], xlat)
            nc.close()

            b = os.path.basename(fn).split("_")
            stamp = f"{b[2]} {b[3]}:{b[4]}:{b[5]}" if len(b) > 5 else f"{b[2]} {b[3]}"

            for (name, _, _), (i, j) in zip(GENERAL_STATIONS, gen_idx):
                gen_rows.append({
                    "Station": name, "UTC_Datetime": stamp,
                    "Precipitation (mm)": float(rainnc[i, j] + rainc[i, j] + rainsh[i, j]),
                    "Temperature (C)": float(t2[i, j] - 273.15),
                    "Wind Speed (m/s)": float(np.hypot(u10[i, j], v10[i, j])),
                    "Pressure (Pa)": float(psfc[i, j]),
                    "Relative Humidity (%)": float(rh2d[i, j]),
                })
            for (name, _, _), (i, j) in zip(ZTD_STATIONS, ztd_idx):
                ztd_rows.append({"Station": name, "UTC_Datetime": stamp,
                                 "ZTD (m)": float(ztd2d[i, j])})

            if k % 20 == 0 or k == len(files):
                print(f"  [{label}] {k}/{len(files)}", flush=True)
        except Exception as e:
            print(f"  [{label}] ERROR {os.path.basename(fn)}: {type(e).__name__}: {e}")

    os.makedirs(out_dir, exist_ok=True)
    for rows, tag in ((gen_rows, "general"), (ztd_rows, "ztd")):
        p = os.path.join(out_dir, f"{tag}_{label}.csv")
        if not rows:
            continue
        cols = list(rows[0].keys())
        with open(p, "w") as fh:
            fh.write(",".join(cols) + "\n")
            for r in rows:
                fh.write(",".join(
                    f'"{r[c]}"' if isinstance(r[c], str) else f"{r[c]:.6f}"
                    for c in cols) + "\n")
        print(f"  wrote {p}  ({len(rows)} rows)")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__); sys.exit(2)
    after, before, out = sys.argv[1:4]
    process(after, "after", out)
    process(before, "before", out)
    print("done")
