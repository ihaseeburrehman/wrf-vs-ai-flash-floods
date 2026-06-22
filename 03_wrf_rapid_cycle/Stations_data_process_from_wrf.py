#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 12:31:45 2025

@author: haseeb.rehman
"""

import glob
from osgeo import gdal
import numpy as np
import os
import pandas as pd
import time
from scipy.spatial import cKDTree
from netCDF4 import Dataset
from wrf import getvar, rh, to_np
import warnings


# Suppress all warnings
warnings.filterwarnings("ignore")
# Base directory
base_dir = "/Users/haseeb.rehman/Documents/Misc/WRF_from_HPC/4th_year/2021_ERA5_cv5/"
before_da_path = os.path.join(base_dir, "Before_DA")
after_da_path = os.path.join(base_dir, "After_DA")
output_before_folder = "/Users/haseeb.rehman/Desktop/For_Animation/4th_year/2021_ERA5_cv5/Before_DA/"
output_after_folder = "/Users/haseeb.rehman/Desktop/For_Animation/4th_year/2021_ERA5_cv5/After_DA/"

# Output file paths
general_excel_file_before = os.path.join(output_before_folder, "general_station_data_before.xlsx")
ztd_excel_file_before = os.path.join(output_before_folder, "ztd_station_data_before.xlsx")
general_excel_file_after = os.path.join(output_after_folder, "general_station_data_after.xlsx")
ztd_excel_file_after = os.path.join(output_after_folder, "ztd_station_data_after.xlsx")

general_stations = [
    ("Briedfeld", 50.12385000, 6.06622000),
    ("Echternach", 49.80310000, 6.44337000),
    ("Ettelbruck", 49.85172000, 6.09754000),
    ("Oberkorn", 49.51220000, 5.90110000),
    ("Remerschen", 49.49100000, 6.34900000),
    ("Findel", 49.63265182, 6.23292867),
    ("Roodt", 49.79450000, 5.82020000),
    ("Hosingen", 49.99314000, 6.10147000),
    ("Useldange", 49.76739000, 5.96748000),
    ("Mamer", 49.63353000, 6.01930000),
    ("Arsdorf", 49.85891000, 5.84868000),
    ("Asselborn", 50.09685689, 5.96960753),
    ("Grevenmacher", 49.68087000, 6.43541000),
    ("Schimpach", 50.00930000, 5.84750000),
    ("Waldbillig", 49.79806000, 6.27730000),
    ("Bettendorf", 49.87410000, 6.20950000),
    ("Fouhren", 49.91445000, 6.19508000),
    ("Beringen", 49.76200000, 6.11179000),
    ("Dahl", 49.93595000, 5.98093000),
    
    # Stations outside Luxembourg (2021 event)
    ("Beitem", 50.900000, 3.117000),
    ("Meyenheim", 47.917000, 7.400000),
    ("Spangdahlem ab", 49.973000, 6.693000),
    ("Kassel calden", 51.408000, 9.378000),
    ("Vatry", 48.776000, 4.184000),
    ("Ernage", 50.583000, 4.683000),
    ("Dusseldorf", 51.289000, 6.767000),
    ("Liege", 50.637000, 5.443000),
    ("Mirecourt", 48.325000, 6.070000),
    ("Frankfurt main", 50.026000, 8.543000),
    ("Oostende", 51.199000, 2.862000),
    ("Zeebrugge", 51.350000, 3.200000),
    ("Fritzlar", 51.115000, 9.286000),
    ("Branches", 47.850000, 3.497000),
    ("Bale mulhouse", 47.590000, 7.530000)
    
    #selected 2018 event outside Luxembourg
    #("Lesquin", 50.562, 3.089),
    #("Augsburg", 48.425, 10.932),
    #("Amberieu", 45.987, 5.328),
    #("Humain", 50.2, 5.25),
    #("Gueret_St_Laurent", 46.183, 1.95),
    #("Buckeburg", 52.279, 9.082),
    #("Saarbrucken", 49.215, 7.11),
    #("Fauville", 49.029, 1.22),
    #("Souche", 46.311, -0.402),
    #("Cochstedt", 51.855, 11.419),
    #("Koksijde", 51.09, 2.653),
    #("Oostende", 51.199, 2.862),
    #("Laage", 53.918, 12.278),
    #("Schonefeld", 52.38, 13.523),
    #("Cap_Ferret", 44.633, -1.25),
    #("Merignac", 44.828, -0.716)
    
    #selected 2018 event outside luxembourg
    #("Tille", 49.454, 2.113),
    #("Leipzig_Halle", 51.424, 12.236),
    #("Saarbrucken", 49.215, 7.11),
    #("Bourges", 47.058, 2.37),
    #("Amberieu", 45.987, 5.328),
   # ("Wunstorf", 52.457, 9.427),
    #("Ernage", 50.583, 4.683),
   # ("Augsburg", 48.425, 10.932),
    #("Alencon_Valframbert", 48.45, 0.117),
    #("Champniers", 45.717, 0.217),
    #("Koksijde", 51.09, 2.653),
    #("Oostende", 51.199, 2.862),
    #("Laage", 53.918, 12.278),
    #("Schonefeld", 52.38, 13.523),
    #("Cap_Ferret", 44.633, -1.25),
    #("Merignac", 44.828, -0.716)

]

ztd_stations = [
    ("D596", 51.200, 8.524),
    ("KLEV", 51.768, 6.142),
    ("FFMJ", 50.091, 8.665),
    ("D624", 50.868, 7.056),
    ("NIKL", 51.141, 4.151),
    ("D402", 48.073, 8.528),
    ("LAIG", 47.842, 4.373),
    ("TRI2", 49.725, 6.618),
    ("CT58", 49.150, 3.044),
    ("BAT1", 50.637, 5.834),
    ("VIT2", 50.317, 6.085),
    ("MABO", 50.075, 5.739),
    ("DBMH", 48.604, 6.364),
    ("SMSP", 49.115, 4.581),
    ("REDU", 50.002, 5.145),
    ("D931", 49.314, 6.746),
]

# Initialize dataframes for both Before_DA and After_DA
dataframes_general_before = {st[0]: [] for st in general_stations}
dataframes_ztd_before = {st[0]: [] for st in ztd_stations}
dataframes_general_after = {st[0]: [] for st in general_stations}
dataframes_ztd_after = {st[0]: [] for st in ztd_stations}

def read_subdataset_explicit(subdatasets, var_name):
    """Return the first subdataset whose Name ends exactly with :var_name."""
    desired_ending = f":{var_name}"
    for s in subdatasets:
        if s[0].endswith(desired_ending):
            ds = gdal.Open(s[0])
            if ds is not None:
                return ds.ReadAsArray()
    return None

def slice_3d_to_2d(arr):
    """Safely slice 4D->2D or 3D->2D for WRF arrays."""
    if arr is None:
        return None
    if arr.ndim == 4:
        return arr[0, 0, :, :]
    elif arr.ndim == 3:
        return arr[0, :, :]
    return arr

def find_nearest_grid_point(xlat, xlong, lat, lon):
    """Find the nearest grid point indices for a given lat/lon."""
    points = np.column_stack((xlat.flatten(), xlong.flatten()))
    tree = cKDTree(points)
    dist, idx = tree.query([lat, lon])
    ny, nx = xlat.shape
    i, j = np.unravel_index(idx, (ny, nx))
    return i, j

# --- ZTD Calculation Functions ---
def compute_ZHD_b(pres_3d, temp_3d, dz):
    pres_3d_hpa = pres_3d / 100.0
    return 1e-6 * np.sum((77.689 * pres_3d_hpa / temp_3d) * dz, axis=0)

def compute_ZHD_t(p_top, h_top, lat):
    return 0.0022767 * p_top * (1 - 0.00266 * np.cos(2 * np.radians(lat)) - 0.00000028 * h_top)

def compute_ZWD(pres_3d, qv_3d, temp_3d, dz):
    e_3d = qv_3d * pres_3d / (0.622 + qv_3d)
    N_wet = 22.1 * (e_3d / temp_3d) + 3.739e5 * (e_3d / (temp_3d * temp_3d))
    return 1e-6 * np.sum(N_wet * dz, axis=0)

def compute_ZTD(pres_3d, temp_3d, qv_3d, dz, p_top, h_top, lat):
    ZHD_b = compute_ZHD_b(pres_3d, temp_3d, dz)
    ZHD_t = compute_ZHD_t(p_top, h_top, lat)
    ZWD = compute_ZWD(pres_3d, qv_3d, temp_3d, dz)
    return ZHD_t + ZHD_b + ZWD

def process_folder(path, dataframes_general, dataframes_ztd, folder_label):
    all_files = sorted(glob.glob(os.path.join(path, "wrfout_d01_*")))
    total_files = len(all_files)
    start_time = time.time()

    for idx, filename in enumerate(all_files, start=1):
        try:
            print(f"\n[{folder_label}] Processing file: {filename}\n")

            # Open with GDAL and netCDF4
            ds = gdal.Open(filename)
            if ds is None:
                print(f"[ERROR] Could not open file: {filename}")
                continue
            subdatasets = ds.GetSubDatasets()
            ncfile = Dataset(filename, 'r')

            # --- Read required arrays ---
            raw_xlat   = read_subdataset_explicit(subdatasets, "XLAT")
            raw_xlong  = read_subdataset_explicit(subdatasets, "XLONG")
            raw_t2     = read_subdataset_explicit(subdatasets, "T2")
            raw_psfc   = read_subdataset_explicit(subdatasets, "PSFC")
            raw_rainnc = read_subdataset_explicit(subdatasets, "RAINNC")
            raw_rainc  = read_subdataset_explicit(subdatasets, "RAINC")
            raw_rainsh = read_subdataset_explicit(subdatasets, "RAINSH")
            raw_u10    = read_subdataset_explicit(subdatasets, "U10")
            raw_v10    = read_subdataset_explicit(subdatasets, "V10")
            raw_p      = read_subdataset_explicit(subdatasets, "P")
            raw_pb     = read_subdataset_explicit(subdatasets, "PB")
            raw_qv     = read_subdataset_explicit(subdatasets, "QVAPOR")
            raw_ph     = read_subdataset_explicit(subdatasets, "PH")
            raw_phb    = read_subdataset_explicit(subdatasets, "PHB")
            raw_t      = read_subdataset_explicit(subdatasets, "T")

            # --- Compute RH using wrf.rh ---
            qv_3d = getvar(ncfile, "QVAPOR", timeidx=0, meta=True)
            pres_3d = getvar(ncfile, "pressure", timeidx=0, meta=True) * 100
            temp_3d = getvar(ncfile, "tk", timeidx=0, meta=True)
            rh_3d = rh(qv_3d, pres_3d, temp_3d, meta=False)
            rh_2d = rh_3d[0, :, :]

            # --- Convert to 2D ---
            xlat   = slice_3d_to_2d(raw_xlat)
            xlong  = slice_3d_to_2d(raw_xlong)
            t2     = slice_3d_to_2d(raw_t2)
            psfc   = slice_3d_to_2d(raw_psfc)
            rainnc = slice_3d_to_2d(raw_rainnc)
            rainc  = slice_3d_to_2d(raw_rainc)
            rainsh = slice_3d_to_2d(raw_rainsh) if raw_rainsh is not None else np.zeros_like(rainnc)
            u10    = slice_3d_to_2d(raw_u10)
            v10    = slice_3d_to_2d(raw_v10)
            p3d    = raw_p
            pb3d   = raw_pb
            qv3d   = raw_qv
            ph     = raw_ph
            phb    = raw_phb
            t3d    = raw_t

            if xlat is None or xlong is None or t2 is None or psfc is None or rainnc is None or rainc is None:
                print(f"[WARNING] Essential arrays missing for file: {filename}")
                continue

            # --- Extract and format date and time ---
            file_basename = os.path.basename(filename)
            parts = file_basename.split("_")
            if len(parts) > 3:
                utc_date = parts[2]
                utc_time = parts[3].replace("_", ":")
                utc_datetime = f"{utc_date} {utc_time}"
            else:
                utc_datetime = "UNKNOWN_DATETIME"

            # --- Process General Stations ---
            for st_name, lat, lon in general_stations:
                i, j = find_nearest_grid_point(xlat, xlong, lat, lon)

                t2_val = t2[i, j] if t2 is not None else None
                psfc_val = psfc[i, j] if psfc is not None else None
                rainnc_val = rainnc[i, j] if rainnc is not None else None
                rainc_val = rainc[i, j] if rainc is not None else None
                rainsh_val = rainsh[i, j] if rainsh is not None else None
                u10_val = u10[i, j] if u10 is not None else None
                v10_val = v10[i, j] if v10 is not None else None
                rh_val = rh_2d[i, j] if rh_2d is not None else None

                out_t2 = t2_val - 273.15 if t2_val is not None else None
                out_precip = (rainnc_val + rainc_val + rainsh_val) if (rainnc_val is not None and rainc_val is not None and rainsh_val is not None) else None
                out_pres = psfc_val if psfc_val is not None else None
                out_wind = np.sqrt(u10_val**2 + v10_val**2) if (u10_val is not None and v10_val is not None) else None
                out_rh = rh_val

                # Updated debug output with RAINNC
                print(f"{st_name}: T2={t2_val:.2f} K, PSFC={out_pres:.2f} Pa, RAINNC={rainnc_val:.2f} mm, RH={out_rh:.2f}%")

                dataframes_general[st_name].append({
                    "UTC_Datetime": utc_datetime,
                    "Precipitation (mm)": out_precip,
                    "Temperature (°C)": out_t2,
                    "Wind Speed (m/s)": out_wind,
                    "Pressure (Pa)": out_pres,
                    "Relative Humidity (%)": out_rh,
                })

            # --- Compute ZTD ---
            full_pressure = p3d + pb3d
            geopotential_height = (ph + phb) / 9.80665
            dz = np.diff(geopotential_height, axis=0)
            theta = t3d + 290.0
            pres_3d = full_pressure / 100.0
            temp_3d = theta * (pres_3d / 1000.0) ** (2.0 / 7.0)
            p_top = full_pressure[-1, :, :] / 100
            h_top = geopotential_height[-1, :, :]
            ztd_2d = compute_ZTD(full_pressure, temp_3d, qv3d, dz, p_top, h_top, xlat)

            # --- Process ZTD Stations ---
            for st_name, lat, lon in ztd_stations:
                i, j = find_nearest_grid_point(xlat, xlong, lat, lon)
                out_ztd = ztd_2d[i, j] if ztd_2d is not None else None

                dataframes_ztd[st_name].append({
                    "UTC_Datetime": utc_datetime,
                    "ZTD (m)": out_ztd,
                })

            # --- Progress and Time Estimation ---
            elapsed_time = time.time() - start_time
            progress = idx / total_files * 100
            estimated_total_time = elapsed_time / idx * total_files
            remaining_time = estimated_total_time - elapsed_time
            elapsed_mins, elapsed_secs = divmod(elapsed_time, 60)
            remaining_mins, remaining_secs = divmod(remaining_time, 60)

            print(f"[INFO] Progress: {progress:.2f}% ({idx}/{total_files})")
            print(f"[INFO] Elapsed Time: {int(elapsed_mins)}m {int(elapsed_secs)}s")
            print(f"[INFO] Estimated Remaining Time: {int(remaining_mins)}m {int(remaining_secs)}s")

            ncfile.close()

        except Exception as e:
            print(f"[ERROR] Error processing {filename}: {e}")

# Process both folders
print("\nProcessing Before_DA folder...")
process_folder(before_da_path, dataframes_general_before, dataframes_ztd_before, "Before_DA")

print("\nProcessing After_DA folder...")
process_folder(after_da_path, dataframes_general_after, dataframes_ztd_after, "After_DA")

# --- Write Before_DA Data ---
print("\n[INFO] Writing Before_DA General station data to Excel...")
with pd.ExcelWriter(general_excel_file_before) as writer:
    for station, records in dataframes_general_before.items():
        pd.DataFrame(records).to_excel(writer, sheet_name=station, index=False)
print(f"[INFO] General station Excel file saved to {general_excel_file_before}")

print("\n[INFO] Writing Before_DA ZTD station data to Excel...")
with pd.ExcelWriter(ztd_excel_file_before) as writer:
    for station, records in dataframes_ztd_before.items():
        pd.DataFrame(records).to_excel(writer, sheet_name=station, index=False)
print(f"[INFO] ZTD Excel file saved to {ztd_excel_file_before}")

# --- Write After_DA Data ---
print("\n[INFO] Writing After_DA General station data to Excel...")
with pd.ExcelWriter(general_excel_file_after) as writer:
    for station, records in dataframes_general_after.items():
        pd.DataFrame(records).to_excel(writer, sheet_name=station, index=False)
print(f"[INFO] General station Excel file saved to {general_excel_file_after}")

print("\n[INFO] Writing After_DA ZTD station data to Excel...")
with pd.ExcelWriter(ztd_excel_file_after) as writer:
    for station, records in dataframes_ztd_after.items():
        pd.DataFrame(records).to_excel(writer, sheet_name=station, index=False)
print(f"[INFO] ZTD Excel file saved to {ztd_excel_file_after}")