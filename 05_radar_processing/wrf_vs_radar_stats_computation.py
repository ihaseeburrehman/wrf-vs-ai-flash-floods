#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compare WRF (ERA5 and GFS) precipitation with Radar data
for July 11-18, 2021 event at 6-hour intervals.
Computes statistics and generates comparison plots.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import h5py
import os
from netCDF4 import Dataset
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------
# Time period (6-hourly: 00, 06, 12, 18 UTC)
start_date = datetime(2021, 7, 11, 0)
end_date = datetime(2021, 7, 18, 18)

# Directories
radar_base_dir = "/Users/haseeb.rehman/Documents/Misc/Data_Datasets/Radar_and_Weather/Belgium_Radar_data_2021"
gfs_wrf_dir = "/Users/haseeb.rehman/Documents/Misc/From_HPC_and_WRF/WRF_from_HPC/1_month_simulation_2021_new_GFS_000_cv5"
era5_wrf_dir = "/Users/haseeb.rehman/Documents/Misc/From_HPC_and_WRF/WRF_from_HPC/4th_year/2021_ERA5_cv5"
output_dir = "/Users/haseeb.rehman/Desktop/For_Animation/4th_Year/Miscs/GFS_vs_ERA5_vs_RADAR_vs_Observed"
os.makedirs(output_dir, exist_ok=True)

# Plot styling - muted vs normal for same model comparison
plt.style.use('seaborn-white')
colors = {
    'observed': '#ca252a',      # dark red
    'radar': 'black',         # Slate Grey 
    'era5_after': '#4366f5',    # Dark Blue
    'era5_no_da': '#5a5a5a',    # Light Blue
    'gfs_after': 'green',     # Dark Green
    'gfs_no_da': '#717171',     
}

label_fontsize = 10  # Match reference
title_fontsize = 12
bar_width = 0.10

# Radar grid parameters (from your reference script)
xsize, ysize = 700, 700

# Ask user if they want Excel files
print("\n" + "="*60)
save_excel = input("Do you want to save Excel files with metrics? (yes/no) [default: no]: ").strip().lower()
save_excel = save_excel in ['yes', 'y']

# Ask user which datasets to include
print("\nWhich datasets do you want to include?")
include_gfs = input("Include GFS (Before/After DA)? (yes/no) [default: yes]: ").strip().lower()
include_gfs = include_gfs not in ['no', 'n']

include_era5 = input("Include ERA5 (Before/After DA)? (yes/no) [default: yes]: ").strip().lower()
include_era5 = include_era5 not in ['no', 'n']

if not include_gfs and not include_era5:
    print("ERROR: At least one of GFS or ERA5 must be included!")
    exit(1)

print(f"\nWill include: {'GFS' if include_gfs else ''} {'ERA5' if include_era5 else ''}")
print("="*60 + "\n")

# Get radar projection information from a sample TIF file
try:
    from osgeo import gdal, osr
    gdal.UseExceptions()
    
    sample_tif = "/Users/haseeb.rehman/Documents/Misc/Belgium_Radar_data_2021/2021/07/14/accum1h/tif/20210714190000.radclim.accum1h.tif"
    ds_sample = gdal.Open(sample_tif)
    radar_geotransform = ds_sample.GetGeoTransform()
    radar_proj_wkt = ds_sample.GetProjection()
    
    # Create coordinate transformer from WGS84 to radar projection
    radar_srs = osr.SpatialReference()
    radar_srs.ImportFromWkt(radar_proj_wkt)
    wgs84_srs = osr.SpatialReference()
    wgs84_srs.ImportFromEPSG(4326)
    coord_transform = osr.CoordinateTransformation(wgs84_srs, radar_srs)
    ds_sample = None
    
    print("Loaded radar projection from TIF file")
    print(f"GeoTransform: {radar_geotransform}")
    use_proper_coords = True
except Exception as e:
    print(f"Warning: Could not load radar projection: {e}")
    print("Falling back to approximate lat/lon grid")
    use_proper_coords = False

# Station coordinates (name, lat, lon) - needed for radar extraction
met_stations = [
    ("Briedfeld", 50.12385, 6.06622), ("Echternach", 49.8031, 6.44337), ("Ettelbruck", 49.85172, 6.09754),
    ("Oberkorn", 49.5122, 5.9011), ("Remerschen", 49.491, 6.349), ("Findel", 49.63265182, 6.23292867),
    ("Roodt", 49.7945, 5.8202), ("Hosingen", 49.99314, 6.10147), ("Useldange", 49.76739, 5.96748),
    ("Mamer", 49.63353, 6.0193), ("Arsdorf", 49.85891, 5.84868), ("Asselborn", 50.09685689, 5.96960753),
    ("Grevenmacher", 49.68087, 6.43541), ("Schimpach", 50.0093, 5.8475), ("Waldbillig", 49.79806, 6.2773),
    ("Bettendorf", 49.8741, 6.2095), ("Fouhren", 49.91445, 6.19508), ("Beringen", 49.762, 6.11179),
    ("Dahl", 49.93595, 5.98093)
]

# Create a dictionary for easy lookup
station_coords = {name: (lat, lon) for name, lat, lon in met_stations}

# Select stations to plot (8 stations for two 2x2 subplot layouts, excluding Findel)
select_stations = ['Ettelbruck', 'Remerschen', 'Oberkorn', 'Bettendorf', 'Echternach', 'Hosingen', 'Mamer', 'Arsdorf']

# ------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------
def aggregate_belgium_radar(radar_base_dir, target_time):
    """
    Aggregate 6 hourly radar files to get 6-hour accumulated precipitation.
    
    Important: Radar files are named by their END time.
    File 20210714060000.radclim.accum1h.hdf = precip from 05:00-06:00
    
    For 6-hour accumulation ending at target_time:
    - We need files with timestamps from (target_time - 5 hours) to target_time
    - This gives us 6 hourly files covering the full 6-hour period
    
    CRITICAL: Radar data is stored as integers scaled by 1000 (i.e., mm * 1000)
    We must apply a scale factor of 0.001 to convert to mm
    """
    accum_data = np.zeros((ysize, xsize), dtype=float)
    file_count = 0
    
    start_time = target_time - timedelta(hours=5)
    period_files = []
    
    # Debug flag for July 15 00:00 and July 14 18:00
    is_debug = (target_time == datetime(2021, 7, 15, 0) or target_time == datetime(2021, 7, 14, 18))
    
    # Ettelbruck coordinates for debugging
    if is_debug:
        ett_lat, ett_lon = 49.85172, 6.09754
        radar_lat_grid = np.linspace(48.5, 51.5, ysize)
        radar_lon_grid = np.linspace(2.5, 8.5, xsize)
        ett_lat_idx = np.argmin(np.abs(radar_lat_grid - ett_lat))
        ett_lon_idx = np.argmin(np.abs(radar_lon_grid - ett_lon))
        print(f"\n  DEBUG: Aggregating for {target_time}")
        print(f"  Ettelbruck coords: ({ett_lat}, {ett_lon}) -> grid indices ({ett_lat_idx}, {ett_lon_idx})")
        print(f"  Time range: {start_time.strftime('%Y-%m-%d %H:%M')} to {target_time.strftime('%Y-%m-%d %H:%M')}")
        files_debug = []
    
    # Collect dates to check
    dates_to_check = set()
    current = start_time
    while current <= target_time:
        dates_to_check.add(current.strftime("%Y/%m/%d"))
        current += timedelta(hours=1)
    
    # Collect all radar files from relevant dates
    for date_str in sorted(dates_to_check):
        radar_dir = os.path.join(radar_base_dir, date_str, "accum1h/hdf")
        
        if not os.path.exists(radar_dir):
            continue
            
        all_files = [f for f in os.listdir(radar_dir) if f.endswith('.accum1h.hdf')]
        
        for f in all_files:
            try:
                file_timestamp = datetime.strptime(f[:14], '%Y%m%d%H%M%S')
                
                if start_time <= file_timestamp <= target_time:
                    file_path = os.path.join(radar_dir, f)
                    with h5py.File(file_path, 'r') as hdf:
                        data = hdf['dataset1']['data1']['data'][:].astype(float)
                        data = np.where(data < 0, 0, data)
                        
                        # CRITICAL FIX: Apply scale factor
                        # Radar data is stored as mm*1000, so we need to divide by 1000
                        # Detect if scaling is needed (values > 1000 indicate scaled data)
                        raw_max = np.nanmax(data) if np.any(data > 0) else 0
                        if raw_max > 1000:
                            scale_factor = 0.001  # Convert from mm*1000 to mm
                            data = data * scale_factor
                            if is_debug and file_count == 0:
                                print(f"  Applying scale factor {scale_factor} (raw_max={raw_max:.0f})")
                        
                        accum_data += data
                        file_count += 1
                        period_files.append(f)
                        
                        if is_debug:
                            val_at_ett = data[ett_lat_idx, ett_lon_idx]
                            files_debug.append((f, file_timestamp, val_at_ett))
            except Exception as e:
                continue
    
    if is_debug and files_debug:
        print(f"\n  Files used ({len(files_debug)}):")
        for fname, ftime, val in sorted(files_debug, key=lambda x: x[1]):
            print(f"    {ftime.strftime('%Y-%m-%d %H:%M')}: {val:.4f} mm (from {fname})")
        total_debug = sum(v for _, _, v in files_debug)
        print(f"  Total at Ettelbruck: {total_debug:.4f} mm")
    
    if file_count > 0:
        return accum_data, file_count
    else:
        return None, 0

def read_wrf_precip_6hr(wrf_dir, target_time, scenario):
    """
    Read WRF 6-hour accumulated precipitation.
    WRF files are named: wrfout_d01_YYYY-MM-DD_HH_00_00
    """
    # For 6-hour accumulation, we need 2 WRF files (start and end of period)
    start_time = target_time - timedelta(hours=6)
    
    # Format filenames
    start_filename = f"wrfout_d01_{start_time.strftime('%Y-%m-%d_%H')}_00_00"
    end_filename = f"wrfout_d01_{target_time.strftime('%Y-%m-%d_%H')}_00_00"
    
    start_path = os.path.join(wrf_dir, scenario, start_filename)
    end_path = os.path.join(wrf_dir, scenario, end_filename)
    
    if not os.path.exists(start_path) or not os.path.exists(end_path):
        print(f"  WRF file not found for {scenario}: {start_filename} or {end_filename}")
        return None
    
    try:
        # Read start time
        nc_start = Dataset(start_path, 'r')
        rainnc_start = nc_start.variables['RAINNC'][0, :, :]
        try:
            rainc_start = nc_start.variables['RAINC'][0, :, :]
            total_start = rainnc_start + rainc_start
        except:
            total_start = rainnc_start
        nc_start.close()
        
        # Read end time
        nc_end = Dataset(end_path, 'r')
        rainnc_end = nc_end.variables['RAINNC'][0, :, :]
        try:
            rainc_end = nc_end.variables['RAINC'][0, :, :]
            total_end = rainnc_end + rainc_end
        except:
            total_end = rainnc_end
        
        # Get lat/lon from end file
        lat = nc_end.variables['XLAT'][0, :, :]
        lon = nc_end.variables['XLONG'][0, :, :]
        nc_end.close()
        
        # Calculate 6-hour accumulation
        precip_6hr = total_end - total_start
        
        return precip_6hr, lat, lon
    
    except Exception as e:
        print(f"  Error reading WRF files for {scenario}: {e}")
        return None

def regrid_to_radar(wrf_data, wrf_lat, wrf_lon, radar_shape=(700, 700)):
    """
    Regrid WRF data to radar grid using nearest neighbor.
    This is a simplified approach - adjust if you need more sophisticated interpolation.
    """
    from scipy.interpolate import griddata
    
    # Flatten WRF coordinates
    wrf_points = np.column_stack((wrf_lon.ravel(), wrf_lat.ravel()))
    wrf_values = wrf_data.ravel()
    
    # Create radar grid (you'll need to define the radar extent)
    # Using approximate Belgian Lambert coordinates converted to lat/lon
    # Adjust these based on your actual radar grid
    radar_lat = np.linspace(48.5, 51.5, radar_shape[0])
    radar_lon = np.linspace(2.5, 8.5, radar_shape[1])
    radar_lon_grid, radar_lat_grid = np.meshgrid(radar_lon, radar_lat)
    
    # Interpolate
    radar_points = np.column_stack((radar_lon_grid.ravel(), radar_lat_grid.ravel()))
    regridded = griddata(wrf_points, wrf_values, radar_points, method='nearest')
    regridded = regridded.reshape(radar_shape)
    
    return regridded

def compute_metrics(predicted, observed):
    """Compute metrics between predicted and observed data."""
    # Flatten and remove NaN and negative values
    pred_flat = predicted.ravel()
    obs_flat = observed.ravel()
    
    # Create mask for valid data (both must be valid and non-negative)
    mask = ~(np.isnan(pred_flat) | np.isnan(obs_flat) | (pred_flat < 0) | (obs_flat < 0))
    pred_valid = pred_flat[mask]
    obs_valid = obs_flat[mask]
    
    if len(pred_valid) == 0:
        return np.nan, np.nan, np.nan, np.nan
    
    # RMSE
    rmse = np.sqrt(mean_squared_error(pred_valid, obs_valid))
    
    # MAE
    mae = np.mean(np.abs(pred_valid - obs_valid))
    
    # MAPE (Symmetric)
    numerator = np.abs(pred_valid - obs_valid)
    denominator = (np.abs(pred_valid) + np.abs(obs_valid)) / 2
    nonzero = denominator > 0.01  # Avoid division by very small numbers
    mape = np.mean((numerator[nonzero] / denominator[nonzero])) * 100 if np.any(nonzero) else np.nan
    
    # Bias
    bias = np.mean(pred_valid - obs_valid)
    
    return rmse, mae, mape, bias

def compute_pod_far(predicted, observed, threshold=0.1):
    """Compute POD and FAR for precipitation."""
    pred_flat = predicted.ravel()
    obs_flat = observed.ravel()
    
    mask = ~(np.isnan(pred_flat) | np.isnan(obs_flat) | (pred_flat < 0) | (obs_flat < 0))
    pred_valid = pred_flat[mask]
    obs_valid = obs_flat[mask]
    
    if len(pred_valid) == 0:
        return np.nan, np.nan
    
    obs_rain = obs_valid > threshold
    pred_rain = pred_valid > threshold
    
    hits = ((pred_rain) & (obs_rain)).sum()
    misses = ((~pred_rain) & (obs_rain)).sum()
    false_alarms = ((pred_rain) & (~obs_rain)).sum()
    
    pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan
    
    return pod, far

# ------------------------------------------------------------------------
# Main Processing
# ------------------------------------------------------------------------
print("Starting WRF vs Radar comparison...")
print(f"Date range: {start_date.strftime('%Y-%m-%d %H:00')} to {end_date.strftime('%Y-%m-%d %H:00')}")
print(f"6-hourly intervals")

# Storage for time series and metrics
time_series_data = []
all_metrics = {
    'gfs_before': {'rmse': [], 'mae': [], 'mape': [], 'bias': [], 'pod': [], 'far': []},
    'gfs_after': {'rmse': [], 'mae': [], 'mape': [], 'bias': [], 'pod': [], 'far': []},
    'era5_before': {'rmse': [], 'mae': [], 'mape': [], 'bias': [], 'pod': [], 'far': []},
    'era5_after': {'rmse': [], 'mae': [], 'mape': [], 'bias': [], 'pod': [], 'far': []}
}

# Storage for station time series (radar and WRF at station coordinates)
station_time_series = {station: [] for station in select_stations}

# Process each 6-hour interval
current_time = start_date
time_count = 0

while current_time <= end_date:
    # Only process 00, 06, 12, 18 UTC
    if current_time.hour not in [0, 6, 12, 18]:
        current_time += timedelta(hours=6)
        continue
    
    # Read aggregated radar data
    radar_data, radar_file_count = aggregate_belgium_radar(radar_base_dir, current_time)
    
    if radar_data is None:
        current_time += timedelta(hours=6)
        continue
    
    print(f"Processing {current_time.strftime('%Y-%m-%d %H:00')}: Radar files={radar_file_count}", end="")
    
    # Extract radar values at station coordinates for station time series
    for station_name in station_time_series.keys():
        if station_name in station_coords:
            lat, lon = station_coords[station_name]
            
            if use_proper_coords:
                # Convert lat/lon to radar projection coordinates
                x_proj, y_proj, _ = coord_transform.TransformPoint(lat, lon)
                
                # Convert projected coordinates to pixel indices
                col = int((x_proj - radar_geotransform[0]) / radar_geotransform[1])
                row = int((y_proj - radar_geotransform[3]) / radar_geotransform[5])
                
                # Extract value from radar grid
                if 0 <= row < ysize and 0 <= col < xsize:
                    radar_at_station = radar_data[row, col]
                else:
                    radar_at_station = np.nan
            else:
                # Fallback to approximate method
                radar_lat_grid = np.linspace(48.5, 51.5, ysize)
                radar_lon_grid = np.linspace(2.5, 8.5, xsize)
                lat_idx = np.argmin(np.abs(radar_lat_grid - lat))
                lon_idx = np.argmin(np.abs(radar_lon_grid - lon))
                radar_at_station = radar_data[lat_idx, lon_idx] if 0 <= lat_idx < ysize and 0 <= lon_idx < xsize else np.nan
            
            station_time_series[station_name].append({
                'datetime': current_time,
                'radar': radar_at_station
            })
            
            # Debug: Print Ettelbruck radar value for July 15 00:00
            if station_name == 'Ettelbruck' and current_time == datetime(2021, 7, 15, 0):
                print(f"\n  DEBUG Ettelbruck @ {current_time}:")
                print(f"    Grid indices: row={row}, col={col}" if use_proper_coords else f"    Using approximate grid")
                print(f"    Radar 6hr total: {radar_at_station:.2f} mm")
                print(f"    (Should be ~24mm based on 6hr_aggregate_from_radar_files.py)")
    
    # Read WRF data for selected scenarios only
    scenarios = {}
    if include_gfs:
        scenarios['gfs_before'] = ('Before_DA', gfs_wrf_dir)
        scenarios['gfs_after'] = ('After_DA', gfs_wrf_dir)
    if include_era5:
        scenarios['era5_before'] = ('Before_DA', era5_wrf_dir)
        scenarios['era5_after'] = ('After_DA', era5_wrf_dir)
    
    wrf_data = {}
    ts_entry = {'datetime': current_time, 'radar': np.nanmean(radar_data)}
    
    for key, (scenario, base_dir) in scenarios.items():
        result = read_wrf_precip_6hr(base_dir, current_time, scenario)
        
        if result is not None:
            precip, lat, lon = result
            
            # Regrid to radar grid
            precip_regridded = regrid_to_radar(precip, lat, lon)
            wrf_data[key] = precip_regridded
            
            # Compute metrics
            rmse, mae, mape, bias = compute_metrics(precip_regridded, radar_data)
            pod, far = compute_pod_far(precip_regridded, radar_data)
            
            all_metrics[key]['rmse'].append(rmse)
            all_metrics[key]['mae'].append(mae)
            all_metrics[key]['mape'].append(mape)
            all_metrics[key]['bias'].append(bias)
            all_metrics[key]['pod'].append(pod)
            all_metrics[key]['far'].append(far)
            
            # Store for time series
            wrf_mean = np.nanmean(precip[precip > 0]) if np.any(precip > 0) else 0
            ts_entry[key] = wrf_mean
        else:
            ts_entry[key] = np.nan
    
    print(f", WRF={len(wrf_data)} scenarios")
    if wrf_data:
        time_series_data.append(ts_entry)
        time_count += 1
    
    current_time += timedelta(hours=6)

print(f"\n{'='*60}")
print(f"Processed {time_count} time steps")
print(f"{'='*60}")

# ------------------------------------------------------------------------
# Compute Overall Statistics
# ------------------------------------------------------------------------
metrics_summary = {}
for key in all_metrics.keys():
    metrics_summary[key] = {
        'RMSE': np.nanmean(all_metrics[key]['rmse']),
        'MAE': np.nanmean(all_metrics[key]['mae']),
        'MAPE': np.nanmean(all_metrics[key]['mape']),
        'Bias': np.nanmean(all_metrics[key]['bias']),
        'POD': np.nanmean(all_metrics[key]['pod']),
        'FAR': np.nanmean(all_metrics[key]['far'])
    }

print("\nOverall Metrics Summary:")
for key, metrics in metrics_summary.items():
    print(f"\n{key.upper()}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")

# ------------------------------------------------------------------------
# Plot 1: Bar Chart Comparison
# ------------------------------------------------------------------------
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
fig.suptitle('WRF vs Radar: 6-Hour Precipitation Metrics (July 11-18, 2021)', fontsize=title_fontsize, fontweight='bold')

# Metrics plot - using SMAPE/100 instead of MAPE
metrics_list = ['RMSE', 'MAE', 'SMAPE/100', 'Bias']
x = np.arange(len(metrics_list)) * 0.6

gfs_before_vals = [metrics_summary['gfs_before']['RMSE'], metrics_summary['gfs_before']['MAE'], metrics_summary['gfs_before']['MAPE']/100, metrics_summary['gfs_before']['Bias']]
gfs_after_vals = [metrics_summary['gfs_after']['RMSE'], metrics_summary['gfs_after']['MAE'], metrics_summary['gfs_after']['MAPE']/100, metrics_summary['gfs_after']['Bias']]
era5_before_vals = [metrics_summary['era5_before']['RMSE'], metrics_summary['era5_before']['MAE'], metrics_summary['era5_before']['MAPE']/100, metrics_summary['era5_before']['Bias']]
era5_after_vals = [metrics_summary['era5_after']['RMSE'], metrics_summary['era5_after']['MAE'], metrics_summary['era5_after']['MAPE']/100, metrics_summary['era5_after']['Bias']]

gap = 0.15
bars1 = axes[0].bar(x - gap/2 - bar_width, gfs_before_vals, bar_width, label='GFS No DA', color=colors['gfs_no_da'], alpha=0.7, edgecolor='black')
bars2 = axes[0].bar(x - gap/2, gfs_after_vals, bar_width, label='GFS After DA', color=colors['gfs_after'], alpha=0.7, edgecolor='black')
bars3 = axes[0].bar(x + gap/2, era5_before_vals, bar_width, label='ERA5 No DA', color=colors['era5_no_da'], alpha=0.7, edgecolor='black')
bars4 = axes[0].bar(x + gap/2 + bar_width, era5_after_vals, bar_width, label='ERA5 After DA', color=colors['era5_after'], alpha=0.7, edgecolor='black')

# Add text inside bars
for bar, val in zip(bars1, gfs_before_vals):
    if not pd.isna(val):
        axes[0].text(bar.get_x() + bar.get_width()/2, val/2, f'{val:.2f}', ha='center', va='center', rotation=90, color='white', fontsize=8, fontweight='bold')
for bar, val in zip(bars2, gfs_after_vals):
    if not pd.isna(val):
        axes[0].text(bar.get_x() + bar.get_width()/2, val/2, f'{val:.2f}', ha='center', va='center', rotation=90, color='black', fontsize=8)
for bar, val in zip(bars3, era5_before_vals):
    if not pd.isna(val):
        axes[0].text(bar.get_x() + bar.get_width()/2, val/2, f'{val:.2f}', ha='center', va='center', rotation=90, color='white', fontsize=8, fontweight='bold')
for bar, val in zip(bars4, era5_after_vals):
    if not pd.isna(val):
        axes[0].text(bar.get_x() + bar.get_width()/2, val/2, f'{val:.2f}', ha='center', va='center', rotation=90, color='black', fontsize=8)

axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics_list, fontsize=label_fontsize)
axes[0].set_ylabel('Metric Value (mm)', fontsize=label_fontsize)
axes[0].set_title('Error Metrics', fontsize=label_fontsize, fontweight='bold')
axes[0].legend(loc='upper right', frameon=True, edgecolor='black', fontsize=10, ncol=2)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].grid(axis='y', alpha=0.3, linestyle='--')

# POD/FAR plot
pod_far_metrics = ['POD', 'FAR']
x_pf = np.arange(len(pod_far_metrics)) * 0.6

gfs_before_pf = [metrics_summary['gfs_before'][m] for m in pod_far_metrics]
gfs_after_pf = [metrics_summary['gfs_after'][m] for m in pod_far_metrics]
era5_before_pf = [metrics_summary['era5_before'][m] for m in pod_far_metrics]
era5_after_pf = [metrics_summary['era5_after'][m] for m in pod_far_metrics]

bars_pf1 = axes[1].bar(x_pf - gap/2 - bar_width, gfs_before_pf, bar_width, label='GFS No DA', color=colors['gfs_no_da'], alpha=0.7, edgecolor='black')
bars_pf2 = axes[1].bar(x_pf - gap/2, gfs_after_pf, bar_width, label='GFS After DA', color=colors['gfs_after'], alpha=0.7, edgecolor='black')
bars_pf3 = axes[1].bar(x_pf + gap/2, era5_before_pf, bar_width, label='ERA5 No DA', color=colors['era5_no_da'], alpha=0.7, edgecolor='black')
bars_pf4 = axes[1].bar(x_pf + gap/2 + bar_width, era5_after_pf, bar_width, label='ERA5 After DA', color=colors['era5_after'], alpha=0.7, edgecolor='black')

# Add text inside bars
for bar, val in zip(bars_pf1, gfs_before_pf):
    if not pd.isna(val):
        axes[1].text(bar.get_x() + bar.get_width()/2, val/2, f'{val:.2f}', ha='center', va='center', rotation=90, color='white', fontsize=8, fontweight='bold')
for bar, val in zip(bars_pf2, gfs_after_pf):
    if not pd.isna(val):
        axes[1].text(bar.get_x() + bar.get_width()/2, val/2, f'{val:.2f}', ha='center', va='center', rotation=90, color='black', fontsize=8)
for bar, val in zip(bars_pf3, era5_before_pf):
    if not pd.isna(val):
        axes[1].text(bar.get_x() + bar.get_width()/2, val/2, f'{val:.2f}', ha='center', va='center', rotation=90, color='white', fontsize=8, fontweight='bold')
for bar, val in zip(bars_pf4, era5_after_pf):
    if not pd.isna(val):
        axes[1].text(bar.get_x() + bar.get_width()/2, val/2, f'{val:.2f}', ha='center', va='center', rotation=90, color='black', fontsize=8)

axes[1].set_xticks(x_pf)
axes[1].set_xticklabels(pod_far_metrics, fontsize=label_fontsize)
axes[1].set_ylabel('Value', fontsize=label_fontsize)
axes[1].set_title('Detection Metrics', fontsize=label_fontsize, fontweight='bold')
axes[1].legend(loc='upper right', frameon=True, edgecolor='black', fontsize=10, ncol=2)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(output_dir, 'wrf_vs_radar_metrics.png'), dpi=300, bbox_inches='tight')
plt.close()
print("\nSaved: wrf_vs_radar_metrics.png")

# ------------------------------------------------------------------------
# Plot 2: Time Series
# ------------------------------------------------------------------------
if len(time_series_data) > 0:
    df_ts = pd.DataFrame(time_series_data)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle('6-Hour Precipitation Time Series: WRF vs Radar (Spatial Mean)', fontsize=title_fontsize, fontweight='bold')
    
    # Build list of columns to plot based on user selection
    ts_cols = []
    ts_colors_line = []
    ts_linestyles_list = []
    ts_labels_list = []
    
    if include_gfs:
        ts_cols.extend(['gfs_before', 'gfs_after'])
        ts_colors_line.extend([colors['gfs_no_da'], colors['gfs_after']])
        ts_linestyles_list.extend(['-', '-'])
        ts_labels_list.extend(['GFS No DA', 'GFS After DA'])
    
    if include_era5:
        ts_cols.extend(['era5_before', 'era5_after'])
        ts_colors_line.extend([colors['era5_no_da'], colors['era5_after']])
        ts_linestyles_list.extend(['-', '-'])
        ts_labels_list.extend(['ERA5 No DA', 'ERA5 After DA'])
    
    # Always add radar
    ts_cols.append('radar')
    ts_colors_line.append(colors['radar'])
    ts_linestyles_list.append('--')
    ts_labels_list.append('Radar')
    
    for col, color, style, label in zip(ts_cols, ts_colors_line, ts_linestyles_list, ts_labels_list):
        if col in df_ts.columns:
            ax.plot(df_ts['datetime'], df_ts[col], 
                   label=label, color=color, linestyle=style, 
                   alpha=0.8, linewidth=1.5)
    
    ax.set_xlabel('Date', fontsize=label_fontsize)
    ax.set_ylabel('6-hr Precip (mm)', fontsize=label_fontsize)
    ax.legend(loc='best', frameon=True, edgecolor='black', fontsize=10, ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='both', alpha=0.3, linestyle='--')
    
    # Format x-axis as GMT style (14-Jul) - horizontal labels
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    # Horizontal labels (no rotation)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'wrf_vs_radar_timeseries.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: wrf_vs_radar_timeseries.png")

# ------------------------------------------------------------------------
# Plot 3: Station Time Series (2x2 subplot with shared legend)
# ------------------------------------------------------------------------
# Observed data files
file_observed_general = '/Users/haseeb.rehman/Documents/Misc/Data_Datasets/Stations_and_Observations/Luxembourg_stations_for_validation/2021_Event/stations_6hr_cumulative.xlsx'

# Station data files (6-hour accumulated from WRF)
gfs_before_station_file = '/Users/haseeb.rehman/Desktop/For_Animation/3rd_Year/1_month_simulation_2021_new_GFS_000_cv5/Before_DA/general_station_data_before.xlsx'
gfs_after_station_file = '/Users/haseeb.rehman/Desktop/For_Animation/3rd_Year/1_month_simulation_2021_new_GFS_000_cv5/After_DA/general_station_data_after.xlsx'
era5_before_station_file = '/Users/haseeb.rehman/Desktop/For_Animation/4th_Year/2021_ERA5_cv5/Before_DA/general_station_data_before.xlsx'
era5_after_station_file = '/Users/haseeb.rehman/Desktop/For_Animation/4th_Year/2021_ERA5_cv5/After_DA/general_station_data_after.xlsx'

print("\n" + "="*60)
print("Generating station time series plots (two 2x2 layouts)...")
print("="*60)

import matplotlib.dates as mdates

# Function to create a single 2x2 figure
def create_station_figure(stations_subset, figure_number):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    legend_handles = []
    legend_labels = []
    
    for idx, station in enumerate(stations_subset):
        if station not in station_coords:
            print(f"Coordinates not found for {station}")
            continue
        
        ax = axes[idx]
        
        try:
            # Read observed data
            df_obs = pd.read_excel(file_observed_general, sheet_name=station)
            df_obs.rename(columns={'Precip(mm)': 'Obs_Precip'}, inplace=True)
            df_obs['UTC_Datetime'] = pd.to_datetime(df_obs['UTC_Datetime'], errors='coerce')
            df_obs = df_obs.sort_values('UTC_Datetime').reset_index(drop=True)
            
            # Read WRF station data
            df_gfs_before = pd.read_excel(gfs_before_station_file, sheet_name=station)
            df_gfs_after = pd.read_excel(gfs_after_station_file, sheet_name=station)
            df_era5_before = pd.read_excel(era5_before_station_file, sheet_name=station)
            df_era5_after = pd.read_excel(era5_after_station_file, sheet_name=station)
            
            # Format WRF datetime
            for df in [df_gfs_before, df_gfs_after, df_era5_before, df_era5_after]:
                if df['UTC_Datetime'].dtype == 'object':
                    df['UTC_Datetime'] = pd.to_datetime(df['UTC_Datetime'], format='%Y-%m-%d %H', errors='coerce')
                else:
                    df['UTC_Datetime'] = pd.to_datetime(df['UTC_Datetime'], errors='coerce')
                df.rename(columns={'Precipitation (mm)': 'Precip'}, inplace=True)
            
            # Filter to date range
            mask_obs = (df_obs['UTC_Datetime'] >= start_date) & (df_obs['UTC_Datetime'] <= end_date)
            df_obs_filtered = df_obs[mask_obs].copy()
            
            mask_wrf = (df_gfs_before['UTC_Datetime'] >= start_date) & (df_gfs_before['UTC_Datetime'] <= end_date)
            df_gfs_before_filtered = df_gfs_before[mask_wrf].copy()
            df_gfs_after_filtered = df_gfs_after[mask_wrf].copy()
            df_era5_before_filtered = df_era5_before[mask_wrf].copy()
            df_era5_after_filtered = df_era5_after[mask_wrf].copy()
            
            # Get radar time series
            df_radar_station = pd.DataFrame(station_time_series[station]) if station in station_time_series else pd.DataFrame()
            
            # Build datasets list
            datasets_station = []
            ts_colors_station = []
            ts_linestyles_station = []
            ts_markers_station = []
            ts_labels_station = []
            
            if include_gfs:
                datasets_station.extend([
                    (df_gfs_before_filtered, 'Precip'),
                    (df_gfs_after_filtered, 'Precip')
                ])
                ts_colors_station.extend([colors['gfs_no_da'], colors['gfs_after']])
                ts_linestyles_station.extend(['-', '-'])
                ts_markers_station.extend(['o', 'o'])  # Circle for GFS
                ts_labels_station.extend(['GFS No DA', 'GFS After DA'])
            
            if include_era5:
                datasets_station.extend([
                    (df_era5_before_filtered, 'Precip'),
                    (df_era5_after_filtered, 'Precip')
                ])
                ts_colors_station.extend([colors['era5_no_da'], colors['era5_after']])
                ts_linestyles_station.extend(['-', '-'])
                ts_markers_station.extend(['s', 's'])  # Square for ERA5
                ts_labels_station.extend(['ERA5 No DA', 'ERA5 After DA'])
            
            # Radar and Observed last
            datasets_station.extend([
                (df_radar_station, 'radar'),
                (df_obs_filtered, 'Obs_Precip')
            ])
            ts_colors_station.extend([colors['radar'], colors['observed']])
            ts_linestyles_station.extend(['--', '-'])
            ts_markers_station.extend(['', '^'])  # No marker for radar and observed
            ts_labels_station.extend(['RADAR', 'Observed'])
            
            # Plot each dataset
            for (df_data, col), color, style, marker, label in zip(datasets_station, ts_colors_station, ts_linestyles_station, ts_markers_station, ts_labels_station):
                if not df_data.empty and col in df_data.columns:
                    datetime_col = 'datetime' if 'datetime' in df_data.columns else 'UTC_Datetime'
                    line, = ax.plot(df_data[datetime_col], df_data[col],
                           label=label, color=color, linestyle=style,
                           marker=marker, markersize=4,
                           alpha=0.8, linewidth=1.5)
                    if idx == 0:
                        legend_handles.append(line)
                        legend_labels.append(label)
            
            ax.set_title(f'Station: {station}', fontsize=label_fontsize, fontweight='bold')
            ax.set_ylabel('6-hr Precip (mm)', fontsize=16)
            ax.tick_params(axis='both', labelsize=13, which='both', direction='out', length=4, width=1)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(axis='both', alpha=0.3, linestyle='--')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            
        except Exception as e:
            import traceback
            print(f"Could not process station {station}: {e}")
            traceback.print_exc()
            ax.set_visible(False)
            continue
    
    # Add shared legend at the bottom
    fig.legend(legend_handles, legend_labels, 
               loc='lower center', 
               bbox_to_anchor=(0.5, 0.02),
               fontsize=11, 
               frameon=True, 
               edgecolor='black',
               fancybox=False,
               ncol=len(legend_labels))
    
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    filename = f'station_timeseries_combined_{figure_number}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename} (2x2 layout, 600 DPI)")
    
    # ========================================================================
    # EXPORT LaTeX/pgfplots version
    # ========================================================================
    if include_era5:  # Only generate LaTeX if ERA5 is included
        latex_filename = f'station_timeseries_{figure_number}.tex'
        latex_file = os.path.join(output_dir, latex_filename)
        
        with open(latex_file, 'w') as f:
            f.write("% LaTeX code for ERA5 vs RADAR vs Observed time series\n")
            f.write("% Required: \\usepackage{pgfplots} \\pgfplotsset{compat=1.18}\n")
            f.write("% Use: \\input{" + latex_filename + "}\n\n")
            f.write("\\begin{figure}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\begin{tikzpicture}\n")
            
            for subplot_idx, station in enumerate(stations_subset):
                if station not in station_coords:
                    continue
                
                try:
                    # Read observed data
                    df_obs = pd.read_excel(file_observed_general, sheet_name=station)
                    df_obs.rename(columns={'Precip(mm)': 'Obs_Precip'}, inplace=True)
                    df_obs['UTC_Datetime'] = pd.to_datetime(df_obs['UTC_Datetime'], errors='coerce')
                    df_obs = df_obs.sort_values('UTC_Datetime').reset_index(drop=True)
                    
                    # Read WRF station data
                    df_era5_before = pd.read_excel(era5_before_station_file, sheet_name=station)
                    df_era5_after = pd.read_excel(era5_after_station_file, sheet_name=station)
                    
                    for df in [df_era5_before, df_era5_after]:
                        if df['UTC_Datetime'].dtype == 'object':
                            df['UTC_Datetime'] = pd.to_datetime(df['UTC_Datetime'], format='%Y-%m-%d %H', errors='coerce')
                        else:
                            df['UTC_Datetime'] = pd.to_datetime(df['UTC_Datetime'], errors='coerce')
                        df.rename(columns={'Precipitation (mm)': 'Precip'}, inplace=True)
                    
                    # Filter to date range
                    mask_obs = (df_obs['UTC_Datetime'] >= start_date) & (df_obs['UTC_Datetime'] <= end_date)
                    df_obs_filtered = df_obs[mask_obs].copy()
                    
                    mask_wrf = (df_era5_before['UTC_Datetime'] >= start_date) & (df_era5_before['UTC_Datetime'] <= end_date)
                    df_era5_before_filtered = df_era5_before[mask_wrf].copy()
                    df_era5_after_filtered = df_era5_after[mask_wrf].copy()
                    
                    # Get radar time series
                    df_radar_station = pd.DataFrame(station_time_series[station]) if station in station_time_series else pd.DataFrame()
                    
                    # Create combined CSV
                    csv_filename = f'station_{station.replace(" ", "_")}_fig{figure_number}.csv'
                    csv_path = os.path.join(output_dir, csv_filename)
                    
                    # Merge all data on datetime
                    df_combined = df_obs_filtered[['UTC_Datetime', 'Obs_Precip']].copy()
                    df_combined = df_combined.merge(df_era5_before_filtered[['UTC_Datetime', 'Precip']].rename(columns={'Precip': 'ERA5_NoDA'}), 
                                                   on='UTC_Datetime', how='outer')
                    df_combined = df_combined.merge(df_era5_after_filtered[['UTC_Datetime', 'Precip']].rename(columns={'Precip': 'ERA5_AfterDA'}), 
                                                   on='UTC_Datetime', how='outer')
                    
                    if not df_radar_station.empty:
                        df_radar_station_renamed = df_radar_station.rename(columns={'datetime': 'UTC_Datetime', 'radar': 'RADAR'})
                        df_combined = df_combined.merge(df_radar_station_renamed[['UTC_Datetime', 'RADAR']], 
                                                       on='UTC_Datetime', how='outer')
                    else:
                        df_combined['RADAR'] = np.nan
                    
                    df_combined = df_combined.sort_values('UTC_Datetime').reset_index(drop=True)
                    
                    # Convert to numeric days for pgfplots
                    df_combined['Days'] = (df_combined['UTC_Datetime'] - start_date).dt.total_seconds() / 86400
                    df_combined['DateLabel'] = df_combined['UTC_Datetime'].dt.strftime('%d-%b')
                    
                    # Save CSV
                    df_combined[['UTC_Datetime', 'Days', 'DateLabel', 'ERA5_NoDA', 'ERA5_AfterDA', 'RADAR', 'Obs_Precip']].to_csv(csv_path, index=False)
                    
                    # Calculate plot position (2x2 grid)
                    row = subplot_idx // 2
                    col = subplot_idx % 2
                    xpos = col * 9
                    ypos = -row * 7.5
                    
                    # Calculate x-axis ticks (every 3 days)
                    max_days = df_combined['Days'].max()
                    xtick_days = [0, 3, 6, 9]  # 0, 3, 6, 9 days from start
                    xtick_labels = []
                    for d in xtick_days:
                        tick_date = start_date + timedelta(days=d)
                        xtick_labels.append(tick_date.strftime('%d-%b'))
                    
                    # Write pgfplot
                    f.write(f"\\begin{{axis}}[\n")
                    f.write(f"    unbounded coords=jump,\n")
                    f.write(f"    at={{({xpos}cm,{ypos}cm)}},\n")
                    f.write(f"    width=8.5cm,\n")
                    f.write(f"    height=6.5cm,\n")
                    f.write(f"    title={{\\textbf{{Station: {station}}}}},\n")
                    f.write(f"    xlabel={{Date (July 2021)}},\n")
                    f.write(f"    ylabel={{6-hr Precip (mm)}},\n")
                    f.write(f"    xmin=0, xmax={max_days:.1f},\n")
                    f.write(f"    ymin=0,\n")
                    f.write(f"    grid=major,\n")
                    f.write(f"    grid style={{dashed, gray}},\n")  # Grey dashed grid
                    f.write(f"    xtick={{{','.join([str(x) for x in xtick_days])}}},\n")
                    f.write(f"    xticklabels={{{','.join(xtick_labels)}}},\n")
                    f.write(f"    xticklabel style={{anchor=east}},\n")  # Horizontal labels
                    f.write(f"    legend pos=north east,\n")
                    f.write(f"    legend cell align=left,\n")
                    f.write(f"    legend style={{font=\\small}},\n")
                    f.write(f"]\n")
                    
                    # Plot datasets with exact styling
                    # ERA5 No DA - grey with square markers
                    f.write(f"\\addplot[gray, thick, mark=square, mark size=2pt] table[x=Days, y=ERA5_NoDA, col sep=comma] {{{csv_filename}}};\n")
                    if subplot_idx == 0:
                        f.write(f"\\addlegendentry{{ERA5 No DA}}\n")
                    
                    # ERA5 After DA - green with square markers
                    f.write(f"\\addplot[green!70!black, thick, mark=square*, mark size=2pt] table[x=Days, y=ERA5_AfterDA, col sep=comma] {{{csv_filename}}};\n")
                    if subplot_idx == 0:
                        f.write(f"\\addlegendentry{{ERA5 After DA}}\n")
                    
                    # RADAR - black dotted
                    f.write(f"\\addplot[black, dotted, thick, mark=none] table[x=Days, y=RADAR, col sep=comma] {{{csv_filename}}};\n")
                    if subplot_idx == 0:
                        f.write(f"\\addlegendentry{{RADAR}}\n")
                    
                    # Observed - red dotted
                    f.write(f"\\addplot[red, dotted, thick, mark=none] table[x=Days, y=Obs_Precip, col sep=comma] {{{csv_filename}}};\n")
                    if subplot_idx == 0:
                        f.write(f"\\addlegendentry{{Observed}}\n")
                    
                    f.write(f"\\end{{axis}}\n\n")
                    
                except Exception as e:
                    print(f"Could not export LaTeX for {station}: {e}")
                    continue
            
            f.write("\\end{tikzpicture}\n")
            f.write(f"\\caption{{ERA5 vs RADAR vs Observed precipitation time series (Figure {figure_number})}}\n")
            f.write(f"\\label{{fig:era5_timeseries_{figure_number}}}\n")
            f.write("\\end{figure}\n")
        
        print(f"📄 Saved LaTeX: {latex_filename}")


# Create two figures with 4 stations each
create_station_figure(select_stations[:4], 1)
create_station_figure(select_stations[4:8], 2)

# ------------------------------------------------------------------------
# Save metrics to Excel (only if user requested)
# ------------------------------------------------------------------------
if save_excel:
    metrics_df = pd.DataFrame(metrics_summary).T
    metrics_df.to_excel(os.path.join(output_dir, 'wrf_vs_radar_metrics.xlsx'))
    print("Saved: wrf_vs_radar_metrics.xlsx")
    
    # Save time series
    if len(time_series_data) > 0:
        df_ts.to_excel(os.path.join(output_dir, 'wrf_vs_radar_timeseries.xlsx'), index=False)
        print("Saved: wrf_vs_radar_timeseries.xlsx")

print("\n" + "="*60)
print("WRF vs Radar comparison complete!")
print(f"Output directory: {output_dir}")
print("="*60)
