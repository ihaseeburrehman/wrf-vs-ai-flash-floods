#!/usr/bin/env python3
"""
Memory-efficient WRF to TUFLOW NetCDF converter.
Writes each timestep immediately instead of accumulating in RAM.
NO DUPLICATES - Sequential indexing from 0, 1, 2, etc.

Author: Haseeb ur Rehman
Email: haseeb.rehman@uni.lu
"""

import xarray as xr
import numpy as np
from datetime import datetime
import pandas as pd
from pyproj import Transformer
import glob
import os
from scipy.interpolate import griddata
import netCDF4 as nc
import rasterio

# Paths
wrf_dir = "/path/to/wrf_outputs/"
dem_path = "/path/to/dem/your_dem.asc"
output_file = "/path/to/output/rain_output.nc"

# Automatically extract DEM grid parameters with full precision
print(f"Reading DEM parameters from: {dem_path}")
with rasterio.open(dem_path) as dem:
    # Get bounds with full precision
    min_x = dem.bounds.left
    max_x = dem.bounds.right
    min_y = dem.bounds.bottom
    max_y = dem.bounds.top
    
    # Get resolution (assuming square pixels)
    resolution = dem.transform[0]  # Cell width
    
    # Get dimensions
    nx = dem.width
    ny = dem.height
    
    print(f"  Bounds: min_x={min_x}, max_x={max_x}")
    print(f"  Bounds: min_y={min_y}, max_y={max_y}")
    print(f"  Resolution: {resolution} meters")
    print(f"  Dimensions: nx={nx}, ny={ny}")

# Cell-center coordinates
x_target = min_x + (np.arange(nx) + 0.5) * resolution
y_target = max_y - (np.arange(ny) + 0.5) * resolution  # Decreasing: top to bottom

# Load WRF files
wrf_files = sorted(glob.glob(os.path.join(wrf_dir, "wrfout_d0*")))
if not wrf_files:
    raise ValueError("No WRF NetCDF files found in directory.")

print(f"Found {len(wrf_files)} WRF files to process")

# Coordinate transformer: WGS84 to Luxembourg 1930
transformer = Transformer.from_crs("EPSG:4326", "EPSG:2169", always_xy=True)

# Extract times first and remove duplicates
print("Extracting timestamps and removing duplicates...")
time_file_map = {}  # Map datetime to file path
for wrf_file in wrf_files:
    try:
        with xr.open_dataset(wrf_file) as ds:
            time_str = str(ds['Times'].values[0]).replace("b'", "").replace("'", "")
            time_dt = pd.to_datetime(time_str, format='%Y-%m-%d_%H:%M:%S')
            
            # Keep only first occurrence of each timestamp
            if time_dt not in time_file_map:
                time_file_map[time_dt] = wrf_file
            else:
                print(f"  Skipping duplicate timestamp: {time_dt} from {os.path.basename(wrf_file)}")
    except Exception as e:
        print(f"Error reading time from {wrf_file}: {e}")

if not time_file_map:
    raise ValueError("No valid times found in WRF files")

# Sort by time and create sequential lists
sorted_times = sorted(time_file_map.keys())
unique_wrf_files = [time_file_map[t] for t in sorted_times]

print(f"Unique timesteps: {len(sorted_times)} (removed {len(wrf_files) - len(sorted_times)} duplicates)")

# Calculate time variable starting from 0
start_time = sorted_times[0]
time_var = np.array([(t - start_time).total_seconds() / 3600.0 for t in sorted_times])

nt = len(time_var)
print(f"Total timesteps: {nt}")
print(f"Time range: {sorted_times[0]} to {sorted_times[-1]}")

# Create NetCDF file structure
print(f"Creating NetCDF file: {output_file}")
ncfile = nc.Dataset(output_file, 'w', format='NETCDF4')

# Create dimensions
time_dim = ncfile.createDimension('time', None)  # Unlimited
x_dim = ncfile.createDimension('x', nx)
y_dim = ncfile.createDimension('y', ny)

# Create coordinate variables
time_var_nc = ncfile.createVariable('time', 'f8', ('time',))
time_var_nc.units = 'hour'
time_var_nc.long_name = 'time'
time_var_nc.axis = 'T'
time_var_nc[:] = time_var

x_var_nc = ncfile.createVariable('x', 'f8', ('x',))
x_var_nc.units = 'm'
x_var_nc.long_name = 'x'
x_var_nc.axis = 'X'
x_var_nc.spatial_ref = 'EPSG:2169'
x_var_nc[:] = x_target

y_var_nc = ncfile.createVariable('y', 'f8', ('y',))
y_var_nc.units = 'm'
y_var_nc.long_name = 'y'
y_var_nc.axis = 'Y'
y_var_nc.spatial_ref = 'EPSG:2169'
y_var_nc[:] = y_target

# Create rainfall variable
rainfall_var = ncfile.createVariable('rainfall_depth', 'f4', ('time', 'y', 'x'), 
                                     zlib=True, complevel=4)
rainfall_var.units = 'mm'
rainfall_var.long_name = 'rainfall_depth'

# Global attributes
ncfile.crs = 'EPSG:2169'
ncfile.spatial_ref = '+proj=tmerc +lat_0=49.8333333333333 +lon_0=6.16666666666667 +k=0.9996 +x_0=80000 +y_0=100000 +ellps=GRS80 +units=m +no_defs'
ncfile.description = f'WRF rainfall data (no duplicates), start time: {start_time}'

# Process each unique WRF file and write immediately
zero_rainfall = np.zeros((ny, nx), dtype=np.float32)

for idx, wrf_file in enumerate(unique_wrf_files):
    print(f"\nProcessing [{idx}/{nt-1}]: {os.path.basename(wrf_file)}")
    print(f"  Time: {sorted_times[idx]}, Hours from start: {time_var[idx]:.2f}")
    
    try:
        ds = xr.open_dataset(wrf_file)
        
        # Use RAINNC
        rainfall = ds['RAINNC']
        if rainfall.isnull().all():
            print(f"  Warning: RAINNC is all NaN, using zeros")
            rainfall_var[idx, :, :] = zero_rainfall
            ds.close()
            continue
        
        # Extract and flatten WRF coordinates and rainfall
        x_wrf = ds['XLONG'].values[0].ravel()
        y_wrf = ds['XLAT'].values[0].ravel()
        rainfall_values = rainfall.values[0].ravel()
        
        # Transform to EPSG:2169
        x_wrf_meters, y_wrf_meters = transformer.transform(x_wrf, y_wrf)
        
        # Create target grid points
        x_grid, y_grid = np.meshgrid(x_target, y_target)
        target_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
        
        # Interpolate
        print("  Interpolating...")
        rainfall_interpolated = griddata(
            (x_wrf_meters, y_wrf_meters),
            rainfall_values,
            target_points,
            method='linear',
            fill_value=0.0
        ).reshape(ny, nx)
        
        # Ensure non-negative
        rainfall_interpolated = np.where(rainfall_interpolated < 0, 0, rainfall_interpolated)
        
        # Write to NetCDF immediately
        print("  Writing to NetCDF...")
        rainfall_var[idx, :, :] = rainfall_interpolated.astype(np.float32)
        
        # Stats
        rainfall_sum = np.sum(rainfall_interpolated)
        non_zero = np.sum(rainfall_interpolated > 0)
        print(f"  Sum: {rainfall_sum:.2f} mm, Non-zero cells: {non_zero}/{nx*ny}")
        print(f"  Min: {np.min(rainfall_interpolated):.2f}, Max: {np.max(rainfall_interpolated):.2f} mm")
        
        ds.close()
        
    except Exception as e:
        print(f"  Error: {e}")
        # Write zeros on error
        rainfall_var[idx, :, :] = zero_rainfall
        continue

# Close NetCDF file
ncfile.close()
print(f"\n✅ NetCDF file saved to {output_file}")
print(f"Total timesteps written: {nt}")
print(f"Timestep indices: 0 to {nt-1}")
