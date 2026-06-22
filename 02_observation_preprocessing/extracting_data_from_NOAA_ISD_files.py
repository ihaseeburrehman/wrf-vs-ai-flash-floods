#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to extract weather parameters from NOAA ISD files for June and July 2018 at 6-hour intervals.
Sheet names will be in the format 'USAFID_StationName' (e.g., '106330_LUXEMBOURG').
"""

import pandas as pd
import os
import numpy as np
import re

# Directory containing ISD CSV files
input_dir = "/Users/haseeb.rehman/Documents/Misc/Luxembourg_stations_for_validation/2016_Event/Stations_other_than_lux"
output_excel = "/Users/haseeb.rehman/Documents/Misc/Luxembourg_stations_for_validation/2016_Event/Stations_other_than_lux/station_weather_data_2016_6hr.xlsx"

# ------------------------
# Parsing Helper Functions
# ------------------------

def parse_temperature(temp_str):
    """Parse temperature string like '+0250,1' => 25.0°C. Returns np.nan if invalid."""
    if pd.isna(temp_str) or temp_str == "+9999,9":
        return np.nan
    try:
        return float(temp_str.split(',')[0]) / 10
    except (ValueError, IndexError):
        return np.nan

def parse_precipitation(precip_str):
    """
    Parse precipitation string like '01,005' => period=1 hour, amount=0.5 mm.
    Returns (np.nan, np.nan) if invalid or 9999 is present.
    """
    if pd.isna(precip_str) or "9999" in precip_str:
        return np.nan, np.nan
    try:
        parts = precip_str.split(',')
        period = int(parts[0])            # e.g. '01' for 1-hour period
        amount = float(parts[1]) / 10     # e.g. '005' => 0.5 mm
        if amount == 999.9:
            amount = np.nan
        return period, amount
    except (ValueError, IndexError):
        return np.nan, np.nan

def parse_wind_speed(wind_str):
    """
    Parse wind string like '010,1,340,0013' => 1.3 m/s (fourth element / 10).
    Returns np.nan if invalid or 9999 is present.
    """
    if pd.isna(wind_str) or "9999" in wind_str:
        return np.nan
    try:
        return float(wind_str.split(',')[3]) / 10
    except (ValueError, IndexError):
        return np.nan

def calculate_rh(temp_c, dew_c):
    """
    Calculate relative humidity (%) from temperature and dew point using
    the Magnus formula for saturation vapor pressure.
    """
    if pd.isna(temp_c) or pd.isna(dew_c):
        return np.nan
    try:
        temp_c, dew_c = float(temp_c), float(dew_c)
        e_dew = 6.1078 * 10 ** ((7.5 * dew_c) / (237.3 + dew_c))
        e_sat = 6.1078 * 10 ** ((7.5 * temp_c) / (237.3 + temp_c))
        return min((e_dew / e_sat) * 100, 100.0)
    except (ValueError, ZeroDivisionError):
        return np.nan

# ------------------------
# Main Processing
# ------------------------

with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
    
    # Loop over all CSV files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(input_dir, filename)
            
            try:
                # Read the CSV file
                df = pd.read_csv(filepath, low_memory=False)
                
                # Extract station name from 'NAME' column
                if 'NAME' in df.columns and not df['NAME'].isna().all():
                    station_name_raw = df['NAME'].iloc[0]
                    station_name = station_name_raw.split(',')[0].strip()
                    
                    # Convert to proper capitalization (only first letter of each word capitalized)
                    station_name = station_name.lower().capitalize()
                else:
                    station_name = "Unknown"
                
                # Construct sheet name
                sheet_name = f"{station_name}"
                # Convert DATE to datetime (UTC)
                df['UTC_Datetime'] = pd.to_datetime(df['DATE'], utc=True)
                
                # Parse precipitation from the "AA1" column
                df[['Precip_Period', 'Precip(mm)']] = pd.DataFrame(
                    df.get('AA1', pd.Series([np.nan] * len(df))).apply(parse_precipitation).tolist(),
                    index=df.index
                )
                
                # Parse temperatures from "TMP" and "DEW"
                df['Temp(2m)'] = df['TMP'].apply(parse_temperature)
                df['Dew_Point'] = df['DEW'].apply(parse_temperature)
                
                # Calculate relative humidity
                df['RH(%)'] = df.apply(lambda row: calculate_rh(row['Temp(2m)'], row['Dew_Point']), axis=1)
                
                # Parse wind speed from "WND"
                df['Wind_Speed(m/s)'] = df['WND'].apply(parse_wind_speed)
                
                # Filter for reqyured month and year
                start_date = pd.Timestamp('2016-07-10', tz='UTC')
                end_date = pd.Timestamp('2016-08-10 23:59:59', tz='UTC')
                df = df[(df['UTC_Datetime'] >= start_date) & (df['UTC_Datetime'] <= end_date)]
                
                if df.empty:
                    print(f"No data for required month and year in {sheet_name}")
                    continue
                
                # Set UTC_Datetime as index for resampling
                df.set_index('UTC_Datetime', inplace=True)
                
                # Resample precipitation by summing over 6-hour intervals
                precip_6hr = df['Precip(mm)'].resample('6H').sum().reset_index()
                precip_6hr.rename(columns={'Precip(mm)': 'Precip(mm)'}, inplace=True)  # Renamed to Precip(mm)
                
                # Resample instantaneous data: take the first record for temperature, RH, and wind speed
                instant_df = df[['Temp(2m)', 'RH(%)', 'Wind_Speed(m/s)']].resample('6H').first().reset_index()
                
                # Merge the resampled datasets on UTC_Datetime
                final_df = pd.merge(precip_6hr, instant_df, on='UTC_Datetime', how='inner')
                
                # Remove timezone and format datetime as 'M/D/YYYY H'
                final_df['UTC_Datetime'] = final_df['UTC_Datetime'].dt.tz_localize(None).dt.strftime('%m/%d/%Y %H')
                
                # Replace any infinite values with NaN
                final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                
                # Reorder columns as requested
                final_df = final_df[[
                    'UTC_Datetime',
                    'Precip(mm)',
                    'Temp(2m)',
                    'RH(%)',
                    'Wind_Speed(m/s)'
                ]]
                
                # Ensure the sheet name is at most 31 characters
                sheet_name = sheet_name[:31]
                final_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                print(f"Processed station: {sheet_name}")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")

print(f"\nAll stations processed. Excel saved to: {output_excel}")