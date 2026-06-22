#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 11:34:12 2025
Modified by Grok 3 to handle errors, optimize downloads, and adjust completeness

@author: haseeb.rehman
"""

import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from math import sqrt
import os
import requests
import sys

# Define your WRF domain
min_lat, max_lat = 44.60498809814453, 54.220977783203125
min_lon, max_lon = -1.324371337890625, 13.7489013671875

# Luxembourg centroid
lux_lat, lux_lon = 49.815, 6.129

# Directory to save downloaded ISD files
isd_dir = "/Users/haseeb.rehman/Documents/Misc/Luxembourg_stations_for_validation/2018_Event/Stations_other_than_lux"
os.makedirs(isd_dir, exist_ok=True)

# Function to calculate distance (km)
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# Function to download ISD data for a station
def download_isd_data(usaf, wban, year=2018):
    station_id = f"{usaf}{wban}"
    url = f"https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{station_id}.csv"
    output_file = os.path.join(isd_dir, f"{station_id}_{year}.csv")
    
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded data for {station_id} to {output_file}")
            return output_file
        else:
            print(f"No data available for {station_id} at {url}")
            return None
    except Exception as e:
        print(f"Error downloading {station_id}: {e}")
        return None

# Function to check data completeness
def check_data_completeness(filepath, start_date='2018-05-20', end_date='2018-06-20'):
    if not filepath or not os.path.exists(filepath):
        return False, 0
    
    try:
        df = pd.read_csv(filepath, low_memory=False)
        df['UTC_Datetime'] = pd.to_datetime(df['DATE'], utc=True)
        df = df[(df['UTC_Datetime'] >= start_date) & (df['UTC_Datetime'] <= end_date)]
        
        if 'TMP' not in df.columns or 'DEW' not in df.columns or 'WND' not in df.columns:
            return False, 0
        
        # Parse fields (include AA1 if available, but don’t require it)
        df['Precip_Period'] = df.get('AA1', pd.Series(np.nan, index=df.index)).apply(
            lambda x: float(x.split(',')[0]) if isinstance(x, str) and '9999' not in x else np.nan
        )
        df['Precip(mm)'] = df.get('AA1', pd.Series(np.nan, index=df.index)).apply(
            lambda x: float(x.split(',')[1])/10 if isinstance(x, str) and '9999' not in x else np.nan
        )
        df['Temp(2m)'] = df['TMP'].apply(lambda x: float(x.split(',')[0])/10 if isinstance(x, str) and x != '+9999,9' else np.nan)
        df['Dew_Point'] = df['DEW'].apply(lambda x: float(x.split(',')[0])/10 if isinstance(x, str) and x != '+9999,9' else np.nan)
        df['Wind_Speed(m/s)'] = df['WND'].apply(lambda x: float(x.split(',')[3])/10 if isinstance(x, str) and '9999' not in x else np.nan)
        
        # Use all data, not just hourly precipitation
        expected_hours = 744  # no of days * 24 hours
        completeness = df[['Temp(2m)', 'Dew_Point', 'Wind_Speed(m/s)']].notna().all(axis=1).sum() / expected_hours
        print(f"Completeness for {filepath}: {completeness:.2%}")
        return completeness >= 0.8, completeness  # 80% threshold
    except Exception as e:
        print(f"Error checking completeness for {filepath}: {e}")
        return False, 0

# Load isd-history.csv
file_path = "/Users/haseeb.rehman/Documents/Misc/isd-history.csv"
isd_history = pd.read_csv(file_path, delimiter=",")
isd_history.columns = isd_history.columns.str.strip()

# Filter for 2021 availability
isd_history['END'] = pd.to_datetime(isd_history['END'], format='%Y%m%d')
isd_history = isd_history[isd_history['END'] >= '2018-12-31']

# Filter stations within domain
domain_stations = isd_history[
    (isd_history['LAT'] >= min_lat) & (isd_history['LAT'] <= max_lat) &
    (isd_history['LON'] >= min_lon) & (isd_history['LON'] <= max_lon) &
    (isd_history['CTRY'].isin(['LU', 'GM', 'FR', 'BE']))
].copy()

# Extract SYNOP stations from assimilation file
assimilation_file = "/Users/haseeb.rehman/WRF/WRFDA/DAT_DIR/conventional_obs/ob_ascii_may_june_2018_3dvar/obs_gts_2018-05-31_18_00.3DVAR"
synop_ids = set()
with open(assimilation_file, 'r') as f:
    for line in f:
        if "FM-12 SYNOP" in line:
            parts = line.split()
            station_id = parts[2].strip()
            if station_id.isdigit():
                synop_ids.add(station_id)

# Fix dtype warning
domain_stations.loc[:, 'USAF'] = domain_stations['USAF'].astype(str)
domain_stations.loc[:, 'WBAN'] = domain_stations['WBAN'].astype(str).str.zfill(5)
domain_stations = domain_stations[~domain_stations['USAF'].isin(synop_ids)]

# Print number of candidate stations
print(f"Number of candidate stations for download: {len(domain_stations)}")

# Check for existing files before downloading
domain_stations['Filepath'] = domain_stations.apply(
    lambda row: os.path.join(isd_dir, f"{row['USAF']}{row['WBAN']}_2018.csv") 
    if os.path.exists(os.path.join(isd_dir, f"{row['USAF']}{row['WBAN']}_2018.csv")) 
    else download_isd_data(row['USAF'], row['WBAN']), axis=1
)
domain_stations[['Has_Complete_Data', 'Completeness']] = domain_stations['Filepath'].apply(
    lambda x: pd.Series(check_data_completeness(x))
)
domain_stations = domain_stations[domain_stations['Has_Complete_Data']]

# Check if any stations remain
if domain_stations.empty:
    print("No stations with complete data found (80% threshold). Try lowering further or checking downloaded files.")
    sys.exit(1)

# Calculate distance from Luxembourg
domain_stations['DIST_FROM_LUX'] = domain_stations.apply(
    lambda row: haversine_distance(row['LAT'], row['LON'], lux_lat, lux_lon), axis=1
)

# Select 10 well-distributed stations using K-means
coords = domain_stations[['LAT', 'LON']].values
kmeans = KMeans(n_clusters=10, random_state=42).fit(coords)
domain_stations['cluster'] = kmeans.labels_

selected_stations = []
for cluster in range(10):
    cluster_stations = domain_stations[domain_stations['cluster'] == cluster]
    if not cluster_stations.empty:
        centroid = kmeans.cluster_centers_[cluster]
        distances = ((cluster_stations['LAT'] - centroid[0])**2 + 
                     (cluster_stations['LON'] - centroid[1])**2)
        closest_station = cluster_stations.loc[distances.idxmin()]
        selected_stations.append(closest_station)

kmeans_df = pd.DataFrame(selected_stations)

# Add 6 more stations (2 per country: BE, GM, FR), far from Luxembourg
additional_stations = []
for country in ['BE', 'GM', 'FR']:
    country_stations = domain_stations[
        (domain_stations['CTRY'] == country) & 
        (domain_stations['DIST_FROM_LUX'] > 100)
    ]
    if not country_stations.empty:
        country_stations = country_stations.sort_values(by='DIST_FROM_LUX', ascending=False)
        additional_stations.extend(country_stations.head(2).to_dict('records'))

additional_df = pd.DataFrame(additional_stations)

# Combine and print
final_df = pd.concat([kmeans_df, additional_df], ignore_index=True)
print("16 Well-Distributed Stations with Complete Data (80% threshold):")
print(final_df[['USAF', 'WBAN', 'STATION NAME', 'CTRY', 'LAT', 'LON', 'ELEV(M)', 'DIST_FROM_LUX', 'Completeness']])

# Save to CSV in isd_dir
output_csv = os.path.join(isd_dir, "selected_stations_extended_complete.csv")
final_df.to_csv(output_csv, index=False)
print(f"Station list saved to {output_csv}")