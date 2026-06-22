#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to download 2018 weather data for specified stations from NOAA NCEI.

@author: haseeb.rehman
Created on Wed Mar 12 2025
"""

import requests
import os

# List of 16 stations from your output from file selected_stations_extended_complete.csv which you get from script getting_required_stations_data_from_noaa_isd.py
stations = [
    {"USAF": "070550", "STATION NAME": "TILLE", "CTRY": "FR"},
    {"USAF": "104690", "STATION NAME": "LEIPZIG_HALLE", "CTRY": "GM"},
    {"USAF": "107080", "STATION NAME": "SAARBRUCKEN", "CTRY": "GM"},
    {"USAF": "072550", "STATION NAME": "BOURGES", "CTRY": "FR"},
    {"USAF": "074820", "STATION NAME": "AMBERIEU", "CTRY": "FR"},
    {"USAF": "103340", "STATION NAME": "WUNSTORF", "CTRY": "GM"},
    {"USAF": "064590", "STATION NAME": "ERNAGE", "CTRY": "BE"},
    {"USAF": "108520", "STATION NAME": "AUGSBURG", "CTRY": "GM"},
    {"USAF": "071390", "STATION NAME": "ALENCON_VALFRAMBERT", "CTRY": "FR"},
    {"USAF": "074200", "STATION NAME": "CHAMPNIERS", "CTRY": "FR"},
    {"USAF": "064000", "STATION NAME": "KOKSIJDE", "CTRY": "BE"},
    {"USAF": "064070", "STATION NAME": "OOSTENDE", "CTRY": "BE"},
    {"USAF": "101720", "STATION NAME": "LAAGE", "CTRY": "GM"},
    {"USAF": "103850", "STATION NAME": "SCHONEFELD", "CTRY": "GM"},
    {"USAF": "075000", "STATION NAME": "CAP_FERRET", "CTRY": "FR"},
    {"USAF": "075100", "STATION NAME": "MERIGNAC", "CTRY": "FR"}

]

# Base URL for NOAA NCEI Global Hourly data for 2018
base_url = "https://www.ncei.noaa.gov/data/global-hourly/access/2018/"

# Output directory
output_dir = "/Users/haseeb.rehman/Documents/Misc/Luxembourg_stations_for_validation/2018_Event/Stations_other_than_lux"
os.makedirs(output_dir, exist_ok=True)

# Default WBAN (adjust if you have specific WBAN values)
default_wban = "99999"

print("Downloading 2018 data for 16 stations:")
for station in stations:
    usaf = str(station['USAF']).zfill(6)  # Ensure 6 digits
    wban = default_wban  # Use 99999 unless you provide specific WBAN
    station_id = f"{usaf}{wban}"
    # Replace slashes and other invalid filename characters in station name
    safe_station_name = station['STATION NAME'].replace('/', '_').replace('\\', '_')
    url = f"{base_url}{station_id}.csv"
    filename = f"{output_dir}/{station_id}_{safe_station_name}.csv"
    
    print(f"Downloading {station_id} ({station['STATION NAME']}) from {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise error for bad status codes
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Saved to {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {station_id}: {e}")

print("\nDownload complete.")

