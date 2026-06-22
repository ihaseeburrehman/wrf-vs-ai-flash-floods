#!/usr/bin/env python3
"""
Python script to download prepbufr files from rda.ucar.edu for 2016-07-10 to 2016-08-10 at 6-hour intervals.
Checks if files are already downloaded and complete (matching server size), skips them, and uses retries for reliability.
Make executable: chmod 755 download_prepbufr.py
"""

import os
import requests
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from urllib.parse import urljoin
import logging

# Configuration
base_url = "https://data-osdf.rda.ucar.edu/ncar/rda/d337000/prepnr/2016/"
output_dir = "/Users/haseeb.rehman/WRF/WRFDA/DAT_DIR/conventional_obs/20160710_20160810_prebufr"
start_date = datetime(2016, 7, 10)
end_date = datetime(2016, 8, 10)
max_workers = 8
max_retries = 3
initial_retry_delay = 2  # seconds

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Generate list of file URLs
filelist = []
current_date = start_date
while current_date <= end_date:
    for hour in ['00', '06', '12', '18']:  # Zero-padded hours
        timestamp = current_date.strftime("%Y%m%d") + hour
        filename = f"prepbufr.gdas.{timestamp}.nr"
        filelist.append((urljoin(base_url, filename), filename))
    current_date += timedelta(days=1)

def download_file(url, filename, retries=max_retries):
    """Download a file if not already present and completely downloaded."""
    output_path = os.path.join(output_dir, filename)

    # Check if file exists and is complete
    try:
        response = requests.head(url, timeout=5)
        if response.status_code == 200:
            server_size = response.headers.get('Content-Length')
            if server_size is not None:
                server_size = int(server_size)
                if os.path.exists(output_path):
                    local_size = os.path.getsize(output_path)
                    if local_size == server_size:
                        logging.info(f"Skipping {filename}: already downloaded and complete ({server_size} bytes)")
                        return  # File is complete, skip download
                    else:
                        logging.info(f"File {filename} exists but incomplete: local={local_size}, server={server_size}")
                else:
                    logging.info(f"File {filename} not found locally")
            else:
                logging.warning(f"No Content-Length header for {filename}, attempting download")
        else:
            logging.warning(f"HEAD request failed for {filename}: status {response.status_code}, attempting download")
    except requests.exceptions.RequestException as e:
        logging.error(f"Error checking {filename}: {e}, attempting download")

    # Download the file
    for attempt in range(retries):
        try:
            logging.info(f"Attempt {attempt + 1} to download {filename}")
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            # Verify downloaded file
            local_size = os.path.getsize(output_path)
            if local_size > 0:
                server_size = response.headers.get('Content-Length')
                if server_size is not None and int(server_size) == local_size:
                    logging.info(f"Saved {filename} to {output_path} (complete, {local_size} bytes)")
                    return
                else:
                    logging.warning(f"Downloaded {filename} but size mismatch: local={local_size}, server={server_size or 'unknown'}")
            else:
                logging.warning(f"Downloaded {filename} is empty")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to download {filename}: {e}")
        time.sleep(initial_retry_delay * (2 ** attempt))
    logging.error(f"Failed to download {filename} after {retries} attempts")

# Download files concurrently
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    executor.map(lambda x: download_file(*x), filelist)

print("Download process complete.")