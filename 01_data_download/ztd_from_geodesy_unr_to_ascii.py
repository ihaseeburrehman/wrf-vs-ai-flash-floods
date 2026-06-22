import os
import zipfile
import gzip
import pandas as pd
from datetime import datetime, timedelta

# Define input and output directories
base_dir = '/Users/haseeb.rehman/WRF/WRFDA/DAT_DIR/ztd_data_July_August_2016/raw'
processed_dir = os.path.join(base_dir, "ztd_processed")

# Specify the months to process (change this if needed)
months_to_process = [7, 8]  # 6 = June, 7 = July

# Create the output directory if it doesn't exist
os.makedirs(processed_dir, exist_ok=True)

# Function to check if a file corresponds to the selected months
def is_selected_month(file_name):
    try:
        parts = file_name.split('.')
        year = int(parts[1])
        day_of_year = int(parts[2])
        date = datetime(year=2000 + year, month=1, day=1) + timedelta(days=day_of_year - 1)
        return date.month in months_to_process  # Check if the month is in the selected list
    except Exception:
        return False

# Function to convert DMS to decimal degrees
def dms_to_decimal(degrees, minutes, seconds):
    return degrees + minutes / 60 + seconds / 3600

# Function to dynamically search for the metadata line
def find_metadata_line(lines):
    for idx, line in enumerate(lines):
        parts = line.strip().split()
        numeric_fields = [v for v in parts if v.replace('.', '', 1).isdigit()]
        if len(numeric_fields) >= 7:  # Check if at least 7 numeric fields exist
            return line.strip(), idx  # Return the valid line and its index
    return None, None

# Function to parse metadata from a valid metadata line
def parse_metadata_line(line):
    try:
        print(f"Processing metadata line: {line}")  # Debugging output
        parts = line.split()
        if len(parts) < 7:
            raise ValueError("Insufficient fields in metadata line")

        lon_deg, lon_min, lon_sec = float(parts[-7]), float(parts[-6]), float(parts[-5])
        longitude = dms_to_decimal(lon_deg, lon_min, lon_sec)

        lat_deg, lat_min, lat_sec = float(parts[-4]), float(parts[-3]), float(parts[-2])
        latitude = dms_to_decimal(lat_deg, lat_min, lat_sec)

        height = float(parts[-1])

        return latitude, longitude, height
    except Exception as e:
        print(f"Error parsing metadata line: {e}")
        return -999, -999, -999

# Function to extract metadata and ZTD data from `.trop` files
def process_trop_file(file_content):
    site_coordinates = {}
    hourly_data = {}

    try:
        lines = file_content.decode('utf-8').splitlines()

        # Dynamically find the metadata line
        metadata_line, line_idx = find_metadata_line(lines)
        if not metadata_line:
            raise ValueError("Metadata line not found in any position")

        latitude, longitude, height = parse_metadata_line(metadata_line)
        site = lines[line_idx].split()[0] if len(lines[line_idx].split()) > 0 else "UNKNOWN"
        site_coordinates[site] = {
            'Latitude': latitude,
            'Longitude': longitude,
            'Height': height,
        }

        # Process ZTD data
        ztd_start_idx = None
        for idx, line in enumerate(lines):
            if line.startswith('*SITE ___EPOCH____ TROTOT'):
                ztd_start_idx = idx + 1
                break
        if ztd_start_idx is None:
            print("ZTD section not found.")
            return {}

        for line in lines[ztd_start_idx:]:
            if line.strip() == "" or line.startswith('%=ENDTRO'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            site = parts[0]
            epoch = parts[1]
            trot = float(parts[2]) / 10.0  # Convert TROTOT from mm to cm

            # Convert epoch to datetime
            try:
                year, day_of_year, seconds = map(int, epoch.split(':'))
                date_time = datetime(year=2000 + year, month=1, day=1) + timedelta(days=day_of_year - 1, seconds=seconds)

                # Filter to include only hourly entries
                if date_time.minute != 0 or date_time.second != 0:
                    continue

                date_time_str = date_time.strftime("%Y-%m-%d_%H:%M:%S")
            except Exception as e:
                print(f"Error parsing epoch: {e}")
                continue

            # Get site metadata or use placeholder values
            latitude = site_coordinates.get(site, {}).get('Latitude', -999)
            longitude = site_coordinates.get(site, {}).get('Longitude', -999)
            height = site_coordinates.get(site, {}).get('Height', -999)

            # Store the data in a dictionary for the given hour
            if date_time_str not in hourly_data:
                hourly_data[date_time_str] = []
            hourly_data[date_time_str].append({
                'Site': site,
                'Latitude': latitude,
                'Longitude': longitude,
                'Height': height,
                'ZTD': trot,
            })
    except Exception as e:
        print(f"Error processing .trop file: {e}")

    return hourly_data

# Main processing function
def main():
    all_hourly_data = {}  # Combine data for all files

    # Iterate through all zip files in the base directory
    for root, _, files in os.walk(base_dir):
        # Skip "for_validation" folder
        if "for_validation" in root:
            continue

        for file in files:
            if file.endswith('.zip'):
                zip_path = os.path.join(root, file)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    for nested_file in zip_ref.namelist():
                        if nested_file.endswith('.gz') and is_selected_month(nested_file):
                            with zip_ref.open(nested_file) as gz_file:
                                with gzip.open(gz_file, 'rb') as trop_file:
                                    file_content = trop_file.read()
                                    # Process the trop file content
                                    hourly_data = process_trop_file(file_content)

                                    # Combine data for all hours
                                    for hour, entries in hourly_data.items():
                                        if hour not in all_hourly_data:
                                            all_hourly_data[hour] = []
                                        all_hourly_data[hour].extend(entries)

    # Generate ASCII files
    for date_time, entries in all_hourly_data.items():
        output_file = os.path.join(processed_dir, f"ob_{date_time.replace(':', '_')}.ascii")
        with open(output_file, 'w') as f:
            constant_values = "-888888.000 -88 200.00"
            for entry in entries:
                f.write(f"FM-114 GPSZD {date_time} {entry['Site']}                                     0      {entry['Latitude']:.3f}                  {entry['Longitude']:.3f}                {entry['Height']:.3f}\n")
                f.write(f"{constant_values}       {entry['ZTD']:.3f}   0  0.800\n")
        print(f"Processed: {output_file}")

if __name__ == "__main__":
    main()
