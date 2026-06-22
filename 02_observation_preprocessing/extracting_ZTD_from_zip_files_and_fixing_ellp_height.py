import os
import zipfile
import gzip
import pandas as pd
from datetime import datetime, timedelta
import pyproj

###############################################################################
# Setup: High-Resolution EGM2008 (1' grid) for Orthometric Height Conversion
###############################################################################

# Define the EGM2008 high-resolution pipeline (1' grid resolution)
pipeline_string = """
+proj=pipeline
+step +proj=unitconvert +xy_in=deg +z_in=m +xy_out=rad +z_out=m
+step +proj=cart +ellps=GRS80
+step +inv +proj=cart +ellps=GRS80
+step +proj=vgridshift +grids=egm08_1.gtx +multiplier=-1
+step +proj=unitconvert +xy_in=rad +z_in=m +xy_out=deg +z_out=m
"""

# Create the Transformer using the updated high-resolution geoid model
try:
    transformer = pyproj.Transformer.from_pipeline(pipeline_string)
except pyproj.exceptions.ProjError as e:
    print(f"Error loading geoid model: {e}")
    transformer = None

###############################################################################
# Define input and output directories
###############################################################################
base_dir = '/Users/haseeb.rehman/WRF/WRFDA/DAT_DIR/ztd_data_may_june_2018/raw/'
processed_dir = os.path.join(base_dir, "ztd_processed")

# Specify the months to process (change this if needed)
months_to_process = [5, 6]  # June & July

# Create the output directory if it doesn't exist
os.makedirs(processed_dir, exist_ok=True)

###############################################################################
# Helper Functions
###############################################################################

def is_selected_month(file_name):
    """Checks if a file belongs to the requested months by parsing the filename."""
    try:
        parts = file_name.split('.')
        year = int(parts[1])
        day_of_year = int(parts[2])
        date = datetime(year=2000 + year, month=1, day=1) + timedelta(days=day_of_year - 1)
        return date.month in months_to_process
    except Exception:
        return False

def dms_to_decimal(degrees, minutes, seconds):
    """Converts degrees-minutes-seconds to decimal degrees."""
    return degrees + minutes / 60 + seconds / 3600

def convert_ellipsoidal_to_orthometric(lat_deg, lon_deg, ellipsoidal_height, transformer):
    """Convert WGS84 ellipsoidal height to corrected orthometric height."""
    if transformer:
        try:
            _, _, H_egm = transformer.transform(lon_deg, lat_deg, ellipsoidal_height)
            return H_egm  # Return the orthometric height
        except Exception as e:
            print(f"Error in height transformation: {e}")
            return ellipsoidal_height  # Fallback if transformation fails
    else:
        return ellipsoidal_height  # Return ellipsoidal height if geoid model is unavailable

def find_metadata_line(lines):
    """Finds the metadata line where station name, lon, lat, and height are located."""
    for idx, line in enumerate(lines):
        parts = line.strip().split()
        numeric_fields = [v for v in parts if v.replace('.', '', 1).isdigit()]
        if len(numeric_fields) >= 7:
            return line.strip(), idx
    return None, None

def parse_metadata_line(line):
    """Parses the metadata line to extract lat, lon, and height."""
    try:
        print(f"Processing metadata line: {line}")  
        parts = line.split()
        if len(parts) < 7:
            raise ValueError("Insufficient fields in metadata line")

        lon_deg, lon_min, lon_sec = float(parts[-7]), float(parts[-6]), float(parts[-5])
        longitude = dms_to_decimal(lon_deg, lon_min, lon_sec)

        lat_deg, lat_min, lat_sec = float(parts[-4]), float(parts[-3]), float(parts[-2])
        latitude = dms_to_decimal(lat_deg, lat_min, lat_sec)

        ellipsoidal_height = float(parts[-1])

        # ✅ Pass transformer correctly
        orth_height = convert_ellipsoidal_to_orthometric(latitude, longitude, ellipsoidal_height, transformer)

        return latitude, longitude, orth_height
    except Exception as e:
        print(f"Error parsing metadata line: {e}")
        return -999, -999, -999

def process_trop_file(file_content):
    """Extracts metadata and ZTD data from a .trop file."""
    site_coordinates = {}
    hourly_data = {}

    try:
        lines = file_content.decode('utf-8').splitlines()

        metadata_line, line_idx = find_metadata_line(lines)
        if not metadata_line:
            raise ValueError("Metadata line not found")

        latitude, longitude, height = parse_metadata_line(metadata_line)
        site = lines[line_idx].split()[0] if len(lines[line_idx].split()) > 0 else "UNKNOWN"

        site_coordinates[site] = {'Latitude': latitude, 'Longitude': longitude, 'Height': height}

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
            trot = float(parts[2]) / 10.0  # Convert mm to cm
            
            try:
                year, day_of_year, seconds = map(int, epoch.split(':'))
                date_time = datetime(year=2000 + year, month=1, day=1) + timedelta(days=day_of_year - 1, seconds=seconds)
                if date_time.minute != 0 or date_time.second != 0:
                    continue
                date_time_str = date_time.strftime("%Y-%m-%d_%H:%M:%S")
            except Exception as e:
                print(f"Error parsing epoch: {e}")
                continue
            
            latitude = site_coordinates.get(site, {}).get('Latitude', -999)
            longitude = site_coordinates.get(site, {}).get('Longitude', -999)
            height = site_coordinates.get(site, {}).get('Height', -999)
            
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

###############################################################################
# Main Processing Function (Fixed)
###############################################################################

def main():
    all_hourly_data = {}

    for root, _, files in os.walk(base_dir):
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
                                    hourly_data = process_trop_file(file_content)
                                    for hour, entries in hourly_data.items():
                                        if hour not in all_hourly_data:
                                            all_hourly_data[hour] = []
                                        all_hourly_data[hour].extend(entries)

    # ✅ **SAVE THE PROCESSED DATA TO FILES** ✅
    for date_time, entries in all_hourly_data.items():
        output_file = os.path.join(processed_dir, f"ob_{date_time.replace(':', '_')}.ascii")
        with open(output_file, 'w') as f:
            constant_values = "-888888.000 -88 200.00"
            for entry in entries:
                f.write(f"FM-114 GPSZD {date_time}      {entry['Site']}                                     0      {entry['Latitude']:.3f}                  {entry['Longitude']:.3f}                {entry['Height']:.3f}\n")
                f.write(f"{constant_values}     {entry['ZTD']:.3f}   0  0.800\n")
        print(f"✅ Processed and saved: {output_file}")

if __name__ == "__main__":
    print("Processing...")
    main()
