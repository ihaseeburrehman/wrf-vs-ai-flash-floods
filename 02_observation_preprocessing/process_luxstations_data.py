#!/usr/bin/env python3
import os
import pandas as pd

# Directory containing the CSV files
input_dir = "/Users/haseeb.rehman/Documents/Misc/Luxembourg_stations_for_validation/2016_event"

# Output Excel file (corrected)
output_file = os.path.join(input_dir, "stations_6hr_cumulative.xlsx")


# Define a function to calculate cumulative precipitation and extract temp/RH for verification hours
def calculate_cumulative_precipitation(df, verification_hours=[0, 6, 12, 18]):
    results = []

    for verification_hour in verification_hours:
        # Find all potential verification times for the given hour
        verification_times = df.index[df.index.hour == verification_hour]
        
        for vt in verification_times:
            # Define the 6-hour window (vt - 5h to vt inclusive)
            start_time = vt - pd.Timedelta(hours=5)
            end_time = vt

            # Extract the data within the window
            window_data = df.loc[start_time:end_time]

            if len(window_data) == 6:  # Ensure we have a complete 6-hour window
                precip_cum = window_data['SUM_NN050'].sum()  # Cumulative precipitation
                temp_val = window_data.iloc[-1]['AVG_TA200']  # Temp at the verification hour
                rh_val = window_data.iloc[-1]['AVG_RH200']  # RH at the verification hour

                # Append the results
                results.append({
                    'UTC_Datetime': vt,
                    'Precip(mm)': precip_cum,
                    'Temp(2m)': temp_val,
                    'RH(%)': rh_val
                })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Add Luxembourg local time column (UTC+2)
    results_df['Lux_Datetime'] = results_df['UTC_Datetime'] + pd.Timedelta(hours=2)
    results_df['Lux_Datetime'] = results_df['Lux_Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    return results_df

# Process each CSV file in the input directory
station_data = {}
for csv_file in os.listdir(input_dir):
    if csv_file.lower().endswith('.csv'):
        station_name = os.path.splitext(csv_file)[0]  # Use filename as station name
        file_path = os.path.join(input_dir, csv_file)

        print(f"Processing file: {csv_file}")  # Debug: Print file being processed

        # Load the CSV file
        df = pd.read_csv(file_path, sep=';', decimal=',', engine='python')

        # Debug: Print columns in the current file
        print(f"Columns in {csv_file}: {df.columns.tolist()}")

        try:
            # Parse date and time columns
            df['datetime_local'] = pd.to_datetime(df['Tag'] + ' ' + df['Stunde'], format='%d.%m.%Y %H:%M')

            # Convert Luxembourg time (UTC+2) to UTC
            df['datetime_utc'] = df['datetime_local'] - pd.Timedelta(hours=2)

            # Set UTC datetime as the index
            df.set_index('datetime_utc', inplace=True)

            # Ensure numeric columns
            df['AVG_RH200'] = pd.to_numeric(df['AVG_RH200'], errors='coerce')
            df['AVG_TA200'] = pd.to_numeric(df['AVG_TA200'], errors='coerce')
            df['SUM_NN050'] = pd.to_numeric(df['SUM_NN050'], errors='coerce')

            # Calculate cumulative precipitation and verification data
            results = calculate_cumulative_precipitation(df)

            # Store the results for this station
            station_data[station_name] = results

        except KeyError as e:
            print(f"KeyError in file: {csv_file}, missing column: {e}")
        except Exception as e:
            print(f"Error processing file {csv_file}: {e}")

# Write all stations to a single Excel file with multiple sheets
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for station_name, data in station_data.items():
        data.to_excel(writer, sheet_name=station_name, index=False)

print("Processing complete. Results saved in:", output_file)

