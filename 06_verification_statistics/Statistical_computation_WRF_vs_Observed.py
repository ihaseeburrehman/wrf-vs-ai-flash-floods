#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compare WRF Before/After DA with observed meteorological and ZTD data,
including new NOAA ISD stations for June-July 2021. Handles NaN values by skipping them.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import seaborn as sns
import os
import warnings


# Suppress all warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------
# 0) Basic Setup
# ------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

base_dir = "/Users/haseeb.rehman/Desktop/For_Animation/4th_Year/2021_ERA5_cv5/statistics_analysis/"
met_dir = os.path.join(base_dir, "Meteorological_variables")
ztd_dir = os.path.join(base_dir, "ztd_variable")

os.makedirs(met_dir, exist_ok=True)
os.makedirs(ztd_dir, exist_ok=True)

colors = {
    'before': '#1b9e77',    # Teal for Before DA
    'after': '#d95f02',     # Orange for After DA
    'observed': '#7570b3',  # Purple for Observed (General)
    'observed_noaa': '#4daf4a'  # Green for NOAA ISD Observed
}

# ------------------------------------------------------------------------
# 1) File Paths (Meteorological)
# ------------------------------------------------------------------------
file_observed_general = '/Users/haseeb.rehman/Documents/Misc/Luxembourg_stations_for_validation/2021_Event/stations_6hr_cumulative.xlsx'
file_observed_noaa    = '/Users/haseeb.rehman/Documents/Misc/Luxembourg_stations_for_validation/2021_Event/Stations_other_than_lux/station_weather_data_june_july_2021_6hr.xlsx'
file_before_DA        = '/Users/haseeb.rehman/Desktop/For_Animation/4th_Year/2021_ERA5_cv5/Before_DA/general_station_data_before.xlsx'
file_after_DA         = '/Users/haseeb.rehman/Desktop/For_Animation/4th_Year/2021_ERA5_cv5/After_DA/general_station_data_after.xlsx'

# ------------------------------------------------------------------------
# 2) File Paths (ZTD)
# ------------------------------------------------------------------------
ztd_file_observed   = '/Users/haseeb.rehman/WRF/WRFDA/DAT_DIR/ztd_data_June_july_2021/for_validation/ztd_data.xlsx'
ztd_file_before_DA  = '/Users/haseeb.rehman/Desktop/For_Animation/4th_Year/2021_ERA5_cv5/Before_DA/ztd_station_data_before.xlsx'
ztd_file_after_DA   = '/Users/haseeb.rehman/Desktop/For_Animation/4th_Year/2021_ERA5_cv5/After_DA/ztd_station_data_after.xlsx'

# ------------------------------------------------------------------------
# 3) Station Names
# ------------------------------------------------------------------------
station_names_general = pd.ExcelFile(file_observed_general).sheet_names
station_names_noaa    = pd.ExcelFile(file_observed_noaa).sheet_names
ztd_station_names     = pd.ExcelFile(ztd_file_observed).sheet_names
print(f"Stations (General Met data): {station_names_general}")
print(f"Stations (NOAA ISD Met data): {station_names_noaa}")
print(f"Stations (ZTD data): {ztd_station_names}")

# ------------------------------------------------------------------------
# 4) Read Meteorological Data
# ------------------------------------------------------------------------
def read_station_data(file_path, station, is_observed=False, is_noaa=False):
    try:
        df = pd.read_excel(file_path, sheet_name=station)
    except ValueError as e:
        print(f"Warning: Sheet '{station}' not found in {file_path}. Skipping.")
        return None
    if is_observed:
        df.rename(columns={'Precip(mm)': 'Obs_Precip(mm)', 'Temp(2m)': 'Obs_T2(C)', 'RH(%)': 'Obs_RH(%)'}, inplace=True)
        if is_noaa:
            # Flexible datetime parsing for NOAA data
            df['UTC_Datetime'] = pd.to_datetime(df['UTC_Datetime'], dayfirst=False, errors='coerce')
        else:
            df['UTC_Datetime'] = pd.to_datetime(df['UTC_Datetime'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
    else:
        df.rename(columns={'UTC_Datetime': 'UTC_Datetime',
                          'Precipitation (mm)': 'Sim_Precip(mm)',
                          'Temperature (°C)': 'Sim_T2(C)',
                          'Relative Humidity (%)': 'Sim_RH(%)'}, inplace=True)
        df['UTC_Datetime'] = pd.to_datetime(df['UTC_Datetime'], format='%Y-%m-%d %H', errors='coerce')
    return df

# ------------------------------------------------------------------------
# 5) Read ZTD Data
# ------------------------------------------------------------------------
def read_ztd_data(file_path, station, is_observed=False):
    try:
        df = pd.read_excel(file_path, sheet_name=station)
    except ValueError as e:
        print(f"Warning: Sheet '{station}' not found in {file_path}. Skipping.")
        return None
    if is_observed:
        df.rename(columns={'UTC_Datetime': 'UTC_Datetime', 'ZTD (m)': 'Obs_ZTD(m)'}, inplace=True)
        df['UTC_Datetime'] = pd.to_datetime(df['UTC_Datetime'], errors='coerce')
    else:
        df.rename(columns={'UTC_Datetime': 'UTC_Datetime', 'ZTD (m)': 'Sim_ZTD(m)'}, inplace=True)
        df['UTC_Datetime'] = pd.to_datetime(df['UTC_Datetime'], errors='coerce')
    return df

# ------------------------------------------------------------------------
# 6) Compute Metrics (Handle NaN Values)
# ------------------------------------------------------------------------
def compute_metrics(df, sim_col, obs_col):
    valid_df = df[[sim_col, obs_col]].dropna()
    if valid_df.empty:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    correlation = valid_df[sim_col].corr(valid_df[obs_col])
    rmse = np.sqrt(mean_squared_error(valid_df[sim_col], valid_df[obs_col]))
    mae = np.mean(np.abs(valid_df[sim_col] - valid_df[obs_col]))
    safe_obs = valid_df[obs_col].replace({0: np.nan})
    mape = np.mean(np.abs((valid_df[sim_col] - valid_df[obs_col]) / safe_obs)) * 100 if not safe_obs.isna().all() else np.nan
    bias = np.mean(valid_df[sim_col] - valid_df[obs_col])
    return correlation, rmse, mae, mape, bias

# ------------------------------------------------------------------------
# 7) MET DATA: Loop Over Stations
# ------------------------------------------------------------------------
variables = [
    ('Sim_Precip(mm)', 'Obs_Precip(mm)', 'Precipitation (mm)'),
    ('Sim_T2(C)',      'Obs_T2(C)',      'Temperature (°C)'),
    ('Sim_RH(%)',      'Obs_RH(%)',      'Relative Humidity (%)')
]

overall_metrics_met = []
improved_stations_met = {var_name: {'Correlation': [], 'RMSE': [], 'MAE': [], 'MAPE': [], 'Bias': []} 
                         for _, _, var_name in variables}

all_station_names = station_names_general + station_names_noaa

for station in all_station_names:
    print(f"\n========== MET Station: {station} ==========")
    if station in station_names_general:
        df_obs = read_station_data(file_observed_general, station, is_observed=True, is_noaa=False)
    elif station in station_names_noaa:
        df_obs = read_station_data(file_observed_noaa, station, is_observed=True, is_noaa=True)
    else:
        print(f"Station {station} not found in observed data. Skipping.")
        continue

    if df_obs is None:
        continue

    df_before = read_station_data(file_before_DA, station, is_observed=False)
    df_after  = read_station_data(file_after_DA,  station, is_observed=False)

    if df_before is None or df_after is None:
        print(f"Skipping {station} due to missing WRF data.")
        continue

    df_merged_before = pd.merge(df_before, df_obs, on='UTC_Datetime', how='inner')
    df_merged_after  = pd.merge(df_after,  df_obs, on='UTC_Datetime', how='inner')

    if df_merged_before.empty or df_merged_after.empty:
        print(f"No overlapping data for {station}. Skipping.")
        continue

    metrics_before = {}
    metrics_after  = {}

    for sim_col, obs_col, var_name in variables:
        corr_b, rmse_b, mae_b, mape_b, bias_b = compute_metrics(df_merged_before, sim_col, obs_col)
        corr_a, rmse_a, mae_a, mape_a, bias_a = compute_metrics(df_merged_after, sim_col, obs_col)

        metrics_before[var_name] = (corr_b, rmse_b, mae_b, mape_b, bias_b)
        metrics_after[var_name]  = (corr_a, rmse_a, mae_a, mape_a, bias_a)

        corr_b_str = f"{corr_b:.2f}" if not pd.isna(corr_b) else "N/A"
        rmse_b_str = f"{rmse_b:.2f}" if not pd.isna(rmse_b) else "N/A"
        mae_b_str = f"{mae_b:.2f}" if not pd.isna(mae_b) else "N/A"
        mape_b_str = f"{mape_b:.2f}" if not pd.isna(mape_b) else "N/A"
        bias_b_str = f"{bias_b:.2f}" if not pd.isna(bias_b) else "N/A"

        corr_a_str = f"{corr_a:.2f}" if not pd.isna(corr_a) else "N/A"
        rmse_a_str = f"{rmse_a:.2f}" if not pd.isna(rmse_a) else "N/A"
        mae_a_str = f"{mae_a:.2f}" if not pd.isna(mae_a) else "N/A"
        mape_a_str = f"{mape_a:.2f}" if not pd.isna(mape_a) else "N/A"
        bias_a_str = f"{bias_a:.2f}" if not pd.isna(bias_a) else "N/A"

        print(f"\n--- {var_name} ---")
        print(f"Before DA: Corr={corr_b_str}, RMSE={rmse_b_str}, MAE={mae_b_str}, MAPE={mape_b_str}%, Bias={bias_b_str}")
        print(f"After  DA: Corr={corr_a_str}, RMSE={rmse_a_str}, MAE={mae_a_str}, MAPE={mape_a_str}%, Bias={bias_a_str}")

        if not pd.isna(corr_a) and not pd.isna(corr_b) and corr_a > corr_b:
            improved_stations_met[var_name]['Correlation'].append(station)
        if not pd.isna(rmse_a) and not pd.isna(rmse_b) and rmse_a < rmse_b:
            improved_stations_met[var_name]['RMSE'].append(station)
        if not pd.isna(mae_a) and not pd.isna(mae_b) and mae_a < mae_b:
            improved_stations_met[var_name]['MAE'].append(station)
        if not pd.isna(mape_a) and not pd.isna(mape_b) and mape_a < mape_b:
            improved_stations_met[var_name]['MAPE'].append(station)
        if not pd.isna(bias_a) and not pd.isna(bias_b) and abs(bias_a) < abs(bias_b):
            improved_stations_met[var_name]['Bias'].append(station)

    fig, ax = plt.subplots(nrows=len(variables), ncols=3, figsize=(18, 5 * len(variables)))
    fig.suptitle(f'Station: {station} - Meteorological Variables', fontsize=16)

    for i, (sim_col, obs_col, var_name) in enumerate(variables):
        corr_b, rmse_b, mae_b, mape_b, bias_b = metrics_before[var_name]
        corr_a, rmse_a, mae_a, mape_a, bias_a = metrics_after[var_name]

        df_before_valid = df_merged_before[[sim_col, obs_col, 'UTC_Datetime']].dropna()
        df_after_valid = df_merged_after[[sim_col, obs_col, 'UTC_Datetime']].dropna()

        if not df_before_valid.empty and not df_after_valid.empty:
            ax[i, 0].scatter(df_before_valid[sim_col], df_before_valid[obs_col],
                            color=colors['before'], alpha=0.6, label='Before DA')
            ax[i, 0].scatter(df_after_valid[sim_col], df_after_valid[obs_col],
                            color=colors['after'], alpha=0.6, label='After DA')
            min_val = min(df_before_valid[obs_col].min(), df_after_valid[obs_col].min())
            max_val = max(df_before_valid[obs_col].max(), df_after_valid[obs_col].max())
            ax[i, 0].plot([min_val, max_val], [min_val, max_val], 'k--', label='1:1 line')
            ax[i, 0].set_title(f'{var_name} Correlation')
            ax[i, 0].set_xlabel('Simulated')
            ax[i, 0].set_ylabel('Observed')
            ax[i, 0].legend(loc='upper right')
            ax[i, 0].grid(True, linestyle='--', alpha=0.6)

        bar_data = [rmse_b, rmse_a, mae_b, mae_a, mape_b/100 if not pd.isna(mape_b) else 0, 
                    mape_a/100 if not pd.isna(mape_a) else 0, bias_b, bias_a]
        bar_labels = ['RMSE', 'RMSE', 'MAE', 'MAE', 'MAPE', 'MAPE', 'Bias', 'Bias']
        bars = ax[i, 1].bar(range(len(bar_data)), bar_data, width=0.1,
                            color=[colors['before'], colors['after']] * 4, alpha=0.7)
        ax[i, 1].set_xticks(range(len(bar_data)))
        ax[i, 1].set_xticklabels(bar_labels, rotation=45, ha="right")
        ax[i, 1].set_title(f'{var_name} Metrics')
        ax[i, 1].set_ylabel('Value')
        ax[i, 1].legend(handles=[bars[0], bars[1]], labels=['Before DA', 'After DA'], loc='upper right')
        ax[i, 1].grid(True, linestyle='--', alpha=0.6)
        for bar, val in zip(bars, [rmse_b, rmse_a, mae_b, mae_a, mape_b/100, mape_a/100, bias_b, bias_a]):
            height = bar.get_height()
            if not pd.isna(val):
                ax[i, 1].text(bar.get_x() + bar.get_width()/2., height/2, f'{val:.2f}',
                            ha='center', va='center', color='white', fontsize=8, rotation=90)

        ax[i, 2].plot(df_before_valid['UTC_Datetime'], df_before_valid[sim_col],
                    label='Sim Before DA', color=colors['before'], alpha=0.7)
        ax[i, 2].plot(df_after_valid['UTC_Datetime'], df_after_valid[sim_col],
                    label='Sim After DA', color=colors['after'], alpha=0.7)
        ax[i, 2].plot(df_before_valid['UTC_Datetime'], df_before_valid[obs_col],
                    label='Observed', color=colors['observed_noaa' if station in station_names_noaa else 'observed'],
                    linestyle='dotted', alpha=0.7)
        ax[i, 2].set_title(f'{var_name} Time Series')
        ax[i, 2].set_xlabel('Datetime')
        ax[i, 2].set_ylabel(var_name)
        ax[i, 2].legend(loc='upper right')
        ax[i, 2].grid(True, linestyle='--', alpha=0.6)
        ax[i, 2].xaxis.set_major_locator(plt.MaxNLocator(nbins=7))
        plt.setp(ax[i, 2].xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(met_dir, f'{station}_met_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    for sim_col, obs_col, var_name in variables:
        corr_b, rmse_b, mae_b, mape_b, bias_b = metrics_before[var_name]
        corr_a, rmse_a, mae_a, mape_a, bias_a = metrics_after[var_name]
        overall_metrics_met.append({
            'Station': station, 
            'Variable': var_name,
            'Before_Corr': corr_b, 'Before_RMSE': rmse_b, 'Before_MAE': mae_b, 'Before_MAPE': mape_b, 'Before_Bias': bias_b,
            'After_Corr': corr_a, 'After_RMSE': rmse_a, 'After_MAE': mae_a, 'After_MAPE': mape_a, 'After_Bias': bias_a,
            'Source': 'NOAA ISD' if station in station_names_noaa else 'General'
        })

# --- Print Improved Stations (Meteorological) ---
print("\n========== Stations with Improved Metrics (Meteorological) After DA ==========")
for var_name, metrics in improved_stations_met.items():
    print(f"\n--- {var_name} ---")
    for metric, stations in metrics.items():
        if stations:
            print(f"{metric} improved at: {', '.join(stations)}")
        else:
            print(f"{metric} improved at: None")

# ------------------------------------------------------------------------------
# 8) MET DATA: Overall Aggregated Metrics
# ------------------------------------------------------------------------------
df_overall_met = pd.DataFrame(overall_metrics_met)
agg_metrics_general = df_overall_met[df_overall_met['Source'] == 'General'].groupby('Variable').mean(numeric_only=True)
agg_metrics_noaa = df_overall_met[df_overall_met['Source'] == 'NOAA ISD'].groupby('Variable').mean(numeric_only=True)

metrics = ['RMSE', 'MAE', 'MAPE', 'Bias']
variables = ['Precipitation (mm)', 'Temperature (°C)', 'Relative Humidity (%)']
bar_width = 0.1
x = np.arange(len(metrics))

# Plot for General Stations
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))
fig.suptitle('Overall Meteorological Metrics (Mean Across General Stations)', fontsize=16)

for i, var in enumerate(variables):
    if var in agg_metrics_general.index:
        before_values = [agg_metrics_general.loc[var, 'Before_RMSE'],
                        agg_metrics_general.loc[var, 'Before_MAE'],
                        agg_metrics_general.loc[var, 'Before_MAPE'] / 100,
                        agg_metrics_general.loc[var, 'Before_Bias']]
        after_values = [agg_metrics_general.loc[var, 'After_RMSE'],
                        agg_metrics_general.loc[var, 'After_MAE'],
                        agg_metrics_general.loc[var, 'After_MAPE'] / 100,
                        agg_metrics_general.loc[var, 'After_Bias']]

        bars1 = axes[i].bar(x - bar_width/2, before_values, bar_width, label='Before DA', color=colors['before'], alpha=0.7)
        bars2 = axes[i].bar(x + bar_width/2, after_values, bar_width, label='After DA', color=colors['after'], alpha=0.7)

        axes[i].set_xticks(x)
        axes[i].set_xticklabels(metrics, rotation=45, ha="right")
        axes[i].set_title(f'{var}')
        axes[i].set_ylabel('Metric Value')
        axes[i].legend(loc='upper right')
        axes[i].grid(True, linestyle='--', alpha=0.6)

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if not pd.isna(height):
                    axes[i].text(bar.get_x() + bar.get_width()/2., height/2, f'{height:.2f}',
                                ha='center', va='center', color='white', fontsize=8, rotation=90)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(met_dir, 'overall_met_metrics_general.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot for NOAA ISD Stations
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))
fig.suptitle('Overall Meteorological Metrics (Mean Across NOAA ISD Stations)', fontsize=16)

for i, var in enumerate(variables):
    if var in agg_metrics_noaa.index:
        before_values = [agg_metrics_noaa.loc[var, 'Before_RMSE'],
                        agg_metrics_noaa.loc[var, 'Before_MAE'],
                        agg_metrics_noaa.loc[var, 'Before_MAPE'] / 100,
                        agg_metrics_noaa.loc[var, 'Before_Bias']]
        after_values = [agg_metrics_noaa.loc[var, 'After_RMSE'],
                        agg_metrics_noaa.loc[var, 'After_MAE'],
                        agg_metrics_noaa.loc[var, 'After_MAPE'] / 100,
                        agg_metrics_noaa.loc[var, 'After_Bias']]

        bars1 = axes[i].bar(x - bar_width/2, before_values, bar_width, label='Before DA', color=colors['before'], alpha=0.7)
        bars2 = axes[i].bar(x + bar_width/2, after_values, bar_width, label='After DA', color=colors['after'], alpha=0.7)

        axes[i].set_xticks(x)
        axes[i].set_xticklabels(metrics, rotation=45, ha="right")
        axes[i].set_title(f'{var}')
        axes[i].set_ylabel('Metric Value')
        axes[i].legend(loc='upper right')
        axes[i].grid(True, linestyle='--', alpha=0.6)

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if not pd.isna(height):
                    axes[i].text(bar.get_x() + bar.get_width()/2., height/2, f'{height:.2f}',
                                ha='center', va='center', color='white', fontsize=8, rotation=90)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(met_dir, 'overall_met_metrics_noaa.png'), dpi=300, bbox_inches='tight')
plt.close()

# ------------------------------------------------------------------------
# 9) ZTD: Loop Over Stations
# ------------------------------------------------------------------------
overall_metrics_ztd = []
improved_stations_ztd = {'ZTD (m)': {'Correlation': [], 'RMSE': [], 'MAE': [], 'MAPE': [], 'Bias': []}}

for station in ztd_station_names:
    print(f"\n========== ZTD Station: {station} ==========")
    df_ztd_obs    = read_ztd_data(ztd_file_observed,   station, is_observed=True)
    df_ztd_before = read_ztd_data(ztd_file_before_DA,  station, is_observed=False)
    df_ztd_after  = read_ztd_data(ztd_file_after_DA,   station, is_observed=False)

    if df_ztd_obs is None or df_ztd_before is None or df_ztd_after is None:
        print(f"Skipping {station} due to missing ZTD data.")
        continue

    df_merged_before_ztd = pd.merge(df_ztd_before, df_ztd_obs, on='UTC_Datetime', how='inner')
    df_merged_after_ztd  = pd.merge(df_ztd_after,  df_ztd_obs, on='UTC_Datetime', how='inner')

    if df_merged_before_ztd.empty or df_merged_after_ztd.empty:
        print(f"No overlapping ZTD data for {station}. Skipping.")
        continue

    sim_col, obs_col = 'Sim_ZTD(m)', 'Obs_ZTD(m)'
    corr_b, rmse_b, mae_b, mape_b, bias_b = compute_metrics(df_merged_before_ztd, sim_col, obs_col)
    corr_a, rmse_a, mae_a, mape_a, bias_a = compute_metrics(df_merged_after_ztd, sim_col, obs_col)

    corr_b_str = f"{corr_b:.2f}" if not pd.isna(corr_b) else "N/A"
    rmse_b_str = f"{rmse_b:.3f}" if not pd.isna(rmse_b) else "N/A"
    mae_b_str = f"{mae_b:.3f}" if not pd.isna(mae_b) else "N/A"
    mape_b_str = f"{mape_b:.2f}" if not pd.isna(mape_b) else "N/A"
    bias_b_str = f"{bias_b:.3f}" if not pd.isna(bias_b) else "N/A"

    corr_a_str = f"{corr_a:.2f}" if not pd.isna(corr_a) else "N/A"
    rmse_a_str = f"{rmse_a:.3f}" if not pd.isna(rmse_a) else "N/A"
    mae_a_str = f"{mae_a:.3f}" if not pd.isna(mae_a) else "N/A"
    mape_a_str = f"{mape_a:.2f}" if not pd.isna(mape_a) else "N/A"
    bias_a_str = f"{bias_a:.3f}" if not pd.isna(bias_a) else "N/A"

    print(f"\n--- Station: {station} - ZTD (m) ---")
    print(f"Before DA: Corr={corr_b_str}, RMSE={rmse_b_str}, MAE={mae_b_str}, MAPE={mape_b_str}%, Bias={bias_b_str}")
    print(f"After  DA: Corr={corr_a_str}, RMSE={rmse_a_str}, MAE={mae_a_str}, MAPE={mape_a_str}%, Bias={bias_a_str}")

    if not pd.isna(corr_a) and not pd.isna(corr_b) and corr_a > corr_b:
        improved_stations_ztd['ZTD (m)']['Correlation'].append(station)
    if not pd.isna(rmse_a) and not pd.isna(rmse_b) and rmse_a < rmse_b:
        improved_stations_ztd['ZTD (m)']['RMSE'].append(station)
    if not pd.isna(mae_a) and not pd.isna(mae_b) and mae_a < mae_b:
        improved_stations_ztd['ZTD (m)']['MAE'].append(station)
    if not pd.isna(mape_a) and not pd.isna(mape_b) and mape_a < mape_b:
        improved_stations_ztd['ZTD (m)']['MAPE'].append(station)
    if not pd.isna(bias_a) and not pd.isna(bias_b) and abs(bias_a) < abs(bias_b):
        improved_stations_ztd['ZTD (m)']['Bias'].append(station)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    fig.suptitle(f'Station: {station} - ZTD Analysis', fontsize=16)

    df_before_valid_ztd = df_merged_before_ztd[[sim_col, obs_col, 'UTC_Datetime']].dropna()
    df_after_valid_ztd = df_merged_after_ztd[[sim_col, obs_col, 'UTC_Datetime']].dropna()

    if not df_before_valid_ztd.empty and not df_after_valid_ztd.empty:
        ax[0].scatter(df_before_valid_ztd[sim_col], df_before_valid_ztd[obs_col],
                    color=colors['before'], alpha=0.6, label='Before DA')
        ax[0].scatter(df_after_valid_ztd[sim_col], df_after_valid_ztd[obs_col],
                    color=colors['after'], alpha=0.6, label='After DA')
        min_val_ztd = min(df_before_valid_ztd[obs_col].min(), df_after_valid_ztd[obs_col].min())
        max_val_ztd = max(df_before_valid_ztd[obs_col].max(), df_after_valid_ztd[obs_col].max())
        ax[0].plot([min_val_ztd, max_val_ztd], [min_val_ztd, max_val_ztd], 'k--', label='1:1 line')
        ax[0].set_title('ZTD Correlation')
        ax[0].set_xlabel('Simulated ZTD (m)')
        ax[0].set_ylabel('Observed ZTD (m)')
        ax[0].legend(loc='upper right')
        ax[0].grid(True, linestyle='--', alpha=0.6)

    bar_data = [rmse_b, rmse_a, mae_b, mae_a, mape_b/100 if not pd.isna(mape_b) else 0, 
                mape_a/100 if not pd.isna(mape_a) else 0, bias_b, bias_a]
    bar_labels = ['RMSE', 'RMSE', 'MAE', 'MAE', 'MAPE', 'MAPE', 'Bias', 'Bias']
    bars = ax[1].bar(range(len(bar_data)), bar_data, width=0.1,
                    color=[colors['before'], colors['after']] * 4, alpha=0.7)
    ax[1].set_xticks(range(len(bar_data)))
    ax[1].set_xticklabels(bar_labels, rotation=45, ha="right")
    ax[1].set_title('ZTD Metrics')
    ax[1].set_ylabel('Metric Value')
    ax[1].legend(handles=[bars[0], bars[1]], labels=['Before DA', 'After DA'], loc='upper right')
    ax[1].grid(True, linestyle='--', alpha=0.6)
    for bar, val in zip(bars, [rmse_b, rmse_a, mae_b, mae_a, mape_b/100, mape_a/100, bias_b, bias_a]):
        height = bar.get_height()
        if not pd.isna(val):
            ax[1].text(bar.get_x() + bar.get_width()/2., height/2, f'{val:.2f}',
                    ha='center', va='center', color='white', fontsize=8, rotation=90)

    ax[2].plot(df_before_valid_ztd['UTC_Datetime'], df_before_valid_ztd[sim_col],
            label='Sim Before DA', color=colors['before'], alpha=0.7)
    ax[2].plot(df_after_valid_ztd['UTC_Datetime'], df_after_valid_ztd[sim_col],
            label='Sim After DA', color=colors['after'], alpha=0.7)
    ax[2].plot(df_before_valid_ztd['UTC_Datetime'], df_before_valid_ztd[obs_col],
            label='Observed', color=colors['observed'], linestyle='dotted', alpha=0.7)
    ax[2].set_title('ZTD Time Series')
    ax[2].set_xlabel('Datetime')
    ax[2].set_ylabel('ZTD (m)')
    ax[2].legend(loc='upper right')
    ax[2].grid(True, linestyle='--', alpha=0.6)
    ax[2].xaxis.set_major_locator(plt.MaxNLocator(nbins=7))
    plt.setp(ax[2].xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(ztd_dir, f'{station}_ztd_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    overall_metrics_ztd.append({
        'Station': station, 
        'Before_Corr': corr_b, 'Before_RMSE': rmse_b, 'Before_MAE': mae_b, 'Before_MAPE': mape_b, 'Before_Bias': bias_b,
        'After_Corr': corr_a, 'After_RMSE': rmse_a, 'After_MAE': mae_a, 'After_MAPE': mape_a, 'After_Bias': bias_a
    })

# --- Print Improved Stations (ZTD) ---
print("\n========== Stations with Improved Metrics (ZTD) After DA ==========")
for var_name, metrics in improved_stations_ztd.items():
    print(f"\n--- {var_name} ---")
    for metric, stations in metrics.items():
        if stations:
            print(f"{metric} improved at: {', '.join(stations)}")
        else:
            print(f"{metric} improved at: None")

# ------------------------------------------------------------------------
# 10) ZTD: Overall Aggregated Metrics
# ------------------------------------------------------------------------
df_overall_ztd = pd.DataFrame(overall_metrics_ztd)
mean_ztd = df_overall_ztd.mean(numeric_only=True)

labels = ['RMSE', 'MAE', 'MAPE', 'Bias']
bar_width = 0.1
x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - bar_width/2,
               [mean_ztd['Before_RMSE'], mean_ztd['Before_MAE'], mean_ztd['Before_MAPE']/100, mean_ztd['Before_Bias']],
               bar_width,
               label='Before DA', color=colors['before'], alpha=0.7)
bars2 = ax.bar(x + bar_width/2,
               [mean_ztd['After_RMSE'], mean_ztd['After_MAE'], mean_ztd['After_MAPE']/100, mean_ztd['After_Bias']],
               bar_width,
               label='After DA', color=colors['after'], alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylabel('Metric Value')
ax.set_title('Overall ZTD Metrics (Mean Across Stations)')
ax.legend(loc='upper right')
ax.grid(True, linestyle='--', alpha=0.6)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if not pd.isna(height):
            ax.text(bar.get_x() + bar.get_width()/2., height/2, f'{height:.2f}',
                    ha='center', va='center', color='white', fontsize=8, rotation=90)

plt.tight_layout()
plt.savefig(os.path.join(ztd_dir, 'overall_ztd_metrics.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nDone! Figures saved in:\n  {met_dir} (Meteorological)\n  {ztd_dir} (ZTD)\n")