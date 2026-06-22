"""FuXi Rapid Update Cycle — mirrors GraphCast rapid cycle scripts.
For each consecutive 6h ERA5 pair [t-6, t0], produces a fresh 6h forecast (t+6).
Uses the FuXi short ONNX model from the ECMWF ai-models asset store.
"""
import sys, os, time, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import xarray as xr
import pandas as pd
import onnxruntime as ort
from pathlib import Path

G="\033[92m"; R="\033[91m"; Y="\033[93m"; C="\033[96m"; B="\033[1m"; RS="\033[0m"

EVENT    = "EVENT_PLACEHOLDER"
BASE     = Path("/scratch/lux0804/fuxi_luxembourg")
ASSETS   = Path("/usr/local/apps/ai-models/0.49/assets/fuxi")
ERA5_DIR = Path(f"/scratch/lux0804/graphcast_luxembourg/data/{EVENT}/era5_mars")
OUT_DIR  = BASE / f"output/{EVENT}/forecasts/rapid_full"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FUXI_LEVELS     = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
FUXI_PL_PARAMS  = ["z", "t", "u", "v", "r"]
FUXI_SFC_PARAMS = ["2t", "10u", "10v", "msl", "tp"]
N_CHANNELS      = len(FUXI_PL_PARAMS) * len(FUXI_LEVELS) + len(FUXI_SFC_PARAMS)  # 70

print(f"\n{B}{C}  FuXi Rapid Update Cycle — {EVENT}{RS}", flush=True)

# ── Load ONNX model ──────────────────────────────────────────────────────────
print(f"  {Y}[1/3]{RS} Loading FuXi short.onnx ...", flush=True)
ort.set_default_logger_severity(3)
opts = ort.SessionOptions()
opts.enable_cpu_mem_arena = False
opts.enable_mem_pattern    = False
opts.enable_mem_reuse      = False
opts.intra_op_num_threads  = 8
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
model = ort.InferenceSession(str(ASSETS / "short.onnx"), sess_options=opts, providers=providers)
active_provider = model.get_providers()[0]
print(f"        Provider: {active_provider}", flush=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
def q_to_rh(q, t, level_hpa):
    """Specific humidity [kg/kg] + temperature [K] + pressure level [hPa] -> RH [%]."""
    p   = float(level_hpa) * 100.0          # Pa
    T_C = t - 273.15                         # Celsius
    e_s = 611.2 * np.exp(17.67 * T_C / (T_C + 243.5))   # saturation vapour pressure [Pa]
    e   = q * p / (0.622 + 0.378 * q)       # actual vapour pressure [Pa]
    return np.clip(100.0 * e / e_s, 0.0, 100.0).astype(np.float32)

def time_embedding(init_pd, freq_h=6):
    """Compute FuXi time embedding for one forecast step. Returns [1, 12]."""
    hours = [pd.Timedelta(hours=h * freq_h) for h in [-1, 0, 1]]
    pts   = [pd.Period(init_pd + h, "H") for h in hours]
    vals  = np.array([(p.day_of_year / 366.0, p.hour / 24.0) for p in pts], dtype=np.float32)
    return np.concatenate([np.sin(vals), np.cos(vals)], axis=-1).reshape(1, -1)  # [1, 12]

def fix_dims(ds):
    rn = {}
    for old, new in [("valid_time","time"), ("latitude","lat"), ("longitude","lon"),
                     ("pressure_level","level"), ("isobaricInhPa","level")]:
        if old in ds.dims: rn[old] = new
    return ds.rename(rn).drop_vars(["valid_time", "step"], errors="ignore")

# ── Load ERA5 (same files as GraphCast) ──────────────────────────────────────
print(f"  {Y}[2/3]{RS} Loading ERA5 data into memory ...", flush=True)

sl_instant = xr.open_dataset(ERA5_DIR / "era5_sl_instant.nc", engine="netcdf4").compute()
sl_accum   = xr.open_dataset(ERA5_DIR / "era5_sl_accum.nc",   engine="netcdf4").compute()
pl_all     = xr.open_dataset(str(ERA5_DIR / "era5_pl.grib"), engine="cfgrib",
                              backend_kwargs={"indexpath": ""}).compute()

sl_instant = fix_dims(sl_instant).isel(lat=slice(None, None, -1))
sl_accum   = fix_dims(sl_accum).isel(lat=slice(None, None, -1))
pl_all     = fix_dims(pl_all).isel(lat=slice(None, None, -1)).isel(level=slice(None, None, -1))
pl_all     = pl_all.assign_coords(level=pl_all.level.astype(np.int32))

# Select FuXi pressure levels
pl_fuxi = pl_all.sel(level=FUXI_LEVELS)

# Pre-compute relative humidity (q → r) for all times
print(f"        Computing relative humidity (q→r) for all timesteps ...", flush=True)
r_arr = np.empty_like(pl_fuxi["q"].values, dtype=np.float32)
for li, lev in enumerate(FUXI_LEVELS):
    r_arr[:, li, :, :] = q_to_rh(
        pl_fuxi["q"].values[:, li, :, :],
        pl_fuxi["t"].values[:, li, :, :],
        lev
    )
r_da   = xr.DataArray(r_arr, coords=pl_fuxi["q"].coords, dims=pl_fuxi["q"].dims)
pl_fuxi = pl_fuxi.assign(r=r_da)

all_times = sorted(
    set(sl_instant.time.values) &
    set(sl_accum.time.values)   &
    set(pl_fuxi.time.values)
)
print(f"        {len(all_times)} common timesteps found", flush=True)

lat_vals   = sl_instant.lat.values   # [721]
lon_vals   = sl_instant.lon.values   # [1440]
level_vals = np.array(FUXI_LEVELS, dtype=np.int32)

def build_channels(t_step):
    """Build [70, 721, 1440] float32 channel array for one ERA5 timestep."""
    pl_t  = pl_fuxi.sel(time=t_step)
    sli_t = sl_instant.sel(time=t_step)
    sla_t = sl_accum.sel(time=t_step)

    chans = []
    for param in FUXI_PL_PARAMS:
        for lev in FUXI_LEVELS:
            chans.append(pl_t[param].sel(level=lev).values.astype(np.float32))

    chans.append(sli_t["t2m"].values.astype(np.float32))   # 2t
    chans.append(sli_t["u10"].values.astype(np.float32))   # 10u
    chans.append(sli_t["v10"].values.astype(np.float32))   # 10v
    chans.append(sli_t["msl"].values.astype(np.float32))   # msl
    # ERA5 tp is in metres; FuXi ONNX expects kg/m² (mm) → ×1000
    chans.append((sla_t["tp"].values.astype(np.float32) * 1000.0))  # tp

    return np.stack(chans, axis=0)  # [70, 721, 1440]

# ── Rapid cycle ───────────────────────────────────────────────────────────────
print(f"  {Y}[3/3]{RS} Starting rapid update cycle ...\n", flush=True)
print(f"  {B}  {'#':<6}{'Init':<22}{'Forecast':<22}{'Total':>8}  Result{RS}")
print("  " + "─" * 75, flush=True)

cycle_num = 0

for i in range(len(all_times) - 1):
    t_minus6 = all_times[i]
    t_zero   = all_times[i + 1]

    dt = (t_zero - t_minus6) / np.timedelta64(1, "h")
    if dt != 6.0:
        continue

    t_plus6     = t_zero + np.timedelta64(6, "h")
    t_zero_str  = str(np.datetime_as_string(t_zero,  unit="h"))
    t_plus6_str = str(np.datetime_as_string(t_plus6, unit="h"))

    if t_plus6 not in sl_accum.time.values:
        continue

    out_file = OUT_DIR / f"forecast_{t_plus6_str.replace(':','').replace('-','')}.nc"
    cycle_num += 1

    if out_file.exists():
        print(f"  {Y}{cycle_num:<6}{RS}{t_zero_str:<22}{t_plus6_str:<22}{'':>8}  {C}skipped{RS}", flush=True)
        continue

    t_start = time.time()
    try:
        # Build [1, 2, 70, 721, 1440] input tensor
        ch_m6   = build_channels(t_minus6)
        ch_zero = build_channels(t_zero)
        fuxi_in = np.stack([ch_m6, ch_zero], axis=0)[np.newaxis]  # [1,2,70,721,1440]

        # Time embedding for t_zero as init time
        temb = time_embedding(pd.Timestamp(t_zero))  # [1, 12]

        # ONNX inference
        (raw_out,) = model.run(None, {"input": fuxi_in, "temb": temb})
        # raw_out: [1, 2, 70, 721, 1440] — take last time slice
        pred = raw_out[0, -1]  # [70, 721, 1440]

        # Parse channels back into variables
        pl_out  = {}
        idx = 0
        for param in FUXI_PL_PARAMS:
            pl_out[param] = np.stack(
                [pred[idx + j] for j in range(len(FUXI_LEVELS))], axis=0
            )  # [13, 721, 1440]
            idx += len(FUXI_LEVELS)
        sfc_out = {p: pred[idx + j] for j, p in enumerate(FUXI_SFC_PARAMS)}

        # tp: mm → metres, clip negatives
        tp_m = np.maximum(0.0, sfc_out["tp"]) / 1000.0  # [721, 1440]

        # Save NetCDF — same structure as GraphCast outputs
        ds_out = xr.Dataset(
            {
                "total_precipitation_6hr": xr.DataArray(
                    tp_m[np.newaxis, np.newaxis], dims=["batch","time","lat","lon"]),
                "2m_temperature": xr.DataArray(
                    sfc_out["2t"][np.newaxis, np.newaxis], dims=["batch","time","lat","lon"]),
                "10m_u_component_of_wind": xr.DataArray(
                    sfc_out["10u"][np.newaxis, np.newaxis], dims=["batch","time","lat","lon"]),
                "10m_v_component_of_wind": xr.DataArray(
                    sfc_out["10v"][np.newaxis, np.newaxis], dims=["batch","time","lat","lon"]),
                "mean_sea_level_pressure": xr.DataArray(
                    sfc_out["msl"][np.newaxis, np.newaxis], dims=["batch","time","lat","lon"]),
                "relative_humidity": xr.DataArray(
                    pl_out["r"][np.newaxis, np.newaxis], dims=["batch","time","level","lat","lon"]),
                "geopotential": xr.DataArray(
                    pl_out["z"][np.newaxis, np.newaxis], dims=["batch","time","level","lat","lon"]),
                "temperature": xr.DataArray(
                    pl_out["t"][np.newaxis, np.newaxis], dims=["batch","time","level","lat","lon"]),
                "u_component_of_wind": xr.DataArray(
                    pl_out["u"][np.newaxis, np.newaxis], dims=["batch","time","level","lat","lon"]),
                "v_component_of_wind": xr.DataArray(
                    pl_out["v"][np.newaxis, np.newaxis], dims=["batch","time","level","lat","lon"]),
            },
            coords={"lat": lat_vals, "lon": lon_vals, "level": level_vals},
        )
        ds_out.to_netcdf(out_file)

        elapsed = time.time() - t_start
        tp_mm   = float(np.nanmax(tp_m)) * 1000.0
        print(f"  {G}{cycle_num:<6}{RS}{t_zero_str:<22}{t_plus6_str:<22}{elapsed:>7.1f}s  {G}max={tp_mm:.1f}mm{RS}", flush=True)

    except Exception as e:
        elapsed = time.time() - t_start
        print(f"  {R}{cycle_num:<6}{RS}{t_zero_str:<22}{t_plus6_str:<22}{elapsed:>7.1f}s  {R}ERROR: {str(e)[:60]}{RS}", flush=True)

print(f"\n{B}{G}  Done: {cycle_num} cycles. Output → {OUT_DIR}{RS}\n", flush=True)
