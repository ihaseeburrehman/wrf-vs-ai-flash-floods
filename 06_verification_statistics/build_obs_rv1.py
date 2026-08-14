#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild the verification observation set for GMD revision 1.

Three corrections relative to the first submission:

1. NOAA ISD precipitation is re-extracted with a period-aware parser
   (isd_precip_fixed_rv1). The original pipeline summed overlapping ISD
   accumulation reports, inflating station totals by 2-70x.

2. Stations co-located (<3 km) with an assimilated SYNOP report in the WRFDA
   ob.ascii files are excluded, so the verification network is independent of
   the assimilated network. GNSS ZTD co-location is NOT disqualifying: ZTD is a
   path-integrated moisture quantity, not a surface observation.

3. Two retained stations carry no usable ISD precipitation and are sourced from
   the national services instead:
       Mirecourt     -> Meteo-France  MIRECOURT-INRAE (88304006), 5.0 km
       Kassel calden -> DWD           Grebenstein     (01750),    4.3 km

Luxembourg AgriMeteo observations are untouched (separate pipeline, unaffected).

Outputs obs_rv1_<event>.csv plus a before/after comparison report.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "noaa_ztd"))
import compare_wrf_gc_fuxi_aifs as C
import isd_precip_fixed_rv1 as F

OBDIR = ("/Users/haseeb.rehman/Documents/Misc/Data_Datasets/Stations_and_Observations/"
         "Luxembourg_stations_for_validation")
ASCII_DIR = ("/Users/haseeb.rehman/WRF/WRFDA/DAT_DIR/data_for_assimilation/"
             "concatenate_2021_event")
MF_DIR = os.path.join(HERE, "external_obs")
DWD_GREB = glob.glob(os.path.join(HERE, "external_obs", "produkt_rr_stunde_*.txt"))

WINDOWS = {
    "2016": ("2016-07-10", "2016-08-06"),
    "2018": ("2018-05-20", "2018-06-20"),
    "2021": ("2021-06-20", "2021-07-20"),
}
ASSIM_RADIUS_KM = 3.0
# Revision decision (GMD RC1, comment 4): verification is restricted to stations
# that are BOTH
#   (a) independent  -- not co-located with any assimilated SYNOP report, and
#   (b) complete     -- supplying >= MIN_COVERAGE of the 6-hourly windows.
# Criterion (b) removes stations whose NOAA ISD records cannot support 6-hourly
# accumulation at all (several report only 12- or 24-hour totals); in the first
# submission these were filled by an erroneous summation of overlapping reports.
KEEP_ALL_STATIONS = False
MIN_COVERAGE = 0.95


# ── assimilated SYNOP locations ─────────────────────────────────────────────
def assimilated_synop_coords():
    seen = {}
    for f in glob.glob(os.path.join(ASCII_DIR, "*.ascii")):
        for L in open(f, errors="ignore"):
            if L.startswith("FM-12 SYNOP"):
                sid = L[33:73].strip()
                try:
                    la, lo = float(L[80:92]), float(L[103:115])
                except ValueError:
                    continue
                if sid and sid not in seen:
                    seen[sid] = (la, lo)
    return np.array(list(seen.values())), list(seen)


def is_assimilated(lat, lon, A):
    km = np.sqrt(((A[:, 0] - lat) * 111.0) ** 2
                 + ((A[:, 1] - lon) * 111.0 * np.cos(np.radians(lat))) ** 2)
    return float(km.min()) < ASSIM_RADIUS_KM, float(km.min())


# ── replacement sources ─────────────────────────────────────────────────────
def mirecourt_6h(t0, t1):
    """Meteo-France MIRECOURT-INRAE hourly RR1 -> 6-hourly, window-END labelled."""
    path = os.path.join(MF_DIR, "MF_88.csv.gz")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path, sep=";", compression="gzip",
                    usecols=["NUM_POSTE", "AAAAMMJJHH", "RR1"], low_memory=False)
    d = d[d["NUM_POSTE"] == 88304006].copy()
    d["t"] = pd.to_datetime(d["AAAAMMJJHH"].astype(str), format="%Y%m%d%H",
                            errors="coerce", utc=True)
    d["RR1"] = pd.to_numeric(d["RR1"], errors="coerce")
    d = d.dropna(subset=["t"]).set_index("t").sort_index()
    s = d["RR1"].resample("6h", label="right", closed="right").sum(min_count=5)
    return s.loc[str(t0):str(pd.Timestamp(t1) + pd.Timedelta(days=1))]


def kassel_6h(t0, t1):
    """DWD Grebenstein (01750) hourly R1 -> 6-hourly, window-END labelled."""
    if not DWD_GREB:
        return None
    d = pd.read_csv(DWD_GREB[0], sep=";", skipinitialspace=True)
    d.columns = [c.strip() for c in d.columns]
    d["t"] = pd.to_datetime(d["MESS_DATUM"].astype(str), format="%Y%m%d%H",
                            errors="coerce", utc=True)
    d["R1"] = pd.to_numeric(d["R1"], errors="coerce").mask(lambda s: s < -100)
    d = d.dropna(subset=["t"]).set_index("t").sort_index()
    s = d["R1"].resample("6h", label="right", closed="right").sum(min_count=5)
    return s.loc[str(t0):str(pd.Timestamp(t1) + pd.Timedelta(days=1))]


REPLACEMENTS = {"Mirecourt": mirecourt_6h, "Kassel calden": kassel_6h}


def main():
    A, _ = assimilated_synop_coords()
    print(f"assimilated SYNOP sites (all cycles): {len(A)}\n")
    report = []

    for ev, (t0, t1) in WINDOWS.items():
        cfg = C.EVENTS[ev]
        old = C.load_obs(cfg)                      # first-submission observations
        old_p = (old.dropna(subset=["Obs_Precip"])
                    .groupby("Station")["Obs_Precip"].sum())

        # ISD raw files for this event
        isd = {}
        pat = os.path.join(OBDIR, f"{ev}_Event", "Stations_other_than_lux", "*.csv")
        for f in sorted(glob.glob(pat)):
            if "complete" in f or "selected" in f:
                continue
            isd[F.station_name(f)] = f

        rows = []
        for stn in sorted(old["Station"].unique()):
            coords = C.STATION_COORDS.get(stn)
            if coords is None:
                continue
            assim, dist = is_assimilated(coords[0], coords[1], A)
            if assim and not KEEP_ALL_STATIONS:
                report.append((ev, stn, "DROPPED (assimilated)",
                               old_p.get(stn, np.nan), np.nan, dist, 0, 0))
                continue

            if stn in REPLACEMENTS:                       # national-service source
                s = REPLACEMENTS[stn](t0, t1)
                src = "MF/DWD"
            elif stn in isd:                              # corrected ISD
                s = F.six_hourly(F.isd_long(isd[stn]), t0, t1)
                src = "ISD-fixed"
            else:                                         # AgriMeteo, unchanged
                sub = old[(old.Station == stn)].dropna(subset=["Obs_Precip"])
                s = pd.Series(sub["Obs_Precip"].values,
                              index=pd.to_datetime(sub["UTC_Datetime"], utc=True))
                src = "AgriMeteo"

            if s is None or len(s) == 0:
                report.append((ev, stn, "no data", old_p.get(stn, np.nan),
                               np.nan, dist, 0, 0))
                continue

            n_valid = int(pd.Series(s.values).notna().sum())
            coverage = n_valid / len(s) if len(s) else 0.0
            if coverage < MIN_COVERAGE:
                report.append((ev, stn, f"DROPPED (coverage {100*coverage:.0f}%)",
                               old_p.get(stn, np.nan), float(np.nansum(s.values)),
                               dist, n_valid, len(s)))
                continue

            for t, v in s.items():
                rows.append({"Station": stn,
                             "UTC_Datetime": pd.Timestamp(t).tz_localize(None)
                                             if pd.Timestamp(t).tzinfo else t,
                             "Obs_Precip": v,
                             "Assimilated": bool(assim)})
            report.append((ev, stn, src, old_p.get(stn, np.nan),
                           float(np.nansum(s.values)), dist, n_valid, len(s)))

        out = pd.DataFrame(rows)
        # merge temperature back from the original observations (unchanged)
        tmp = old[["Station", "UTC_Datetime", "Obs_T2m"]].copy()
        tmp["UTC_Datetime"] = pd.to_datetime(tmp["UTC_Datetime"])
        out = out.merge(tmp, on=["Station", "UTC_Datetime"], how="left")
        path = os.path.join(HERE, f"obs_rv1_{ev}.csv")
        out.to_csv(path, index=False)
        print(f"{ev}: wrote {len(out)} rows, {out.Station.nunique()} stations -> {os.path.basename(path)}")

    rep = pd.DataFrame(report, columns=["Event", "Station", "Source",
                                        "Old_total_mm", "New_total_mm", "km_to_assim",
                                        "valid_windows", "total_windows"])
    rep.to_csv(os.path.join(HERE, "obs_rv1_report.csv"), index=False)
    print("\n" + "=" * 78)
    print("BEFORE / AFTER station precipitation totals")
    print("=" * 78)
    for ev in WINDOWS:
        r = rep[rep.Event == ev]
        print(f"\n--- {ev} ---")
        print(f"{'Station':18}{'Source':22}{'old_mm':>9}{'new_mm':>9}{'ratio':>8}{'coverage':>12}")
        for _, x in r.iterrows():
            ratio = (x.Old_total_mm / x.New_total_mm
                     if x.New_total_mm and x.New_total_mm > 0 else np.nan)
            o = f"{x.Old_total_mm:9.1f}" if pd.notna(x.Old_total_mm) else f"{'-':>9}"
            n = f"{x.New_total_mm:9.1f}" if pd.notna(x.New_total_mm) else f"{'-':>9}"
            rr = f"{ratio:8.1f}" if pd.notna(ratio) else f"{'-':>8}"
            cov = (f"{x.valid_windows}/{x.total_windows}"
                   f" {100*x.valid_windows/x.total_windows:.0f}%"
                   if x.total_windows else "-")
            print(f"{x.Station:18}{x.Source:22}{o}{n}{rr}{cov:>12}")


if __name__ == "__main__":
    main()
