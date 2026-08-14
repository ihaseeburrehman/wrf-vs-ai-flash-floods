#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Period-aware NOAA ISD precipitation extraction (GMD revision 1).

Background
----------
The original extraction (extracting_data_from_NOAA_ISD_files.py) parsed the ISD
AA1 field into (period, amount) but then discarded the period and did

    df['Precip(mm)'].resample('6H').sum()

ISD AAx fields report *overlapping* accumulations (1, 3, 6, 12 or running 24 h),
often several per hour, so this double-counts badly: stations reporting a running
24-hour total every hour were inflated by up to 70x.

Correct approach
----------------
For each 6-hour window [T-6h, T):
  1. prefer a single period==6 report at T                (exact 6-h total)
  2. else sum period==1 reports inside the window         (needs >=5 of 6 hours)
  3. else sum period==3 reports inside the window         (needs both halves)
  4. else NaN  -- missing, never fabricated as zero

All AA1..AA4 slots are scanned, since the period varies by slot. ISD quality
codes 2,3,6,7 (erroneous/suspect) are rejected; depth 9999 is missing.

Output is labelled by the END of the accumulation window (valid time), i.e.
value at T is the total over (T-6h, T].
"""

import os
import re
import numpy as np
import pandas as pd

AA_SLOTS = ["AA1", "AA2", "AA3", "AA4"]
BAD_QC = {"2", "3", "6", "7"}


def _parse_aa(val):
    """Return (period_hours, depth_mm) or None."""
    if not isinstance(val, str):
        return None
    p = val.split(",")
    if len(p) < 2:
        return None
    try:
        period = int(p[0])
        raw = p[1].strip()
        if raw in ("9999", "99999"):
            return None
        depth = float(raw) / 10.0
    except (ValueError, IndexError):
        return None
    if period <= 0 or depth < 0 or depth > 2000:
        return None
    if len(p) >= 4 and p[3].strip() in BAD_QC:
        return None
    return period, depth


def isd_long(csv_path):
    """Read one ISD station CSV -> long frame [t, period, depth]."""
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    if "DATE" not in df.columns:
        return pd.DataFrame(columns=["t", "period", "depth"])
    t = pd.to_datetime(df["DATE"], errors="coerce", utc=True)
    recs = []
    for slot in AA_SLOTS:
        if slot not in df.columns:
            continue
        parsed = df[slot].map(_parse_aa)
        ok = parsed.notna()
        if not ok.any():
            continue
        sub = pd.DataFrame({
            "t": t[ok].values,
            "period": [x[0] for x in parsed[ok]],
            "depth": [x[1] for x in parsed[ok]],
        })
        recs.append(sub)
    if not recs:
        return pd.DataFrame(columns=["t", "period", "depth"])
    out = pd.concat(recs, ignore_index=True).dropna(subset=["t"])
    # concat of datetime64 values can drop tz; restore UTC awareness
    out["t"] = pd.to_datetime(out["t"], utc=True)
    return out.drop_duplicates(subset=["t", "period"]).sort_values("t")


def six_hourly(long_df, t0, t1):
    """Correct 6-hourly totals on windows ENDING at 00/06/12/18 UTC.

    Returns Series indexed by window end time; NaN where unobtainable.
    """
    idx = pd.date_range(pd.Timestamp(t0, tz="UTC"), pd.Timestamp(t1, tz="UTC"),
                        freq="6h")
    if long_df.empty:
        return pd.Series(np.nan, index=idx, name="Precip_mm")

    d = long_df.set_index("t")
    out = {}
    for end in idx:
        start = end - pd.Timedelta(hours=6)

        # 1) exact 6-hour report at the window end (allow +/-30 min)
        near = d[(d.index >= end - pd.Timedelta(minutes=30)) &
                 (d.index <= end + pd.Timedelta(minutes=30)) &
                 (d["period"] == 6)]
        if len(near):
            out[end] = float(near["depth"].max())
            continue

        win = d[(d.index > start) & (d.index <= end)]

        # 2) hourly reports
        h1 = win[win["period"] == 1]
        if len(h1) >= 5:
            out[end] = float(h1.groupby(h1.index.floor("h"))["depth"].max().sum())
            continue

        # 3) three-hourly reports
        h3 = win[win["period"] == 3]
        if len(h3) >= 2:
            g = h3.groupby(h3.index.floor("3h"))["depth"].max()
            if len(g) >= 2:
                out[end] = float(g.sum())
                continue

        out[end] = np.nan

    return pd.Series(out, name="Precip_mm").reindex(idx)


def station_name(csv_path):
    try:
        df = pd.read_csv(csv_path, dtype=str, nrows=50, low_memory=False)
        if "NAME" in df.columns and df["NAME"].notna().any():
            return str(df["NAME"].dropna().iloc[0]).split(",")[0].strip().lower().capitalize()
    except Exception:
        pass
    base = os.path.basename(csv_path)
    m = re.match(r"\d+_(.+)\.csv", base)
    return (m.group(1) if m else base).lower().capitalize()
