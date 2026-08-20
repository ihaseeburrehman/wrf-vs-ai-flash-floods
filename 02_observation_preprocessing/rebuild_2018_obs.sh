#!/usr/bin/env bash
# Rebuild the 2018 assimilation observations from the original NCEP PREPBUFR.
#
# Why: the little_r conversion used for the original 2018 run lost every
# surface report type (SYNOP, METAR, BUOY, SHIP, mesonet), leaving an
# observation set that was ~95 % satellite winds with no surface anchor.
# The PREPBUFR files themselves are complete, so the observations are
# regenerated here.
#
# Per cycle:
#   1. PREPBUFR -> little_r        (prepbufr2littleR.exe, built locally)
#   2. little_r -> obs_gts .3DVAR  (obsproc.exe: domain + time filtering)
#   3. append GNSS ZTD, fix TOTAL and GPSZD in the header (merge_ztd.py)
#   4. delete the ~700 MB little_r intermediate immediately
#
# Output: concatenated_YYYY-MM-DD_HH:00:00.ascii  in $OUT_DIR

set -uo pipefail
export MACOSX_DEPLOYMENT_TARGET=14.0

WORK=/Users/haseeb.rehman/pb2lr_local
PB_DIR="/Users/haseeb.rehman/WRF/WRFDA/DAT_DIR/conventional_obs/ob_ascii_may_june_2018_prebufr/New Folder With Items"
ZTD_DIR=/Users/haseeb.rehman/WRF/WRFDA/DAT_DIR/ztd_data_may_june_2018/ztd_processed
OUT_DIR=/Users/haseeb.rehman/WRF/WRFDA/DAT_DIR/data_for_assimilation/fix_concatenate_2018_event
OBSPROC=/Users/haseeb.rehman/WRF/WRFDA/var/obsproc
STAGE="$WORK/stage"
LOG="$WORK/rebuild_2018.log"

mkdir -p "$OUT_DIR" "$STAGE"
: > "$LOG"

TEMPLATE="$WORK/namelist.obsproc.backup"

write_namelist() {   # $1 little_r path, $2 YYYY-MM-DD, $3 HH
  local lr="$1" d="$2" h="$3"
  /opt/homebrew/Caskroom/miniconda/base/bin/python3 - "$lr" "$d" "$h" "$TEMPLATE" <<'PY' > "$OBSPROC/namelist.obsproc"
import sys, datetime
lr, d, h, tmpl_path = sys.argv[1:5]
t  = datetime.datetime.strptime(f"{d}_{h}", "%Y-%m-%d_%H")
lo = (t - datetime.timedelta(hours=1)).strftime("%Y-%m-%d_%H:%M")
hi = (t + datetime.timedelta(hours=1)).strftime("%Y-%m-%d_%H:%M")
s = open(tmpl_path).read()
import re
s = re.sub(r"obs_gts_filename\s*=\s*'[^']*'", f"obs_gts_filename = '{lr}'", s)
s = re.sub(r"time_window_min\s*=\s*'[^']*'", f"time_window_min  = '{lo}'", s)
s = re.sub(r"time_analysis\s*=\s*'[^']*'",   f"time_analysis    = '{t:%Y-%m-%d_%H:%M}'", s)
s = re.sub(r"time_window_max\s*=\s*'[^']*'", f"time_window_max  = '{hi}'", s)
print(s)
PY
}

ok=0; fail=0; total=$(ls "$PB_DIR"/prepbufr.gdas.*.nr 2>/dev/null | wc -l | tr -d ' ')
echo "rebuilding $total cycles -> $OUT_DIR" | tee -a "$LOG"

for pb in "$PB_DIR"/prepbufr.gdas.*.nr; do
  [ -e "$pb" ] || continue
  base=$(basename "$pb"); dt=${base#prepbufr.gdas.}; dt=${dt%.nr}
  Y=${dt:0:4}; M=${dt:4:2}; D=${dt:6:2}; H=${dt:8:2}
  stamp="${Y}-${M}-${D}_${H}:00:00"
  out="$OUT_DIR/concatenated_${stamp}.ascii"
  ztd="$ZTD_DIR/ob_${Y}-${M}-${D}_${H}_00_00.ascii"

  if [ -s "$out" ]; then echo "skip  $stamp" | tee -a "$LOG"; ok=$((ok+1)); continue; fi

  # 1. PREPBUFR -> little_r
  cd "$WORK" || exit 1
  rm -f bufrfile pbufr.table
  rm -f prepbufr2littleR.txt prepbufr2littleR_*.txt 2>/dev/null
  cp "$pb" bufrfile
  ./prepbufr2littleR.exe >/dev/null 2>&1
  if [ ! -s prepbufr2littleR.txt ]; then
    echo "FAIL  $stamp  conversion empty" | tee -a "$LOG"; fail=$((fail+1)); continue
  fi
  nsyn=$(LC_ALL=C grep -oE "FM-12" prepbufr2littleR.txt 2>/dev/null | wc -l | tr -d ' ')
  mv prepbufr2littleR.txt "$STAGE/OBS_$dt"
  rm -f prepbufr2littleR_*.txt pbufr.table bufrfile

  # 2. little_r -> obs_gts
  write_namelist "$STAGE/OBS_$dt" "${Y}-${M}-${D}" "$H"
  cd "$OBSPROC" || exit 1
  rm -f obs_gts_*.3DVAR
  ./obsproc.exe > "$STAGE/obsproc_$dt.log" 2>&1
  gts=$(ls -t "$OBSPROC"/obs_gts_*.3DVAR 2>/dev/null | head -1)
  if [ -z "${gts:-}" ] || [ ! -s "$gts" ]; then
    echo "FAIL  $stamp  obsproc empty (see $STAGE/obsproc_$dt.log)" | tee -a "$LOG"
    rm -f "$STAGE/OBS_$dt"; fail=$((fail+1)); continue
  fi

  # 3. append ZTD + fix header
  if [ -s "$ztd" ]; then
    "$WORK/merge_ztd.py" "$gts" "$ztd" "$out" >> "$LOG" 2>&1 || { echo "FAIL  $stamp  merge" | tee -a "$LOG"; fail=$((fail+1)); rm -f "$STAGE/OBS_$dt"; continue; }
  else
    cp "$gts" "$out"
    echo "  note: no ZTD file for $stamp" | tee -a "$LOG"
  fi

  # 4. clean up
  rm -f "$STAGE/OBS_$dt" "$OBSPROC"/obs_gts_*.3DVAR
  nS=$(LC_ALL=C grep -c "^FM-12 " "$out" 2>/dev/null || echo 0)
  nZ=$(LC_ALL=C grep -c "^FM-114" "$out" 2>/dev/null || echo 0)
  echo "ok    $stamp  little_r FM-12=$nsyn  ascii SYNOP=$nS  ZTD=$nZ" | tee -a "$LOG"
  ok=$((ok+1))
done

echo "" | tee -a "$LOG"
echo "done: $ok ok, $fail failed  ->  $OUT_DIR" | tee -a "$LOG"
cp "$TEMPLATE" "$OBSPROC/namelist.obsproc" 2>/dev/null   # restore original namelist
