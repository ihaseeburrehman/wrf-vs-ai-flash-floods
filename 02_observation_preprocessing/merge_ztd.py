#!/usr/bin/env python3
"""Append GNSS ZTD reports to an OBSPROC ob.ascii file and fix its header.

Mirrors the original process_ztd_ref.sh: the OBSPROC header already carries the
correct counts for every conventional platform, so only two fields change,
TOTAL and GPSZD. Everything else is left byte-for-byte alone, and the
fixed-width column layout that WRFDA expects is preserved.

Usage:  merge_ztd.py  <obsproc_ascii>  <ztd_ascii>  <output>
"""

import os
import re
import sys


def set_field(line, name, value):
    """Replace 'NAME =   n' keeping the original field width."""
    pat = re.compile(rf"({re.escape(name)}\s*=)(\s*)(\d+)")
    m = pat.search(line)
    if not m:
        return line, False
    width = len(m.group(2)) + len(m.group(3))     # spaces + digits
    new = str(value).rjust(width)
    if len(new) > width:                          # number grew: eat one space
        new = str(value).rjust(width)
    return line[:m.start(2)] + new + line[m.end(3):], True


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    src, ztd, out = sys.argv[1], sys.argv[2], sys.argv[3]
    for p in (src, ztd):
        if not os.path.exists(p):
            print(f"missing: {p}")
            sys.exit(1)

    with open(src, errors="ignore") as fh:
        lines = fh.readlines()
    with open(ztd, errors="ignore") as fh:
        ztd_lines = [L for L in fh.readlines() if L.strip()]

    # each ZTD observation is a header record plus one data record
    n_ztd = sum(1 for L in ztd_lines if L.startswith("FM-114"))
    if n_ztd == 0:
        print("no FM-114 records in ZTD file")
        sys.exit(1)

    # locate the header block: everything before the first report
    first = next((i for i, L in enumerate(lines) if L.startswith("FM-")), len(lines))
    header, body = lines[:first], lines[first:]

    # current TOTAL
    total = None
    for L in header:
        m = re.search(r"TOTAL\s*=\s*(\d+)", L)
        if m:
            total = int(m.group(1))
            break
    if total is None:
        print("could not read TOTAL from header")
        sys.exit(1)

    new_total = total + n_ztd
    done_total = done_ztd = False
    for i, L in enumerate(header):
        if not done_total:
            L2, hit = set_field(L, "TOTAL", new_total)
            if hit:
                header[i] = L2
                L = L2
                done_total = True
        if not done_ztd:
            L2, hit = set_field(L, "GPSZD", n_ztd)
            if hit:
                header[i] = L2
                done_ztd = True

    if not (done_total and done_ztd):
        print(f"header update incomplete (TOTAL={done_total}, GPSZD={done_ztd})")
        sys.exit(1)

    with open(out, "w") as fh:
        fh.writelines(header)
        fh.writelines(body)
        fh.writelines(ztd_lines)

    print(f"TOTAL {total} -> {new_total}, GPSZD 0 -> {n_ztd}")


if __name__ == "__main__":
    main()
