import os
import glob
import numpy as np
from datetime import datetime, timedelta
from osgeo import gdal, osr

# ---- Radar TIF folders ----
tif_dirs = [
    "/Users/haseeb.rehman/Documents/Misc/Belgium_Radar_data_2021/2021/07/14/accum1h/tif/",
    "/Users/haseeb.rehman/Documents/Misc/Belgium_Radar_data_2021/2021/07/15/accum1h/tif/"
]

# ---- Output folder ----
out_dir = "/Users/haseeb.rehman/Documents/Misc/Belgium_Radar_data_2021/6h_tif"
os.makedirs(out_dir, exist_ok=True)

# ---- Get reference metadata from first tif ----
sample = glob.glob(os.path.join(tif_dirs[0], "*.tif"))[0]
ref = gdal.Open(sample)
gt = ref.GetGeoTransform()
# read existing projection from sample; do NOT force EPSG:2169
proj_wkt = ref.GetProjection() or ""   # empty string => no CRS will be written
ref = None

# ---- Aggregate 6 hours ----
def aggregate_6h(target):
    start = target - timedelta(hours=5)
    stack = []

    for d in tif_dirs:
        files = glob.glob(os.path.join(d, "*.tif"))
        for f in files:
            name = os.path.basename(f)
            dt = datetime.strptime(name[:14], "%Y%m%d%H%M%S")
            if start <= dt <= target:
                ds = gdal.Open(f)
                band = ds.GetRasterBand(1)

                # read declared scale/offset (may be None)
                declared_scale = band.GetScale()
                declared_offset = band.GetOffset() or 0.0

                # read raw array (preserve float for inference)
                arr = band.ReadAsArray().astype(np.float32)
                arr[arr < 0] = 0.0

                # infer scale when not provided or when it's 1.0 but values look scaled
                if declared_scale is None or float(declared_scale) == 1.0:
                    raw_max = np.nanmax(arr)
                    # heuristic: if values are >1000 treat them as mm*1000 and use 0.001
                    if raw_max > 1000:
                        inferred_scale = 0.001
                        # debug message (optional)
                        print(f"Inferred scale {inferred_scale} for {os.path.basename(f)} (raw_max={raw_max:.0f})")
                    else:
                        inferred_scale = 1.0
                    scale = inferred_scale
                else:
                    scale = float(declared_scale)

                # apply scale/offset
                arr = arr * scale + float(declared_offset)
                stack.append(arr)
                ds = None

    if not stack:
        return None
    return np.sum(stack, axis=0)

# ---- Loop 6-hour steps ----
t = datetime(2021, 7, 14, 6)
end = datetime(2021, 7, 15, 0)

while t <= end:
    data = aggregate_6h(t)
    if data is not None:
        out_name = t.strftime("%Y_%m_%d_%H") + ".tif"
        out_path = os.path.join(out_dir, out_name)

        driver = gdal.GetDriverByName("GTiff")
        out = driver.Create(out_path, data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
        out.SetGeoTransform(gt)
        # preserve source projection if present; otherwise skip (no CRS assigned)
        if proj_wkt:
            out.SetProjection(proj_wkt)

        # ensure written array is float32 and set float NoData
        band = out.GetRasterBand(1)
        band.WriteArray(data.astype(np.float32))
        band.SetNoDataValue(float(-9999.0))
        # optional: declare no extra scale (we already applied it)
        try:
            band.SetScale(1.0)
            band.SetOffset(0.0)
        except Exception:
            pass

        out = None

        # print max with two decimals to confirm decimal handling
        max_val = np.nanmax(data)
        print(f"Saved: {out_name}, max={max_val:.2f}")
    else:
        print("No data for:", t)

    t += timedelta(hours=6)
