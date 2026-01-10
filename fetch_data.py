import os, sys, datetime, xarray as xr
import numpy as np
from meteodatalab import ogd_api

# --- Configuration ---
LAT_TARGET, LON_TARGET = 46.81, 6.94
CORE_VARS = ["T", "U", "V", "P"]
CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_nearest_profile(ds, lat_target, lon_target):
    """Your Original Vertical Profile Extraction Logic"""
    if ds is None: return None
    data = ds if isinstance(ds, xr.DataArray) else ds[list(ds.data_vars)[0]]
    lat_coord = 'latitude' if 'latitude' in data.coords else 'lat'
    lon_coord = 'longitude' if 'longitude' in data.coords else 'lon'
    horiz_dims = data.coords[lat_coord].dims
    dist = (data[lat_coord] - lat_target)**2 + (data[lon_coord] - lon_target)**2
    flat_idx = dist.argmin().values
    if len(horiz_dims) == 1:
        profile = data.isel({horiz_dims[0]: flat_idx})
    else:
        profile = data.stack(gp=horiz_dims).isel(gp=flat_idx)
    return profile.squeeze().compute()

def main():
    force = os.getenv("FORCE_REFRESH") == "true"
    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    latest_run = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    times_to_try = [latest_run - datetime.timedelta(hours=i*3) for i in range(4)]

    for ref_time in times_to_try:
        time_tag = ref_time.strftime('%Y%m%d_%H%M')
        cache_path = os.path.join(CACHE_DIR, f"profile_{time_tag}.nc")

        # Skip if already downloaded (unless forced)
        if os.path.exists(cache_path) and not force:
            print(f">>> Run {time_tag} found in cache. Skipping download.")
            return

        print(f"--- Attempting Run: {ref_time.strftime('%H:%M')} UTC ---")
        try:
            profile_data = {}
            for var in CORE_VARS:
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                     reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
                res = get_nearest_profile(ogd_api.get_from_ogd(req), LAT_TARGET, LON_TARGET)
                if res is None or res.size < 5: raise ValueError(f"Empty {var}")
                profile_data[var] = res
            
            # Fetch Humidity with fallback
            for hum_var in ["RELHUM", "QV"]:
                try:
                    req_h = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=hum_var,
                                           reference_datetime=ref_time, horizon="P0DT0H", perturbed=False)
                    res_h = get_nearest_profile(ogd_api.get_from_ogd(req_h), LAT_TARGET, LON_TARGET)
                    if res_h is not None and res_h.size >= 5:
                        profile_data["HUM"], profile_data["HUM_TYPE"] = res_h, hum_var
                        break
                except: continue
            
            if "HUM" not in profile_data: raise ValueError("No Humidity data found")

            # --- PREPARE FOR SAVING ---
            # Merge into a dataset
            ds = xr.Dataset({v: profile_data[v] for v in CORE_VARS + ["HUM"]})
            
            # WIPE ALL METADATA (This is what caused the truth value errors)
            # We want just the raw numbers and coordinates
            ds.attrs = {"HUM_TYPE": profile_data["HUM_TYPE"], "ref_time": ref_time.isoformat()}
            for v in ds.data_vars: ds[v].attrs = {}
            for c in ds.coords: ds[c].attrs = {}

            ds.to_netcdf(cache_path)
            print(f">>> SUCCESS: Downloaded and cached {time_tag}")
            return 

        except Exception as e:
            print(f"Run {ref_time.strftime('%H:%M')} incomplete: {e}")
            continue

    print("Error: Could not retrieve any complete model runs.")
    sys.exit(1)

if __name__ == "__main__":
    main()
