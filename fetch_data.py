import os, sys, datetime, json, xarray as xr
import numpy as np
from meteodatalab import ogd_api

# --- Configuration ---
CORE_VARS = ["T", "U", "V", "P"]
HUM_VARS = ["RELHUM", "QV"]
CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_iso_horizon(total_hours):
    """Converts integer hours to ISO8601 duration string (e.g. 26 -> 'P1DT2H')"""
    days = total_hours // 24
    hours = total_hours % 24
    return f"P{days}DT{hours}H"

def get_location_indices(ds, locations):
    """
    Calculates the nearest grid indices for all locations at once.
    Handles both unstructured (cell) and regular (y, x) ICON grids.
    """
    lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
    lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
    
    indices = {}
    grid_lat = ds[lat_name].values
    grid_lon = ds[lon_name].values
    
    for name, coords in locations.items():
        # Euclidean distance to find nearest grid point
        dist = (grid_lat - coords['lat'])**2 + (grid_lon - coords['lon'])**2
        idx = np.unravel_index(np.argmin(dist), dist.shape)
        indices[name] = idx
    return indices

def main():
    # 0. Load locations
    if not os.path.exists("locations.json"):
        print("Error: locations.json not found.")
        return
    with open("locations.json", "r") as f:
        locations = json.load(f)

    # 1. Determine Model Run (ICON-CH1 runs every 3 hours)
    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    ref_time = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    
    # Logic for max horizon: 03 UTC run goes to 45h, others to 33h
    max_h = 45 if ref_time.hour == 3 else 33
    horizons = range(0, max_h + 1, 2)
    time_tag = ref_time.strftime('%Y%m%d_%H%M')

    print(f"--- ICON-CH1 Run: {time_tag} | Max Horizon: {max_h}h ---")

    # Indices are cached once the first file of the run is opened
    cached_indices = None

    # 2. Main Loop: Iterate by Horizon (Lead Time)
    for h_int in horizons:
        iso_h = get_iso_horizon(h_int)
        valid_time = ref_time + datetime.timedelta(hours=h_int)
        
        # Check if we actually need to download this horizon 
        # (Check if at least one location is missing this file)
        locations_to_process = []
        for name in locations.keys():
            path = os.path.join(CACHE_DIR, f"{name}_{time_tag}_H{h_int:02d}.nc")
            if not os.path.exists(path):
                locations_to_process.append(name)
        
        if not locations_to_process:
            continue

        print(f"\nHorizon +{h_int:02d}h: Fetching domain fields for {len(locations_to_process)} locations...")
        
        domain_fields = {}
        hum_type_found = None

        try:
            # A. Fetch Core Variables (Full Domain)
            for var in CORE_VARS:
                req = ogd_api.Request(
                    collection="ogd-forecasting-icon-ch1", 
                    variable=var,
                    reference_datetime=ref_time, 
                    horizon=iso_h, 
                    perturbed=False
                )
                domain_fields[var] = ogd_api.get_from_ogd(req)
            
            # B. Fetch Humidity (Try RELHUM, then QV)
            for hv in HUM_VARS:
                try:
                    req_h = ogd_api.Request(
                        collection="ogd-forecasting-icon-ch1", 
                        variable=hv,
                        reference_datetime=ref_time, 
                        horizon=iso_h, 
                        perturbed=False
                    )
                    res_h = ogd_api.get_from_ogd(req_h)
                    if res_h is not None:
                        domain_fields["HUM"] = res_h
                        hum_type_found = hv
                        break
                except: continue

            if "HUM" not in domain_fields:
                print(f"  [!] Missing humidity for horizon {h_int}, skipping.")
                continue

            # C. Determine grid indices once per model run
            if cached_indices is None:
                cached_indices = get_location_indices(domain_fields[CORE_VARS[0]], locations)

            # D. Extract Profiles and Save
            for name in locations_to_process:
                idx = cached_indices[name]
                cache_path = os.path.join(CACHE_DIR, f"{name}_{time_tag}_H{h_int:02d}.nc")

                loc_data = {}
                for var_name, ds_field in domain_fields.items():
                    # Handle native ICON grid (cell) vs regular grid (y, x)
                    if 'cell' in ds_field.dims:
                        subset = ds_field.isel(cell=idx[0])
                    else:
                        subset = ds_field.isel(y=idx[0], x=idx[1])
                    
                    # FORCE SQUEEZE: This removes the 'number' (ensemble) dimension
                    # Crucial to prevent IndexErrors in plotting
                    loc_data[var_name] = subset.squeeze().compute()

                # Merge into unified dataset
                ds_final = xr.Dataset(loc_data)
                ds_final.attrs = {
                    "location": name,
                    "HUM_TYPE": hum_type_found, 
                    "ref_time": ref_time.isoformat(),
                    "horizon_h": h_int,
                    "valid_time": valid_time.isoformat()
                }

                # Wipe attributes that cause GRIB/NetCDF conversion conflicts
                for v in ds_final.data_vars: ds_final[v].attrs = {}
                for c in ds_final.coords: ds_final[c].attrs = {}

                ds_final.to_netcdf(cache_path)
                print(f"    -> Saved: {name}")

        except Exception as e:
            print(f"  [ERROR] Horizon +{h_int}h failed: {e}")

if __name__ == "__main__":
    main()
