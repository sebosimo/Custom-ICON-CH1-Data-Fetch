import os, sys, datetime, json, xarray as xr
import numpy as np
from meteodatalab import ogd_api

# --- Configuration ---
CORE_VARS = ["T", "U", "V", "P", "QV"]
CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_iso_horizon(total_hours):
    days = total_hours // 24
    hours = total_hours % 24
    return f"P{days}DT{hours}H"

def get_location_indices(ds, locations):
    print(f"DEBUG: Entering get_location_indices", flush=True)
    lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
    lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
    
    print(f"DEBUG: Grid coordinates identified as: {lat_name}, {lon_name}", flush=True)
    
    indices = {}
    grid_lat = ds[lat_name].values
    grid_lon = ds[lon_name].values
    
    print(f"DEBUG: Coordinate array shape: {grid_lat.shape}", flush=True)
    
    for name, coords in locations.items():
        dist = (grid_lat - coords['lat'])**2 + (grid_lon - coords['lon'])**2
        flat_min_idx = np.argmin(dist)
        # unravel_index turns a flat index into a coordinate tuple (e.g. (402,) or (10, 20))
        idx = np.unravel_index(flat_min_idx, grid_lat.shape)
        indices[name] = idx
    
    sample_key = list(indices.keys())[0]
    print(f"DEBUG: Sample calculated index for {sample_key}: {indices[sample_key]}", flush=True)
    return indices

def main():
    if not os.path.exists("locations.json"):
        print("ERROR: locations.json missing", flush=True)
        return
    with open("locations.json", "r") as f:
        locations = json.load(f)

    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    ref_time = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    
    max_h = 45 if ref_time.hour == 3 else 33
    horizons = range(0, max_h + 1, 2)
    time_tag = ref_time.strftime('%Y%m%d_%H%M')

    print(f"--- STARTING RUN: {time_tag} ---", flush=True)

    cached_indices = None

    for h_int in horizons:
        iso_h = get_iso_horizon(h_int)
        print(f"\n>>> PROCESSING STEP: +{h_int}h", flush=True)
        
        domain_fields = {}
        try:
            for var in CORE_VARS:
                print(f"DEBUG: Fetching {var} from API...", flush=True)
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                     reference_datetime=ref_time, horizon=iso_h, perturbed=False)
                res = ogd_api.get_from_ogd(req)
                if res is None:
                    print(f"DEBUG: Variable {var} returned None", flush=True)
                    continue
                domain_fields[var] = res

            if not domain_fields:
                print(f"DEBUG: No variables fetched for this step, skipping.", flush=True)
                continue

            if cached_indices is None:
                cached_indices = get_location_indices(domain_fields[list(domain_fields.keys())[0]], locations)

            for name in locations.keys():
                idx = cached_indices[name]
                print(f"DEBUG: Extracting location '{name}' with index {idx}", flush=True)
                
                loc_data = {}
                for var_name, ds_field in domain_fields.items():
                    print(f"  DEBUG: Processing {var_name}. Dims: {list(ds_field.dims)}", flush=True)
                    
                    # Logic Check
                    spatial_dim = None
                    for d in ['ncells', 'cell', 'values', 'index']:
                        if d in ds_field.dims:
                            spatial_dim = d
                            break
                    
                    print(f"  DEBUG: Spatial dim identified as: {spatial_dim}", flush=True)

                    # THIS IS THE SUSPECTED CRASH POINT
                    if spatial_dim:
                        print(f"  DEBUG: Attempting 1D selection on {spatial_dim} using idx[0]={idx[0]}", flush=True)
                        subset = ds_field.isel({spatial_dim: idx[0]})
                    else:
                        print(f"  DEBUG: No 1D dim found. Attempting 2D selection (y,x) using idx[0]={idx[0]}, idx[1]={idx[1]}", flush=True)
                        subset = ds_field.isel(y=idx[0], x=idx[1])
                    
                    loc_data[var_name] = subset.squeeze().compute()

                # Saving logic
                ds_final = xr.Dataset(loc_data)
                ds_final.attrs = {"location": name, "ref_time": ref_time.isoformat(), "horizon_h": h_int}
                out_path = os.path.join(CACHE_DIR, f"{name}_{time_tag}_H{h_int:02d}.nc")
                ds_final.to_netcdf(out_path)
                print(f"  SUCCESS: Saved {name}", flush=True)

        except Exception as e:
            print(f"\n!!! CRITICAL FAILURE AT +{h_int}h !!!", flush=True)
            print(f"Error Type: {type(e).__name__}", flush=True)
            print(f"Error Message: {e}", flush=True)
            # In a diagnostic run, we want to stop and see the log
            sys.exit(1)

if __name__ == "__main__":
    main()
