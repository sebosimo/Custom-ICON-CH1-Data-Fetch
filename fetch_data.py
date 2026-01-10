import os, sys, datetime, json, xarray as xr
import numpy as np
from meteodatalab import ogd_api

# --- Configuration ---
CORE_VARS = ["T", "U", "V", "P"]
HUM_VARS = ["RELHUM", "QV"]
CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_iso_horizon(total_hours):
    days = total_hours // 24
    hours = total_hours % 24
    return f"P{days}DT{hours}H"

def get_location_indices(ds, locations):
    """Calculates the nearest grid indices for all locations."""
    lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
    lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
    
    indices = {}
    grid_lat = ds[lat_name].values
    grid_lon = ds[lon_name].values
    
    for name, coords in locations.items():
        # Euclidean distance
        dist = (grid_lat - coords['lat'])**2 + (grid_lon - coords['lon'])**2
        # Returns the flat integer index of the closest cell
        indices[name] = int(np.argmin(dist))
    return indices

def main():
    if not os.path.exists("locations.json"):
        print("Error: locations.json not found.")
        return
    with open("locations.json", "r") as f:
        locations = json.load(f)

    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    ref_time = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    
    max_h = 45 if ref_time.hour == 3 else 33
    horizons = range(0, max_h + 1, 2)
    time_tag = ref_time.strftime('%Y%m%d_%H%M')

    print(f"--- ICON-CH1 Run: {time_tag} | Max Horizon: {max_h}h ---")

    cached_indices = None

    for h_int in horizons:
        iso_h = get_iso_horizon(h_int)
        valid_time = ref_time + datetime.timedelta(hours=h_int)
        
        locations_to_process = [n for n in locations.keys() 
                               if not os.path.exists(os.path.join(CACHE_DIR, f"{n}_{time_tag}_H{h_int:02d}.nc"))]
        
        if not locations_to_process:
            continue

        print(f"\nHorizon +{h_int:02d}h: Fetching domain fields...")
        domain_fields = {}
        hum_type_found = None

        try:
            for var in CORE_VARS:
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                     reference_datetime=ref_time, horizon=iso_h, perturbed=False)
                domain_fields[var] = ogd_api.get_from_ogd(req)
            
            for hv in HUM_VARS:
                try:
                    req_h = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=hv,
                                           reference_datetime=ref_time, horizon=iso_h, perturbed=False)
                    res_h = ogd_api.get_from_ogd(req_h)
                    if res_h is not None:
                        domain_fields["HUM"], hum_type_found = res_h, hv
                        break
                except: continue

            if "HUM" not in domain_fields: continue

            if cached_indices is None:
                cached_indices = get_location_indices(domain_fields[CORE_VARS[0]], locations)

            for name in locations_to_process:
                flat_idx = cached_indices[name]
                cache_path = os.path.join(CACHE_DIR, f"{name}_{time_tag}_H{h_int:02d}.nc")

                loc_data = {}
                for var_name, ds_field in domain_fields.items():
                    # ROBUST DIMENSION DETECTION
                    # Find the dimension that represents the grid cells
                    spatial_dim = None
                    for dim in ['ncells', 'cell', 'values', 'index']:
                        if dim in ds_field.dims:
                            spatial_dim = dim
                            break
                    
                    if spatial_dim:
                        subset = ds_field.isel({spatial_dim: flat_idx})
                    else:
                        # Fallback for structured grids (y, x) if ever switched
                        # This assumes the flat index corresponds to a flattened (y,x)
                        subset = ds_field.stack(grid=['y','x']).isel(grid=flat_idx)
                    
                    loc_data[var_name] = subset.squeeze().compute()

                ds_final = xr.Dataset(loc_data)
                ds_final.attrs = {
                    "location": name, "HUM_TYPE": hum_type_found, 
                    "ref_time": ref_time.isoformat(), "horizon_h": h_int,
                    "valid_time": valid_time.isoformat()
                }

                for v in ds_final.data_vars: ds_final[v].attrs = {}
                for c in ds_final.coords: ds_final[c].attrs = {}

                ds_final.to_netcdf(cache_path)
                print(f"    -> Saved: {name}")

        except Exception as e:
            print(f"  [ERROR] Horizon +{h_int}h failed: {e}")

if __name__ == "__main__":
    main()
