import os, sys, datetime, json, xarray as xr
import numpy as np
from meteodatalab import ogd_api

# --- Configuration ---
CORE_VARS = ["T", "U", "V", "P"]
HUM_VARS = ["RELHUM", "QV"]
CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_iso_horizon(total_hours):
    """Converts integer hours to ISO8601 duration string."""
    days = total_hours // 24
    hours = total_hours % 24
    return f"P{days}DT{hours}H"

def get_location_indices(ds, locations):
    """
    Calculates the grid indices (x, y) or (cell) for all locations at once.
    This is done once per model run to save CPU time.
    """
    lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
    lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
    
    indices = {}
    # Flatten the grid coordinates for easy distance calculation
    grid_lat = ds[lat_name].values
    grid_lon = ds[lon_name].values
    
    for name, coords in locations.items():
        # Simple Euclidean distance (fine for small local areas like CH)
        dist = (grid_lat - coords['lat'])**2 + (grid_lon - coords['lon'])**2
        idx = np.unravel_index(np.argmin(dist), dist.shape)
        indices[name] = idx
    return indices

def main():
    if not os.path.exists("locations.json"):
        print("Error: locations.json not found.")
        return
    with open("locations.json", "r") as f:
        locations = json.load(f)

    # 1. Determine latest model run
    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    ref_time = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    
    # 03 UTC run goes to 45h, others to 33h
    max_h = 45 if ref_time.hour == 3 else 33
    horizons = range(0, max_h + 1, 2)
    time_tag = ref_time.strftime('%Y%m%d_%H%M')

    print(f"--- ICON-CH1 Run: {time_tag} | Max Horizon: {max_h}h ---")

    # We store indices here once we've opened the first file
    cached_indices = None

    # 2. Loop through lead times
    for h_int in horizons:
        iso_h = get_iso_horizon(h_int)
        print(f"\nHorizon +{h_int:02d}h: Fetching domain fields...")
        
        domain_fields = {}
        hum_type = None

        try:
            # A. Fetch Core Variables (T, U, V, P)
            for var in CORE_VARS:
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                     reference_datetime=ref_time, horizon=iso_h, perturbed=False)
                domain_fields[var] = ogd_api.get_from_ogd(req)
            
            # B. Fetch Humidity
            for hv in HUM_VARS:
                try:
                    req_h = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=hv,
                                           reference_datetime=ref_time, horizon=iso_h, perturbed=False)
                    res_h = ogd_api.get_from_ogd(req_h)
                    if res_h is not None:
                        domain_fields["HUM"], hum_type = res_h, hv
                        break
                except: continue

            if not domain_fields or "HUM" not in domain_fields:
                print(f"  [!] Missing data for horizon {h_int}, skipping.")
                continue

            # C. Calculate indices once for this model grid geometry
            if cached_indices is None:
                cached_indices = get_location_indices(domain_fields[CORE_VARS[0]], locations)

            # D. Extract and save profiles for each location
            for name, idx in cached_indices.items():
                cache_path = os.path.join(CACHE_DIR, f"{name}_{time_tag}_H{h_int:02d}.nc")
                if os.path.exists(cache_path): continue

                # Extract the vertical column (isel handles 1D or 2D grids)
                loc_data = {}
                for var_name, ds_field in domain_fields.items():
                    # ICON grid might be (cell) or (y, x). This handles both.
                    if len(idx) == 1: # Unstructured/Cell
                        loc_data[var_name] = ds_field.isel(cell=idx[0]).compute()
                    else: # Lat/Lon grid
                        loc_data[var_name] = ds_field.isel(y=idx[0], x=idx[1]).compute()

                # Build dataset
                ds_final = xr.Dataset(loc_data)
                ds_final.attrs = {
                    "location": name,
                    "HUM_TYPE": hum_type, 
                    "ref_time": ref_time.isoformat(),
                    "horizon_h": h_int,
                    "valid_time": (ref_time + datetime.timedelta(hours=h_int)).isoformat()
                }

                # Clear metadata to avoid NetCDF write errors
                for v in ds_final.data_vars: ds_final[v].attrs = {}
                for c in ds_final.coords: ds_final[c].attrs = {}

                ds_final.to_netcdf(cache_path)
                print(f"    -> Saved: {name}")

        except Exception as e:
            print(f"  [ERROR] Horizon +{h_int}h: {e}")

if __name__ == "__main__":
    main()
