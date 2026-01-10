import os, sys, datetime, json, xarray as xr
from meteodatalab import ogd_api

# --- Configuration ---
CACHE_DIR = "cache_data"
LOCATIONS_FILE = "locations.json"
os.makedirs(CACHE_DIR, exist_ok=True)

# Define horizons for 2h steps (e.g., 0h, 2h, 4h... up to 32h)
HORIZONS = [f"P0DT{h}H" for h in range(0, 33, 2)] 
CORE_VARS = ["T", "U", "V", "P"]

def get_nearest_profile(ds, lat_target, lon_target):
    if ds is None: return None
    data = ds if isinstance(ds, xr.DataArray) else ds[list(ds.data_vars)[0]]
    lat_coord = 'latitude' if 'latitude' in data.coords else 'lat'
    lon_coord = 'longitude' if 'longitude' in data.coords else 'lon'
    horiz_dims = data.coords[lat_coord].dims
    dist = (data[lat_coord] - lat_target)**2 + (data[lon_coord] - lon_target)**2
    flat_idx = dist.argmin().values
    profile = data.stack(gp=horiz_dims).isel(gp=flat_idx) if len(horiz_dims) > 1 else data.isel({horiz_dims[0]: flat_idx})
    return profile.squeeze().compute()

def main():
    with open(LOCATIONS_FILE, 'r') as f:
        locations = json.load(f)

    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    ref_time = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    
    for name, coords in locations.items():
        print(f"\n>>> Processing Location: {name}")
        loc_data_list = []
        
        for horizon in HORIZONS:
            try:
                profile_vars = {}
                for var in CORE_VARS:
                    req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                         reference_datetime=ref_time, horizon=horizon)
                    res = get_nearest_profile(ogd_api.get_from_ogd(req), coords['lat'], coords['lon'])
                    profile_vars[var] = res
                
                # Fetch Humidity fallback
                for hum_var in ["RELHUM", "QV"]:
                    try:
                        req_h = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=hum_var,
                                               reference_datetime=ref_time, horizon=horizon)
                        res_h = get_nearest_profile(ogd_api.get_from_ogd(req_h), coords['lat'], coords['lon'])
                        if res_h is not None:
                            profile_vars["HUM"] = res_h
                            break
                    except: continue

                # Merge time step
                ds_step = xr.Dataset(profile_vars)
                ds_step = ds_step.expand_dims(time=[ref_time + datetime.timedelta(hours=int(horizon[4:-1]))])
                loc_data_list.append(ds_step)
                print(f"  Fetched horizon: {horizon}")

            except Exception as e:
                print(f"  Error fetching {horizon} for {name}: {e}")

        if loc_data_list:
            final_ds = xr.concat(loc_data_list, dim="time")
            cache_path = os.path.join(CACHE_DIR, f"{name}_latest.nc")
            final_ds.to_netcdf(cache_path)
            print(f"  Saved: {cache_path}")

if __name__ == "__main__":
    main()
