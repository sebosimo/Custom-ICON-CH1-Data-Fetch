import os, sys
import datetime, json, xarray as xr
import numpy as np
import warnings
import requests

# Set GRIB definitions for COSMO/ICON BEFORE importing libraries that might use them
# Using absolute path directly as a fallback/fix
COSMO_DEFS = r"C:\Users\sebas\.conda\envs\weather_final\share\eccodes-cosmo-resources\definitions"
STANDARD_DEFS = os.path.join(sys.prefix, "Library", "share", "eccodes", "definitions")

defs_to_use = []
if os.path.exists(COSMO_DEFS):
    defs_to_use.append(COSMO_DEFS)
if os.path.exists(STANDARD_DEFS):
    defs_to_use.append(STANDARD_DEFS)

if defs_to_use:
    final_def_path = ":".join(defs_to_use)
    os.environ["GRIB_DEFINITION_PATH"] = final_def_path
    os.environ["ECCODES_DEFINITION_PATH"] = final_def_path
    print(f"Set GRIB definitions to: {final_def_path}", flush=True)

from meteodatalab import ogd_api

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
# Variables needed for Point Traces
VARS_TRACES = ["T", "U", "V", "P", "QV"]
# Variables needed for Wind Maps
VARS_MAPS = ["U", "V", "HHL"] 

CACHE_DIR_TRACES = "cache_data"
CACHE_DIR_MAPS = "cache_wind"
STATIC_DIR = "static_data"
HHL_FILENAME = "vertical_constants_icon-ch1-eps.grib2"
HGRID_FILENAME = "horizontal_constants_icon-ch1-eps.grib2"
STAC_ASSETS_URL = "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-forecasting-icon-ch1/assets"

# Levels definition
WIND_LEVELS = [
    {"name": "10m_AGL",   "h": 10,   "type": "AGL"},
    {"name": "800m_AGL",  "h": 800,  "type": "AGL"},
    {"name": "1500m_AMSL","h": 1500, "type": "AMSL"},
    {"name": "2000m_AMSL","h": 2000, "type": "AMSL"},
    {"name": "3000m_AMSL","h": 3000, "type": "AMSL"},
    {"name": "4000m_AMSL","h": 4000, "type": "AMSL"},
]

os.makedirs(CACHE_DIR_TRACES, exist_ok=True)
os.makedirs(CACHE_DIR_MAPS, exist_ok=True)

def get_iso_horizon(total_hours):
    days = total_hours // 24
    hours = total_hours % 24
    return f"P{days}DT{hours}H"

def sanitize_name(name):
    """Sanitizes location names for Windows paths (removes umlauts)."""
    n = name.replace("ü", "ue").replace("ö", "oe").replace("ä", "ae") \
            .replace("Ü", "Ue").replace("Ö", "Oe").replace("Ä", "Ae").replace("ß", "ss")
    # Keep only alphanumeric, dash, underscore
    clean = "".join(c for c in n if c.isalnum() or c in ('-', '_'))
    return clean if clean else "unnamed"

def log(msg):
    with open("debug_log.txt", "a") as f:
        f.write(f"{datetime.datetime.now()}: {msg}\n")
    print(msg, flush=True)

def download_static_files():
    """Ensures static constants files (HHL) are available."""
    os.makedirs(STATIC_DIR, exist_ok=True)
    hhl_path = os.path.join(STATIC_DIR, HHL_FILENAME)
    
    if not os.path.exists(hhl_path):
        log(f"Downloading static HHL file from {STAC_ASSETS_URL}...")
        try:
            # 1. Get asset URL from STAC
            resp = requests.get(STAC_ASSETS_URL)
            resp.raise_for_status()
            assets = resp.json()["assets"]
            
            # Find the vertical constants file
            file_url = None
            for asset in assets:
                if asset.get("id") == HHL_FILENAME:
                    file_url = asset.get("href")
                    break
            
            if not file_url:
                log("Error: Could not find HHL file URL in STAC response.")
            else:
                # 2. Download the file
                log(f"Fetching {file_url}...")
                file_resp = requests.get(file_url, stream=True)
                file_resp.raise_for_status()
                
                with open(hhl_path, 'wb') as f:
                    for chunk in file_resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                log("Static HHL file download complete.")
        
        except Exception as e:
            log(f"Failed to download static HHL file: {e}")

    # --- Horizontal Constants ---
    hgrid_path = os.path.join(STATIC_DIR, HGRID_FILENAME)
    if not os.path.exists(hgrid_path):
        log(f"Downloading static HGRID file from {STAC_ASSETS_URL}...")
        try:
            # Re-fetch or reuse assets if we were smarter, but re-fetch is safe
            resp = requests.get(STAC_ASSETS_URL)
            resp.raise_for_status()
            assets = resp.json()["assets"]
            
            file_url = None
            for asset in assets:
                if asset.get("id") == HGRID_FILENAME:
                    file_url = asset.get("href")
                    break
            
            if file_url:
                log(f"Fetching {file_url}...")
                file_resp = requests.get(file_url, stream=True)
                file_resp.raise_for_status()
                with open(hgrid_path, 'wb') as f:
                    for chunk in file_resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                log("Static HGRID file download complete.")
        except Exception as e:
            log(f"Failed to download static HGRID file: {e}")

def load_static_hhl():
    """Loads the static HHL field."""
    hhl_path = os.path.join(STATIC_DIR, HHL_FILENAME)
    if not os.path.exists(hhl_path):
        return None
    
    try:
        # Open with cfgrib engine, backend_kwargs to avoid locking issues if possible
        # On Windows, cfgrib creates temp indices.
        ds = xr.open_dataset(hhl_path, engine='cfgrib')
        
        # Check for likely variable names
        if 'h' in ds:
            hhl = ds['h'].load() # Load into memory so we can close file
        elif 'HHL' in ds:
            hhl = ds['HHL'].load()
        else:
             # Fallback: check data vars
             vars = list(ds.data_vars)
             if vars: 
                 hhl = ds[vars[0]].load()
             else:
                 hhl = None
        
        ds.close() # Explicitly close
        return hhl

    except Exception as e:
        log(f"Error loading HHL: {e}")
        return None

def load_static_grid():
    """Loads text lat/lon from HGRID file."""
    hgrid_path = os.path.join(STATIC_DIR, HGRID_FILENAME)
    if not os.path.exists(hgrid_path):
        return None
    try:
        # Open with cfgrib
        ds = xr.open_dataset(hgrid_path, engine='cfgrib', backend_kwargs={'indexpath': ''})
        # We expect tlat/tlon OR CLAT/CLON
        grid = {}
        if 'tlat' in ds: grid['lat'] = ds['tlat'].load()
        elif 'CLAT' in ds: grid['lat'] = ds['CLAT'].load()
        
        if 'tlon' in ds: grid['lon'] = ds['tlon'].load()
        elif 'CLON' in ds: grid['lon'] = ds['CLON'].load()
        
        ds.close()
        
        if 'lat' in grid and 'lon' in grid:
            return grid
        return None
    except Exception as e:
        log(f"Error loading HGRID: {e}")
        return None

def is_run_complete_locally(time_tag, locations, max_h):
    """Checks if the very last file of a run exists."""
    last_loc = list(locations.keys())[-1]
    safe_last = sanitize_name(last_loc)
    check_trace = os.path.join(CACHE_DIR_TRACES, time_tag, safe_last, f"H{max_h:02d}.nc")
    check_map = os.path.join(CACHE_DIR_MAPS, time_tag, f"wind_maps_H{max_h:02d}.nc")
    return os.path.exists(check_trace) and os.path.exists(check_map)

def process_traces(domain_fields, locations, time_tag, h_int, ref_time):
    """Extracts point data for specific locations."""
    sample = list(domain_fields.values())[0]
    lat_n = 'latitude' if 'latitude' in sample.coords else 'lat'
    lon_n = 'longitude' if 'longitude' in sample.coords else 'lon'
    lats, lons = sample[lat_n].values, sample[lon_n].values
    
    indices = {n: int(np.argmin((lats-c['lat'])**2+(lons-c['lon'])**2)) for n, c in locations.items()}

    for name, flat_idx in indices.items():
        safe_name = sanitize_name(name)
        loc_dir = os.path.join(CACHE_DIR_TRACES, time_tag, safe_name)
        os.makedirs(loc_dir, exist_ok=True)
        cache_path = os.path.join(loc_dir, f"H{h_int:02d}.nc")
        
        if os.path.exists(cache_path): continue

        loc_vars = {}
        for var_name in VARS_TRACES:
            if var_name not in domain_fields: continue
            ds = domain_fields[var_name]
            s_dim = ds[lat_n].dims[0]
            profile = ds.squeeze().isel({s_dim: flat_idx}).compute()
            
            if len(profile.dims) > 0:
                v_dim = profile.dims[0]
                profile = profile.rename({v_dim: 'level'})
            
            loc_vars[var_name] = profile.drop_vars([c for c in profile.coords if c not in profile.dims])

        ds_final = xr.Dataset(loc_vars)
        ds_final.attrs = {
            "location": name, "HUM_TYPE": "QV", 
            "ref_time": ref_time.isoformat(), 
            "horizon_h": h_int, 
            "valid_time": (ref_time + datetime.timedelta(hours=h_int)).isoformat()
        }
        for v in ds_final.data_vars: ds_final[v].attrs = {}
        ds_final.to_netcdf(cache_path)

def process_wind_maps(domain_fields, time_tag, h_int, ref_time):
    """Interpolates wind and saves map NC."""
    output_dir = os.path.join(CACHE_DIR_MAPS, time_tag)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"wind_maps_H{h_int:02d}.nc")
    if os.path.exists(output_path): return

    if "U" not in domain_fields or "V" not in domain_fields:
        return # Cannot process
    if "HHL" not in domain_fields:
        log(f" Warning: HHL missing for H{h_int:02d}. Skipping maps.")
        return

    # Imports locally to avoid top-level require
    from metpy.interpolate import interpolate_to_isosurface
    from metpy.units import units

    u = domain_fields["U"].squeeze()
    v = domain_fields["V"].squeeze()
    hhl = domain_fields["HHL"].squeeze() # Now guaranteed to be static HHL if passed
    
    # If HHL has no time dimension but U/V do, that's fine as long as spatial dims match.
    # U/V might have (ncells). HHL might have (generalVerticalLayer, ncells).

    
    # Calculate HFL (Height Full Levels)
    # HHL likely (81, ncells), U/V (80, ncells)
    try:
        # Check dim name consistency
        z_dim = hhl.dims[0]
        # Slice logic: Depends on order. 
        # Safest: Use 0:-1 and 1:None manually
        z_f = (hhl.isel({z_dim: slice(0,-1)}).values + hhl.isel({z_dim: slice(1,None)}).values) / 2
        
        h_surf = hhl.isel({z_dim: -1}) # Assuming last is surface
    except Exception as e:
        log(f" HHL processing error: {e}")
        return

    log(f"DEBUG: Processing Wind Maps for H+{h_int}...")
    out_ds_dict = {}
    
    # --- Prepare for Interpolation (Numpy Fallback) ---
    # We use numpy to avoid MetPy/Xarray unit/indexing clashes on unstructured grids
    
    # 1. Dequantify / Get Numpy Arrays
    def to_np(da):
        if hasattr(da, "metpy"):
            da = da.metpy.dequantify()
        return da.values

    np_u = to_np(u)
    np_v = to_np(v)
    np_z = to_np(xr.DataArray(z_f, coords=u.coords, dims=u.dims)) # Reuse logic to align shapes if needed
    
    # We assume dim 0 is vertical because we constructed (or read) it that way
    vert_axis = 0
    if u.dims[vert_axis] != 'generalVerticalLayer':
         if u.dims[1] == 'generalVerticalLayer': vert_axis = 1

    # MetPy interpolate_to_isosurface expects vertical dim as axis 0.
    # Transpose if needed.
    if vert_axis != 0:
        np_u = np_u.T
        np_v = np_v.T
        np_z = np_z.T
        # if AGL mode uses h_surf, check its shape too? h_surf is 1D (ncells).
        # if vert_axis was 1, u is (N, 80). np_u becomes (80, N).
        # h_surf stays (N). (80, N) - (N) works.
    
    for lvl in WIND_LEVELS:
        h_target = lvl['h']
        mode = lvl['type']
        name_key = lvl['name']
        
        try:
            if mode == 'AGL':
                 # For AGL, we need H_surf.
                 np_h_surf = to_np(h_surf)
                 # Numpy broadcasting: (80, N) - (N,) -> (80, N)
                 np_field_z = np_z - np_h_surf
            else: # AMSL
                 np_field_z = np_z
            
            # Interpolate (Assumes axis 0 is vertical)
            res_u = interpolate_to_isosurface(np_field_z, np_u, h_target)
            res_v = interpolate_to_isosurface(np_field_z, np_v, h_target)
            
            # Re-wrap in DataArray
            # Find the non-vertical dim from ORIGINAL u
            spatial_dim = u.dims[1-vert_axis] 
            
            coords_dict = {spatial_dim: u[spatial_dim]}
            if 'latitude' in u.coords: coords_dict['latitude'] = u.coords['latitude']
            if 'longitude' in u.coords: coords_dict['longitude'] = u.coords['longitude']
            
            # Result is (N) [spatial dim], no matter if input was (Z, N)
            da_res_u = xr.DataArray(res_u, dims=[spatial_dim], coords=coords_dict, name=f"u_{name_key}")
            da_res_v = xr.DataArray(res_v, dims=[spatial_dim], coords=coords_dict, name=f"v_{name_key}")

            out_ds_dict[f"u_{name_key}"] = da_res_u
            out_ds_dict[f"v_{name_key}"] = da_res_v
            log(f" - Level {name_key} done")
        except Exception as e:
            log(f" - Level {name_key} failed: {e}")
            pass # Skip level

    if out_ds_dict:
        log(f"Saving wind map to {output_path}")
        ds_out = xr.Dataset(out_ds_dict)
        for v_c in ds_out:
            if hasattr(ds_out[v_c].data, 'magnitude'):
                ds_out[v_c] = ds_out[v_c].metpy.dequantify()
        ds_out.to_netcdf(output_path)

def main():
    log("Main start...")
    if not os.path.exists("locations.json"):
        log("locations.json missing")
        return
    with open("locations.json", "r", encoding="utf-8") as f: locations = json.load(f)
    log(f"Loaded {len(locations)} locations")

    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    latest_run = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    
    # We will try the latest two cycles
    targets = [latest_run, latest_run - datetime.timedelta(hours=3)]
    
    selected_run = None
    log("--- CHECKING FOR AVAILABLE RUNS ---")
    
    for run in targets:
        tag = run.strftime('%Y%m%d_%H%M')
        max_h = 45 if run.hour == 3 else 33
        
        # 1. Check Local Complete
        if is_run_complete_locally(tag, locations, max_h):
             log(f"Run {tag}: [OK] Already fully cached.")
             # We might still want to proceed if we want to fill gaps? 
             # But the logic says return.
             if run == targets[0]: return 
             continue
        
        # 2. Check Server Availability
        try:
             log(f"Checking server for {tag} (Available check)...")
             
             # Check start
             req_start = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable="T",
                                 reference_datetime=run, horizon="P0DT0H", perturbed=False)
             
             # Check end (to ensure upload is mostly complete)
             iso_end = get_iso_horizon(max_h)
             req_end = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable="T",
                                 reference_datetime=run, horizon=iso_end, perturbed=False)

             log("- Checking Start Horizon...")
             # Use get_asset_urls to check availability WITHOUT downloading/opening the file
             urls_start = ogd_api.get_asset_urls(req_start)
             
             log("- Checking End Horizon...")
             urls_end = ogd_api.get_asset_urls(req_end)
             
             start_ok = len(urls_start) > 0
             end_ok = len(urls_end) > 0

             if start_ok and end_ok:
                 log(f"Run {tag}: [NEW] NEW DATA READY (Start & End confirmed)!")
                 selected_run = run
                 break
             else:
                 log(f"Run {tag}: Incomplete (Start={start_ok}, End={end_ok}).")

        except Exception as e:
             # Catch specific "list index out of range" which means NO DATA for this run/horizon yet (probably)
             if "list index out of range" in str(e):
                 log(f"Run {tag}: [ERR] Data not available yet (Index error).")
             else:
                 log(f"Run {tag}: [ERR] Server says 'not ready'. Error: {e}")
             pass

    if not selected_run:
        log("RESULT: No new runs to download.")
        cleanup_old_runs()
        return

    ref_time = selected_run
    time_tag = ref_time.strftime('%Y%m%d_%H%M')
    max_h = 45 if ref_time.hour == 3 else 33
    horizons = range(0, max_h + 1, 1)
    
    # Ensure static files are present
    download_static_files()
    
    # Load HHL once
    hhl_field = load_static_hhl()
    
    log(f"HHL Field loaded: {hhl_field is not None}")
    if hhl_field is None:
        log("Warning: Could not load static HHL. Wind maps will likely fail.")

    # Load Grid once
    grid_coords = load_static_grid()
    log(f"Grid Coords loaded: {grid_coords is not None}")
    if grid_coords is None:
        log("Warning: Could not load Grid Coordinates. Point traces will fail (KeyError: lat).")
    elif hhl_field is not None:
         # Also inject coords into HHL so it can serve as a sample
         n_grid = grid_coords['lat'].shape[0]
         match_dim = None
         for d in hhl_field.dims:
             if hhl_field.sizes[d] == n_grid:
                 match_dim = d
                 break
         
         if match_dim:
             hhl_field = hhl_field.assign_coords({
                 "latitude": (match_dim, grid_coords['lat'].values),
                 "longitude": (match_dim, grid_coords['lon'].values)
             })

    log(f"\n--- PROCESSING RUN: {time_tag} ---")
    
    vars_all = list(set(VARS_TRACES + VARS_MAPS))
    
    # We remove HHL because it is static
    vars_to_fetch = [v for v in vars_all if v != "HHL"]
    
    for h_int in horizons:
        # Check if needed
        traces_needed = False
        last_loc = list(locations.keys())[-1]
        safe_last = sanitize_name(last_loc)
        if not os.path.exists(os.path.join(CACHE_DIR_TRACES, time_tag, safe_last, f"H{h_int:02d}.nc")):
             traces_needed = True
        
        maps_needed = False
        if not os.path.exists(os.path.join(CACHE_DIR_MAPS, time_tag, f"wind_maps_H{h_int:02d}.nc")):
             maps_needed = True

        if not traces_needed and not maps_needed: 
             continue

        log(f"Fetching +{h_int:02d}h...")
        iso_h = get_iso_horizon(h_int)
        
        domain_fields = {}
        
        # Inject static HHL for maps if available
        if hhl_field is not None:
            domain_fields["HHL"] = hhl_field

        for var in vars_to_fetch:
            # Print separate progress dot potentially or handle failure
            try:
                req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable=var,
                                     reference_datetime=ref_time, horizon=iso_h, perturbed=False)
                
                # Manual fetch to avoid ogd_api hang/locking
                urls = ogd_api.get_asset_urls(req)
                if urls:
                    target_url = urls[0]
                    temp_filename = f"temp_{var}_{time_tag}_{h_int:02d}.grib2"
                    
                    # Log only on error or verbose
                    # log(f"Fetching {var}...") 

                    with requests.get(target_url, stream=True) as r:
                        r.raise_for_status()
                        with open(temp_filename, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                    
                    # Open, load, close, delete
                    # backend_kwargs={'indexpath': ''} prevents .idx file creation which can cause locking/delays
                    ds_raw = xr.open_dataset(temp_filename, engine='cfgrib', backend_kwargs={'indexpath': ''})
                    
                    # We expect the variable to be in the dataset. 
                    # Note: cfgrib might rename variables (e.g. 't' instead of 'T'). 
                    # But usually for ICON/COSMO output it matches or we find it.
                    # OGD variable 'T' often maps to 't' in GRIB.
                    # We will try exact match, then lower case.
                    
                    data_var = None
                    if var in ds_raw:
                        data_var = ds_raw[var]
                    elif var.lower() in ds_raw:
                        data_var = ds_raw[var.lower()]
                    elif var.upper() in ds_raw:
                        data_var = ds_raw[var.upper()]
                    else:
                        # Fallback: take the first data variable
                        vars_in_ds = list(ds_raw.data_vars)
                        if vars_in_ds:
                            data_var = ds_raw[vars_in_ds[0]]
                    
                    if data_var is not None:
                        # Load into memory
                        data_var = data_var.load()
                        
                        # Attach Coordinates if available (Crucial for unstructured grid)
                        if grid_coords:
                             # We need to find which dimension matches the grid size
                             n_grid = grid_coords['lat'].shape[0]
                             match_dim = None
                             for d in data_var.dims:
                                 if data_var.sizes[d] == n_grid:
                                     match_dim = d
                                     break
                             
                             if match_dim:
                                 # Ensure grid coords use this dim name
                                 lat_c = grid_coords['lat']
                                 lon_c = grid_coords['lon']
                                 
                                 # If grid coords have completely different dim name, we might need to rename or swap
                                 # Usually grid output from cfgrib is already 'values' or similar. 
                                 # We can just pass the array as a new coordinate with the matching dim name.
                                 
                                 data_var = data_var.assign_coords({
                                     "latitude": (match_dim, lat_c.values),
                                     "longitude": (match_dim, lon_c.values)
                                 })
                             else:
                                 log(f"DEBUG: Could not find dim matching grid size {n_grid} in {data_var.sizes}")

                        domain_fields[var] = data_var
                    
                    ds_raw.close()
                    try:
                        os.remove(temp_filename)
                    except:
                        pass
            except Exception as e:
                log(f"({var} err: {e})")
                pass
        
        if domain_fields:
            # Check if we have enough for Traces
            # Traces need VARS_TRACES. HHL is NOT in VARS_TRACES usually (only for maps)
            # VARS_TRACES = ["T", "U", "V", "P", "QV"]
            
            if traces_needed:
                 if any(v in domain_fields for v in VARS_TRACES):
                     log(f"DEBUG: Processing Traces. Domain vars: {list(domain_fields.keys())}")
                     # Inspect logic
                     sample = list(domain_fields.values())[0]
                     log(f"DEBUG: Sample coords: {list(sample.coords)}")
                     process_traces(domain_fields, locations, time_tag, h_int, ref_time)
            
            if maps_needed:
                 # Check HHL/Wind
                 if "HHL" in domain_fields and "U" in domain_fields and "V" in domain_fields:
                     process_wind_maps(domain_fields, time_tag, h_int, ref_time)
                 else:
                     missing = []
                     if "HHL" not in domain_fields: missing.append("HHL (Static)")
                     if "U" not in domain_fields: missing.append("U")
                     if "V" not in domain_fields: missing.append("V")
                     log(f"[Skip Maps: Missing {','.join(missing)}]")
            
            log(f"H+{h_int:02d} Done")
        else:
            log(f"H+{h_int:02d} Failed (No data)")
    
    # Cleanup after processing
    cleanup_old_runs()

def cleanup_old_runs():
    """
    Removes runs older than RETENTION_DAYS (env var).
    If RETENTION_DAYS is not set, NO cleanup is performed (local default).
    """
    try:
        days_str = os.environ.get("RETENTION_DAYS")
        if not days_str:
            # If not running in CI (or no env var), we might want a default or just return.
            # For now, let's look for a local default or return to avoid deleting user data unexpectedly.
            return 
        
        days_to_keep = int(days_str)
    except ValueError:
        print(f"Warning: Invalid RETENTION_DAYS '{days_str}', skipping cleanup.")
        return

    print(f"--- CLEANING UP OLD DATA (Retention: {days_to_keep} days) ---")
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_to_keep)
    
    # We look at folder names YYYYMMDD_HHMM
    dirs_to_check = [CACHE_DIR_TRACES, CACHE_DIR_MAPS]
    
    for d in dirs_to_check:
        if not os.path.exists(d): continue
        
        for item in os.listdir(d):
            path = os.path.join(d, item)
            
            # 1. Handle Directories (Runs)
            if os.path.isdir(path):
                # Parse timestamp from folder name
                try:
                    # Expected format: YYYYMMDD_HHMM
                    dt = datetime.datetime.strptime(item, "%Y%m%d_%H%M")
                    dt = dt.replace(tzinfo=datetime.timezone.utc)
                    
                    if dt < cutoff:
                        print(f"Deleting old run: {path}")
                        import shutil
                        shutil.rmtree(path)
                except ValueError:
                    # Not a timestamped folder, skip
                    pass
            
            # 2. Handle Orphaned Files (e.g. invalid downloads, old logs)
            elif os.path.isfile(path):
                 # Check modification time
                 mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path), tz=datetime.timezone.utc)
                 if mtime < cutoff:
                     print(f"Deleting old orphaned file: {path}")
                     try:
                        os.remove(path)
                     except Exception as e:
                        print(f"Failed to remove {path}: {e}")

if __name__ == "__main__":
    main()
