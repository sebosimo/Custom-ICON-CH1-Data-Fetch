import os, sys, datetime, json, xarray as xr
import numpy as np
import traceback # Added for sound data
from meteodatalab import ogd_api

# --- Configuration ---
CORE_VARS = ["T", "U", "V", "P", "QV"]
CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_iso_horizon(total_hours):
    days = total_hours // 24
    hours = total_hours % 24
    return f"P{days}DT{hours}H"

def main():
    if not os.path.exists("locations.json"):
        print("ERROR: locations.json missing", flush=True)
        return
    with open("locations.json", "r") as f:
        locations = json.load(f)

    # Calculate latest run
    now = datetime.datetime.now(datetime.timezone.utc)
    base_hour = (now.hour // 3) * 3
    ref_time = now.replace(hour=base_hour, minute=0, second=0, microsecond=0)
    
    # Check if the run is too fresh (less than 90 mins old)
    # ICON-CH1 usually takes ~1.5 hours to start appearing on OGD
    time_since_run = (now - ref_time).total_seconds() / 60
    print(f"--- RUN INFO ---", flush=True)
    print(f"Target Run: {ref_time.strftime('%Y-%m-%d %H:%M')} UTC", flush=True)
    print(f"Time since run start: {time_since_run:.1f} minutes", flush=True)

    horizons = range(0, 33, 2)
    time_tag = ref_time.strftime('%Y%m%d_%H%M')

    for h_int in horizons:
        iso_h = get_iso_horizon(h_int)
        print(f"\n>>> ATTEMPTING STEP: +{h_int}h ({iso_h})", flush=True)
        
        try:
            # 1. FETCH - This is where the crash happens
            print(f"DEBUG: Calling ogd_api.get_from_ogd for variable 'T'...", flush=True)
            req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", 
                                 variable="T",
                                 reference_datetime=ref_time, 
                                 horizon=iso_h, 
                                 perturbed=False)
            
            # This call downloads and indexes the GRIB
            ds_t = ogd_api.get_from_ogd(req)
            
            if ds_t is None:
                print(f"RESULT: API returned None (Data not ready yet).", flush=True)
                continue
                
            print(f"RESULT: Successfully fetched T. Dims: {list(ds_t.dims)}", flush=True)
            
            # If T works, we would proceed to others...
            # For this diagnostic, we stop here if successful to save time
            print(f"DEBUG: Data for this step is healthy. Moving to next check.", flush=True)

        except Exception:
            print(f"\n!!! CAPTURED CRITICAL TRACEBACK !!!", flush=True)
            # THIS IS THE SOUND DATA: It prints the exact line in the library that failed
            traceback.print_exc(file=sys.stdout)
            print(f"!!! END OF TRACEBACK !!!\n", flush=True)
            
            # If the error is an IndexError, it's almost certainly a "Not Ready" issue on the server
            if isinstance(sys.exc_info()[1], IndexError):
                print("INTERPRETATION: The GRIB file likely exists but is empty or has a broken header (Incomplete Upload).", flush=True)
            
            # Stop the script so we don't spam the log
            sys.exit(1)

if __name__ == "__main__":
    main()
