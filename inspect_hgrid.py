import xarray as xr
import os

path = os.path.join("static_data", "horizontal_constants_icon-ch1-eps.grib2")
if not os.path.exists(path):
    print(f"File not found: {path}")
else:
    try:
        ds = xr.open_dataset(path, engine='cfgrib', backend_kwargs={'indexpath': ''})
        print(ds)
        print("Data Vars:", list(ds.data_vars))
        print("Coords:", list(ds.coords))
    except Exception as e:
        print(e)
