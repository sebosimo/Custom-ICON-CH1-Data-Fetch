import datetime
import os, sys
# Set GRIB definitions so ogd_api doesn't complain if it needs them
COSMO_DEFS = r"C:\Users\sebas\.conda\envs\weather_final\share\eccodes-cosmo-resources\definitions"
if os.path.exists(COSMO_DEFS):
    os.environ["GRIB_DEFINITION_PATH"] = COSMO_DEFS
    os.environ["ECCODES_DEFINITION_PATH"] = COSMO_DEFS

from meteodatalab import ogd_api

ref_time = datetime.datetime(2026, 1, 21, 18, 0, tzinfo=datetime.timezone.utc)
req = ogd_api.Request(collection="ogd-forecasting-icon-ch1", variable="T",
                     reference_datetime=ref_time, horizon="P0DT22H", perturbed=False)

print("Calling get_asset_urls...")
urls = ogd_api.get_asset_urls(req)
print(f"URLs found: {len(urls)}")
