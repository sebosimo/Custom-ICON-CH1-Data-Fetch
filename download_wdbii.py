
import os
import requests
import zipfile
import io
import shutil

# Target directory
CARTOPY_DIR = os.path.expanduser('~/.local/share/cartopy/shapefiles')
WDBII_DIR = os.path.join(CARTOPY_DIR, 'wdbii')

# URL for GSHHG (contains WDBII)
# Using a specific version to ensure compatibility
URL = "http://www.soest.hawaii.edu/pwessel/gshhg/gshhg-shp-2.3.7.zip"

def main():
    print(f"Checking for WDBII data in {WDBII_DIR}...")
    
    # Check if we need to download
    # We need rivers, high resolution (h)
    # Expected path: wdbii/river/WDBII_river_h_L01.shp (structure usually inside zip is WDBII_shp/h/WDBII_river_h_L01.shp)
    
    # Let's just download and extract if the wdbii dir is empty or missing
    if os.path.exists(WDBII_DIR) and os.listdir(WDBII_DIR):
        print("WDBII directory exists and is not empty. Checking for river files...")
        # Check specifically for river h
        river_dir = os.path.join(WDBII_DIR, "river", "h")
        if os.path.exists(river_dir) and len(os.listdir(river_dir)) > 0:
             print("River files seem present. Skipping download.")
             return

    print(f"Downloading GSHHG/WDBII data from {URL}...")
    print("This might take a while (approx 170MB)...")
    
    try:
        r = requests.get(URL, stream=True)
        r.raise_for_status()
        
        print("Download complete. Extracting...")
        
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            # List files to verify structure
            # Structure in zip: gshhg-shp-2.3.7/WDBII_shp/h/WDBII_river_h_L01.shp
            
            for file in z.namelist():
                if "WDBII_shp" in file and "river" in file and "/h/" in file and file.endswith(".shp"):
                     # Extract this file
                     # We want to place it in: .../wdbii/river/h/
                     
                     filename = os.path.basename(file)
                     target_dir = os.path.join(WDBII_DIR, "river", "h")
                     os.makedirs(target_dir, exist_ok=True)
                     
                     target_path = os.path.join(target_dir, filename)
                     
                     print(f"Extracting {filename} to {target_path}")
                     with open(target_path, 'wb') as f:
                         f.write(z.read(file))
                         
                     # Also extract .dbf and .shx if they exist (usually do)
                     base_no_ext = os.path.splitext(file)[0]
                     for ext in ['.dbf', '.shx', '.prj']:
                         sibling = base_no_ext + ext
                         if sibling in z.namelist():
                             s_filename = os.path.basename(sibling)
                             s_target = os.path.join(target_dir, s_filename)
                             with open(s_target, 'wb') as sf:
                                 sf.write(z.read(sibling))

        print("Extraction complete.")
        
    except Exception as e:
        print(f"Download/Extraction failed: {e}")

if __name__ == "__main__":
    main()
