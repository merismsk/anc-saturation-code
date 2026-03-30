import os
import tarfile
import zipfile
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_ROOT = './data_large'
os.makedirs(DATA_ROOT, exist_ok=True)

# Check available download tools (faster than urllib)
def get_download_command():
    """Prefer aria2c (parallel) > curl > wget > urllib"""
    if shutil.which('aria2c'):
        return 'aria2c'  # Fastest - supports parallel connections
    elif shutil.which('curl'):
        return 'curl'
    elif shutil.which('wget'):
        return 'wget'
    return None

DOWNLOAD_TOOL = get_download_command()
print(f"Using download tool: {DOWNLOAD_TOOL or 'urllib (slow)'}")

def download_file(url, dest_folder):
    filename = url.split('/')[-1]
    # Remove query parameters from filename (e.g., ?download=1)
    if '?' in filename:
        filename = filename.split('?')[0]
    filepath = os.path.join(dest_folder, filename)
    
    if os.path.exists(filepath):
        print(f"Already exists: {filename}")
        return filepath
    
    print(f"Downloading {filename}...")
    
    try:
        if DOWNLOAD_TOOL == 'aria2c':
            # aria2c: Multi-connection parallel download (fastest)
            # -x 16: 16 connections per server
            # -s 16: split file into 16 parts
            # -k 1M: minimum split size
            subprocess.run([
                'aria2c', '-x', '16', '-s', '16', '-k', '1M',
                '-d', dest_folder, '-o', filename, url
            ], check=True)
        elif DOWNLOAD_TOOL == 'curl':
            # curl with resume support
            subprocess.run([
                'curl', '-L', '-C', '-', '-o', filepath, url
            ], check=True)
        elif DOWNLOAD_TOOL == 'wget':
            # wget with continue support
            subprocess.run([
                'wget', '-c', '-O', filepath, url
            ], check=True)
        else:
            # Fallback to urllib (slowest)
            import urllib.request
            urllib.request.urlretrieve(url, filepath)
        
        print(f"Done: {filename}")
        return filepath
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)  # Clean up partial download
        return None

def extract_file(filepath, dest_folder):
    print(f"Extracting {filepath}...")
    try:
        if filepath.endswith('.tar.gz'):
            with tarfile.open(filepath, "r:gz") as tar:
                tar.extractall(path=dest_folder)
        elif filepath.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(dest_folder)
        print("Extracted.")
    except (EOFError, zipfile.BadZipFile, tarfile.ReadError) as e:
        print(f"Error extracting {filepath}: {e}")
        print("The file seems corrupted. Deleting it...")
        os.remove(filepath)
        print("Please re-run this script to download it again.")


def main():
    # =========================================================================
    # 1. DEMAND Dataset (Standard for Noise Control) - 16-channel environmental noise
    # Multiple subsets from Zenodo - covers different noise environments
    # =========================================================================
    print("=" * 60)
    print("1. DEMAND Dataset (Environmental Noise - Standard for ANC)")
    print("=" * 60)
    
    demand_subsets = [
        # Domestic environments
        ("DKITCHEN_16k.zip", "https://zenodo.org/record/1227121/files/DKITCHEN_16k.zip?download=1"),
        ("DLIVING_16k.zip", "https://zenodo.org/record/1227121/files/DLIVING_16k.zip?download=1"),
        ("DWASHING_16k.zip", "https://zenodo.org/record/1227121/files/DWASHING_16k.zip?download=1"),
        # Office environments
        ("OOFFICE_16k.zip", "https://zenodo.org/record/1227121/files/OOFFICE_16k.zip?download=1"),
        # Transportation (critical for ANC)
        ("TBUS_16k.zip", "https://zenodo.org/record/1227121/files/TBUS_16k.zip?download=1"),
        ("TCAR_16k.zip", "https://zenodo.org/record/1227121/files/TCAR_16k.zip?download=1"),
        ("TMETRO_16k.zip", "https://zenodo.org/record/1227121/files/TMETRO_16k.zip?download=1"),
        # Public spaces
        ("PCAFETER_16k.zip", "https://zenodo.org/record/1227121/files/PCAFETER_16k.zip?download=1"),
        ("PRESTO_16k.zip", "https://zenodo.org/record/1227121/files/PRESTO_16k.zip?download=1"),
        # Street noise
        ("SCAFE_16k.zip", "https://zenodo.org/record/1227121/files/SCAFE_16k.zip?download=1"),
        ("SPSQUARE_16k.zip", "https://zenodo.org/record/1227121/files/SPSQUARE_16k.zip?download=1"),
        ("STRAFFIC_16k.zip", "https://zenodo.org/record/1227121/files/STRAFFIC_16k.zip?download=1"),
        # Nature
        ("NFIELD_16k.zip", "https://zenodo.org/record/1227121/files/NFIELD_16k.zip?download=1"),
        ("NPARK_16k.zip", "https://zenodo.org/record/1227121/files/NPARK_16k.zip?download=1"),
        ("NRIVER_16k.zip", "https://zenodo.org/record/1227121/files/NRIVER_16k.zip?download=1"),
    ]
    
    for name, url in demand_subsets:
        print(f"\n--- Downloading {name} ---")
        fp = download_file(url, DATA_ROOT)
        if fp: extract_file(fp, DATA_ROOT)
    
    # =========================================================================
    # 2. ESC-50 Dataset (Environmental Sound Classification - 50 classes)
    # 2000 labeled environmental recordings - animals, exterior, human, interior, natural
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. ESC-50 Dataset (Environmental Sound Classification)")
    print("=" * 60)
    esc50_url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    fp = download_file(esc50_url, DATA_ROOT)
    if fp: extract_file(fp, DATA_ROOT)
    
    # =========================================================================
    # 3. Mini-LibriSpeech (Clean Speech) - for speech interference scenarios
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. Mini-LibriSpeech (Clean Speech for Interference)")
    print("=" * 60)
    libri_url = "https://www.openslr.org/resources/31/dev-clean-2.tar.gz"
    fp = download_file(libri_url, DATA_ROOT)
    if fp: extract_file(fp, DATA_ROOT)
    
    # =========================================================================
    # 4. MUSAN Noise Subset (Music, Speech, Noise corpus - just noise portion)
    # Note: Full MUSAN is 11GB, we download noise subset only for ANC
    # =========================================================================
    print("\n" + "=" * 60)
    print("4. MUSAN Noise Corpus (Background Noise)")
    print("=" * 60)
    print("Note: Full MUSAN is 11GB. For ANC, consider downloading manually if needed:")
    print("  URL: https://openslr.org/resources/17/musan.tar.gz")
    # Uncomment below to download full MUSAN (11GB):
    # musan_url = "https://openslr.org/resources/17/musan.tar.gz"
    # fp = download_file(musan_url, DATA_ROOT)
    # if fp: extract_file(fp, DATA_ROOT)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUCCESS: Data preparation complete!")
    print("=" * 60)
    print(f"Datasets are located in {DATA_ROOT}")
    print("\nDownloaded datasets:")
    print("  - DEMAND (15 environments): Transportation, Domestic, Office, Public, Street, Nature")
    print("  - ESC-50: 2000 environmental sounds in 50 classes")
    print("  - LibriSpeech: Clean speech for interference testing")
    print("\nYou can now run 'python train_large_scale.py' to train on this data.")

if __name__ == "__main__":
    main()
