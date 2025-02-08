import os
import requests
import zipfile
import sys
from pathlib import Path
from tqdm import tqdm

# Get the absolute path to the project root
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from config import ProjectConfig
except ImportError as e:
    print(f"Error importing config: {e}")
    print(f"Python path: {sys.path}")
    print(f"Project root: {PROJECT_ROOT}")
    sys.exit(1)

def download_opus_dataset(config: ProjectConfig):
    """Download parallel corpus from OPUS"""
    try:
        # Using CCAligned corpus for Hindi-English
        base_url = "https://object.pouta.csc.fi/OPUS-CCAligned/v1/moses"
        files_to_download = [
            f"{base_url}/en-hi.txt.zip",
            f"{base_url}/hi-en.txt.zip"
        ]
        
        # Create data directory if it doesn't exist
        data_dir = Path(config.DATA_DIR) / 'opus'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        for url in files_to_download:
            filename = url.split('/')[-1]
            file_path = data_dir / filename
            
            # Download file
            print(f"Downloading from: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error for bad status codes
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as file, tqdm(
                desc=f"Downloading {filename}",
                total=total_size,
                unit='iB',
                unit_scale=True
            ) as pbar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    pbar.update(size)
            
            # Extract files
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
                
        print(f"Datasets downloaded and extracted to {data_dir}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")
        print("Available OPUS datasets can be found at: http://opus.nlpl.eu/CCAligned.php")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    config = ProjectConfig()
    download_opus_dataset(config) 