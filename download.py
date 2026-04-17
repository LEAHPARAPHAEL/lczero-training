import os
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from tqdm import tqdm

def download_first_n_files(index_url, target_dir, n):
    """
    Downloads the first n files from a web directory index.
    """
    # 1. Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    print(f"Fetching index from: {index_url}")

    # 2. Fetch the HTML content of the index page
    try:
        response = requests.get(index_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch the URL: {e}")
        return

    # 3. Parse the HTML to find all the file links
    soup = BeautifulSoup(response.text, 'html.parser')
    file_links = []
    
    for a_tag in soup.find_all('a'):
        href = a_tag.get('href')
        # Filter out parent directories and sorting links
        if href and not href.startswith('?') and href != '../':
            # specifically target the .tar files based on your example
            if href.endswith('.tar'): 
                file_links.append(href)

    # 4. Limit to the first n files
    files_to_download = file_links[:n]
    
    if not files_to_download:
        print("No valid .tar files found on the page.")
        return

    print(f"Found {len(file_links)} files. Downloading the first {len(files_to_download)}...")

    # 5. Download each file with a progress bar
    for file_name in files_to_download:
        file_url = urljoin(index_url, file_name)
        
        # Clean up the filename in case there are URL parameters
        clean_file_name = os.path.basename(urlparse(file_url).path)
        file_path = os.path.join(target_dir, clean_file_name)

        # Skip if file already exists
        if os.path.exists(file_path):
            print(f"Skipping {clean_file_name} (already exists)")
            continue

        print(f"\nDownloading: {clean_file_name}")
        
        # Stream the download so we don't load a 500MB file entirely into RAM
        with requests.get(file_url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as f, tqdm(
                desc=clean_file_name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)

    print("\nAll downloads complete!")

# --- Usage Example ---
if __name__ == "__main__":
    # URL of the index site
    URL = "https://data.lczero.org/files/training_data/test91/"
    
    # Where you want to save the files (can be absolute or relative)
    DESTINATION_FOLDER = "./lczero_training_data"
    
    # Number of files you want to download
    NUMBER_OF_FILES = 3
    
    download_first_n_files(URL, DESTINATION_FOLDER, NUMBER_OF_FILES)