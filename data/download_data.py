"""Download the raw dataset into data/raw/ (run from the repo root: python data/download_data.py).

Source: UCI Machine Learning Repository, "Bike Sharing" (Fanaee-T & Gama, 2013), hour.csv.
This copy is mirrored in Allen Downey's bayesian-analysis-recipes repository.
"""
from pathlib import Path
from urllib.request import urlretrieve

FILES = {
    "hour.csv": "https://raw.githubusercontent.com/AllenDowney/bayesian-analysis-recipes/master/datasets/bikes/hour.csv",
}

raw = Path(__file__).parent / "raw"
raw.mkdir(exist_ok=True)
for name, url in FILES.items():
    print(f"Downloading {name} ...")
    urlretrieve(url, raw / name)
print("Done.")
