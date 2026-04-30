"""Download Global City Air Quality (hourly) from Kaggle via kagglehub."""
from pathlib import Path

import kagglehub

# Put files next to this script (override Kaggle cache location for this project)
OUT = Path(__file__).resolve().parent / "globaldata"
OUT.mkdir(parents=True, exist_ok=True)

path = kagglehub.dataset_download(
    "ibrahimqasimi/global-city-air-quality-hourly-data",
    output_dir=str(OUT),
)
print("Path to dataset files:", path)
