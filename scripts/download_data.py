"""Download the GPVS-Faults dataset (Bakdi et al., Mendeley Data, CC BY 4.0).

DOI: 10.17632/n76t439f65.1
The CSV archive is downloaded and extracted to data/raw/.
(The .mat files are MATLAB v7.3 tables and are not easily read outside MATLAB.)
"""
import zipfile
from pathlib import Path

import requests

DATASET_ID = "n76t439f65"
API_URL = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    meta = requests.get(API_URL, timeout=60).json()
    for f in meta["files"]:
        name = f["filename"]
        if name not in ("CSV_Files.zip", "ReadMe.pdf"):
            continue  # .mat files duplicate the CSV content
        out = RAW_DIR / name
        if out.exists() and out.stat().st_size == f["size"]:
            print(f"[skip] {name}")
            continue
        url = f["content_details"]["download_url"]
        print(f"[get ] {name} ({f['size'] / 1e6:.1f} MB)")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(out, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)

    zip_path = RAW_DIR / "CSV_Files.zip"
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            if member.lower().endswith(".csv") and not (RAW_DIR / Path(member).name).exists():
                with z.open(member) as src, open(RAW_DIR / Path(member).name, "wb") as dst:
                    dst.write(src.read())
    print("CSV files:", sorted(p.name for p in RAW_DIR.glob("*.csv")))


if __name__ == "__main__":
    main()
