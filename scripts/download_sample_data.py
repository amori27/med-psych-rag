#!/usr/bin/env python3
"""Downloads PMC Open Access sample articles for development."""

import requests
from pathlib import Path

SAMPLE_DOCS = {
    "gad_guidelines": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12345/pdf/",
    "depression_review": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC67890/pdf/",
}

DATA_DIR = Path("data/sample")
DATA_DIR.mkdir(parents=True, exist_ok=True)

for name, url in SAMPLE_DOCS.items():
    print(f"Downloading {name}...")
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            path = DATA_DIR / f"{name}.pdf"
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"  Saved to {path}")
        else:
            print(f"  Failed (HTTP {r.status_code})")
    except Exception as e:
        print(f"  Error: {e}")

print("Done.")
