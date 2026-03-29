"""
Extract bloom (NDCI proxy) time series for ALL 16 weirs.
==========================================================
Replicates the pilot approach (pilot_hapcheon_bloom.json) across
every weir in weir_inventory.json.

Sensors:
  - Landsat 5/7 (2000-2012, pre-weir baseline)
  - Landsat 8/9 (2013-2025, post-weir)
  - Sentinel-2   (2017-2025, true NDCI)

Output:
  - Per-weir JSON  -> output/bloom_data/<name_en>.json
  - Combined CSV   -> output/all_weirs_bloom_summary.csv

Usage:
    python extract_all_weirs.py
    python extract_all_weirs.py --test          # first 2 weirs only
    python extract_all_weirs.py --weir 합천창녕보  # single weir
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import ee

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
INVENTORY_PATH = BASE_DIR / "data" / "weir_inventory.json"
OUTPUT_DIR = BASE_DIR / "output" / "bloom_data"
COMBINED_CSV = BASE_DIR / "output" / "all_weirs_bloom_summary.csv"

# ── GEE Auth ───────────────────────────────────────────────────
SERVICE_ACCOUNT = "your-service-account@project.iam.gserviceaccount.com"
KEY_PATH = (
    "your-gee-service-key.json"  # Replace with your GEE service key path
    
)
GEE_PROJECT = "your-gee-project-id"

# ── Constants ──────────────────────────────────────────────────
BUFFER_M = 5000
BLOOM_MONTHS = (5, 10)  # May-October inclusive
LANDSAT_PRE_RANGE = (2000, 2012)
LANDSAT_POST_RANGE = (2013, 2025)
SENTINEL2_RANGE = (2017, 2025)
SCALE_LANDSAT = 30   # metres
SCALE_S2 = 20        # metres
MAX_RETRIES = 1


# ── GEE Initialization ────────────────────────────────────────

def init_gee() -> None:
    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_PATH)
    ee.Initialize(credentials=credentials, project=GEE_PROJECT)
    print("[GEE] Initialized successfully.")


# ── Cloud Masking ──────────────────────────────────────────────

def mask_landsat(image: ee.Image) -> ee.Image:
    """Mask clouds (bit 4) and cloud shadow (bit 3) via QA_PIXEL."""
    qa = image.select("QA_PIXEL")
    mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    return image.updateMask(mask)


def mask_s2(image: ee.Image) -> ee.Image:
    """Mask opaque clouds (bit 10) and cirrus (bit 11) via QA60."""
    qa = image.select("QA60")
    mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return image.updateMask(mask)


# ── NDCI Computation ──────────────────────────────────────────

def add_ndci_landsat57(image: ee.Image) -> ee.Image:
    """Landsat 5/7: NDCI proxy = (NIR - Red) / (NIR + Red).
    SR_B3 = red, SR_B4 = nir. scale=0.0000275, offset=-0.2
    """
    red = image.select("SR_B3").multiply(0.0000275).add(-0.2)
    nir = image.select("SR_B4").multiply(0.0000275).add(-0.2)
    ndci = nir.subtract(red).divide(nir.add(red)).rename("NDCI")
    return image.addBands(ndci)


def add_ndci_landsat89(image: ee.Image) -> ee.Image:
    """Landsat 8/9: NDCI proxy = (NIR - Red) / (NIR + Red).
    SR_B4 = red, SR_B5 = nir. scale=0.0000275, offset=-0.2
    """
    red = image.select("SR_B4").multiply(0.0000275).add(-0.2)
    nir = image.select("SR_B5").multiply(0.0000275).add(-0.2)
    ndci = nir.subtract(red).divide(nir.add(red)).rename("NDCI")
    return image.addBands(ndci)


def add_ndci_s2(image: ee.Image) -> ee.Image:
    """Sentinel-2: true NDCI = (B5 - B4) / (B5 + B4).
    B4 = red 665nm, B5 = rededge 705nm. scale=0.0001
    """
    red = image.select("B4").multiply(0.0001)
    re1 = image.select("B5").multiply(0.0001)
    ndci = re1.subtract(red).divide(re1.add(red)).rename("NDCI")
    return image.addBands(ndci)


# ── Collection Builders ───────────────────────────────────────

def _bloom_filter(col: ee.ImageCollection, year: int) -> ee.ImageCollection:
    """Filter to bloom season (May-Oct) of a given year."""
    return (col
            .filterDate(f"{year}-05-01", f"{year}-11-01")
            .filter(ee.Filter.calendarRange(BLOOM_MONTHS[0], BLOOM_MONTHS[1], "month")))


def get_landsat57_col(geom: ee.Geometry, year: int) -> ee.ImageCollection:
    """Merged Landsat 5 + 7 collection for one bloom season."""
    l5 = (ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
          .filterBounds(geom).map(mask_landsat).map(add_ndci_landsat57))
    l7 = (ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
          .filterBounds(geom).map(mask_landsat).map(add_ndci_landsat57))
    merged = l5.merge(l7)
    return _bloom_filter(merged, year)


def get_landsat89_col(geom: ee.Geometry, year: int) -> ee.ImageCollection:
    """Merged Landsat 8 + 9 collection for one bloom season."""
    l8 = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
          .filterBounds(geom).map(mask_landsat).map(add_ndci_landsat89))
    l9 = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
          .filterBounds(geom).map(mask_landsat).map(add_ndci_landsat89))
    merged = l8.merge(l9)
    return _bloom_filter(merged, year)


def get_s2_col(geom: ee.Geometry, year: int) -> ee.ImageCollection:
    """Sentinel-2 SR Harmonized collection for one bloom season."""
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(geom)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
           .map(mask_s2)
           .map(add_ndci_s2))
    return _bloom_filter(col, year)


# ── Stats Extraction ──────────────────────────────────────────

def _combined_reducer() -> ee.Reducer:
    """median + mean + stdDev + count in one call."""
    return (ee.Reducer.median()
            .combine(ee.Reducer.mean(), sharedInputs=True)
            .combine(ee.Reducer.stdDev(), sharedInputs=True)
            .combine(ee.Reducer.count(), sharedInputs=True))


def extract_stats(col: ee.ImageCollection, geom: ee.Geometry,
                  scale: int) -> dict:
    """Reduce an NDCI image collection to spatial-temporal stats.

    Returns dict with keys: NDCI_median, NDCI_mean, NDCI_stdDev, NDCI_count
    (or None values if collection is empty).
    """
    composite = col.select("NDCI").reduce(_combined_reducer())
    raw = composite.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=scale,
        maxPixels=1e8,
    ).getInfo()
    return raw


def parse_stats(raw: dict | None) -> dict:
    """Normalise GEE reducer output to clean keys."""
    if raw is None:
        return {"median": None, "mean": None, "std": None, "count": 0}
    return {
        "median": raw.get("NDCI_median"),
        "mean":   raw.get("NDCI_mean"),
        "std":    raw.get("NDCI_stdDev"),
        "count":  int(raw.get("NDCI_count") or 0),
    }


# ── Per-Weir Extraction ──────────────────────────────────────

def extract_one_weir(weir: dict) -> dict:
    """Extract full time series for a single weir.

    Returns a dict matching the pilot JSON structure:
    {
        weir, lat, lon, buffer_m, completion_year,
        landsat_pre_weir: [{year, count, median, mean, std}, ...],
        landsat_post_weir: [...],
        sentinel2: [...]
    }
    """
    name_kr = weir["name_kr"]
    lat, lon = weir["lat"], weir["lon"]
    comp_year = weir["completion_year"]

    geom = ee.Geometry.Point([lon, lat]).buffer(BUFFER_M)

    result = {
        "weir": name_kr,
        "lat": lat,
        "lon": lon,
        "buffer_m": BUFFER_M,
        "completion_year": comp_year,
        "landsat_pre_weir": [],
        "landsat_post_weir": [],
        "sentinel2": [],
    }

    # --- Landsat pre-weir (2000-2012): L5/L7 ---
    for year in range(LANDSAT_PRE_RANGE[0], LANDSAT_PRE_RANGE[1] + 1):
        raw = _safe_extract(
            lambda y=year: extract_stats(get_landsat57_col(geom, y), geom, SCALE_LANDSAT),
            label=f"{name_kr}/landsat_pre/{year}",
        )
        stats = parse_stats(raw)
        result["landsat_pre_weir"].append({"year": year, **stats})

    # --- Landsat post-weir (2013-2025): L8/L9 ---
    for year in range(LANDSAT_POST_RANGE[0], LANDSAT_POST_RANGE[1] + 1):
        raw = _safe_extract(
            lambda y=year: extract_stats(get_landsat89_col(geom, y), geom, SCALE_LANDSAT),
            label=f"{name_kr}/landsat_post/{year}",
        )
        stats = parse_stats(raw)
        result["landsat_post_weir"].append({"year": year, **stats})

    # --- Sentinel-2 (2017-2025) ---
    for year in range(SENTINEL2_RANGE[0], SENTINEL2_RANGE[1] + 1):
        raw = _safe_extract(
            lambda y=year: extract_stats(get_s2_col(geom, y), geom, SCALE_S2),
            label=f"{name_kr}/sentinel2/{year}",
        )
        stats = parse_stats(raw)
        result["sentinel2"].append({"year": year, **stats})

    return result


# ── Error Handling ────────────────────────────────────────────

def _safe_extract(fn, label: str = ""):
    """Call fn() with one retry on GEE errors."""
    for attempt in range(1 + MAX_RETRIES):
        try:
            return fn()
        except ee.ee_exception.EEException as exc:
            if attempt < MAX_RETRIES:
                print(f"  [WARN] {label}: GEE error, retrying in 5s... ({exc})")
                time.sleep(5)
            else:
                print(f"  [ERROR] {label}: skipped after {MAX_RETRIES + 1} attempts. {exc}")
                return None
        except Exception as exc:
            if attempt < MAX_RETRIES:
                print(f"  [WARN] {label}: error, retrying in 5s... ({exc})")
                time.sleep(5)
            else:
                print(f"  [ERROR] {label}: skipped. {exc}")
                traceback.print_exc()
                return None


# ── CSV Builder ───────────────────────────────────────────────

CSV_COLUMNS = [
    "weir", "year", "sensor", "ndci_median", "ndci_mean", "ndci_std",
    "count", "period",
]


def weir_result_to_rows(wr: dict) -> list[dict]:
    """Flatten a per-weir result dict into CSV rows."""
    rows = []
    comp = wr["completion_year"]

    for entry in wr["landsat_pre_weir"]:
        rows.append({
            "weir": wr["weir"],
            "year": entry["year"],
            "sensor": "landsat57",
            "ndci_median": entry["median"],
            "ndci_mean": entry["mean"],
            "ndci_std": entry["std"],
            "count": entry["count"],
            "period": "pre" if entry["year"] < comp else "post",
        })

    for entry in wr["landsat_post_weir"]:
        rows.append({
            "weir": wr["weir"],
            "year": entry["year"],
            "sensor": "landsat89",
            "ndci_median": entry["median"],
            "ndci_mean": entry["mean"],
            "ndci_std": entry["std"],
            "count": entry["count"],
            "period": "pre" if entry["year"] < comp else "post",
        })

    for entry in wr["sentinel2"]:
        rows.append({
            "weir": wr["weir"],
            "year": entry["year"],
            "sensor": "sentinel2",
            "ndci_median": entry["median"],
            "ndci_mean": entry["mean"],
            "ndci_std": entry["std"],
            "count": entry["count"],
            "period": "pre" if entry["year"] < comp else "post",
        })

    return rows


# ── Main Pipeline ─────────────────────────────────────────────

def run_pipeline(weirs: list[dict]) -> None:
    """Process all weirs and save outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_csv_rows: list[dict] = []
    total = len(weirs)

    for idx, weir in enumerate(weirs, 1):
        name_kr = weir["name_kr"]
        name_en = weir["name_en"]
        print(f"\n{'='*60}")
        print(f"[{idx}/{total}] {name_kr} ({name_en})")
        print(f"  lat={weir['lat']}, lon={weir['lon']}, "
              f"completion={weir['completion_year']}")
        print(f"{'='*60}")

        t0 = time.time()

        try:
            result = extract_one_weir(weir)
        except Exception as exc:
            print(f"  [FATAL] {name_kr}: entire weir skipped. {exc}")
            traceback.print_exc()
            continue

        elapsed = time.time() - t0

        # Save per-weir JSON
        safe_name = name_en.replace(" ", "_").replace("-", "_")
        json_path = OUTPUT_DIR / f"{safe_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Count valid observations
        n_pre = sum(1 for e in result["landsat_pre_weir"] if e["count"] > 0)
        n_post = sum(1 for e in result["landsat_post_weir"] if e["count"] > 0)
        n_s2 = sum(1 for e in result["sentinel2"] if e["count"] > 0)
        print(f"  Saved: {json_path.name}")
        print(f"  Valid years: pre={n_pre}, post={n_post}, S2={n_s2}")
        print(f"  Elapsed: {elapsed:.1f}s")

        # Accumulate CSV rows
        all_csv_rows.extend(weir_result_to_rows(result))

    # Save combined CSV
    COMBINED_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(COMBINED_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(all_csv_rows)

    print(f"\n{'='*60}")
    print(f"Pipeline complete.")
    print(f"  Per-weir JSONs: {OUTPUT_DIR}/")
    print(f"  Combined CSV:   {COMBINED_CSV}")
    print(f"  Total rows:     {len(all_csv_rows)}")
    print(f"{'='*60}")


# ── CLI ───────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract bloom NDCI time series for all 16 weirs via GEE",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Process only the first 2 weirs (quick validation)",
    )
    parser.add_argument(
        "--weir", type=str, default=None,
        help="Process a single weir by Korean name (e.g. 합천창녕보)",
    )
    parser.add_argument(
        "--inventory", type=str, default=str(INVENTORY_PATH),
        help="Path to weir_inventory.json",
    )
    args = parser.parse_args()

    # Load inventory
    with open(args.inventory, encoding="utf-8") as f:
        weirs = json.load(f)

    # Filter
    if args.weir:
        weirs = [w for w in weirs if w["name_kr"] == args.weir]
        if not weirs:
            print(f"[ERROR] Weir '{args.weir}' not found in inventory.")
            sys.exit(1)
    elif args.test:
        weirs = weirs[:2]

    print(f"[INFO] Will process {len(weirs)} weir(s).")

    # Init GEE
    init_gee()

    # Run
    run_pipeline(weirs)


if __name__ == "__main__":
    main()
