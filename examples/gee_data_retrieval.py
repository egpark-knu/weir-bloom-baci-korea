"""
Example: Retrieving Bloom Proxy Data from Google Earth Engine
=============================================================
Demonstrates how to extract bloom-season median NDVI for a single
weir–control pair using the Landsat Collection 2 archive.

Prerequisites:
    pip install earthengine-api
    earthengine authenticate   # one-time browser authentication

This script extracts water-masked, cloud-free NDVI for:
  - Hapcheon Weir (Nakdong River) — treatment site
  - Hapcheon Upstream — paired control site
  - Years 2000–2025, bloom season (May–October)
"""

import ee
import json

# ─── Initialize GEE ───
ee.Authenticate()     # opens browser; skip if already authenticated
ee.Initialize(project="your-gee-project-id")   # replace with your project ID

# ─── Site Definition ───
# Hapcheon Weir (example): 5-km-radius buffer around weir centroid
WEIR_SITE = {
    "name": "Hapcheon",
    "lat": 35.5641,
    "lon": 128.1728,
    "radius_m": 5000,
}

# Paired upstream control: 15–20 km above weir, beyond backwater zone
CONTROL_SITE = {
    "name": "Hapcheon_Control",
    "lat": 35.6451,
    "lon": 128.0528,
    "radius_m": 5000,
}


# ─── Helper Functions ───

def make_buffer(site):
    """Create a circular buffer geometry from site definition."""
    return ee.Geometry.Point([site["lon"], site["lat"]]).buffer(site["radius_m"])


def mask_landsat_sr(image):
    """Mask clouds and cloud shadows using CFMask QA_PIXEL band."""
    qa = image.select("QA_PIXEL")
    cloud_shadow = qa.bitwiseAnd(1 << 3).eq(0)
    cloud = qa.bitwiseAnd(1 << 4).eq(0)
    return image.updateMask(cloud_shadow).updateMask(cloud)


def apply_water_mask(image, water_mask):
    """Restrict analysis to persistent water pixels (JRC GSW >= 50%)."""
    return image.updateMask(water_mask)


def compute_ndvi(image, red_band, nir_band, scale, offset):
    """Compute NDVI from scaled surface reflectance."""
    red = image.select(red_band).multiply(scale).add(offset)
    nir = image.select(nir_band).multiply(scale).add(offset)
    return image.addBands(nir.subtract(red).divide(nir.add(red)).rename("NDVI"))


# ─── JRC Global Surface Water Mask ───
# Pekel et al. (2016), Nature — pixels with >= 50% water occurrence
jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
water_mask = jrc.select("occurrence").gte(50)

# ─── Landsat Collections ───
COLLECTIONS = {
    "L5": {
        "id": "LANDSAT/LT05/C02/T1_L2",
        "years": range(2000, 2012),
        "red": "SR_B3", "nir": "SR_B4",
        "scale": 0.0000275, "offset": -0.2,
    },
    "L7": {
        "id": "LANDSAT/LE07/C02/T1_L2",
        "years": range(2000, 2024),
        "red": "SR_B3", "nir": "SR_B4",
        "scale": 0.0000275, "offset": -0.2,
    },
    "L8": {
        "id": "LANDSAT/LC08/C02/T1_L2",
        "years": range(2013, 2026),
        "red": "SR_B4", "nir": "SR_B5",
        "scale": 0.0000275, "offset": -0.2,
    },
    "L9": {
        "id": "LANDSAT/LC09/C02/T1_L2",
        "years": range(2022, 2026),
        "red": "SR_B4", "nir": "SR_B5",
        "scale": 0.0000275, "offset": -0.2,
    },
}


def extract_bloom_season_ndvi(site, year):
    """
    Extract bloom-season (May–Oct) median NDVI for a site in a given year.

    Returns:
        dict with keys: site, year, ndvi_median, n_obs
    """
    geom = make_buffer(site)
    start = f"{year}-05-01"
    end = f"{year}-11-01"

    # Merge all available Landsat sensors for this year
    images = ee.ImageCollection([])
    for key, col in COLLECTIONS.items():
        if year in col["years"]:
            ic = (
                ee.ImageCollection(col["id"])
                .filterDate(start, end)
                .filterBounds(geom)
                .map(mask_landsat_sr)
                .map(lambda img, c=col: compute_ndvi(
                    img, c["red"], c["nir"], c["scale"], c["offset"]
                ))
                .map(lambda img: apply_water_mask(img, water_mask))
                .select("NDVI")
            )
            images = images.merge(ic)

    # Compute bloom-season median
    n_obs = images.size().getInfo()
    if n_obs == 0:
        return {"site": site["name"], "year": year, "ndvi_median": None, "n_obs": 0}

    median_ndvi = (
        images.median()
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=30,
            maxPixels=1e9,
        )
        .get("NDVI")
        .getInfo()
    )

    return {
        "site": site["name"],
        "year": year,
        "ndvi_median": round(median_ndvi, 6) if median_ndvi else None,
        "n_obs": n_obs,
    }


# ─── Main Extraction ───
if __name__ == "__main__":
    results = []

    for year in range(2000, 2026):
        print(f"Processing {year}...", end=" ")

        weir_result = extract_bloom_season_ndvi(WEIR_SITE, year)
        ctrl_result = extract_bloom_season_ndvi(CONTROL_SITE, year)

        results.append(weir_result)
        results.append(ctrl_result)

        w = weir_result["ndvi_median"]
        c = ctrl_result["ndvi_median"]
        diff = (w - c) if (w and c) else None
        print(f"Weir={w}, Control={c}, Diff={diff}")

    # Save to JSON
    output_path = "hapcheon_bloom_timeseries.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} records to {output_path}")
    print("\nExpected pattern:")
    print("  Pre-weir (2000-2011): Weir-Control difference ~ 0")
    print("  Post-weir (2013-2025): Weir-Control difference > 0 (bloom intensification)")
