"""
HAB-Weir BACI Analysis — Configuration
=======================================
Central configuration for paths and analysis parameters.

Adjust PROJECT_DIR and GEE_PROJECT to match your local environment.
"""

from pathlib import Path

# ─── Project Paths ───
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"

# ─── Google Earth Engine ───
GEE_PROJECT = "your-gee-project-id"          # Replace with your GEE project
# GEE_SERVICE_KEY = "/path/to/service-key.json"  # Optional: for service-account auth
# GEE_SERVICE_ACCOUNT = "your-sa@project.iam.gserviceaccount.com"

# ─── Weir Inventory ───
WEIR_INVENTORY = DATA_DIR / "weir_inventory.json"
CONTROL_REACHES = DATA_DIR / "control_reaches.json"

# ─── Analysis Parameters ───
BLOOM_MONTHS = [5, 6, 7, 8, 9, 10]           # May–October (bloom season)
NEGATIVE_CONTROL_MONTHS = [12, 1, 2]          # Dec–Feb (winter negative control)
COMPLETION_YEAR = 2012                         # All 16 weirs completed by end of 2012

# ─── Sensor Periods ───
PERIODS = {
    "pre_weir": (2000, 2011),
    "transition": (2012, 2012),
    "post_weir": (2013, 2025),
    "sentinel2_overlap": (2017, 2025),
}

# ─── Output ───
OUTPUT_DIR = PROJECT_DIR / "output"
BLOOM_DATA_DIR = OUTPUT_DIR / "bloom_data"
FIGURES_DIR = OUTPUT_DIR / "figures"
