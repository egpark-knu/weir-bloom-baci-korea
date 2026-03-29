# Satellite-Based BACI Analysis of Weir-Induced Algal Bloom Intensification in South Korea

Code and data for:

> **Park, E.G.** (2026). *Satellite-based quasi-experimental evidence of weir-induced algal bloom intensification across South Korea's Four Major Rivers*. Submitted to *Water Research*.

## Overview

This repository provides the analysis pipeline for a 25-year (2000–2025) satellite-based Before–After Control–Impact (BACI) event-study assessment of algal bloom dynamics at South Korea's 16 Four Major Rivers Restoration Project (FMRP) weirs.

**Key findings:**
- Weir construction was associated with a pooled average increase of **+0.012 NDVI (+17%)** in bloom-season bloom proxy intensity
- 11 of 16 weirs showed positive treatment effects; 6 were individually significant (*p* < 0.05)
- The effect persisted over 13 years with no significant attenuation
- Robustness confirmed via 9 independent tests (parallel trends, permutation placebo, leave-one-out, cross-sensor validation, winter negative control, BSTS CausalImpact, etc.)

## Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Data Extraction (Google Earth Engine)                       │
│     gee_bloom_pipeline.py  — Bloom index definitions & masking  │
│     extract_water_masked.py — JRC water-masked NDVI/NDCI        │
│     extract_all_weirs.py    — 16 weir treatment sites           │
│     extract_control_reaches.py — 16 upstream control sites      │
│     extract_winter_control.py  — Winter negative-control data   │
│                                                                 │
│  2. BACI Analysis                                               │
│     baci_analysis.py       — DiD estimation + event-study       │
│     full_analysis.py       — Complete pipeline (extract→analyze) │
│                                                                 │
│  3. Robustness & Sensitivity                                    │
│     robustness_analysis.py  — 9-test robustness battery         │
│     sensitivity_analysis.py — Leave-one-out, dose-response      │
│     causal_impact_analysis.py — Bayesian structural time-series │
│     climate_confounders.py  — Climate covariate analysis        │
│     analyze_winter_control.py — Winter negative-control test    │
│                                                                 │
│  4. Supplementary                                               │
│     create_study_area_map.py — Figure 1 study area map          │
│     cem_matching.py         — Coarsened exact matching           │
│     analyze_all_weirs.py    — Per-weir descriptive analysis     │
│     susceptibility_index.py — Bloom susceptibility scoring      │
└─────────────────────────────────────────────────────────────────┘
```

## Data

All satellite data are publicly available through Google Earth Engine:

| Dataset | GEE Collection ID | Resolution | Period |
|---------|-------------------|------------|--------|
| Landsat 5 TM | `LANDSAT/LT05/C02/T1_L2` | 30 m | 2000–2011 |
| Landsat 7 ETM+ | `LANDSAT/LE07/C02/T1_L2` | 30 m | 2000–2024 |
| Landsat 8 OLI | `LANDSAT/LC08/C02/T1_L2` | 30 m | 2013–2025 |
| Landsat 9 OLI-2 | `LANDSAT/LC09/C02/T1_L2` | 30 m | 2022–2025 |
| Sentinel-2 MSI | `COPERNICUS/S2_SR_HARMONIZED` | 10 m | 2017–2025 |
| JRC Global Surface Water | `JRC/GSW1_4/GlobalSurfaceWater` | 30 m | — |

### Weir Inventory

`data/weir_inventory.json` contains the 16 FMRP weirs with coordinates, completion dates, and structural specifications. `data/control_reaches.json` defines the 16 paired upstream control sites (15–20 km above each weir, beyond the backwater influence zone).

## Quick Start

### 1. Prerequisites

```bash
pip install -r requirements.txt
earthengine authenticate   # one-time GEE authentication
```

### 2. Configure

Edit `config.py` to set your GEE project ID:

```python
GEE_PROJECT = "your-gee-project-id"
```

### 3. Run the Example

```bash
python examples/gee_data_retrieval.py
```

This extracts bloom-season NDVI for the Hapcheon Weir (Nakdong River) and its paired upstream control for 2000–2025, demonstrating the data retrieval pipeline.

### 4. Full Analysis

```bash
# Extract water-masked bloom data for all 32 sites
python scripts/extract_water_masked.py --target all

# Run BACI analysis
python scripts/baci_analysis.py

# Run robustness battery
python scripts/robustness_analysis.py

# Run Bayesian CausalImpact
python scripts/causal_impact_analysis.py
```

## Study Design

- **Design:** Before–After Control–Impact (BACI) with 16 treatment–control pairs
- **Treatment sites:** 5-km-radius buffer around each weir centroid
- **Control sites:** 15–20 km upstream, beyond estimated backwater zone
- **Pre-period:** 2000–2011 (12 years)
- **Post-period:** 2013–2025 (13 years)
- **Transition year:** 2012 (excluded)
- **Bloom proxy:** Bloom-season (May–Oct) median NDVI on water pixels
- **Cross-validation:** Sentinel-2 NDCI (2017–2025)

## Robustness Battery

| Test | Purpose |
|------|---------|
| Parallel-trends verification | Validate pre-intervention comparability |
| Detrended event-study | Remove pre-existing trend influence |
| Permutation placebo (*n*=1,000) | Test against random treatment dates |
| Leave-one-out (*n*=16) | Check sensitivity to individual weirs |
| Cross-sensor validation | Landsat NDVI vs. Sentinel-2 NDCI agreement |
| Winter negative control | Confirm seasonal specificity |
| Temporal persistence | Test for effect attenuation over time |
| BSTS CausalImpact | Independent Bayesian estimator |
| Dose–response | Storage capacity vs. effect magnitude |

## Citation

```
Park, E.G. (2026). Satellite-based quasi-experimental evidence of weir-induced
algal bloom intensification across South Korea's Four Major Rivers.
Water Research (submitted).
```

## License

MIT License — see [LICENSE](LICENSE).
