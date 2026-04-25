#
"""
Google Earth Engine Data Download Script - Indonesia Mangroves
Downloads 69-band training tiles (.tif) for Indonesia.

Two-pass approach:
  Pass 1: Scan all longitude strips, collect candidate tiles with quality
          metrics derived from ESA WorldCover class histograms.
  Pass 2: Apply quality filters, randomize, download top N tiles.

Quality Filters (tuned for Indonesia mangrove detection, ~20 km tiles):
  - Minimum mangrove coverage:   >= 0.1 % — thin coastal fringes on large cells
  - Maximum water coverage:      <= 80 %  — archipelago / long-coast scenes
  - Minimum land-cover classes:  >= 2     — mangrove + at least one other class
  - Maximum bare/built-up:       <= 30 %  — allow some peri-urban / bare coast

Band layout (69 dimensions per pixel):
  Bands  1-4  : Sentinel-2 optical (Red, Green, Blue, NIR)   — Features
  Bands  5-68 : Google Satellite Embeddings (64-dim)          — Features
  Band   69   : ESA WorldCover multi-class label              — Label

Datasets used:
  1. ESA WorldCover v200          — class labels (mangrove = 95, trees = 10, etc.)
  2. Copernicus S2_SR_HARMONIZED  — optical imagery (RGB + NIR)
     + GOOGLE/SATELLITE_EMBEDDING — 64-dim learned features
  3. FAO/GAUL/2015/level0         — country boundaries
"""

import ee
import time
import geemap
import os
import sys
import json
import random


# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
TOTAL_TILES = 100
RANDOM_SEED = 42

# Cache Pass-1 scan results so later runs skip straight to download.
# Delete this file (or set RESCAN=1) to force a fresh scan.
CANDIDATES_CACHE_PATH = os.path.join(os.getcwd(), 'candidates_cache.json')

# Quality filter thresholds (tuned for ~20 km grid cells + archipelago)
MIN_MANGROVE_PCT = 0.1       # >= 0.1% mangrove (~40 px of 100 m in a 20 km tile)
MAX_WATER_PCT = 80.0         # <= 80% ESA open water (class 80)
MIN_LAND_CLASSES = 2         # >= 2 distinct ESA classes in tile
MAX_BARE_BUILT_PCT = 30.0    # bare (60) + built-up (50) <= 30%

DEBUG_FIRST_FEATURE = True   # print first tile's properties once for inspection


def _normalize_esa_histogram(hist):
    """Coerce frequencyHistogram keys to int class codes (getInfo may use str or int)."""
    out = {}
    for k, v in hist.items():
        try:
            code = int(float(k))
        except (TypeError, ValueError):
            continue
        try:
            n = float(v)
        except (TypeError, ValueError):
            continue
        out[code] = out.get(code, 0.0) + n
    return out


def evaluate_tile_quality(features, esa_label_image):
    """Compute per-tile ESA class histograms and return tiles passing filters.

    Uses ee.Reducer.frequencyHistogram() so all class fractions and diversity
    are derived from a single server-side reduceRegions call per strip.

    Returns list of dicts with coords and quality metrics.
    """
    tile_histograms = esa_label_image.reduceRegions(
        collection=features,
        reducer=ee.Reducer.frequencyHistogram(),
        scale=100,
        tileScale=4,
    )

    tile_histograms = tile_histograms.map(
        lambda f: f.setGeometry(f.geometry().transform('EPSG:4326', 1))
    )

    tile_data = tile_histograms.toList(500).getInfo()

    global DEBUG_FIRST_FEATURE
    if DEBUG_FIRST_FEATURE and tile_data:
        sample_props = tile_data[0].get('properties', {})
        print(f"  [debug] first feature property keys: {list(sample_props.keys())}")
        print(f"  [debug] first feature properties: {sample_props}")
        DEBUG_FIRST_FEATURE = False

    passed = []
    for feat in tile_data:
        props = feat.get('properties', {})
        hist = props.get('lc_class') or props.get('histogram') or props.get('Map') or {}
        if not isinstance(hist, dict) or not hist:
            continue

        hist = _normalize_esa_histogram(hist)
        total = sum(hist.values())
        if total == 0:
            continue

        # ESA WorldCover v200 classes:
        #   10=Trees  20=Shrub  30=Grass  40=Crop  50=Built-up
        #   60=Bare   70=Snow   80=Water  90=Wetland  95=Mangrove  100=Moss
        mangrove_pct = hist.get(95, 0) / total * 100
        water_pct = hist.get(80, 0) / total * 100
        bare_built_pct = (hist.get(60, 0) + hist.get(50, 0)) / total * 100
        num_classes = len(hist)

        if mangrove_pct < MIN_MANGROVE_PCT:
            continue
        if water_pct > MAX_WATER_PCT:
            continue
        if bare_built_pct > MAX_BARE_BUILT_PCT:
            continue
        if num_classes < MIN_LAND_CLASSES:
            continue

        passed.append({
            'coords': feat['geometry']['coordinates'],
            'mangrove_pct': round(mangrove_pct, 1),
            'water_pct': round(water_pct, 1),
            'bare_built_pct': round(bare_built_pct, 1),
            'num_classes': num_classes,
        })

    return passed


def download_tiles():
    """Main pipeline: scan, filter, randomize, and download training tiles."""

    # ------------------------------------------------------------------
    # 0. Initialise Earth Engine
    # ------------------------------------------------------------------
    ee.Initialize()
    print("Initializing Filtered Direct-to-Disk Pipeline...")

    # ------------------------------------------------------------------
    # 1. Base Geometry — Indonesia via FAO GAUL
    # ------------------------------------------------------------------
    countries = ee.FeatureCollection("FAO/GAUL/2015/level0")
    indonesia = countries.filter(ee.Filter.eq('ADM0_NAME', 'Indonesia'))
    indo_geom = indonesia.geometry()

    # ------------------------------------------------------------------
    # 2. ESA WorldCover
    # ------------------------------------------------------------------
    esa = ee.ImageCollection('ESA/WorldCover/v200').first()

    esa_all_classes = esa.select('Map').byte().rename('ESA_MultiClass_Label')
    esa_mangrove_filter = esa.eq(95)
    esa_label = esa.select('Map').byte().rename('lc_class')

    # ------------------------------------------------------------------
    # 3. Build the 69-Band Training Stack
    # ------------------------------------------------------------------
    indo_bounds = ee.Geometry.Rectangle([95, -11, 141, 6])

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(indo_bounds)
          .filterDate('2023-01-01', '2024-01-01')
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
          .median())

    s2_optical = s2.select(['B4', 'B3', 'B2', 'B8']).rename(
        ['Red', 'Green', 'Blue', 'NIR']
    )

    embeddings = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                  .filterBounds(indo_bounds)
                  .filterDate('2023-01-01', '2024-01-01')
                  .mosaic())

    training_stack = s2_optical.addBands(embeddings).addBands(esa_all_classes)

    # ------------------------------------------------------------------
    # 4. PASS 1 — Scan all strips, collect quality-filtered candidates
    #    (skipped when candidates_cache.json exists, unless RESCAN=1)
    # ------------------------------------------------------------------
    lon_start, lon_end, lon_step = 95, 141, 5
    lat_min, lat_max = -11, 6

    all_candidates = []
    force_rescan = os.environ.get('RESCAN') == '1'

    if os.path.exists(CANDIDATES_CACHE_PATH) and not force_rescan:
        with open(CANDIDATES_CACHE_PATH, 'r') as f:
            all_candidates = json.load(f)
        print(f"\n========== PASS 1 SKIPPED (loaded cache) ==========")
        print(f"Loaded {len(all_candidates)} candidates from {CANDIDATES_CACHE_PATH}")
        print("  (delete this file or run with RESCAN=1 to force a fresh scan)")
        # Jump straight to Pass 2 below
        return _run_pass_2(all_candidates, training_stack)

    print("\n========== PASS 1: Scanning tiles & applying quality filters ==========")
    print(f"  Filters: mangrove >= {MIN_MANGROVE_PCT}%, "
          f"water <= {MAX_WATER_PCT}%, "
          f"bare+built <= {MAX_BARE_BUILT_PCT}%, "
          f"classes >= {MIN_LAND_CLASSES}")

    for lon in range(lon_start, lon_end, lon_step):
        strip_left = lon
        strip_right = min(lon + lon_step, lon_end)
        strip_box = ee.Geometry.Rectangle([strip_left, lat_min, strip_right, lat_max])
        strip_geom = indo_geom.intersection(strip_box, 100)

        print(f"\n--- Scanning strip: {strip_left}° to {strip_right}° ---")

        strip_bounds = strip_geom.bounds()
        base_grid = strip_bounds.coveringGrid('EPSG:3857', 20480)
        indo_grid = base_grid.filterBounds(strip_geom)

        try:
            # Pre-filter: keep only tiles containing any mangrove pixel
            tiles_with_max = esa_mangrove_filter.reduceRegions(
                collection=indo_grid,
                reducer=ee.Reducer.max(),
                scale=100,
                tileScale=4,
            )
            mangrove_tiles = tiles_with_max.filter(ee.Filter.gt('max', 0))

            n_mangrove = mangrove_tiles.size().getInfo()
            print(f"  Tiles with any mangrove: {n_mangrove}")

            if n_mangrove == 0:
                continue

            # Quality filter via class histogram
            strip_passed = evaluate_tile_quality(mangrove_tiles, esa_label)
            print(f"  Tiles passing quality filters: {len(strip_passed)}")

            for t in strip_passed:
                t['strip'] = f'{strip_left}-{strip_right}'
            all_candidates.extend(strip_passed)

        except Exception as e:
            print(f"  Strip failed: {e}")
            continue

    print(f"\n========== PASS 1 COMPLETE ==========")
    print(f"Total candidate tiles: {len(all_candidates)}")

    if not all_candidates:
        print("No tiles passed quality filters. Try relaxing the thresholds.")
        return

    try:
        with open(CANDIDATES_CACHE_PATH, 'w') as f:
            json.dump(all_candidates, f)
        print(f"Cached candidates → {CANDIDATES_CACHE_PATH}")
    except Exception as e:
        print(f"Warning: failed to write cache file: {e}")

    return _run_pass_2(all_candidates, training_stack)


def _run_pass_2(all_candidates, training_stack):
    """Shuffle candidates, take TOTAL_TILES, download each."""
    # ------------------------------------------------------------------
    # 5. PASS 2 — Randomize selection and download
    # ------------------------------------------------------------------
    random.seed(RANDOM_SEED)
    random.shuffle(all_candidates)
    to_download = all_candidates[:TOTAL_TILES]

    avg_mangrove = sum(t['mangrove_pct'] for t in to_download) / len(to_download)
    avg_water = sum(t['water_pct'] for t in to_download) / len(to_download)
    strips_hit = len(set(t['strip'] for t in to_download))

    print(f"\nRandomly selected {len(to_download)} tiles (seed={RANDOM_SEED}):")
    print(f"  Avg mangrove: {avg_mangrove:.1f}%")
    print(f"  Avg water:    {avg_water:.1f}%")
    print(f"  Strips represented: {strips_hit}")

    out_dir = os.path.join(os.getcwd(), 'Indonesia_Training_Dataset')
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n========== PASS 2: Downloading {len(to_download)} tiles ==========")

    for i, tile in enumerate(to_download):
        filename = os.path.join(out_dir, f'ID_Training_Tile_{i+1:03d}.tif')

        if os.path.exists(filename):
            print(f"[Tile {i+1}] Already exists, skipping.")
            continue

        print(f"[Tile {i+1}/{len(to_download)}] Downloading "
              f"(mangrove={tile['mangrove_pct']}%, water={tile['water_pct']}%, "
              f"classes={tile['num_classes']}, strip={tile['strip']})...")

        try:
            geom = ee.Geometry.Polygon(tile['coords'])
            geemap.download_ee_image(
                image=training_stack,
                filename=filename,
                region=geom,
                crs='EPSG:3857',
                scale=10,
            )
            print(f"  Saved: {filename}")
        except Exception as e:
            print(f"  Failed: {e}")

    print(f"\nDone! {len(to_download)} tiles downloaded to: {out_dir}")


if __name__ == '__main__':
    download_tiles()
