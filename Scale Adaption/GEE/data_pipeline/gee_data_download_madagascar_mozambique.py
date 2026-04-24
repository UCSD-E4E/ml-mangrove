import ee
import time
import geemap
import os
import random

ee.Authenticate()
ee.Initialize(project='e4e-mangrove')

print("Initializing Filtered Direct-to-Disk Pipeline...")

# ==========================================
# 1. Base Geometry
# ==========================================
countries  = ee.FeatureCollection("FAO/GAUL/2015/level0")
madagascar = countries.filter(ee.Filter.eq('ADM0_NAME', 'Madagascar'))
mozambique = countries.filter(ee.Filter.eq('ADM0_NAME', 'Mozambique'))

study_region  = madagascar.merge(mozambique).union(maxError=1000)
madmoz_bounds = study_region.geometry().bounds()

# ==========================================
# 2. Define the Target Label (ESA WorldCover)
# ==========================================
esa = ee.ImageCollection('ESA/WorldCover/v200').first()

# A. The FULL 11-Class Label (This becomes Band 69)
# Values include: 10(Trees), 50(Built-up), 80(Water), 95(Mangroves), etc.
esa_all_classes = esa.select('Map').byte().rename('ESA_MultiClass_Label')

# B. The STRICT Mangrove Filter (Class 95)
# We use this purely to find the coastal bounding boxes
esa_mangrove_filter = esa.eq(95)

# ==========================================
# 3. FILTER THE GRID (The Gatekeeper)
# ==========================================
base_grid = madmoz_bounds.coveringGrid('EPSG:3857', 20480)

# Scan the grid using the STRICT Class 95 filter to drop the inland noise
tiles_with_stats = esa_mangrove_filter.reduceRegions(
    collection=base_grid,
    reducer=ee.Reducer.max(),
    scale=100,
    tileScale=4
)
# ONLY keep tiles that have true ESA Mangroves
mangrove_tiles = tiles_with_stats.filter(ee.Filter.gt('max', 0))

# ==========================================
# 4. Quality Filter (Water / Mangrove Fraction)
# ==========================================
# Pull per-tile ESA class histograms to exclude ocean-dominated tiles
# and tiles with negligible mangrove coverage.
print("Computing class histograms for quality filtering...")
class_stats = esa_all_classes.reduceRegions(
    collection=mangrove_tiles,
    reducer=ee.Reducer.frequencyHistogram(),
    scale=100,
    tileScale=4
)
all_stats = class_stats.toList(500).getInfo()
print(f"Candidate tiles: {len(all_stats)}")

WATER_MAX    = 0.65  # drop tiles that are >65% open water
MANGROVE_MIN = 0.03  # drop tiles with <3% mangrove coverage

def parse_hist(feature):
    raw = feature['properties'].get('histogram', {})
    return {int(float(k)): int(v) for k, v in raw.items()}

kept = []
for feat in all_stats:
    hist  = parse_hist(feat)
    total = sum(hist.values())
    if hist.get(80, 0) / total > WATER_MAX:
        continue
    if hist.get(95, 0) / total < MANGROVE_MIN:
        continue
    kept.append(feat)

print(f"Tiles after quality filter: {len(kept)}")

# ==========================================
# 5. Build the 69-Band Training Stack
# ==========================================
s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(madmoz_bounds)
      .filterDate('2023-01-01', '2024-01-01')
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
      .median())

s2_optical = s2.select(['B4', 'B3', 'B2', 'B8']).rename(['Red', 'Green', 'Blue', 'NIR'])

embeddings = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
              .filterBounds(madmoz_bounds)
              .filterDate('2023-01-01', '2024-01-01')
              .mosaic())

# Stack: Optical (1-4) + Embeddings (5-68) + FULL Multi-Class Labels (69)
training_stack = s2_optical.addBands(embeddings).addBands(esa_all_classes)

# ==========================================
# 6. Direct-to-Disk Download Loop
# ==========================================
# Shuffle and cap at 100 tiles
random.seed(42)
random.shuffle(kept)
download_batch = kept[:100]

out_dir = os.path.join(os.getcwd(), 'MadMoz_Training_Dataset')
os.makedirs(out_dir, exist_ok=True)
print(f"Starting direct download to: {out_dir}")

total_tiles = len(download_batch)
for i, feat in enumerate(download_batch):
    geom     = ee.Geometry.Polygon(feat['geometry']['coordinates'], 'EPSG:3857').transform('EPSG:4326', 1)
    filename = os.path.join(out_dir, f'MadMoz_Training_Tile_{i + 1:03d}.tif')

    print(f"[{i+1}/{total_tiles}] Downloading directly to disk. This might take a few minutes...")

    try:
        geemap.download_ee_image(
            image=training_stack,
            filename=filename,
            region=geom,
            crs='EPSG:3857',
            scale=10
        )
        print(f"Success! Saved: {filename}")
    except Exception as e:
        print(f"Failed to download tile {i+1}. Error: {e}")

print("\nDownload script finished!")
