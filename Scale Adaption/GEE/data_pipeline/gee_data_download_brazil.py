import ee
import geemap
import os

def get_coords_gps(tile):
        return ee.Feature(tile).geometry().transform('EPSG:4326', 1).coordinates()

def add_metrics(feat):
    """
    add water and tree filters (so if a tile is 70% water or 70% tree, then discard)
    """
    hist = ee.Dictionary(feat.get('histogram'))
    total = ee.Number(hist.values().reduce(ee.Reducer.sum()))
    total = ee.Algorithms.If(total, total, 1)
    
    water = ee.Number(hist.get('80', 0)).divide(total).multiply(100)
    tree  = ee.Number(hist.get('10', 0)).divide(total).multiply(100)

    keep = water.lt(70).And(tree.lt(70))
    keep_num = ee.Number(keep)
    return feat.set({'keep': keep_num})

def sample_hundred(coords_array, training_stack):
    """
    finds the stats of all the coords/existing mangrove tiles, applies a filter
    to only keep the tiles that satisfy the conditions and then randomly
    samples 100 of those tiles into the variable sampled
    """
    tiles_fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Polygon(coords)) for coords in coords_array
    ])

    stats = training_stack.select('ESA_MultiClass_Label').reduceRegions(
        collection=tiles_fc,
        reducer=ee.Reducer.frequencyHistogram(),
        scale=100,
        tileScale=4
    )
    # print(stats.first().getInfo())

    filtered = stats.map(add_metrics).filter(ee.Filter.eq('keep', 1))
    # print(filtered.size().getInfo())

    sampled = filtered.randomColumn('rand').sort('rand').limit(100)
    return sampled

def main():
    """
    Controls the main workflow of downloading all the brazil tiles
    """
    ee.Authenticate()
    ee.Initialize(project='ee-mydong')
    print("Initializing Filtered Direct-to-Disk Pipeline...")

    # ==========================================
    # 1. Base Geometry
    # ==========================================
    countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    brazil = countries.filter(ee.Filter.eq('country_na', 'Brazil'))
    brazil_bounds = brazil.geometry().bounds()

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
    base_grid = brazil_bounds.coveringGrid('EPSG:3857', 20480)

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
    # 4. Build the 69-Band Training Stack
    # ==========================================
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(brazil_bounds)
        .filterDate('2023-01-01', '2024-01-01')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
        .median())

    s2_optical = s2.select(['B4', 'B3', 'B2', 'B8']).rename(['Red', 'Green', 'Blue', 'NIR'])

    embeddings = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                .filterBounds(brazil_bounds)
                .filterDate('2023-01-01', '2024-01-01')
                .mosaic())

    # Stack: Optical (1-4) + Embeddings (5-68) + FULL Multi-Class Labels (69)
    training_stack = s2_optical.addBands(embeddings).addBands(esa_all_classes)

    # ==========================================
    # 5. Direct-to-Disk Download Loop
    # ==========================================
    print("Calculating tile intersections via Raster Engine...")
    tile_list = mangrove_tiles.toList(500)

    tile_coordinates = tile_list.map(get_coords_gps)
    coords_array = tile_coordinates.getInfo()

    total_tiles = len(coords_array)
    print(f"\nSUCCESS! Total Intersecting Tiles Found: {total_tiles}")

    # creating the output directory
    out_dir = os.path.join(os.getcwd(), 'Brazil_Training_Dataset')
    os.makedirs(out_dir, exist_ok=True)
    print(f"Starting direct download to: {out_dir}")

    # only use the first 100 randomly sampled tiles given the by function sample_hundred
    selected_tiles = sample_hundred(coords_array, training_stack).toList(100).getInfo()

    # WARNING: [0:1] slices the array to only download the VERY FIRST tile for testing.
    # Once it succeeds, remove the slice [0:1] to loop through all ~98 tiles.
    for i, feat in enumerate(selected_tiles[:1]):
        geom = ee.Geometry(feat['geometry'])
        filename = os.path.join(out_dir, f'BR_Training_Tile_{i + 1:03d}.tif')
        
        print(f"[{i+1}/{100}] Downloading directly to disk. This might take a few minutes...")

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

    print("After filter: ", len(selected_tiles)) 
    print("\nDownload script finished!")

if __name__ == "__main__":
    main()