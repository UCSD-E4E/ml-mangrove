import ee
import time
import geemap
import os

def main():
    ee.Authenticate()
    ee.Initialize(project='e4e-mangrove-493601')

    # ==========================================
    # 1. Base Geometry
    # ==========================================
    # Define a custom bounding box
    # Format: [min_longitude, min_latitude, max_longitude, max_latitude]
    # region_bounds = ee.Geometry.Rectangle([88.0, 21.5, 91.2, 22.5])
    region_bounds = ee.Geometry.Rectangle([86.5, 18.5, 94.25, 22.7])

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
    # 3. FILTER THE GRID (Mangrove + Water Filter + Randomize)
    # ==========================================
    base_grid = region_bounds.coveringGrid('EPSG:3857', 20480)

    # 1. Gatekeeper 1: Must contain Mangroves (Class 95)
    tiles_with_mangroves = esa_mangrove_filter.reduceRegions(
        collection=base_grid,
        reducer=ee.Reducer.max(),
        scale=100,
        tileScale=4 
    ).filter(ee.Filter.gt('max', 0))

    # 2. Gatekeeper 2: Water Filter (< 80% Water)
    water_mask = esa.eq(80) # 1 = Water, 0 = Not Water

    def calculate_water_fraction(feature):
        # The mean of a 0/1 mask gives the exact percentage of water
        water_stats = water_mask.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=feature.geometry(),
            scale=100,
            maxPixels=1e9
        )
        return feature.set('water_fraction', water_stats.get('Map'))

    # Apply calculation and drop tiles with >= 80% water
    filtered_tiles = tiles_with_mangroves.map(calculate_water_fraction) \
                                         .filter(ee.Filter.lt('water_fraction', 0.80))

    # 3. Randomize and Limit to 100 tiles
    # randomColumn() assigns a random float to each tile, allowing us to shuffle them
    final_target_tiles = filtered_tiles.randomColumn('random_id') \
                                       .sort('random_id') \
                                       .limit(100)

    # ==========================================
    # 4. Build the 69-Band Training Stack
    # ==========================================
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(region_bounds)
        .filterDate('2023-01-01', '2024-01-01')
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
        .median())

    s2_optical = s2.select(['B4', 'B3', 'B2', 'B8']).rename(['Red', 'Green', 'Blue', 'NIR'])

    embeddings = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                .filterBounds(region_bounds)
                .filterDate('2023-01-01', '2024-01-01')
                .mosaic())

    # Stack: Optical (1-4) + Embeddings (5-68) + FULL Multi-Class Labels (69)
    training_stack = s2_optical.addBands(embeddings).addBands(esa_all_classes)

    
    # ==========================================
    # 5. Direct-to-Disk Download Loop
    # ==========================================
    print("Calculating tile intersections via Raster Engine...")
    
    tile_list = final_target_tiles.toList(100) 

    def get_coords_gps(tile):
        return ee.Feature(tile).geometry().transform('EPSG:4326', 1).coordinates()

    tile_coordinates = tile_list.map(get_coords_gps)
    coords_array = tile_coordinates.getInfo()

    total_tiles = len(coords_array)
    print(f"\nSUCCESS! Total Intersecting Tiles Found: {total_tiles}")

    out_dir = os.path.join(os.getcwd(), 'EastIndia_Bangladesh_Training_Dataset')
    os.makedirs(out_dir, exist_ok=True)
    print(f"Starting direct download to: {out_dir}")

    # Full Loop
    # for i, coords in enumerate(coords_array[0:1]): 
    for i, coords in enumerate(coords_array):
        geom = ee.Geometry.Polygon(coords)
        
        filename = os.path.join(out_dir, f'IB_Tile_{i + 1:03d}.tif')
        
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
    

if (__name__ == "__main__"):
    main()