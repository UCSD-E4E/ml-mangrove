# E4E Global Mangrove Observatory

The E4E Global Mangrove Observatory is an open-source interactive 3D visualization platform that communicates the output of E4E's mangrove machine learning pipeline to researchers, engineers, and the public. It ingests raw segmentation predictions from Sentinel-2 satellite imagery, converts them into a web-optimized geospatial format, and renders them as a navigable, data-rich map experience directly in the browser — no plugins, no installs.

The platform covers the Florida coastline as its first deployment area, with the architecture designed to extend to any region where E4E's SegFormer predictions are available.

---

## What It Does

A visitor lands on a rotating satellite globe of Florida. They click Explore and the camera flies down to the coast. From there, every patch of mangrove forest, wetland, tree cover, and built-up land that the model has identified is rendered as a colored polygon on top of high-resolution satellite imagery. Mangrove polygons extrude vertically based on a computed health index, so the healthiest stands literally stand taller. Hovering over any polygon surfaces its class, health score, area, and source tile. A live analytics sidebar aggregates everything visible in the current viewport in real time as the user pans and zooms.

---

## Tech Stack

**Next.js 16** serves as the application framework. It handles routing, static asset delivery, and the build pipeline. Server-side rendering is disabled for the map component because MapLibre GL JS and Deck.gl both require a browser environment with WebGL support.

**React 19** powers the component tree. The project uses React 19's ref-as-prop pattern throughout rather than the deprecated `forwardRef` API.

**TypeScript 5** is used across the entire codebase including configuration files.

**MapLibre GL JS 5** provides the satellite basemap and camera system. MapTiler supplies the satellite tile imagery via an API key. The map is initialized in Mercator projection with a 30-degree pitch for a slight perspective tilt. Globe projection is intentionally disabled — see the Known Limitations section for a detailed explanation.

**Deck.gl 9** renders the segmentation polygon layer. It runs in non-interleaved mode, maintaining a separate WebGL canvas positioned over MapLibre's canvas. A `TileLayer` fetches tiles on demand from the PMTiles archive, decodes them using the MVT specification, and passes GeoJSON features to a `GeoJsonLayer` for rendering with per-class colors, edge glow, and 3D extrusion.

**PMTiles** is the tile delivery format. The entire Florida dataset is packaged into a single binary archive that supports HTTP byte-range requests. The browser fetches only the specific tile bytes it needs for the current viewport, with no tile server required.

**Framer Motion** drives all panel entrance animations and the animated number counters in the analytics sidebar.

**GSAP** handles the landing overlay fade-in sequence and the button pulse animation.

**Tailwind CSS 4** provides all utility styling.

**Python** (with rasterio, shapely, numpy, and pyproj) powers the data preparation scripts that convert raw GeoTIFF predictions into the GeoJSON source file.

**tippecanoe** (via WSL on Windows, or natively on Linux and macOS) packages the GeoJSON into the PMTiles archive.

---

## Repository Structure

The Observatory lives entirely within the `Observatory/` directory of the broader ml-mangrove repository.

`pipeline/` contains the two Python scripts responsible for preparing the data. `vectorize.py` is the main conversion script and `validate.py` is the quality assurance tool.

`data/` holds the intermediate and final data outputs from the pipeline: the raw GeoJSON FeatureCollection, the PMTiles archive, and a centroid scatter plot used during validation.

`web/` is the Next.js application. Inside it, `app/` holds the root layout and the single page component. `components/` holds every UI component. `lib/` holds shared configuration and the Deck.gl layer factory. `public/` serves the PMTiles archive as a static asset.

---

## Data Pipeline

### Vectorization

`pipeline/vectorize.py` is the entry point for data preparation. It reads every GeoTIFF from the `Scale Adaption/GEE/Florida_Training_Dataset/` directory. Each tile is a 2048×2048 pixel raster at 10-meter resolution with 69 bands: four Sentinel-2 optical bands (RGBN), 64 AlphaEarth satellite embedding dimensions, and a final band containing ESA WorldCover class labels.

The script extracts the label band and identifies all connected regions belonging to any of the eight active land cover classes. For each connected region it computes:

- A **health index** between 0 and 1, derived from the NDVI of the pixel values in bands 1 and 4 (red and near-infrared), normalized and clipped to the valid range. Higher values indicate denser, healthier vegetation.
- The **polygon area** in square meters, reprojected to a local equal-area coordinate system before measurement.
- The **class name** and **class index** from the ESA label mapping used in training.
- The **data year** (currently 2023 for all features).
- The **source tile identifier** so features can be traced back to their origin GeoTIFF.

All polygons are reprojected to WGS84 and written to `data/florida_mangroves.geojson` as a single GeoJSON FeatureCollection. The raw output is approximately 1.6 GB.

The eight active classes are: Tree Cover, Grassland, Cropland, Built-up, Bare or Sparse Vegetation, Water, Wetland, and Mangrove. Three ESA classes (Shrubland, Snow and Ice, Moss and Lichen) are not present in Florida imagery and are mapped to an ignore index during training and excluded from vectorization.

### Validation

`pipeline/validate.py` runs quality checks on the GeoJSON output. It verifies that every feature contains all required properties, samples 200 geometries to check spatial validity, confirms that all centroids fall within Florida's bounding box, prints a per-class statistics table to the console, and saves a dark-mode centroid scatter plot colored by class to `data/florida_centroids.png`. Run this after vectorization before packaging tiles.

### Tile Packaging

tippecanoe converts the GeoJSON into a PMTiles archive. It generates tiles from zoom level 0 through 14, preserves all polygon properties on every feature, and places all features in a layer named `landcover`. At lower zoom levels, tippecanoe automatically simplifies geometries and merges small features to keep individual tiles within a manageable byte budget. The resulting archive is approximately 342 MB.

Place the `.pmtiles` file in `web/public/` so Next.js serves it as a static asset. Next.js's static file server supports HTTP Range requests natively, which is what the PMTiles client uses to fetch individual tiles without downloading the entire archive.

---

## Frontend Architecture

### Page Shell

`app/page.tsx` is the single entry point. It owns all top-level state: whether the landing screen is active, which year is selected, what feature the cursor is hovering over, and which features are currently visible in the viewport. It passes state down as props and receives updates through callbacks. The map component is loaded via `dynamic()` with SSR disabled.

### Map Component

`components/MangroveGlobe.tsx` is the core of the application. It initializes MapLibre GL JS with the satellite basemap, creates the Deck.gl overlay, and manages the camera system.

One important implementation detail: MapLibre adds a CSS class to its container element that sets `position: relative`, which overrides Tailwind's `position: absolute` and causes the container to collapse to zero height. The fix is a two-div wrapper. An outer div holds the absolute positioning and fills the viewport. An inner div, which MapLibre actually mounts to, fills 100% of the outer div's dimensions. This way MapLibre's CSS override only affects the inner div, which still gets its size from its correctly-positioned parent.

On every `moveend` event, the component calls Deck.gl's `pickObjects` across the full canvas dimensions to collect all rendered features currently on screen and reports them to the parent via the `onFeaturesChange` callback. This is how the analytics sidebar stays in sync with the viewport.

Click handling computes the centroid of a clicked polygon and flies the camera to it. The centroid calculation handles both Polygon and MultiPolygon geometry types and includes a guard against NaN coordinates before calling `flyTo`.

### Landing Overlay

`components/LandingOverlay.tsx` is the full-screen cinematic intro. It shows the E4E badge, the platform title, a brief tagline, and an Explore Florida call-to-action button. GSAP drives a staggered fade-in on mount. Clicking the button dismisses the overlay with a Framer Motion exit animation and flies the camera to the Florida coastline at zoom level 8.

### Analytics Panel

`components/AnalyticsPanel.tsx` is a frosted-glass sidebar pinned to the right edge. It receives the current viewport's feature list from the parent and computes:

- Total land cover area in hectares across all visible polygons
- Total mangrove area in hectares
- Mean health index across all visible polygons
- Total polygon count
- A per-class breakdown showing each class's percentage of the visible total and raw polygon count, with a proportional stacked bar visualization

All numeric values animate smoothly using Framer Motion springs whenever the viewport changes. When no features have loaded yet, the panel shows a placeholder message. The panel slides in from the right using a spring entrance animation after the landing overlay is dismissed.

### Time Slider

`components/TimeSlider.tsx` is a bottom-center control for temporal navigation. The current dataset only covers 2023, so the slider displays a static year badge. The component is fully wired to the `selectedYear` state in the page shell and is ready for multi-year filtering once additional prediction years are ingested through the pipeline.

### Hover Tooltip

`components/HoverTooltip.tsx` is a small card that follows the cursor position. When the pointer moves over a polygon, it displays the land cover class name, a color-coded health index progress bar, the polygon's area in hectares, the data year, and the source tile identifier. It disappears when the cursor leaves a polygon.

### Class Configuration

`lib/classConfig.ts` is the single source of truth for the class taxonomy. It defines the name, RGB color tuple, hex color code, and emoji icon for each of the eight land cover classes. Both the layer builder and the analytics panel import from this file. Adding a new class or changing a color only requires an edit in one place.

Water polygons are rendered fully transparent (zero alpha on fill and stroke). The satellite imagery beneath open water is far more informative than a colored polygon overlay, and rendering the water mask was visually cluttering the coastline.

### Layer Builder

`lib/layerBuilder.ts` exports a factory function that returns a configured Deck.gl `TileLayer`. It maintains a singleton `PMTiles` instance so the archive header is only fetched once per session regardless of how many times the layer is rebuilt. For each tile, it calls `pmtiles.getZxy(z, x, y)` to fetch the raw tile bytes via HTTP Range request, then passes them through `@loaders.gl/mvt`'s `MVTLoader` with `wgs84` coordinate output so the resulting GeoJSON is in standard longitude and latitude.

Each tile's features are rendered via a `GeoJsonLayer` sublayer configured with:

- Per-class fill colors from `classConfig`
- A faint edge glow per class
- 3D extrusion for Mangrove features only, with height calculated as `health_index × 500` meters
- Physically-based material settings (ambient, diffuse, shininess) for the extruded surfaces
- Auto-highlighting on hover
- Update triggers on `selectedYear` so the layer rebuilds correctly when the year filter changes

---

## Setup and Configuration

### Prerequisites

Running the data pipeline requires Python with rasterio, shapely, numpy, and pyproj installed. The `mangrove` Anaconda environment in the broader ml-mangrove repository has these dependencies.

Running the tile packaging step requires tippecanoe. On macOS install it via Homebrew. On Linux install it via apt or build from source. On Windows, run it inside WSL with Ubuntu.

Running the frontend requires Node.js 20 or later and npm.

### Environment Variables

Create a file at `web/.env.local` with the following two variables:

`NEXT_PUBLIC_MAPTILER_KEY` is your MapTiler API key. A free tier account at maptiler.com provides enough satellite tile quota for development.

`NEXT_PUBLIC_PMTILES_URL` is the path to the PMTiles archive relative to the web root. It defaults to `/florida_mangroves.pmtiles` which resolves to the file in `web/public/`. Change this if you are hosting the archive externally, for example on a CDN or cloud storage bucket with CORS configured.

### Running the Pipeline

Activate the `mangrove` Anaconda environment, then run `vectorize.py` from the repository root. It will write the GeoJSON to `Observatory/data/florida_mangroves.geojson`. Follow up with `validate.py` to confirm the output before proceeding.

Once validation passes, run tippecanoe against the GeoJSON to produce the PMTiles archive. Copy or symlink the result to `web/public/florida_mangroves.pmtiles`.

### Running the Frontend

Navigate to `Observatory/web/`, run `npm install` to install dependencies, then run `npm run dev` to start the development server. The application will be available at `http://localhost:3000`. For a production build, run `npm run build` followed by `npm start`.

---

## Known Limitations

**Globe projection is disabled.** MapLibre GL JS 5 supports a globe rendering mode, and the original design called for the landing experience to show a true 3D globe. Deck.gl's non-interleaved overlay mode cannot synchronize correctly with MapLibre's spherical coordinate math at low zoom levels, causing polygon misalignment. The interleaved mode (which renders Deck.gl inside MapLibre's WebGL pipeline and would fix the coordinate mismatch) silently halts MapLibre's render loop in the current combination of MapLibre 5 and Deck.gl 9. This is a known compatibility gap between the two libraries at these versions. The map still renders with a perspective pitch and satellite imagery, which preserves much of the intended cinematic quality. This should be revisited as both libraries update.

**The analytics panel uses rendered-canvas picking.** Deck.gl's `pickObjects` queries the rendered canvas rather than a spatial index over the underlying data. This works well at moderate zoom levels, but at very high zoom with dense polygon coverage the call can become expensive. A future improvement would maintain an in-memory spatial index (such as rbush) updated as new tiles load, enabling viewport queries without touching the GPU.

**Health index mean includes all classes.** The mean health index displayed in the analytics panel is computed across all visible polygons, including non-vegetation classes where the NDVI-based health index has no ecological meaning. Restricting this metric to Mangrove features specifically would make it more meaningful.

**Single year dataset.** The time slider UI is functional but the underlying data only covers 2023. Multi-year temporal analysis requires running the prediction pipeline over Sentinel-2 imagery from multiple years and ingesting the results through the same vectorization and packaging workflow.

---

## Contributing

The codebase follows the conventions of the broader ml-mangrove repository. All frontend code is in TypeScript with strict mode enabled. New land cover classes should be added exclusively in `lib/classConfig.ts`. Layer rendering logic lives in `lib/layerBuilder.ts` and should not be duplicated in components. Any changes to the GeoJSON schema (adding or renaming feature properties) need to be reflected in both the Python pipeline and the TypeScript `HoverInfo` interface in `lib/layerBuilder.ts`.

For questions about the ML pipeline that produces the segmentation predictions, see the `Scale Adaption/GEE/` directory and the SegFormer training notebook.
