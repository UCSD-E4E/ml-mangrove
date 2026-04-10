# Global Mangrove Observatory — Platform Documentation

This document describes every part of the Observatory web platform in detail. It is written for developers who want to understand how the system works, extend it with new features, or debug unexpected behaviour. The platform is a GPU-accelerated geospatial visualisation tool that renders satellite-derived mangrove land-cover maps over a 3D interactive globe.

---

## Table of Contents

1. [What the Platform Does](#1-what-the-platform-does)
2. [Technology Stack](#2-technology-stack)
3. [Repository Layout](#3-repository-layout)
4. [Data: PMTiles and MVT](#4-data-pmtiles-and-mvt)
5. [Application Lifecycle and Stage Machine](#5-application-lifecycle-and-stage-machine)
6. [The Map Engine](#6-the-map-engine)
7. [Tile Loading and Rendering Pipeline](#7-tile-loading-and-rendering-pipeline)
8. [Class Taxonomy and Styling](#8-class-taxonomy-and-styling)
9. [Hover Interaction](#9-hover-interaction)
10. [Analytics Panel](#10-analytics-panel)
11. [Landing Experience](#11-landing-experience)
12. [Loading Screen](#12-loading-screen)
13. [Time Slider](#13-time-slider)
14. [Performance Architecture](#14-performance-architecture)
15. [Environment Variables](#15-environment-variables)

---

## 1. What the Platform Does

The Observatory is a web application that lets users explore a machine-learning-generated map of Florida's mangrove ecosystems at 10-metre resolution. The underlying data comes from Sentinel-2 satellite imagery processed through a SegFormer segmentation model. The model classifies every 10-metre pixel in the scene into one of eight land-cover categories: mangrove, tree cover, grassland, cropland, built-up, bare or sparse vegetation, water, and wetland.

The result of that classification is stored as a collection of vector polygon tiles. On the web, the user can fly over the coast of Florida, zoom into specific regions, and see each land-cover class rendered as a coloured polygon mask directly on top of real satellite imagery. At higher zoom levels the masks extrude upward into three dimensions, with mangrove polygons rising to heights that reflect the ecosystem's health index — healthy mangrove forests appear tall and vibrant, while degraded areas appear low and flat. Every other class is extruded to a fixed representative height so the terrain reads clearly when tilted. Clicking on any polygon flies the camera to that location and zooming out returns to the overview.

A panel on the right side of the screen continuously updates with statistics about the land cover currently visible in the viewport: total area, mangrove area specifically, mean health index across all visible mangrove polygons, and a breakdown of polygon counts by class. A tooltip follows the cursor and shows the class name, health index, area, data year, and tile identifier for whatever polygon the user is hovering over.

---

## 2. Technology Stack

The platform is built on Next.js 16 with React 19 and written entirely in TypeScript. The map base layer is provided by MapLibre GL, which handles the satellite tile imagery, camera controls, and the underlying WebGL context. On top of MapLibre, Deck.gl version 9 is used as a GPU-accelerated overlay for rendering the vector polygon masks. Deck.gl is what makes it possible to render millions of polygon vertices at interactive frame rates without the browser stalling.

The land-cover data is served from a single self-contained PMTiles archive. PMTiles is a file format that stores a pyramid of Mapbox Vector Tiles inside one HTTP-range-request-friendly binary file. The browser fetches only the portions of the file that correspond to the current viewport, so there is no tile server needed at runtime — the archive can be hosted on any static file server or CDN.

Animations outside the map (the landing screen, analytics panel, tooltips) use Framer Motion for physics-based spring transitions and GSAP for the sequenced entry animations on the landing overlay. The UI is styled with Tailwind CSS v4.

---

## 3. Repository Layout

Inside the `Observatory/web` directory the project follows the Next.js App Router convention. The `app` directory contains `layout.tsx`, which sets up global metadata and imports MapLibre's CSS, and `page.tsx`, which is the root component that orchestrates the entire application.

The `components` directory holds six UI components: `MangroveGlobe`, `AnalyticsPanel`, `HoverTooltip`, `LandingOverlay`, `LoadingScreen`, and `TimeSlider`. Each component owns a clearly scoped piece of the interface.

The `lib` directory contains two modules. `classConfig.ts` defines the land-cover taxonomy including class names, RGB colours, CSS hex values, emoji icons, and the extrusion height table. `layerBuilder.ts` builds the Deck.gl layer that loads and renders the PMTiles data, and exports the `warmPMTiles` utility used during the loading screen.

The `public` directory holds the PMTiles archive file `florida_mangroves.pmtiles`, which is 358 MB and is served directly by Next.js as a static asset.

---

## 4. Data: PMTiles and MVT

Understanding the data format is essential to understanding how the rendering works.

**Mapbox Vector Tiles (MVT)** are the individual tile units. Each tile is a binary protobuf that describes vector geometries (polygons, lines, points) and their properties within a specific geographic tile boundary. The tile coordinate system uses a three-number index: zoom level, x column, and y row. At zoom level zero, the entire world fits in one tile. Each additional zoom level doubles the grid in both dimensions, so zoom level 10 has a grid of 1024 by 1024 tiles. The Observatory's PMTiles archive contains a layer named `landcover` inside each tile, with polygon features that carry properties like `class_name`, `class_idx`, `health_index`, `area_sqm`, `year`, and `tile_id`.

**PMTiles** wraps the entire MVT pyramid into a single file using a spatial index that maps tile coordinates to byte ranges. The browser sends an HTTP Range request for the file header and root index (usually around 512 KB) and then follows byte-range pointers to fetch individual tiles on demand. This means the first time you visit a tile's geographic area, the browser fetches just that tile's byte range. Subsequent visits to the same area at the same zoom level are served from the browser's HTTP cache. The `warmPMTiles` function in `layerBuilder.ts` pre-fetches the file header during the loading screen, so by the time the user clicks "Explore Florida" the root index is already cached and the first batch of tile requests are nearly instantaneous.

The archive's `minZoom` and `maxZoom` metadata tell us which zoom levels actually contain data. The application reads these values from the PMTiles header during warm-up and uses the `minZoom` to compute the correct `zoomOffset` parameter for Deck.gl, which is explained in the tile loading section below.

---

## 5. Application Lifecycle and Stage Machine

The application runs through three distinct stages: `loading`, `landing`, and `exploring`. This progression is managed by a single `stage` state variable in `page.tsx`.

When the page first loads, the stage is `loading`. The map and the PMTiles header begin initialising immediately in the background, behind a full-screen loading overlay. This means the map is never waiting for the user — it starts loading the moment the browser executes the JavaScript.

The `LoadingScreen` component is rendered during this stage. It shows an animated progress counter that crawls smoothly up to 85 percent while the map initialises. As soon as MapLibre fires its `load` event, the `onReady` callback sets `mapReady` to true, which signals the loading screen to jump to 100 percent, wait briefly, and then fade out. After the fade, the stage advances to `landing`.

During the `landing` stage, the globe is visible and auto-rotating slowly. The `LandingOverlay` appears with the project title, a short description, and an "Explore Florida" button. The globe rotation is driven by a requestAnimationFrame loop that increments the map bearing by 0.025 degrees on every frame. When the user interacts with the map directly (panning or zooming) the rotation is suspended automatically.

When the user clicks "Explore Florida", the stage advances to `exploring`. The overlay fades out and the camera flies to the Florida coastline at zoom level 8. The analytics panel and time slider slide into view. From this point on the user is in full interactive mode.

The `MangroveGlobe` component is always mounted, even during the loading and landing stages. This is a deliberate choice: unmounting and remounting the map would require reinitialising the WebGL context, which is expensive. The map runs continuously in the background so it is always warm and responsive the moment the user enters the exploring stage.

---

## 6. The Map Engine

`MangroveGlobe` is the component responsible for the interactive globe. It initialises a MapLibre map instance pointed at a MapTiler satellite tile style, which provides the photographic base imagery. Deck.gl is attached to the same WebGL context via a `MapboxOverlay`, which means Deck.gl renders its vector layers directly into the MapLibre pipeline rather than as a separate canvas on top. This interleaved approach avoids the visual artefacts that would occur if two separate WebGL contexts were composited.

The component uses React refs extensively. The map instance, the Deck.gl overlay, the requestAnimationFrame handle, and the hover state all live in refs rather than state. Refs allow these values to be read and mutated synchronously without triggering React re-renders. The only piece of UI state managed inside the component is the `hovered` value for the tooltip, which is the only thing that needs to cause a DOM update in response to user interaction.

**Auto-rotation** is implemented as a requestAnimationFrame loop that runs from mount to unmount. On every frame it reads `rotatingRef.current` (which mirrors the `rotating` prop) and if rotation is active it increments the map bearing by 0.025 degrees. The use of a ref rather than a prop inside the loop prevents the loop from needing to be recreated when the prop changes. The loop is started once on mount and cancelled once on unmount. When the user initiates any map movement, the `map.on('movestart')` handler sets `rotatingRef.current` to false, pausing rotation for the duration of the session.

**Feature picking** is how the analytics panel gets updated. After the user finishes a pan or zoom (the `moveend` event fires), a 300-millisecond timer starts. When the timer fires without being interrupted by another `moveend`, it calls Deck.gl's `pickObjects` method on the full viewport rectangle. This method returns all features whose geometry intersects the viewport, without requiring any additional GPU draw calls — Deck.gl maintains a separate picking buffer for exactly this purpose. The resulting array of feature objects is passed up to `page.tsx` via the `onFeaturesChange` callback, which stores them and passes them down to `AnalyticsPanel`.

**Fly-to on click** is handled by extracting the geometric centroid of the clicked polygon and calling MapLibre's `flyTo` with a zoom of 13 and a pitch of 60 degrees. The centroid calculation averages all the vertices of the first ring of the polygon's exterior, which gives a good enough approximation for the polygon sizes in this dataset.

---

## 7. Tile Loading and Rendering Pipeline

The core of the data rendering is in `layerBuilder.ts`. The function `buildMangroveLayer` constructs and returns a Deck.gl `TileLayer` configured to load from the PMTiles archive.

**TileLayer configuration** controls which tiles are loaded and when. The `minZoom` of 0 means the layer is visible at any camera zoom level. The `maxZoom` of 14 means that even if the user zooms to level 18, the layer will use zoom-14 tiles rather than requesting tiles beyond the resolution of the dataset. The `extent` option is set to the approximate bounding box of Florida, which prevents the layer from wasting requests on tiles covering the Atlantic Ocean, the Gulf of Mexico, or other states.

The `zoomOffset` parameter is computed dynamically from the PMTiles header. When Deck.gl decides which tiles to request, it takes the current viewport zoom and adds `zoomOffset` to determine the tile zoom. A positive offset means the layer loads finer-grained tiles than the viewport zoom alone would suggest. If the PMTiles archive starts at zoom level 8 (meaning no data exists below that), and the user is viewing Florida at viewport zoom level 4, then without an offset the layer would request zoom-4 tiles which do not exist in the archive. With an offset of 4, it would request zoom-8 tiles, which do exist. The formula `max(2, pmtilesMinZoom - 4)` chooses an offset that ensures even the widest Florida overview view (around zoom 4) hits real tile data.

The `refinementStrategy` is set to `'best-available'`. This tells Deck.gl that when a requested tile is still loading, it should display the best available tile from any zoom level already in its cache. In practice this means the globe never goes blank during a zoom transition: you see either a slightly blurry lower-resolution version of the current view (from a parent tile) or a slightly cropped higher-resolution version (from a child tile), and when the correct tile loads it replaces the placeholder seamlessly.

The `maxCacheSize` of 512 controls how many tiles Deck.gl keeps in VRAM. When the cache is full and a new tile arrives, the tile that was used least recently is evicted. A larger cache means more tiles survive as the user pans around, reducing the number of network fetches on revisited areas.

**Tile fetching** happens in the `getTileData` callback. For each tile that Deck.gl determines it needs, this function calls `pmtiles.getZxy(z, x, y)` to fetch the tile's byte range from the archive. If the archive returns no data for that coordinate (which happens when a tile genuinely does not exist at that zoom level), the function returns null and Deck.gl skips rendering for that tile. When data is returned, it is decoded from the MVT binary format using `@loaders.gl/mvt`'s `MVTLoader`. The `worker: true` option causes this decoding to happen on a background web worker thread, which means the main thread is never blocked by protobuf parsing and animation stays smooth even when many tiles are arriving simultaneously.

**Sub-layer rendering** is controlled by the `renderSubLayers` callback, which Deck.gl calls once per loaded tile. This function receives the tile's properties and the parsed GeoJSON features, and returns a Deck.gl `GeoJsonLayer` that renders those features. The decision about whether to extrude polygons into 3D is made here based on the tile's zoom level. At zoom 11 and above, `shouldExtrude` becomes true and the layer switches from flat 2D rendering to extruded 3D rendering.

The `getElevation` accessor checks the feature's class name. If the class is Mangrove, the height is `health_index * 500`. The health index ranges from 0 to 1, so mangrove polygons can reach up to 500 units tall in map space, which at globe scale is a dramatic and visually meaningful exaggeration. All other classes use the fixed height values from the `CLASS_ELEVATION` table in `classConfig.ts`. Built-up land is the tallest fixed class at 150 units, representing urban density. Tree cover is 120 units. Wetland is 60. Grassland and cropland are low. Water and bare vegetation are at or near ground level.

The `getFillColor` accessor returns a pre-cached RGBA tuple for each class name. Water features are rendered fully transparent (alpha 0) because the satellite base map already shows the water clearly, and overlaying a colour on top would be redundant. All other classes use their assigned colour at alpha 220, which allows the satellite imagery to show faintly through the mask, giving the viewer geographic context even while looking at classified land cover.

The `stroked: false` setting disables polygon outlines entirely. This is important for two reasons. First, it saves a GPU draw pass. Second, when vector tiles are clipped to tile boundaries, polygon outlines would appear along the clipping edges, creating visible grid seams across the map. With strokes disabled, the tile boundaries are invisible.

---

## 8. Class Taxonomy and Styling

`classConfig.ts` is the single source of truth for everything related to the eight land-cover classes. Any component that needs to know a class's colour, icon, elevation, or display name reads from this file.

The eight classes are: Tree Cover (ESA WorldCover code 10), Grassland (30), Cropland (40), Built-up (50), Bare or Sparse Vegetation (60), Water (80), Wetland (90), and Mangrove (95). These codes correspond directly to the ESA WorldCover classification scheme used during model training. ESA codes not present in Florida (Shrubland, Snow/Ice, Moss/Lichen) are excluded.

The `CLASS_COLORS` object maps each class name to an RGB triplet. These colours are chosen to be perceptually distinct and to carry semantic meaning: tree cover is deep forest green, built-up is brick red, water is ocean blue, mangrove is teal. The colours also mirror those used in the training notebook so that the visualisation is consistent with the model's internal class representations.

The colour lookup is performance-critical because Deck.gl calls `getFillColor` for every visible feature on every rendered frame. To avoid allocating a new array on every call, all eight RGBA tuples are pre-computed at module load time into a `FILL_COLOR_CACHE` dictionary. The `getColor` function then returns the pre-computed tuple by key lookup, which is a single hash table read with zero allocation. This pattern is deliberately applied because JavaScript's garbage collector is sensitive to frequent small allocations, and at the scale of thousands of polygons rendered at 60 frames per second, even tiny per-frame allocations accumulate into noticeable pauses.

The `CLASS_ELEVATION` table assigns each class a fixed extrusion height in "map metres" (these are not geographic metres — they are units in Deck.gl's coordinate space, exaggerated for visual effect). The Mangrove entry is set to zero because its height is computed dynamically from `health_index` rather than from this table.

---

## 9. Hover Interaction

When the user moves the mouse over the map, Deck.gl fires an `onHover` event at the native browser mousemove rate, which can be 100 to 200 events per second. Passing each of these events directly to React's state would cause 100 to 200 `setState` calls per second, each triggering a React reconciliation and DOM update. At that frequency, the browser would spend most of its time reconciling rather than rendering frames, causing visible lag.

The solution is a requestAnimationFrame-based throttle. The `onHover` handler stores the latest hover info in a ref called `pendingHoverRef`, then checks whether an animation frame is already scheduled. If not, it schedules one. When the frame fires, it reads the latest value from `pendingHoverRef` (which may be a different hover event than the one that scheduled the frame) and calls `setHovered` once. This ensures that React state updates happen at most once per display frame regardless of how many mousemove events arrive between frames.

The `hovered` state lives inside `MangroveGlobe` rather than in `page.tsx`. This is a deliberate performance choice. If hover state were in `page.tsx`, every hover event would trigger a re-render of `page.tsx` and all of its children, including the MapLibre and Deck.gl containers. Even though MapLibre and Deck.gl are imperative and their internal state would not change, React would still run their render functions and reconcile their virtual DOM, which adds overhead on every frame. By keeping hover state inside `MangroveGlobe`, only the `HoverTooltip` component re-renders on each hover event, which is the minimum necessary.

`HoverTooltip` is rendered inside `MangroveGlobe`'s JSX return, positioned absolutely over the map. It receives the `hovered` object and uses Framer Motion's `AnimatePresence` to fade and scale the tooltip in and out as the user moves between features. The tooltip is offset 14 pixels down and to the right of the cursor so it never covers the polygon being inspected. The `pointer-events: none` CSS style ensures the tooltip does not intercept mouse events that should reach the map beneath it.

---

## 10. Analytics Panel

`AnalyticsPanel` is the statistics sidebar that appears on the right side of the screen during the exploring stage. It receives the array of visible features from `page.tsx` and computes four metrics.

The total area is the sum of all features' `area_sqm` properties converted from square metres to hectares. The mangrove area is the same sum filtered to only features where `class_name` is `'Mangrove'`. The mean health index is the arithmetic mean of `health_index` across all features that have a non-null value. The polygon count is simply the length of the features array.

The class breakdown section groups features by class name, counts them, sorts by count descending, and renders a stacked bar chart and a legend. Each class is shown with its colour, icon, name, and count.

All numeric values animate smoothly using Framer Motion's `useSpring` hook. When the visible features change after a pan or zoom, the numbers do not jump instantly to their new values — they spring toward the new value over about 300 milliseconds, which makes the panel feel responsive without being jarring.

The panel slides in from the right side of the screen when the exploring stage begins, using a Framer Motion spring animation with stiffness 120 and damping 20. The panel is scrollable if the class breakdown list is taller than the viewport.

---

## 11. Landing Experience

`LandingOverlay` is the welcome screen that appears after loading completes, while the globe is auto-rotating beneath it. It is built entirely with GSAP animations that run sequentially on mount.

The Engineers for Exploration badge fades in first. Then the title "Global Mangrove Observatory" slides down from above. The tagline about Sentinel-2 and deep learning fades in shortly after. Finally the "Explore Florida" button scales up from slightly smaller than its final size, then enters an infinite gentle pulse animation (scaling between 1.0 and 1.04 on a one-second loop) to draw attention.

When the user clicks the button, GSAP immediately kills the pulse animation, fades the entire overlay to transparent over half a second, and then calls the `onExplore` callback. This transitions the application to the exploring stage, which flies the camera to the Florida coast and reveals the analytics panel.

---

## 12. Loading Screen

`LoadingScreen` is a full-viewport overlay that sits above the map while the WebGL context, MapLibre style, and PMTiles header are initialising. It displays a large numeric progress counter and a thin progress bar at the bottom of the screen.

The progress animation is a simple exponential ease: every 50 milliseconds, the counter moves 6 percent of the remaining distance to the target value. The target is 85 while the map is still loading, and 100 once the `ready` prop becomes true. This easing function means the counter approaches 85 quickly at first and then slows as it gets close, creating the characteristic "almost there" crawl that signals background work is in progress.

Once `ready` is true, the counter snaps to 100. After 250 milliseconds the component begins its exit animation: a Framer Motion opacity and blur transition that takes 700 milliseconds. When the exit animation completes, the `onComplete` callback fires, which advances the application to the landing stage.

The status messages shown below the counter are keyed to progress thresholds: connecting to satellite data at the start, fetching the tile index around 25 percent, warming up the map around 55 percent, and the final "Ready." message near completion.

---

## 13. Time Slider

`TimeSlider` is a control at the bottom of the screen that allows navigating between data years. The current implementation only has data for 2023, so the previous and next buttons are disabled and the slider input is hidden. The component is fully prepared to support multiple years: it receives `yearMin`, `yearMax`, `selectedYear`, and `onYearChange` props, and renders prev/next buttons and a range slider when `yearMin !== yearMax`.

When the year changes, `page.tsx` passes the new value down to `MangroveGlobe` as `selectedYear`, which triggers the `useEffect` in `MangroveGlobe` that rebuilds the Deck.gl layer with the new year. The `updateTriggers` in the `GeoJsonLayer` ensure that only the `getFillColor` and `getElevation` accessors are re-evaluated when the year changes, rather than reprocessing the entire tile geometry.

---

## 14. Performance Architecture

The platform is designed around several interlocking performance techniques, each targeting a specific bottleneck.

**Pre-warming the tile index.** The PMTiles header and root spatial index are fetched as early as possible — before the map has even finished loading. This means the first tile requests, which arrive when the user clicks "Explore Florida", can be resolved by consulting the already-cached index rather than fetching it over the network.

**Web worker tile decoding.** Every MVT tile arrives as a binary buffer and must be parsed into GeoJSON features. This parsing is done inside a web worker via `@loaders.gl/mvt` with `worker: true`. Offloading this work to a background thread means the main thread (and therefore the animation and rendering loop) is never blocked by tile parsing, even when a dozen tiles arrive simultaneously during a rapid zoom.

**Stable layer identity.** The Deck.gl `TileLayer` has the stable string ID `'mangrove-landcover'`. When `buildMangroveLayer` is called again (only when `selectedYear` changes), Deck.gl sees a layer with the same ID and performs a diff against the previous props rather than tearing down and recreating the layer from scratch. The tile cache is preserved across layer updates, so already-loaded tiles do not need to be re-fetched.

**Zero-allocation colour lookup.** The `getColor` function returns a pre-computed tuple from a module-level cache object. It performs no allocation on each call. Since Deck.gl calls `getFillColor` for every visible polygon on every rendered frame, this matters at scale.

**RAF-throttled hover.** Capping hover events to one per animation frame prevents React from scheduling more re-renders than the display can show. The tooltip update rate matches the display refresh rate exactly, eliminating wasted work.

**Scope-limited state updates.** Hover state lives inside `MangroveGlobe` and does not propagate to `page.tsx`. Feature picking results are debounced to 300 milliseconds and only sent to `page.tsx` after the user finishes moving. These two design choices ensure that the most frequent user interactions (moving the mouse, panning the map) cause the minimum possible amount of React reconciliation work.

**VRAM tile cache.** Deck.gl keeps up to 512 tiles in GPU memory. Tiles evicted from the cache are not destroyed — they are evicted from VRAM but remain in the browser's HTTP cache. If the user pans back to a recently evicted area, the tile data is re-uploaded from memory rather than re-fetched over the network.

**Bounding-box extent.** Setting the `extent` parameter restricts tile requests to Florida's geographic bounds. Without this, the TileLayer would request tiles for the entire world at the current zoom level, most of which would return empty from the PMTiles archive. The extent saves both network requests and PMTiles range lookups.

---

## 15. Environment Variables

The application reads two environment variables at build time.

`NEXT_PUBLIC_MAPTILER_KEY` is the API key for MapTiler's satellite tile service. This key is embedded in the map style URL. Without a valid key the base satellite imagery will not load. MapTiler offers a free tier for development.

`NEXT_PUBLIC_PMTILES_URL` overrides the default PMTiles file location, which is `/florida_mangroves.pmtiles` (the file in the `public` directory). This variable can be set to a CDN URL pointing to the archive for production deployments, which avoids serving a 358 MB file from the Next.js application server.

Both variables must be prefixed with `NEXT_PUBLIC_` because they are accessed in client-side code (the browser cannot read server-only environment variables). They are typically stored in a `.env.local` file at the root of the `Observatory/web` directory and are not committed to version control.
