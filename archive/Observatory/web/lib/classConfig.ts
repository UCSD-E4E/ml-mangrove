// Class taxonomy — mirrors segformer_training.ipynb exactly
// ESA values 20 (Shrubland), 70 (Snow/Ice), 100 (Moss/Lichen) absent in Florida → excluded

export const CLASS_NAMES = [
  'Tree Cover',      // 0  ESA 10
  'Grassland',       // 1  ESA 30
  'Cropland',        // 2  ESA 40
  'Built-up',        // 3  ESA 50
  'Bare/Sparse Veg', // 4  ESA 60
  'Water',           // 5  ESA 80
  'Wetland',         // 6  ESA 90
  'Mangrove',        // 7  ESA 95
] as const

export type ClassName = (typeof CLASS_NAMES)[number]

// RGB tuples [r, g, b] — sourced from CLASS_COLORS in the training notebook
export const CLASS_COLORS: Record<ClassName, [number, number, number]> = {
  'Tree Cover':      [45,  138,  78],
  'Grassland':       [200, 217, 111],
  'Cropland':        [227, 200, 120],
  'Built-up':        [192,  57,  43],
  'Bare/Sparse Veg': [212, 180, 131],
  'Water':           [ 36, 113, 163],
  'Wetland':         [118, 215, 196],
  'Mangrove':        [ 26, 188, 156],
}

// Hex equivalents for CSS usage
export const CLASS_HEX: Record<ClassName, string> = {
  'Tree Cover':      '#2d8a4e',
  'Grassland':       '#c8d96f',
  'Cropland':        '#e3c878',
  'Built-up':        '#c0392b',
  'Bare/Sparse Veg': '#d4b483',
  'Water':           '#2471a3',
  'Wetland':         '#76d7c4',
  'Mangrove':        '#1abc9c',
}

export const CLASS_ICONS: Record<ClassName, string> = {
  'Tree Cover':      '🌳',
  'Grassland':       '🌿',
  'Cropland':        '🌾',
  'Built-up':        '🏗',
  'Bare/Sparse Veg': '🪨',
  'Water':           '💧',
  'Wetland':         '🌱',
  'Mangrove':        '🌴',
}

// Pre-computed RGBA tuples — allocated once at module load, reused on every feature render.
// getFillColor is called for every visible feature on every frame; avoid heap allocation here.
const FILL_COLOR_CACHE: Record<string, [number, number, number, number]> = {}
for (const [name, [r, g, b]] of Object.entries(CLASS_COLORS)) {
  FILL_COLOR_CACHE[name] = [r, g, b, 220]
}
const FALLBACK_COLOR: [number, number, number, number] = [255, 255, 255, 220]

// Extrusion heights (metres, visually exaggerated for globe scale).
// Mangrove height is dynamic (health_index × 500) — its entry here is unused.
export const CLASS_ELEVATION: Record<ClassName, number> = {
  'Mangrove':        0,    // overridden by health_index × 500
  'Tree Cover':      120,
  'Wetland':         60,
  'Built-up':        150,
  'Grassland':       25,
  'Cropland':        15,
  'Bare/Sparse Veg': 8,
  'Water':           0,
}

/** Returns a pre-computed [r, g, b, a] tuple for a class name — zero allocation. */
export function getColor(className: string): [number, number, number, number] {
  return FILL_COLOR_CACHE[className] ?? FALLBACK_COLOR
}

/** Health index → color: green > 0.6, yellow 0.3–0.6, red < 0.3 */
export function healthColor(index: number): string {
  if (index > 0.6) return '#22c55e'   // green-500
  if (index > 0.3) return '#eab308'   // yellow-500
  return '#ef4444'                     // red-500
}
