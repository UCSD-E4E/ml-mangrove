'use client'

import { useState, useCallback, useEffect, useRef, useImperativeHandle } from 'react'
import maplibregl from 'maplibre-gl'
import { MapboxOverlay } from '@deck.gl/mapbox'
import { buildMangroveLayer, HoverInfo } from '@/lib/layerBuilder'
import HoverTooltip from '@/components/HoverTooltip'

const MAPTILER_KEY = process.env.NEXT_PUBLIC_MAPTILER_KEY ?? ''

export interface GlobeHandle {
  flyTo: (lng: number, lat: number, zoom: number) => void
}

interface MangroveGlobeProps {
  // React 19: ref is passed as a regular prop
  ref?: React.Ref<GlobeHandle>
  rotating: boolean
  selectedYear: number
  onFeaturesChange: (features: any[]) => void
  onReady?: () => void
}

function getCentroid(geometry: { type: string; coordinates: any }): [number, number] | null {
  let ring: number[][] | undefined
  if (geometry.type === 'MultiPolygon') {
    ring = geometry.coordinates?.[0]?.[0]
  } else {
    ring = geometry.coordinates?.[0]
  }
  if (!ring?.length) return null
  const lng = ring.reduce((s: number, c: number[]) => s + c[0], 0) / ring.length
  const lat = ring.reduce((s: number, c: number[]) => s + c[1], 0) / ring.length
  if (!isFinite(lng) || !isFinite(lat)) return null
  return [lng, lat]
}

export default function MangroveGlobe({
  ref,
  rotating,
  selectedYear,
  onFeaturesChange,
  onReady,
}: MangroveGlobeProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef = useRef<maplibregl.Map | null>(null)
  const overlayRef = useRef<MapboxOverlay | null>(null)
  const rafRef = useRef<number>(0)
  const rotatingRef = useRef(rotating)
  const pickTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const hoverRafRef = useRef<number | null>(null)
  const pendingHoverRef = useRef<HoverInfo | null>(null)
  // Track whether a zoom gesture is in progress so we can skip pickObjects during zoom
  const zoomingRef = useRef(false)

  // Hover state is internal — changes here never cause page.tsx to re-render
  const [hovered, setHovered] = useState<HoverInfo | null>(null)

  // Keep rotatingRef in sync. When rotation is re-enabled, restart the RAF loop
  // if it isn't already running.
  useEffect(() => {
    rotatingRef.current = rotating
    if (rotating && rafRef.current === 0 && mapRef.current) {
      const map = mapRef.current
      const rotate = () => {
        if (rotatingRef.current && mapRef.current) {
          map.setBearing((map.getBearing() + 0.025) % 360)
          rafRef.current = requestAnimationFrame(rotate)
        } else {
          rafRef.current = 0
        }
      }
      rafRef.current = requestAnimationFrame(rotate)
    }
  }, [rotating])

  // Expose flyTo via ref (React 19 style)
  useImperativeHandle(ref, () => ({
    flyTo: (lng, lat, zoom) => {
      mapRef.current?.flyTo({
        center: [lng, lat],
        zoom,
        pitch: 60,
        duration: 2500,
        essential: true,
      })
    },
  }))

  // RAF-throttled hover: Deck.gl fires onHover at native mousemove rate (100–200/sec).
  // Funnelling through rAF caps React setState calls at the display refresh rate (~60fps).
  // setHovered is stable (from useState), so this callback never changes — the layer is
  // never rebuilt due to hover activity.
  const throttledHover = useCallback((info: HoverInfo | null) => {
    pendingHoverRef.current = info
    if (hoverRafRef.current !== null) return
    hoverRafRef.current = requestAnimationFrame(() => {
      setHovered(pendingHoverRef.current)
      hoverRafRef.current = null
    })
  }, []) // no deps — setHovered is stable

  // Rebuild Deck.gl layer when year changes
  useEffect(() => {
    if (!overlayRef.current) return
    overlayRef.current.setProps({
      layers: [
        buildMangroveLayer({
          selectedYear,
          onHover: throttledHover,
          onClick: (info) => {
            if (!mapRef.current || !info.object) return
            const center = getCentroid(info.object.geometry)
            if (!center) return
            mapRef.current.flyTo({ center, zoom: 13, pitch: 60, duration: 1800, essential: true })
          },
        }),
      ],
    })
  }, [selectedYear, throttledHover])

  // Initialise map once on mount
  useEffect(() => {
    if (!containerRef.current) return

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: `https://api.maptiler.com/maps/satellite/style.json?key=${MAPTILER_KEY}`,
      center: [-82, 27],
      zoom: 3,
      pitch: 30,
      bearing: 0,
    })
    mapRef.current = map

    map.on('error', (e) => console.error('[Globe] map error:', e))

    map.on('load', () => {
      // Deck.gl overlay
      const overlay = new MapboxOverlay({
        interleaved: true,
        layers: [
          buildMangroveLayer({
            selectedYear,
            onHover: throttledHover,
            onClick: (info) => {
              if (!info.object) return
              const center = getCentroid(info.object.geometry)
              if (!center) return
              map.flyTo({ center, zoom: 13, pitch: 60, duration: 1800, essential: true })
            },
          }),
        ],
      })
      map.addControl(overlay as any)
      overlayRef.current = overlay
      onReady?.()
    })

    // Auto-rotation RAF loop — only calls setBearing when actually rotating,
    // so it has zero cost when the user is interacting with the map.
    const rotate = () => {
      if (rotatingRef.current && mapRef.current) {
        mapRef.current.setBearing((mapRef.current.getBearing() + 0.025) % 360)
        rafRef.current = requestAnimationFrame(rotate)
      } else {
        rafRef.current = 0
      }
    }
    if (rotatingRef.current) {
      rafRef.current = requestAnimationFrame(rotate)
    }

    map.on('movestart', () => { rotatingRef.current = false })

    map.on('zoomstart', () => { zoomingRef.current = true })
    map.on('zoomend', () => { zoomingRef.current = false })

    map.on('moveend', () => {
      // Skip the expensive pickObjects call when the movement was a zoom gesture.
      // Features in the analytics panel are irrelevant during mid-zoom anyway.
      if (zoomingRef.current) return

      if (pickTimerRef.current) clearTimeout(pickTimerRef.current)
      pickTimerRef.current = setTimeout(() => {
        if (!overlayRef.current) return
        const canvas = map.getCanvas()
        // Pick only the centre 60% of the canvas to avoid expensive edge polygons
        // and reduce the search space significantly at high zoom levels.
        const pw = Math.round(canvas.clientWidth * 0.6)
        const ph = Math.round(canvas.clientHeight * 0.6)
        const px = Math.round((canvas.clientWidth - pw) / 2)
        const py = Math.round((canvas.clientHeight - ph) / 2)
        const picked = overlayRef.current.pickObjects({ x: px, y: py, width: pw, height: ph })
        onFeaturesChange(picked.map((p: any) => p.object).filter(Boolean))
      }, 400)
    })

    return () => {
      cancelAnimationFrame(rafRef.current)
      if (pickTimerRef.current) clearTimeout(pickTimerRef.current)
      map.remove()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <div className="absolute inset-0">
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
      <HoverTooltip info={hovered} />
    </div>
  )
}
