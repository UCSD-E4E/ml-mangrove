'use client'

import { useState, useRef, useCallback } from 'react'
import dynamic from 'next/dynamic'
import type { GlobeHandle } from '@/components/MangroveGlobe'
import LandingOverlay from '@/components/LandingOverlay'
import AnalyticsPanel from '@/components/AnalyticsPanel'
import TimeSlider from '@/components/TimeSlider'
import HoverTooltip from '@/components/HoverTooltip'
import type { HoverInfo } from '@/lib/layerBuilder'
import { AnimatePresence } from 'framer-motion'

// MapLibre + Deck.gl must be client-only (they need window/WebGL)
const MangroveGlobe = dynamic(() => import('@/components/MangroveGlobe'), { ssr: false })

const YEAR_MIN = 2023
const YEAR_MAX = 2023

export default function Page() {
  const [isLanding, setIsLanding] = useState(true)
  const [hovered, setHovered] = useState<HoverInfo | null>(null)
  const [visibleFeatures, setVisibleFeatures] = useState<any[]>([])
  const [selectedYear, setSelectedYear] = useState(2023)
  const globeRef = useRef<GlobeHandle>(null)

  const handleExplore = useCallback(() => {
    setIsLanding(false)
    globeRef.current?.flyTo(-80.5, 25.5, 8)
  }, [])

  const handleFeaturesChange = useCallback((features: any[]) => {
    setVisibleFeatures(features)
  }, [])

  return (
    <div className="relative w-full bg-[#0d1117]" style={{ height: '100dvh' }}>
      {/* Full-screen map */}
      <MangroveGlobe
        ref={globeRef}
        rotating={isLanding}
        selectedYear={selectedYear}
        onHover={setHovered}
        onFeaturesChange={handleFeaturesChange}
      />

      {/* Landing screen */}
      <AnimatePresence>
        {isLanding && <LandingOverlay onExplore={handleExplore} />}
      </AnimatePresence>

      {/* Post-landing UI */}
      <AnimatePresence>
        {!isLanding && (
          <>
            <AnalyticsPanel features={visibleFeatures} />
            <TimeSlider
              yearMin={YEAR_MIN}
              yearMax={YEAR_MAX}
              selectedYear={selectedYear}
              onYearChange={setSelectedYear}
            />
          </>
        )}
      </AnimatePresence>

      {/* Hover tooltip — always present, visibility controlled by hovered state */}
      <HoverTooltip info={hovered} />
    </div>
  )
}
