'use client'

import { motion } from 'framer-motion'

interface TimeSliderProps {
  yearMin: number
  yearMax: number
  selectedYear: number
  onYearChange: (year: number) => void
}

export default function TimeSlider({ yearMin, yearMax, selectedYear, onYearChange }: TimeSliderProps) {
  const multiYear = yearMax > yearMin

  return (
    <motion.div
      initial={{ y: 80, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ type: 'spring', stiffness: 120, damping: 20, delay: 0.2 }}
      className="fixed bottom-6 left-1/2 -translate-x-1/2 z-10
                 bg-black/50 backdrop-blur-2xl border border-white/10 rounded-2xl
                 px-8 py-4 flex flex-col items-center gap-1
                 shadow-[0_8px_32px_rgba(0,0,0,0.5)]"
    >
      {/* Controls row */}
      <div className="flex items-center gap-5">
        <button
          onClick={() => multiYear && onYearChange(Math.max(yearMin, selectedYear - 1))}
          disabled={!multiYear || selectedYear <= yearMin}
          className="text-slate-400 hover:text-white disabled:opacity-20 transition-colors text-lg"
          aria-label="Previous year"
        >
          ◀
        </button>

        {/* Year display */}
        <div className="flex flex-col items-center">
          <span className="text-4xl font-black text-white tabular-nums tracking-tight leading-none">
            {selectedYear}
          </span>
        </div>

        <button
          onClick={() => multiYear && onYearChange(Math.min(yearMax, selectedYear + 1))}
          disabled={!multiYear || selectedYear >= yearMax}
          className="text-slate-400 hover:text-white disabled:opacity-20 transition-colors text-lg"
          aria-label="Next year"
        >
          ▶
        </button>
      </div>

      {/* Slider track — only shown for multi-year data */}
      {multiYear && (
        <input
          type="range"
          min={yearMin}
          max={yearMax}
          value={selectedYear}
          onChange={(e) => onYearChange(Number(e.target.value))}
          className="w-48 accent-teal-400"
        />
      )}

      {/* Caption */}
      <p className="text-xs text-slate-500 tracking-wider mt-0.5">
        Sentinel-2 · ESA WorldCover
      </p>
    </motion.div>
  )
}
