'use client'

import { useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface LoadingScreenProps {
  ready: boolean
  onComplete: () => void
}

const STATUS_MESSAGES = [
  { at:  0, text: 'Connecting to satellite data...' },
  { at: 25, text: 'Fetching tile index...' },
  { at: 55, text: 'Warming up the map...' },
  { at: 80, text: 'Almost ready...' },
  { at: 99, text: 'Ready.' },
]

export default function LoadingScreen({ ready, onComplete }: LoadingScreenProps) {
  const [progress, setProgress] = useState(0)
  const [visible, setVisible] = useState(true)
  const onCompleteRef = useRef(onComplete)
  onCompleteRef.current = onComplete

  useEffect(() => {
    let animFrame: number

    const tick = () => {
      setProgress((prev) => {
        // Hold at 85% until the ready signal arrives, then run to 100
        const target = ready ? 100 : 85
        const next = prev + (target - prev) * 0.06

        // Snap to target when close enough to avoid infinite crawl
        if (Math.abs(target - next) < 0.15) return target
        return next
      })
      animFrame = requestAnimationFrame(tick)
    }

    animFrame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(animFrame)
  }, [ready])

  // When progress hits 100, wait a beat then exit
  useEffect(() => {
    if (progress < 100) return
    const t = setTimeout(() => {
      setVisible(false)
      // Give the exit animation time to finish before unmounting
      setTimeout(() => onCompleteRef.current(), 700)
    }, 250)
    return () => clearTimeout(t)
  }, [progress])

  const statusText =
    [...STATUS_MESSAGES].reverse().find((s) => progress >= s.at)?.text ??
    STATUS_MESSAGES[0].text

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          initial={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.7, ease: [0.4, 0, 0.2, 1] }}
          className="fixed inset-0 z-50 bg-[#0d1117] flex flex-col items-center justify-center select-none"
        >
          {/* E4E label */}
          <p className="text-xs font-semibold tracking-[0.3em] uppercase text-teal-400/50 mb-16">
            Engineers for Exploration · UCSD
          </p>

          {/* Percentage counter */}
          <div
            className="font-black tabular-nums text-white leading-none"
            style={{ fontSize: 'clamp(5rem, 18vw, 12rem)', fontVariantNumeric: 'tabular-nums' }}
          >
            {Math.floor(progress)}
          </div>

          {/* Status message */}
          <p className="mt-4 mb-16 text-sm text-white/30 tracking-wide h-5">
            {statusText}
          </p>

          {/* Progress bar — pinned to bottom edge */}
          <div className="absolute bottom-0 left-0 right-0 h-px bg-white/5">
            <div
              className="h-full bg-teal-500 transition-none"
              style={{ width: `${progress}%` }}
            />
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
