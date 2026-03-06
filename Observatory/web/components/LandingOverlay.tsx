'use client'

import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'

interface LandingOverlayProps {
  onExplore: () => void
}

export default function LandingOverlay({ onExplore }: LandingOverlayProps) {
  const overlayRef = useRef<HTMLDivElement>(null)
  const badgeRef = useRef<HTMLDivElement>(null)
  const titleRef = useRef<HTMLHeadingElement>(null)
  const taglineRef = useRef<HTMLParagraphElement>(null)
  const buttonRef = useRef<HTMLButtonElement>(null)
  const glowTweenRef = useRef<gsap.core.Tween | null>(null)

  useEffect(() => {
    const ctx = gsap.context(() => {
      const tl = gsap.timeline()

      tl.fromTo(badgeRef.current, { opacity: 0, y: 10 }, { opacity: 1, y: 0, duration: 0.4, ease: 'power2.out' })
        .fromTo(titleRef.current, { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 0.7, ease: 'power3.out' }, '-=0.1')
        .fromTo(taglineRef.current, { opacity: 0, y: 15 }, { opacity: 1, y: 0, duration: 0.5, ease: 'power2.out' }, '-=0.3')
        .fromTo(buttonRef.current, { opacity: 0, scale: 0.9 }, { opacity: 1, scale: 1, duration: 0.4, ease: 'back.out(1.4)' }, '-=0.1')
        .then(() => {
          // Infinite subtle pulse on button
          glowTweenRef.current = gsap.to(buttonRef.current, {
            scale: 1.04,
            duration: 1,
            ease: 'sine.inOut',
            yoyo: true,
            repeat: -1,
          })
        })
    })
    return () => ctx.revert()
  }, [])

  const handleExplore = () => {
    glowTweenRef.current?.kill()
    gsap.to(overlayRef.current, {
      opacity: 0,
      duration: 0.5,
      ease: 'power2.inOut',
      onComplete: onExplore,
    })
  }

  return (
    <div
      ref={overlayRef}
      className="fixed inset-0 z-20 flex flex-col items-center justify-center bg-black/40 backdrop-blur-sm"
    >
      {/* E4E badge */}
      <div ref={badgeRef} className="mb-10 opacity-0">
        <span className="text-xs font-semibold tracking-[0.25em] uppercase text-teal-400 border border-teal-400/30 bg-teal-400/10 rounded-full px-4 py-1.5">
          Engineers for Exploration · UCSD
        </span>
      </div>

      {/* Title */}
      <h1
        ref={titleRef}
        className="opacity-0 text-center text-5xl md:text-6xl lg:text-7xl font-black text-white tracking-tight leading-tight mb-4"
      >
        Global Mangrove
        <br />
        <span className="text-teal-400">Observatory</span>
      </h1>

      {/* Tagline */}
      <p
        ref={taglineRef}
        className="opacity-0 text-center text-lg text-slate-400 italic max-w-md mb-12 leading-relaxed"
      >
        Mapping the world&apos;s mangroves at 10-metre resolution
        <br />
        using Sentinel-2 &amp; deep learning.
      </p>

      {/* Explore button */}
      <button
        ref={buttonRef}
        onClick={handleExplore}
        className="opacity-0 group flex items-center gap-3 px-8 py-4 rounded-full bg-teal-500 hover:bg-teal-400 text-black font-bold text-base transition-colors duration-200 shadow-[0_0_40px_rgba(20,184,166,0.4)]"
      >
        <span>Explore Florida</span>
        <svg
          className="w-4 h-4 transition-transform duration-200 group-hover:translate-x-1"
          fill="none" stroke="currentColor" viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M9 5l7 7-7 7" />
        </svg>
      </button>

      {/* Subtle scroll hint at bottom */}
      <div className="absolute bottom-8 flex flex-col items-center gap-2 opacity-40">
        <span className="text-xs text-slate-400 tracking-widest uppercase">Scroll to zoom</span>
        <div className="w-px h-6 bg-gradient-to-b from-slate-400 to-transparent" />
      </div>
    </div>
  )
}
