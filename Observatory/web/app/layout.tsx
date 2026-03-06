import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import 'maplibre-gl/dist/maplibre-gl.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'E4E Global Mangrove Observatory',
  description:
    'Mapping the world\'s mangroves at 10-metre resolution using Sentinel-2 imagery and deep learning. Built by Engineers for Exploration, UC San Diego.',
  openGraph: {
    title: 'E4E Global Mangrove Observatory',
    description: 'Interactive 3D mangrove mapping at 10-metre resolution.',
    type: 'website',
  },
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full overflow-hidden">
      <body
        className={`${inter.className} h-full overflow-hidden bg-[#0d1117] text-white antialiased`}
      >
        {children}
      </body>
    </html>
  )
}
