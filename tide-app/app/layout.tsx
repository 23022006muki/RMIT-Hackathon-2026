import type { Metadata } from 'next'
import type { ReactNode } from 'react'
import { TideCloakProvider } from '@tidecloak/nextjs'
import tcConfig from '../tidecloak.json'
import './globals.css'

export const metadata: Metadata = {
  title: 'Signal Commons · RMIT Hackathon 2026',
  description: 'Share multilingual AI safety signals while keeping original prompts under TideCloak protection.',
}

export default function RootLayout({ children }: { children: ReactNode }) {
  return <html lang="en"><body><TideCloakProvider config={tcConfig}>{children}</TideCloakProvider></body></html>
}
