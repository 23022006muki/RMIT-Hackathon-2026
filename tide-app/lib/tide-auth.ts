import type { NextRequest } from 'next/server'
import { verifyTideCloakToken, type TidecloakConfig } from '@tidecloak/nextjs/server'
import rawConfig from '../tidecloak.json'

export type VerifiedUser = { id: string; name: string }

export async function authenticatedUser(request: NextRequest): Promise<VerifiedUser | null> {
  const header = request.headers.get('authorization')
  if (!header?.startsWith('Bearer ')) return null
  const token = header.slice(7)
  if (!token) return null
  try {
    const payload = await verifyTideCloakToken(rawConfig as TidecloakConfig, token, ['appUser']) as Record<string, unknown> | null
    if (!payload) return null
    const id = payload.vuid ?? payload.sub
    if (typeof id !== 'string' || !id) return null
    return { id, name: String(payload.preferred_username ?? 'Contributor') }
  } catch {
    return null
  }
}
