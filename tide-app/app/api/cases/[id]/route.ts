import { NextRequest, NextResponse } from 'next/server'
import { authenticatedUser } from '../../../../lib/tide-auth'
import { getOwnedCase } from '../../../../lib/records'

export const runtime = 'nodejs'

export async function GET(request: NextRequest, context: { params: Promise<{ id: string }> }) {
  const user = await authenticatedUser(request)
  if (!user) return NextResponse.json({ error: 'TideCloak login required' }, { status: 401 })
  const { id } = await context.params
  const entry = await getOwnedCase(id, user.id)
  if (!entry) return NextResponse.json({ error: 'Case unavailable to this identity' }, { status: 404 })
  return NextResponse.json({ id: entry.id, encryptedPrompt: entry.encryptedPrompt, label: entry.label,
    language: entry.language, modelId: entry.modelId })
}
