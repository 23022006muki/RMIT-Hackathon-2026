import { NextRequest, NextResponse } from 'next/server'
import { authenticatedUser } from '../../../lib/tide-auth'
import { listOwnedCases, saveCase } from '../../../lib/records'
import { consumeReceipt } from '../../../lib/receipt'

export const runtime = 'nodejs'

export async function GET(request: NextRequest) {
  const user = await authenticatedUser(request)
  if (!user) return NextResponse.json({ error: 'TideCloak login required' }, { status: 401 })
  const cases = (await listOwnedCases(user.id)).map(({ encryptedPrompt: _ciphertext, owner: _owner, ...safe }) => safe)
  return NextResponse.json({ cases })
}

export async function POST(request: NextRequest) {
  const user = await authenticatedUser(request)
  if (!user) return NextResponse.json({ error: 'TideCloak login required' }, { status: 401 })
  let body: { encryptedPrompt?: unknown; receipt?: unknown }
  try { body = await request.json() } catch { return NextResponse.json({ error: 'Invalid JSON' }, { status: 400 }) }
  if (typeof body.encryptedPrompt !== 'string' || body.encryptedPrompt.length < 40 || body.encryptedPrompt.length > 250_000) {
    return NextResponse.json({ error: 'TideCloak ciphertext required' }, { status: 400 })
  }
  const receipt = typeof body.receipt === 'string' ? consumeReceipt(body.receipt, user.id) : null
  if (!receipt) return NextResponse.json({ error: 'Valid classifier receipt required' }, { status: 403 })
  const entry = await saveCase({ owner: user.id, language: receipt.language, label: receipt.label,
    modelId: receipt.modelId, encryptedPrompt: body.encryptedPrompt })
  return NextResponse.json({ id: entry.id, label: entry.label, createdAt: entry.createdAt }, { status: 201 })
}
