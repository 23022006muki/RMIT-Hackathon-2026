import { NextRequest, NextResponse } from 'next/server'
import { classify } from '../../../lib/classifier'
import { issueReceipt } from '../../../lib/receipt'
import { authenticatedUser } from '../../../lib/tide-auth'

export const runtime = 'nodejs'

export async function POST(request: NextRequest) {
  const user = await authenticatedUser(request)
  if (!user) return NextResponse.json({ error: 'TideCloak login required' }, { status: 401 })
  let body: { text?: unknown; language?: unknown }
  try { body = await request.json() } catch { return NextResponse.json({ error: 'Invalid JSON' }, { status: 400 }) }
  if (typeof body.text !== 'string' || body.text.trim().length < 10 || body.text.length > 20_000) {
    return NextResponse.json({ error: 'Prompt must contain 10–20,000 characters' }, { status: 400 })
  }
  if (typeof body.language !== 'string' || !['pt', 'fr', 'it', 'en'].includes(body.language)) {
    return NextResponse.json({ error: 'Choose pt, fr, it or en' }, { status: 400 })
  }
  try {
    const result = await classify(body.text, body.language)
    const language = body.language as 'pt' | 'fr' | 'it' | 'en'
    const receipt = issueReceipt({ owner: user.id, language, label: result.label, modelId: result.modelId })
    return NextResponse.json({ ...result, receipt })
  } catch (error) {
    const unavailable = error instanceof Error && error.message === 'MODEL_UNAVAILABLE'
    console.error('Classifier unavailable:', error instanceof Error ? error.message : 'unknown error')
    return NextResponse.json({ error: unavailable ? 'Kaggle-trained model is not configured' : 'Classifier failed' }, { status: 503 })
  }
}
