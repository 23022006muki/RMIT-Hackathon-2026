import { NextResponse } from 'next/server'
import { getCounts } from '../../../lib/records'

export const runtime = 'nodejs'

export async function GET() {
  return NextResponse.json(await getCounts())
}
