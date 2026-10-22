import { createHmac, randomBytes, timingSafeEqual } from 'node:crypto'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import path from 'node:path'

const lifetimeMs = 5 * 60 * 1000
const usedReceipts = new Map<string, number>()
const keyPath = path.join(process.cwd(), '.data', 'receipt.key')

function signingKey(): Buffer {
  const configured = process.env.RECEIPT_SIGNING_KEY
  if (configured) {
    const key = Buffer.from(configured, 'hex')
    if (key.length < 32) throw new Error('RECEIPT_SIGNING_KEY must be at least 32 random bytes in hex')
    return key
  }
  try { return readFileSync(keyPath) } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error
  }
  mkdirSync(path.dirname(keyPath), { recursive: true })
  try { writeFileSync(keyPath, randomBytes(32), { flag: 'wx', mode: 0o600 }) } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error
  }
  return readFileSync(keyPath)
}

export type ClassificationReceipt = {
  owner: string
  language: 'pt' | 'fr' | 'it' | 'en'
  label: 'BENIGN' | 'JAILBREAK' | 'PROMPT_INJECTION'
  modelId: string
  expiresAt: number
}

export function issueReceipt(data: Omit<ClassificationReceipt, 'expiresAt'>): string {
  const payload = Buffer.from(JSON.stringify({ ...data, expiresAt: Date.now() + lifetimeMs })).toString('base64url')
  const signature = createHmac('sha256', signingKey()).update(payload).digest('base64url')
  return `${payload}.${signature}`
}

export function verifyReceipt(receipt: string, owner: string): ClassificationReceipt | null {
  if (receipt.length > 4_000) return null
  const parts = receipt.split('.')
  if (parts.length !== 2) return null
  const expected = createHmac('sha256', signingKey()).update(parts[0]).digest()
  let supplied: Buffer
  try { supplied = Buffer.from(parts[1], 'base64url') } catch { return null }
  if (supplied.length !== expected.length || !timingSafeEqual(supplied, expected)) return null
  try {
    const data = JSON.parse(Buffer.from(parts[0], 'base64url').toString('utf8')) as ClassificationReceipt
    if (data.owner !== owner || !Number.isFinite(data.expiresAt) || data.expiresAt < Date.now()) return null
    if (!['pt', 'fr', 'it', 'en'].includes(data.language)) return null
    if (!['BENIGN', 'JAILBREAK', 'PROMPT_INJECTION'].includes(data.label)) return null
    if (!data.modelId) return null
    return data
  } catch { return null }
}

export function consumeReceipt(receipt: string, owner: string): ClassificationReceipt | null {
  const data = verifyReceipt(receipt, owner)
  if (!data) return null
  for (const [signature, expiresAt] of usedReceipts) {
    if (expiresAt < Date.now()) usedReceipts.delete(signature)
  }
  if (usedReceipts.has(receipt)) return null
  usedReceipts.set(receipt, data.expiresAt)
  return data
}
