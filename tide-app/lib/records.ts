import { randomUUID } from 'node:crypto'
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import path from 'node:path'

export type StoredCase = {
  id: string
  owner: string
  language: 'pt' | 'fr' | 'it' | 'en'
  label: 'BENIGN' | 'JAILBREAK' | 'PROMPT_INJECTION'
  encryptedPrompt: string
  modelId: string
  createdAt: string
}

const databasePath = path.join(process.cwd(), '.data', 'cases.json')

async function readCases(): Promise<StoredCase[]> {
  try {
    const parsed: unknown = JSON.parse(await readFile(databasePath, 'utf8'))
    return Array.isArray(parsed) ? parsed as StoredCase[] : []
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return []
    throw error
  }
}

let writes: Promise<unknown> = Promise.resolve()

export async function saveCase(record: Omit<StoredCase, 'id' | 'createdAt'>): Promise<StoredCase> {
  const entry: StoredCase = { ...record, id: randomUUID(), createdAt: new Date().toISOString() }
  writes = writes.then(async () => {
    const all = await readCases()
    all.push(entry)
    await mkdir(path.dirname(databasePath), { recursive: true })
    const pendingPath = `${databasePath}.${randomUUID()}.tmp`
    await writeFile(pendingPath, JSON.stringify(all, null, 2), 'utf8')
    await rename(pendingPath, databasePath)
  })
  try { await writes } catch (error) { writes = Promise.resolve(); throw error }
  return entry
}

export async function listOwnedCases(owner: string): Promise<StoredCase[]> {
  return (await readCases()).filter(item => item.owner === owner).reverse()
}

export async function getOwnedCase(id: string, owner: string): Promise<StoredCase | null> {
  return (await readCases()).find(item => item.id === id && item.owner === owner) ?? null
}

export async function withdrawOwnedCase(id: string, owner: string): Promise<boolean> {
  let withdrawn = false
  writes = writes.then(async () => {
    const all = await readCases()
    const remaining = all.filter(item => item.id !== id || item.owner !== owner)
    if (remaining.length === all.length) return
    await mkdir(path.dirname(databasePath), { recursive: true })
    const pendingPath = `${databasePath}.${randomUUID()}.tmp`
    await writeFile(pendingPath, JSON.stringify(remaining, null, 2), 'utf8')
    await rename(pendingPath, databasePath)
    withdrawn = true
  })
  try { await writes } catch (error) { writes = Promise.resolve(); throw error }
  return withdrawn
}

export async function getCounts() {
  const all = await readCases()
  const byLabel = { BENIGN: 0, JAILBREAK: 0, PROMPT_INJECTION: 0 }
  const byLanguage = { pt: 0, fr: 0, it: 0, en: 0 }
  for (const item of all) {
    if (item.label in byLabel && item.language in byLanguage) {
      byLabel[item.label] += 1
      byLanguage[item.language] += 1
    }
  }
  return { total: all.length, byLabel, byLanguage }
}
