'use client'

import { useCallback, useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useTideCloak } from '@tidecloak/nextjs'

type Language = 'pt' | 'fr' | 'it' | 'en'
type Label = 'BENIGN' | 'JAILBREAK' | 'PROMPT_INJECTION'
type CaseSummary = { id: string; language: Language; label: Label; modelId: string; createdAt: string }
type Classification = { label: Label; modelId: string; language: Language; truncated: boolean; receipt: string }
type Stats = { total: number; byLabel: Record<Label, number>; byLanguage: Record<Language, number> }

async function responseBody(response: Response): Promise<Record<string, unknown>> {
  const data: unknown = await response.json()
  return data && typeof data === 'object' ? data as Record<string, unknown> : {}
}

export default function HomePage() {
  const router = useRouter()
  const { authenticated, isInitializing, getToken, getValueFromIdToken, logout, doEncrypt, doDecrypt } = useTideCloak()
  const [prompt, setPrompt] = useState('')
  const [language, setLanguage] = useState<Language>('en')
  const [busy, setBusy] = useState(false)
  const [revealBusy, setRevealBusy] = useState<string | null>(null)
  const [withdrawBusy, setWithdrawBusy] = useState<string | null>(null)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [classification, setClassification] = useState<Classification | null>(null)
  const [cases, setCases] = useState<CaseSummary[]>([])
  const [revealed, setRevealed] = useState<Record<string, string>>({})
  const [stats, setStats] = useState<Stats | null>(null)

  const loadCases = useCallback(async () => {
    const token = await getToken()
    if (!token) return
    const response = await fetch('/api/cases', { headers: { Authorization: `Bearer ${token}` }, cache: 'no-store' })
    const data = await responseBody(response)
    if (response.ok && Array.isArray(data.cases)) setCases(data.cases as CaseSummary[])
  }, [getToken])

  const loadStats = useCallback(async () => {
    const response = await fetch('/api/stats', { cache: 'no-store' })
    if (response.ok) setStats(await response.json() as Stats)
  }, [])

  useEffect(() => {
    if (!isInitializing && !authenticated) router.replace('/')
  }, [authenticated, isInitializing, router])

  useEffect(() => {
    if (!authenticated) return
    void loadCases().catch(() => setError('Could not load your contributions.'))
    void loadStats().catch(() => {})
  }, [authenticated, loadCases, loadStats])

  async function contribute() {
    setError('')
    setMessage('')
    setClassification(null)
    if (prompt.trim().length < 10) { setError('Enter at least 10 characters.'); return }
    setBusy(true)
    try {
      const token = await getToken()
      if (!token) throw new Error('Please sign in again.')

      const classified = await fetch('/api/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ text: prompt, language }),
      })
      const result = await responseBody(classified)
      if (!classified.ok) throw new Error(String(result.error || 'Classification failed.'))
      const outcome = result as Classification
      setClassification(outcome)

      const encrypted = await doEncrypt([{ data: prompt, tags: ['message'] }]) as unknown
      if (!Array.isArray(encrypted) || typeof encrypted[0] !== 'string') throw new Error('TideCloak encryption did not return ciphertext.')
      const saved = await fetch('/api/cases', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${token}` },
        body: JSON.stringify({ encryptedPrompt: encrypted[0], receipt: outcome.receipt }),
      })
      const savedResult = await responseBody(saved)
      if (!saved.ok) throw new Error(String(savedResult.error || 'Could not save sealed contribution.'))
      setPrompt('')
      setMessage('Classified and sealed. Your original is available only through your Tide identity.')
      await Promise.all([loadCases(), loadStats()])
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Contribution failed.')
    } finally {
      setBusy(false)
    }
  }

  async function reveal(id: string) {
    setError('')
    if (revealed[id]) { setRevealed(previous => { const next = { ...previous }; delete next[id]; return next }); return }
    setRevealBusy(id)
    try {
      const token = await getToken()
      if (!token) throw new Error('Please sign in again.')
      const response = await fetch(`/api/cases/${encodeURIComponent(id)}`, {
        headers: { Authorization: `Bearer ${token}` }, cache: 'no-store',
      })
      const data = await responseBody(response)
      if (!response.ok) throw new Error(String(data.error || 'Could not retrieve contribution.'))
      if (typeof data.encryptedPrompt !== 'string') throw new Error('Stored ciphertext is invalid.')
      const plain = await doDecrypt([{ encrypted: data.encryptedPrompt, tags: ['message'] }]) as unknown
      if (!Array.isArray(plain) || typeof plain[0] !== 'string') throw new Error('TideCloak could not decrypt this contribution.')
      setRevealed(previous => ({ ...previous, [id]: plain[0] }))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Decryption failed.')
    } finally {
      setRevealBusy(null)
    }
  }

  async function withdraw(id: string) {
    if (!window.confirm('Withdraw this contribution? Its ciphertext and classification will be removed from this research commons.')) return
    setError('')
    setWithdrawBusy(id)
    try {
      const token = await getToken()
      if (!token) throw new Error('Please sign in again.')
      const response = await fetch(`/api/cases/${encodeURIComponent(id)}`, {
        method: 'DELETE', headers: { Authorization: `Bearer ${token}` },
      })
      if (!response.ok) {
        const data = await responseBody(response)
        throw new Error(String(data.error || 'Could not withdraw contribution.'))
      }
      setRevealed(previous => { const next = { ...previous }; delete next[id]; return next })
      setMessage('Contribution withdrawn from the research commons.')
      await Promise.all([loadCases(), loadStats()])
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Withdrawal failed.')
    } finally {
      setWithdrawBusy(null)
    }
  }

  if (isInitializing || !authenticated) return <main className="workspace"><p>Checking Tide identity…</p></main>
  const name = String(getValueFromIdToken('preferred_username') || 'Contributor')

  return <main>
    <header className="topbar"><a className="brand" href="/"><span className="brand-mark">◈</span> SIGNAL COMMONS</a><span className="topline">PRIVATE CONTRIBUTOR WORKSPACE</span></header>
    <div className="workspace">
      <div className="workspace-head"><div><p className="eyebrow">SIGNED IN WITH TIDECLOAK</p><h1>Hello, {name}.</h1><p>Share a research signal. Keep the source under your identity.</p></div><button className="quiet-button" onClick={() => void logout()}>Sign out ↗</button></div>
      <div className="work-grid">
        <section className="card"><span className="overline">01 / CONTRIBUTE</span><h2>Classify and seal a prompt</h2><p>Our Challenge 1 model reads your text for classification. TideCloak then encrypts the original in your browser before the app stores it.</p>
          <label className="field" htmlFor="prompt">Original prompt</label><textarea id="prompt" className="textbox" value={prompt} maxLength={20000} onChange={event => setPrompt(event.target.value)} placeholder="Paste a multilingual prompt for analysis…" />
          <div className="row"><label className="field" htmlFor="language">Language</label><select id="language" value={language} onChange={event => setLanguage(event.target.value as Language)}><option value="en">English</option><option value="pt">Português</option><option value="fr">Français</option><option value="it">Italiano</option></select><span className="note">{prompt.length} / 20,000 characters</span></div>
          <button className="primary" disabled={busy || prompt.trim().length < 10} onClick={() => void contribute()}>{busy ? 'Classifying and sealing…' : 'Contribute securely'} <span>↗</span></button>
          {classification && <div className="result"><span className="overline">CHALLENGE 1 RESULT</span><strong>{classification.label.replace('_', ' ')}</strong><small>Model: {classification.modelId}{classification.truncated ? ' · Input truncated to model window' : ''}</small></div>}
          {message && <div className="result" role="status">{message}</div>}
          {error && <div className="error" role="alert">{error}</div>}
        </section>
        <div><section className="card privacy"><span className="overline">THE TRUST CONSTRAINT</span><h2>Your text, your key.</h2><p>Only ciphertext is saved with the case. Classification metadata powers shared counts. Sign in as the same Tide identity to retrieve and decrypt your original.</p><span className="pill">TIDECLOAK SELF ENCRYPTION</span></section>
          <section className="card" style={{ marginTop: 22 }}><span className="overline">COMMUNITY VIEW</span><h2>Patterns, not raw text</h2><div className="stats-compact"><div><b>{stats?.total ?? '—'}</b><span>SEALED PROMPTS</span></div><div><b>{stats?.byLabel.JAILBREAK ?? '—'}</b><span>JAILBREAK SIGNALS</span></div><div><b>{stats?.byLabel.PROMPT_INJECTION ?? '—'}</b><span>INJECTION SIGNALS</span></div></div></section></div>
      </div>
      <section className="card" style={{ marginTop: 22 }}><span className="overline">02 / YOUR CONTRIBUTIONS</span><h2>Revisit the originals you own</h2><p>Only your own cases appear here. The server returns ciphertext after checking your Tide token; decryption happens in this browser.</p>
        <div className="case-list">{cases.length === 0 ? <p>No contributions yet.</p> : cases.map(entry => <article className="case-item" key={entry.id}><strong>{entry.label.replace('_', ' ')} <span className="pill">{entry.language.toUpperCase()}</span></strong><small>{new Date(entry.createdAt).toLocaleString()} · {entry.modelId}</small><div className="case-actions"><button className="quiet-button" disabled={revealBusy === entry.id || withdrawBusy === entry.id} onClick={() => void reveal(entry.id)}>{revealed[entry.id] ? 'Hide original' : revealBusy === entry.id ? 'Decrypting…' : 'Decrypt my original'}</button><button className="quiet-button danger-button" disabled={withdrawBusy === entry.id} onClick={() => void withdraw(entry.id)}>{withdrawBusy === entry.id ? 'Withdrawing…' : 'Withdraw contribution'}</button></div>{revealed[entry.id] && <div className="revealed">{revealed[entry.id]}</div>}</article>)}</div>
      </section>
    </div>
  </main>
}
