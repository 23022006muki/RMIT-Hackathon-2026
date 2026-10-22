'use client'

import { useEffect, useState } from 'react'
import { useRouter } from 'next/navigation'
import { useTideCloak } from '@tidecloak/nextjs'
import config from '../tidecloak.json'

type Stats = { total: number; byLabel: Record<string, number>; byLanguage: Record<string, number> }

export default function Landing() {
  const { login, authenticated } = useTideCloak()
  const router = useRouter()
  const [stats, setStats] = useState<Stats | null>(null)
  const [loginError, setLoginError] = useState('')
  const configured = Boolean((config as { realm?: string }).realm)

  async function signIn() {
    setLoginError('')
    try { await login() } catch { setLoginError('Tide sign-in could not start. Check that the TideCloak server is running.') }
  }

  useEffect(() => { if (authenticated) router.replace('/home') }, [authenticated, router])
  useEffect(() => { fetch('/api/stats').then(response => response.json()).then(setStats).catch(() => {}) }, [])

  return <main>
    <header className="topbar"><div className="brand"><span className="brand-mark">◈</span> SIGNAL COMMONS</div><span className="topline">RMIT HACKATHON 2026 · UEL × TIDE</span></header>
    <section className="hero">
      <div className="hero-copy">
        <p className="eyebrow">A new way to study AI safety together</p>
        <h1>Share the signal.<br/><em>Keep the source.</em></h1>
        <p className="lead">A multilingual research commons for risky AI prompts. Contributors share useful patterns while the raw text stays encrypted under their own Tide identity.</p>
        <div className="hero-actions"><button className="primary" onClick={() => void signIn()} disabled={!configured}>Contribute a prompt <span>↗</span></button><a className="secondary" href="#how">See how it works ↓</a></div>
        {!configured && <p className="setup-note">TideCloak realm setup is needed before sign in. Follow the app README to initialize it.</p>}
        {loginError && <p className="error" role="alert">{loginError}</p>}
      </div>
      <div className="hero-panel">
        <div className="panel-head"><span className="live-dot"/> COMMUNITY SIGNAL <span className="panel-caption">AGGREGATE ONLY</span></div>
        <div className="big-number">{stats?.total ?? '—'} <span>sealed contributions</span></div>
        <div className="mini-bars">{(['BENIGN', 'JAILBREAK', 'PROMPT_INJECTION'] as const).map(label => <div className="bar-row" key={label}><span>{label.replace('_', ' ')}</span><div className="bar-track"><div style={{ width: stats?.total ? `${100 * (stats.byLabel[label] || 0) / stats.total}%` : '0%' }}/></div><b>{stats?.byLabel[label] ?? 0}</b></div>)}</div>
        <div className="language-grid">{(['pt', 'fr', 'it', 'en'] as const).map(language => <div key={language}><span>{language.toUpperCase()}</span><b>{stats?.byLanguage[language] ?? 0}</b></div>)}</div>
        <p className="panel-foot">Research patterns are visible. Original prompts remain private to their contributor.</p>
      </div>
    </section>
    <section className="steps" id="how"><div className="section-kicker">THE TRUST SHIFT</div><h2>Collaboration without a central raw prompt archive.</h2><div className="step-grid">
      <article><span>01 / CLASSIFY</span><h3>Detect across languages</h3><p>The Challenge 1 model labels Portuguese, French, Italian and English prompts as benign, jailbreak or prompt injection.</p></article>
      <article><span>02 / SEAL</span><h3>Keep the original yours</h3><p>TideCloak encrypts the raw prompt in the browser. The server stores ciphertext and a classification record.</p></article>
      <article><span>03 / LEARN</span><h3>See the shared picture</h3><p>Researchers see aggregate patterns without gaining a copy of each contributor’s original text.</p></article>
    </div></section>
    <footer>Built for RMIT Hackathon 2026 · Challenge 1 × Challenge 2 <a href="https://tide.org/unlocked/#challenge">About UNLOCKED ↗</a></footer>
  </main>
}
