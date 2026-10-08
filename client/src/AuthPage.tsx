import { useState } from 'react'
import type { FormEvent } from 'react'

type AuthPageProps = { mode: 'login' | 'register' }
type AuthResponse = { access_token: string; username: string }

export default function AuthPage({ mode }: AuthPageProps) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      const response = await fetch(`/api/${mode}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      })
      const data = await response.json() as AuthResponse & { error?: string }
      if (!response.ok) throw new Error(data.error || 'Account request failed')
      localStorage.setItem('lynx_token_v2', data.access_token)
      localStorage.setItem('lynx_username_v2', data.username)
      window.location.assign('/')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Account request failed')
    } finally { setBusy(false) }
  }

  return (
    <main className="auth-page">
      <a className="brand auth-brand" href="/" aria-label="Lynx home"><span className="brand-mark">L</span><span>lynx<span className="brand-dot">.</span></span></a>
      <section className="auth-card">
        <span className="eyebrow">YOUR PRIVATE PHOTO LIBRARY</span>
        <h1>{mode === 'register' ? 'Create your account' : 'Welcome back'}</h1>
        <p>{mode === 'register' ? 'Sign up to upload photos and search your collection.' : 'Sign in to continue to your photo library.'}</p>
        <form onSubmit={submit}>
          <label>Username<input autoComplete="username" minLength={3} maxLength={32} pattern="[A-Za-z0-9_.-]+" value={username} onChange={(event) => setUsername(event.target.value)} required /></label>
          <label>Password<input type="password" autoComplete={mode === 'login' ? 'current-password' : 'new-password'} minLength={8} maxLength={128} value={password} onChange={(event) => setPassword(event.target.value)} required /></label>
          {error && <div className="notice error-notice">{error}</div>}
          <button type="submit" disabled={busy}>{busy ? 'Please wait…' : mode === 'register' ? 'Create account' : 'Sign in'}</button>
        </form>
        <div className="auth-switch">{mode === 'register' ? <>Already have an account? <a href="/login">Sign in</a></> : <>New to Lynx? <a href="/register">Create an account</a></>}</div>
      </section>
    </main>
  )
}
