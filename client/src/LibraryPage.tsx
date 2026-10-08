import { useEffect, useState } from 'react'
import ResultImage from './ResultImage'

type LibraryImage = { filename: string; image_url: string; created_at: string }
type LibraryResponse = { images: LibraryImage[]; page: number; page_size: number; total_images: number; total_pages: number }

export default function LibraryPage() {
  const token = localStorage.getItem('lynx_token_v2') || ''
  const username = localStorage.getItem('lynx_username_v2') || ''
  const [page, setPage] = useState(1)
  const [library, setLibrary] = useState<LibraryResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!token) { setLoading(false); return }
    let active = true
    setLoading(true)
    setError('')
    fetch(`/api/library?page=${page}&page_size=12`, { headers: { Authorization: `Bearer ${token}` } })
      .then(async (response) => {
        const data = await response.json() as LibraryResponse & { error?: string }
        if (!response.ok) throw new Error(data.error || 'Could not load your library')
        return data
      })
      .then((data) => { if (active) setLibrary(data) })
      .catch((cause: unknown) => { if (active) setError(cause instanceof Error ? cause.message : 'Could not load your library') })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [page, token])

  async function signOut() {
    try { await fetch('/api/logout', { method: 'POST', headers: { Authorization: `Bearer ${token}` } }) } catch { /* local sign-out still works */ }
    localStorage.removeItem('lynx_token_v2')
    localStorage.removeItem('lynx_username_v2')
    window.location.assign('/login')
  }

  if (!token) return <main className="auth-page"><a className="brand auth-brand" href="/">lynx<span className="brand-dot">.</span></a><section className="auth-card"><span className="eyebrow">SIGN IN REQUIRED</span><h1>Your photos, your library.</h1><p>Sign in to view your uploaded photos.</p><a className="primary-link" href="/login">Sign in</a><div className="auth-switch">New to Lynx? <a href="/register">Create an account</a></div></section></main>

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Lynx home"><span className="brand-mark">L</span><span>lynx<span className="brand-dot">.</span></span></a>
        <div className="account-actions"><span className="topbar-note"><span className="status-dot" /> {username}</span><a href="/">Search</a><button type="button" onClick={signOut}>Sign out</button></div>
      </header>
      <section className="library-page-heading"><span className="eyebrow"><span /> YOUR COLLECTION</span><h1>My Library</h1><p>Your uploaded photos, ready to search.</p></section>
      <section className="library-content" aria-live="polite">
        {error && <div className="notice error-notice">{error}</div>}
        {loading && <div className="loading-state"><span className="spinner" /> Loading your photos…</div>}
        {!loading && !error && library && library.images.length > 0 && <>
          <div className="results-heading"><div><span className="eyebrow">YOUR PHOTOS</span><h2>{library.total_images} {library.total_images === 1 ? 'photo' : 'photos'}</h2></div><span className="result-count">Page {library.page} of {library.total_pages}</span></div>
          <div className="image-grid">{library.images.map((image, index) => <article className="image-card" key={image.filename}><div className="image-wrap"><ResultImage url={image.image_url} token={token} alt={`Library photo ${index + 1}`} /></div><div className="card-caption"><span>{new Date(image.created_at).toLocaleDateString()}</span><span className="library-file" title={image.filename}>{image.filename}</span></div></article>)}</div>
          <nav className="library-pagination" aria-label="Library pages"><button type="button" onClick={() => setPage((current) => Math.max(1, current - 1))} disabled={page <= 1 || loading}>← Previous</button><span>Page {library.page} of {library.total_pages}</span><button type="button" onClick={() => setPage((current) => current + 1)} disabled={page >= library.total_pages || loading}>Next →</button></nav>
        </>}
        {!loading && !error && library?.total_images === 0 && <div className="library-empty"><div className="empty-art" aria-hidden="true"><span>▧</span></div><h2>Your library is empty</h2><p>Upload photos to start building your searchable collection.</p><a className="primary-link" href="/">Go upload photos</a></div>}
      </section>
      <footer><span>LYNX <i>•</i> VISION-LANGUAGE IMAGE RETRIEVAL</span><span>Powered by CLIP embeddings</span></footer>
    </main>
  )
}
