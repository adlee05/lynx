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
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null)
  const images = library?.images ?? []
  const isViewerOpen = selectedIndex !== null

  useEffect(() => {
    if (!isViewerOpen) return
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    function onKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') setSelectedIndex(null)
      if (event.key === 'ArrowLeft') setSelectedIndex((current) => current === null ? null : Math.max(0, current - 1))
      if (event.key === 'ArrowRight') setSelectedIndex((current) => current === null ? null : Math.min(images.length - 1, current + 1))
    }
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      document.body.style.overflow = previousOverflow
    }
  }, [isViewerOpen, images.length])

  const dateGroups = images.reduce<{ key: string; title: string; images: { image: LibraryImage; index: number }[] }[]>((groups, image, index) => {
    const date = new Date(image.created_at)
    const key = `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}`
    let group = groups[groups.length - 1]
    if (!group || group.key !== key) {
      const today = new Date()
      const yesterday = new Date(today)
      yesterday.setDate(today.getDate() - 1)
      const isToday = date.toDateString() === today.toDateString()
      const isYesterday = date.toDateString() === yesterday.toDateString()
      group = {
        key,
        title: isToday ? 'Today' : isYesterday ? 'Yesterday' : date.toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' }),
        images: [],
      }
      groups.push(group)
    }
    group.images.push({ image, index })
    return groups
  }, [])

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
          {dateGroups.map((group) => <section className="library-date-group" key={group.key}><h3>{group.title}<span>{group.images.length} {group.images.length === 1 ? 'photo' : 'photos'}</span></h3><div className="image-grid">{group.images.map(({ image, index }) => <article className="image-card" key={image.filename}><div className="image-wrap"><button className="gallery-image-button" type="button" onClick={() => setSelectedIndex(index)} aria-label={`View photo uploaded ${new Date(image.created_at).toLocaleString()}`}><ResultImage url={image.image_url} token={token} alt={`Library photo ${index + 1}`} /></button></div><div className="card-caption"><span className="library-file" title={image.filename}>{image.filename}</span><span aria-hidden="true">↗</span></div></article>)}</div></section>)}
          <nav className="library-pagination" aria-label="Library pages"><button type="button" onClick={() => setPage((current) => Math.max(1, current - 1))} disabled={page <= 1 || loading}>← Previous</button><span>Page {library.page} of {library.total_pages}</span><button type="button" onClick={() => setPage((current) => current + 1)} disabled={page >= library.total_pages || loading}>Next →</button></nav>
        </>}
        {!loading && !error && library?.total_images === 0 && <div className="library-empty"><div className="empty-art" aria-hidden="true"><span>▧</span></div><h2>Your library is empty</h2><p>Upload photos to start building your searchable collection.</p><a className="primary-link" href="/">Go upload photos</a></div>}
      </section>
      {selectedIndex !== null && images[selectedIndex] && <div className="gallery-lightbox" role="presentation" onClick={() => setSelectedIndex(null)}>
        <section className="gallery-viewer" role="dialog" aria-modal="true" aria-label="Photo viewer" onClick={(event) => event.stopPropagation()}>
          <div className="gallery-viewer-toolbar"><span>{selectedIndex + 1} / {images.length} on this page</span><button type="button" onClick={() => setSelectedIndex(null)} aria-label="Close photo viewer">×</button></div>
          <div className="gallery-viewer-stage">
            <button className="gallery-step previous" type="button" aria-label="Previous photo" disabled={selectedIndex === 0} onClick={() => setSelectedIndex((current) => current === null ? null : Math.max(0, current - 1))}>‹</button>
            <div className="gallery-full-image"><ResultImage url={images[selectedIndex].image_url} token={token} alt={`Full size photo ${selectedIndex + 1}`} /></div>
            <button className="gallery-step next" type="button" aria-label="Next photo" disabled={selectedIndex === images.length - 1} onClick={() => setSelectedIndex((current) => current === null ? null : Math.min(images.length - 1, current + 1))}>›</button>
          </div>
          <div className="gallery-viewer-caption">Uploaded {new Date(images[selectedIndex].created_at).toLocaleString()}</div>
        </section>
      </div>}
      <footer><span>LYNX <i>•</i> VISION-LANGUAGE IMAGE RETRIEVAL</span><span>Powered by CLIP embeddings</span></footer>
    </main>
  )
}
