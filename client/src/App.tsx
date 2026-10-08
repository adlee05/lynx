import { useState } from 'react'
import type { FormEvent } from 'react'
import AuthPage from './AuthPage'
import LibraryPage from './LibraryPage'
import ResultImage from './ResultImage'
import './App.css'

type SearchResult = { rank: number; filename: string; image_url: string; score: number }
type SearchResponse = { query: string; results: SearchResult[]; best_score?: number; min_score?: number }
type SearchMode = 'text' | 'image'

const suggestions = ['a dog playing in water', 'a street at night', 'people riding bicycles']

function SearchPage() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<SearchMode>('text')
  const [queryImage, setQueryImage] = useState<File | null>(null)
  const [uploadImages, setUploadImages] = useState<File[]>([])
  const [uploadProgress, setUploadProgress] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [searchedQuery, setSearchedQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const token = localStorage.getItem('lynx_token_v2') || ''
  const username = localStorage.getItem('lynx_username_v2') || ''

  if (!token) {
    return <main className="auth-page"><a className="brand auth-brand" href="/" aria-label="Lynx home"><span className="brand-mark">L</span><span>lynx<span className="brand-dot">.</span></span></a><section className="auth-card"><span className="eyebrow">SIGN IN REQUIRED</span><h1>Your photos, your library.</h1><p>Create an account or sign in to upload photos and search your private collection.</p><a className="primary-link" href="/login">Sign in</a><div className="auth-switch">New to Lynx? <a href="/register">Create an account</a></div></section></main>
  }

  async function handleUpload() {
    if (uploadImages.length === 0 || loading) return
    setLoading(true); setError(''); setNotice('')
    const failures: string[] = []
    let uploaded = 0
    for (const [index, file] of uploadImages.entries()) {
      setUploadProgress(`Uploading ${index + 1} of ${uploadImages.length}: ${file.name}`)
      try {
        const response = await fetch('/api/upload', {
          method: 'POST', headers: { Authorization: `Bearer ${token}`, 'Content-Type': file.type || 'application/octet-stream' },
          body: file,
        })
        const data = await response.json() as { error?: string }
        if (!response.ok) throw new Error(data.error || `Upload failed (${response.status})`)
        uploaded++
      } catch (cause) {
        failures.push(`${file.name}: ${cause instanceof Error ? cause.message : 'upload failed'}`)
      }
    }
    setNotice(uploaded > 0 ? `Added ${uploaded} ${uploaded === 1 ? 'photo' : 'photos'} to your library.` : '')
    if (failures.length > 0) setError(`Could not upload ${failures.length} ${failures.length === 1 ? 'photo' : 'photos'}:\n${failures.join('\n')}`)
    setUploadImages([])
    setUploadProgress('')
    const input = document.getElementById('library-upload') as HTMLInputElement | null
    if (input) input.value = ''
    setLoading(false)
  }

  async function handleSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (loading) return
    if (mode === 'text' && !query.trim()) return
    if (mode === 'image' && !queryImage) { setError('Choose a photo to search with first.'); return }
    setLoading(true); setError(''); setNotice(''); setResults([])
    try {
      let response: Response
      if (mode === 'image' && queryImage) {
        response = await fetch('/api/search/image', {
          method: 'POST', headers: { Authorization: `Bearer ${token}`, 'Content-Type': queryImage.type || 'application/octet-stream' }, body: queryImage,
        })
      } else {
        response = await fetch('/api/search', {
          method: 'POST', headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
          body: JSON.stringify({ query: query.trim(), top_k: 10 }),
        })
      }
      const data = await response.json() as SearchResponse & { error?: string }
      if (!response.ok) throw new Error(data.error || 'Search failed')
      setResults(data.results); setSearchedQuery(data.query)
      if (data.results.length === 0) {
        if (data.best_score !== undefined) setNotice(`The closest match scored ${data.best_score.toFixed(3)} cosine similarity, below Lynx’s relevance cutoff. Try a more specific description or upload more photos.`)
        else setNotice('Your library has no indexed photos yet. Upload photos, then search again.')
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not reach Lynx. Check that the services are running.')
    } finally { setLoading(false) }
  }

  async function signOut() {
    try { await fetch('/api/logout', { method: 'POST', headers: { Authorization: `Bearer ${token}` } }) } catch { /* local sign-out still works */ }
    localStorage.removeItem('lynx_token_v2'); localStorage.removeItem('lynx_username_v2')
    window.location.assign('/login')
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Lynx home"><span className="brand-mark">L</span><span>lynx<span className="brand-dot">.</span></span></a>
        <div className="account-actions"><span className="topbar-note"><span className="status-dot" /> {username}</span><a href="/library">My Library</a><button type="button" onClick={signOut}>Sign out</button></div>
      </header>

      <section className="hero">
        <div className="eyebrow"><span /> SEARCH YOUR PHOTO LIBRARY</div>
        <h1>Find the image<br /><span>you have in mind.</span></h1>
        <p className="intro">Upload your photos, then find them with a description or a similar image.</p>
        <div className="mode-tabs"><button className={mode === 'text' ? 'active' : ''} onClick={() => setMode('text')} type="button">Describe it</button><button className={mode === 'image' ? 'active' : ''} onClick={() => setMode('image')} type="button">Use an image</button></div>
        <form className="search-form" onSubmit={handleSearch}>
          {mode === 'text' ? <><span className="search-icon" aria-hidden="true">⌕</span><input aria-label="Describe the image you are looking for" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Try “a dog playing in water”…" /></> : <><span className="search-icon" aria-hidden="true">▧</span><input className="file-input" aria-label="Choose an image to find similar photos" type="file" accept="image/jpeg,image/png,image/webp" onChange={(event) => setQueryImage(event.target.files?.[0] || null)} /></>}
          <button type="submit" disabled={loading || (mode === 'text' ? !query.trim() : !queryImage)}>{loading ? <><span className="spinner" /> Searching</> : <>Search <span aria-hidden="true">↗</span></>}</button>
        </form>
        {mode === 'text' && <div className="suggestions" aria-label="Example searches"><span>Try</span>{suggestions.map((suggestion) => <button key={suggestion} type="button" onClick={() => setQuery(suggestion)}>{suggestion}</button>)}</div>}
      </section>

      <section className="library-panel"><div><span className="eyebrow">YOUR LIBRARY</span><p>Add JPG, PNG, or WebP photos (up to 10 MB each) to make them searchable.</p></div><div className="upload-controls"><label className="choose-file" htmlFor="library-upload">{uploadImages.length === 0 ? 'Choose photos' : `${uploadImages.length} photos selected`}</label><input id="library-upload" type="file" accept="image/jpeg,image/png,image/webp" multiple onChange={(event) => setUploadImages(Array.from(event.target.files || []))} /><button type="button" onClick={handleUpload} disabled={loading || uploadImages.length === 0}>{loading ? 'Uploading…' : `Upload ${uploadImages.length || ''} photos`}</button></div>{uploadProgress && <span className="upload-progress">{uploadProgress}</span>}</section>

      <section className="results-section" aria-live="polite">
        {error && <div className="notice error-notice">{error}</div>}{notice && <div className="notice success-notice">{notice}</div>}
        {loading && <div className="loading-state"><span className="spinner" /> Processing photos…</div>}
        {!loading && results.length > 0 && <><div className="results-heading"><div><span className="eyebrow">YOUR RESULTS</span><h2>Images for “{searchedQuery}”</h2></div><span className="result-count">{results.length} matches</span></div><div className="image-grid">{results.map((result) => <article className="image-card" key={result.filename}><div className="image-wrap"><ResultImage url={result.image_url} token={token} alt={`Search result ${result.rank}`} /><span className="rank-badge">{String(result.rank).padStart(2, '0')}</span></div><div className="card-caption"><span>{result.filename}</span><span className="score" title="Cosine similarity">{result.score.toFixed(3)}</span></div></article>)}</div></>}
        {!loading && results.length === 0 && !error && <div className="empty-state"><div className="empty-art" aria-hidden="true"><span>⌕</span><i /><b /></div><p>Your search results will appear here.</p><span>Search with a description or an image.</span></div>}
      </section>
      <footer><span>LYNX <i>•</i> VISION-LANGUAGE IMAGE RETRIEVAL</span><span>Powered by CLIP embeddings</span></footer>
    </main>
  )
}

export default function App() {
  const path = window.location.pathname
  if (path === '/login' || path === '/register') {
    return <AuthPage mode={path.slice(1) as 'login' | 'register'} />
  }
  if (path === '/library') return <LibraryPage />
  return <SearchPage />
}
