import { useState } from 'react'
import type { FormEvent } from 'react'
import './App.css'

type SearchResult = {
  rank: number
  filename: string
  image_url: string
  score: number
}

type SearchResponse = {
  query: string
  results: SearchResult[]
}

const suggestions = ['a dog playing in water', 'a street at night', 'people riding bicycles']

function App() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [searchedQuery, setSearchedQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const cleanQuery = query.trim()
    if (!cleanQuery || loading) return

    setLoading(true)
    setError('')
    setResults([])
    try {
      const response = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: cleanQuery, top_k: 10 }),
      })
      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || `Search failed (${response.status})`)
      }
      const data: SearchResponse = await response.json()
      setResults(data.results)
      setSearchedQuery(data.query)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Could not reach Lynx. Check that the Go and Python services are running.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Lynx home">
          <span className="brand-mark">L</span>
          <span>lynx<span className="brand-dot">.</span></span>
        </a>
        <span className="topbar-note"><span className="status-dot" /> Semantic image search</span>
      </header>

      <section className="hero">
        <div className="eyebrow"><span /> SEARCH BY MEANING</div>
        <h1>Find the image<br /><span>you have in mind.</span></h1>
        <p className="intro">Describe a moment, a place, or an idea. Lynx finds images that match what you mean.</p>

        <form className="search-form" onSubmit={handleSearch}>
          <span className="search-icon" aria-hidden="true">⌕</span>
          <input
            aria-label="Describe the image you are looking for"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Try “a dog playing in water”…"
          />
          <button type="submit" disabled={loading || !query.trim()}>
            {loading ? <><span className="spinner" /> Searching</> : <>Search <span aria-hidden="true">↗</span></>}
          </button>
        </form>

        <div className="suggestions" aria-label="Example searches">
          <span>Try</span>
          {suggestions.map((suggestion) => (
            <button key={suggestion} type="button" onClick={() => setQuery(suggestion)}>{suggestion}</button>
          ))}
        </div>
      </section>

      <section className="results-section" aria-live="polite">
        {error && <div className="notice error-notice">{error}</div>}
        {loading && <div className="loading-state"><span className="spinner" /> Finding images related to “{query.trim()}”</div>}
        {!loading && results.length > 0 && (
          <>
            <div className="results-heading">
              <div><span className="eyebrow">YOUR RESULTS</span><h2>Images for “{searchedQuery}”</h2></div>
              <span className="result-count">{results.length} matches</span>
            </div>
            <div className="image-grid">
              {results.map((result) => (
                <article className="image-card" key={result.filename}>
                  <div className="image-wrap">
                    <img src={result.image_url} alt={`Search result ${result.rank} for ${searchedQuery}`} loading="lazy" />
                    <span className="rank-badge">{String(result.rank).padStart(2, '0')}</span>
                  </div>
                  <div className="card-caption">
                    <span>{result.filename}</span>
                    <span className="score" title="Cosine similarity">{result.score.toFixed(3)}</span>
                  </div>
                </article>
              ))}
            </div>
          </>
        )}
        {!loading && !error && results.length === 0 && (
          <div className="empty-state">
            <div className="empty-art" aria-hidden="true"><span>⌕</span><i /><b /></div>
            <p>Your search results will appear here.</p>
            <span>Start with a description above.</span>
          </div>
        )}
      </section>

      <footer><span>LYNX <i>•</i> VISION-LANGUAGE IMAGE RETRIEVAL</span><span>Powered by CLIP embeddings</span></footer>
    </main>
  )
}

export default App
