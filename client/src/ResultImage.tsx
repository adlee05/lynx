import { useEffect, useState } from 'react'

export default function ResultImage({ url, token, alt }: { url: string; token: string; alt: string }) {
  const [source, setSource] = useState('')
  useEffect(() => {
    let objectUrl = ''
    fetch(url, { headers: { Authorization: `Bearer ${token}` } })
      .then((response) => {
        if (!response.ok) throw new Error('Image request failed')
        return response.blob()
      })
      .then((blob) => {
        objectUrl = URL.createObjectURL(blob)
        setSource(objectUrl)
      })
      .catch(() => setSource(''))
    return () => { if (objectUrl) URL.revokeObjectURL(objectUrl) }
  }, [url, token])
  return source ? <img src={source} alt={alt} loading="lazy" /> : <div className="image-loading">Image unavailable</div>
}
