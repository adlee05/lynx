# Lynx

Semantic image retrieval using pretrained CLIP embeddings and Qdrant vector search.

## Runtime architecture

- React provides separate sign-in and registration pages, photo upload, text search, and image-to-image search.
- Go handles authentication, sessions, upload storage, PostgreSQL metadata, and Qdrant search.
- PostgreSQL stores users, password hashes, sessions, and image ownership metadata.
- Qdrant stores CLIP vectors with an `owner_id` payload so every search is scoped to the signed-in account.
- The Python service loads CLIP once and only creates normalized text or image embeddings.

The search slider sets a minimum cosine similarity score. It filters weak matches; it is not a calibrated probability that a result is correct.

## Run locally

Start PostgreSQL and Qdrant:

```bash
docker compose -f compose.yaml up -d
```

In a second terminal, activate the Python environment and start the CLIP service:

```bash
cd embedding-service
source .venv/bin/activate
uvicorn api:app --reload --port 8000
```

Start the Go API in another terminal:

```bash
cd backend
go run .
```

Start the React client:

```bash
cd client
bun run dev
```

Open the Vite URL, create an account, upload at least two photos, then search the library with text or a query photo. Uploaded image files are stored under `backend/uploads/`; accounts and image metadata are stored in PostgreSQL; vectors are stored in Qdrant.

Local PostgreSQL defaults are `lynx` / `lynx_dev` with database `lynx`, published on host port `5433` to avoid conflicting with a local PostgreSQL server on `5432`. Configure `DATABASE_URL`, `QDRANT_URL`, `QDRANT_API_KEY`, or `EMBEDDING_SERVICE_URL` to use other services.
