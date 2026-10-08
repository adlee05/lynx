# Lynx

Semantic image retrieval using pretrained CLIP embeddings and Qdrant vector search.

## Runtime architecture

- React provides separate sign-in and registration pages, photo upload, text search, and image-to-image search.
- Go handles authentication, sessions, upload storage, PostgreSQL metadata, and Qdrant search.
- PostgreSQL stores users, password hashes, sessions, and image ownership metadata.
- Qdrant stores CLIP vectors with an `owner_id` payload so every search is scoped to the signed-in account.
- The Python service loads CLIP once and only creates normalized text or image embeddings.

Text search returns up to 10 results with a minimum cosine similarity of `0.19`; image-to-image search uses `0.35` because an image query should match its visual neighbors more closely. These fixed cutoffs filter weak matches; cosine similarity is not a calibrated probability that a result is correct. They can be adjusted in `backend/api.go` as the demo dataset changes.

## Run locally

### Run the complete stack with Docker Compose

Build and start the client, Go API, CLIP embedding service, PostgreSQL, and Qdrant:

```bash
docker compose -f compose.yaml up --build -d
```

Open <http://localhost:3000>. The first start downloads the pretrained CLIP weights and may take a while; later starts reuse the model cache. PostgreSQL data, Qdrant vectors, uploaded photos, and model weights are kept in named Docker volumes. The embedding container uses CPU inference for portability.

To inspect service startup or stop the stack:

```bash
docker compose -f compose.yaml logs -f
docker compose -f compose.yaml down
```

`down` keeps the named data volumes. To remove all data as well, use `docker compose -f compose.yaml down -v`.

### Run services directly on the host

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
