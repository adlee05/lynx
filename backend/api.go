package main

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"database/sql"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"
)

type API struct {
	db           *sql.DB
	httpClient   *http.Client
	embeddingURL string
	qdrantBase   string
	uploadDir    string
}

type User struct {
	ID       int64
	Username string
}

type EmbeddingResponse struct {
	Embedding []float32 `json:"embedding"`
}

type SearchResult struct {
	Rank     int     `json:"rank"`
	Filename string  `json:"filename"`
	ImageURL string  `json:"image_url"`
	Score    float64 `json:"score"`
}

type SearchResponse struct {
	Query   string         `json:"query"`
	Results []SearchResult `json:"results"`
}

type LibraryImage struct {
	Filename  string    `json:"filename"`
	ImageURL  string    `json:"image_url"`
	CreatedAt time.Time `json:"created_at"`
}

type LibraryResponse struct {
	Images      []LibraryImage `json:"images"`
	Page        int            `json:"page"`
	PageSize    int            `json:"page_size"`
	TotalImages int            `json:"total_images"`
	TotalPages  int            `json:"total_pages"`
}

var usernamePattern = regexp.MustCompile(`^[A-Za-z0-9_.-]{3,32}$`)

func newAPI(db *sql.DB) *API {
	return &API{
		db:           db,
		httpClient:   &http.Client{Timeout: 75 * time.Second},
		embeddingURL: envOr("EMBEDDING_SERVICE_URL", "http://127.0.0.1:8000"),
		qdrantBase:   strings.TrimRight(envOr("QDRANT_URL", "http://127.0.0.1:6333"), "/"),
		uploadDir:    envOr("LYNX_UPLOAD_DIR", "uploads"),
	}
}

func envOr(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func (a *API) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/health", a.health)
	mux.HandleFunc("POST /api/register", a.register)
	mux.HandleFunc("POST /api/login", a.login)
	mux.HandleFunc("POST /api/logout", a.logout)
	mux.HandleFunc("POST /api/upload", a.upload)
	mux.HandleFunc("GET /api/library", a.myLibrary)
	mux.HandleFunc("POST /api/search", a.searchText)
	mux.HandleFunc("POST /api/search/image", a.searchImage)
	mux.HandleFunc("GET /api/images/{filename}", a.getImage)
	return mux
}

func (a *API) myLibrary(w http.ResponseWriter, r *http.Request) {
	user, err := a.authenticatedUser(r)
	if err != nil {
		writeError(w, http.StatusUnauthorized, err.Error())
		return
	}

	page := 1
	if raw := r.URL.Query().Get("page"); raw != "" {
		page, err = strconv.Atoi(raw)
		if err != nil || page < 1 {
			writeError(w, http.StatusBadRequest, "page must be a positive integer")
			return
		}
	}
	pageSize := 12
	if raw := r.URL.Query().Get("page_size"); raw != "" {
		pageSize, err = strconv.Atoi(raw)
		if err != nil || pageSize < 1 {
			writeError(w, http.StatusBadRequest, "page_size must be a positive integer")
			return
		}
		if pageSize > 48 {
			pageSize = 48
		}
	}

	var total int
	if err := a.db.QueryRowContext(r.Context(), "SELECT COUNT(*) FROM images WHERE owner_id=$1", user.ID).Scan(&total); err != nil {
		log.Printf("count library images: %v", err)
		writeError(w, http.StatusInternalServerError, "Could not load your library")
		return
	}
	totalPages := 0
	if total > 0 {
		totalPages = (total + pageSize - 1) / pageSize
		if page > totalPages {
			page = totalPages
		}
	}

	images := make([]LibraryImage, 0, pageSize)
	rows, err := a.db.QueryContext(r.Context(), `
		SELECT filename, created_at FROM images
		WHERE owner_id=$1
		ORDER BY created_at DESC, id DESC
		LIMIT $2 OFFSET $3`, user.ID, pageSize, (page-1)*pageSize)
	if err != nil {
		log.Printf("query library images: %v", err)
		writeError(w, http.StatusInternalServerError, "Could not load your library")
		return
	}
	defer rows.Close()
	for rows.Next() {
		var image LibraryImage
		if err := rows.Scan(&image.Filename, &image.CreatedAt); err != nil {
			writeError(w, http.StatusInternalServerError, "Could not load your library")
			return
		}
		image.ImageURL = "/api/images/" + image.Filename
		images = append(images, image)
	}
	if err := rows.Err(); err != nil {
		writeError(w, http.StatusInternalServerError, "Could not load your library")
		return
	}
	writeJSON(w, http.StatusOK, LibraryResponse{
		Images: images, Page: page, PageSize: pageSize, TotalImages: total, TotalPages: totalPages,
	})
}

func (a *API) health(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
	defer cancel()
	if err := a.db.PingContext(ctx); err != nil {
		writeError(w, http.StatusServiceUnavailable, "PostgreSQL is unavailable")
		return
	}
	response, err := a.httpClient.Get(a.embeddingURL + "/health")
	if err != nil || response.StatusCode != http.StatusOK {
		writeError(w, http.StatusServiceUnavailable, "Embedding service is unavailable")
		return
	}
	defer response.Body.Close()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = io.Copy(w, response.Body)
}

type credentials struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

func decodeJSON(w http.ResponseWriter, r *http.Request, target any, limit int64) error {
	r.Body = http.MaxBytesReader(w, r.Body, limit)
	return json.NewDecoder(r.Body).Decode(target)
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func writeError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]string{"error": message})
}

func (a *API) register(w http.ResponseWriter, r *http.Request) {
	var input credentials
	if err := decodeJSON(w, r, &input, 1<<20); err != nil {
		writeError(w, http.StatusBadRequest, "Invalid account request")
		return
	}
	if !usernamePattern.MatchString(input.Username) || len(input.Password) < 8 || len(input.Password) > 128 {
		writeError(w, http.StatusBadRequest, "Use a 3–32 character username and password of at least 8 characters")
		return
	}
	salt := make([]byte, 16)
	if _, err := rand.Read(salt); err != nil {
		writeError(w, http.StatusInternalServerError, "Could not create account")
		return
	}
	_, err := a.db.ExecContext(r.Context(),
		"INSERT INTO users(username, salt, password_hash) VALUES($1, $2, $3)",
		input.Username, salt, derivePasswordKey(input.Password, salt),
	)
	if err != nil {
		if isUniqueViolation(err) {
			writeError(w, http.StatusConflict, "That username is already taken")
		} else {
			log.Printf("create user: %v", err)
			writeError(w, http.StatusInternalServerError, "Could not create account")
		}
		return
	}
	a.createSession(w, r, input.Username)
}

func (a *API) login(w http.ResponseWriter, r *http.Request) {
	var input credentials
	if err := decodeJSON(w, r, &input, 1<<20); err != nil {
		writeError(w, http.StatusBadRequest, "Invalid login request")
		return
	}
	var user User
	var salt, expected []byte
	err := a.db.QueryRowContext(r.Context(),
		"SELECT id, username, salt, password_hash FROM users WHERE username=$1", input.Username,
	).Scan(&user.ID, &user.Username, &salt, &expected)
	if err != nil || !hmac.Equal(derivePasswordKey(input.Password, salt), expected) {
		writeError(w, http.StatusUnauthorized, "Incorrect username or password")
		return
	}
	a.issueSession(w, r, user)
}

func derivePasswordKey(password string, salt []byte) []byte {
	const rounds = 310_000
	mac := hmac.New(sha256.New, []byte(password))
	_, _ = mac.Write(append(append([]byte{}, salt...), 0, 0, 0, 1))
	u := mac.Sum(nil)
	result := append([]byte(nil), u...)
	for i := 1; i < rounds; i++ {
		mac = hmac.New(sha256.New, []byte(password))
		_, _ = mac.Write(u)
		u = mac.Sum(nil)
		for j := range result {
			result[j] ^= u[j]
		}
	}
	return result
}

func tokenDigest(token string) []byte {
	hash := sha256.Sum256([]byte(token))
	return hash[:]
}

func (a *API) createSession(w http.ResponseWriter, r *http.Request, username string) {
	var user User
	if err := a.db.QueryRowContext(r.Context(), "SELECT id, username FROM users WHERE username=$1", username).
		Scan(&user.ID, &user.Username); err != nil {
		writeError(w, http.StatusInternalServerError, "Could not create session")
		return
	}
	a.issueSession(w, r, user)
}

func (a *API) issueSession(w http.ResponseWriter, r *http.Request, user User) {
	bytes := make([]byte, 32)
	if _, err := rand.Read(bytes); err != nil {
		writeError(w, http.StatusInternalServerError, "Could not create session")
		return
	}
	token := base64.RawURLEncoding.EncodeToString(bytes)
	_, err := a.db.ExecContext(r.Context(),
		"INSERT INTO sessions(token_hash, user_id, expires_at) VALUES($1, $2, NOW() + INTERVAL '30 days')",
		tokenDigest(token), user.ID,
	)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "Could not create session")
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{
		"access_token": token, "token_type": "bearer", "username": user.Username,
	})
}

func (a *API) authenticatedUser(r *http.Request) (User, error) {
	header := r.Header.Get("Authorization")
	scheme, token, ok := strings.Cut(header, " ")
	if !ok || !strings.EqualFold(scheme, "Bearer") || token == "" {
		return User{}, errors.New("sign in to continue")
	}
	var user User
	err := a.db.QueryRowContext(r.Context(), `
		SELECT users.id, users.username
		FROM sessions JOIN users ON users.id=sessions.user_id
		WHERE sessions.token_hash=$1 AND sessions.expires_at > NOW()
	`, tokenDigest(token)).Scan(&user.ID, &user.Username)
	if err != nil {
		return User{}, errors.New("invalid or expired session")
	}
	return user, nil
}

func (a *API) logout(w http.ResponseWriter, r *http.Request) {
	user, err := a.authenticatedUser(r)
	if err != nil {
		writeError(w, http.StatusUnauthorized, err.Error())
		return
	}
	_, token, _ := strings.Cut(r.Header.Get("Authorization"), " ")
	_, err = a.db.ExecContext(r.Context(), "DELETE FROM sessions WHERE token_hash=$1", tokenDigest(token))
	if err != nil {
		writeError(w, http.StatusInternalServerError, "Could not sign out")
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"message": "signed out", "username": user.Username})
}

func (a *API) embed(ctx context.Context, path string, body []byte, contentType string) ([]float32, error) {
	var reader io.Reader
	if body == nil {
		reader = strings.NewReader(`{"text":""}`)
	} else {
		reader = bytes.NewReader(body)
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, a.embeddingURL+path, reader)
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", contentType)
	response, err := a.httpClient.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		message, _ := io.ReadAll(io.LimitReader(response.Body, 2048))
		return nil, fmt.Errorf("embedding service: %s", strings.TrimSpace(string(message)))
	}
	var output EmbeddingResponse
	if err := json.NewDecoder(io.LimitReader(response.Body, 1<<20)).Decode(&output); err != nil {
		return nil, err
	}
	if len(output.Embedding) != 512 {
		return nil, fmt.Errorf("expected a 512-dimensional embedding, got %d", len(output.Embedding))
	}
	return output.Embedding, nil
}

type textSearchRequest struct {
	Query    string   `json:"query"`
	TopK     int      `json:"top_k"`
	MinScore *float64 `json:"min_score"`
}

func (a *API) searchText(w http.ResponseWriter, r *http.Request) {
	user, err := a.authenticatedUser(r)
	if err != nil {
		writeError(w, http.StatusUnauthorized, err.Error())
		return
	}
	var input textSearchRequest
	if err := decodeJSON(w, r, &input, 1<<20); err != nil {
		writeError(w, http.StatusBadRequest, "Invalid search request")
		return
	}
	input.Query = strings.TrimSpace(input.Query)
	if input.Query == "" || len(input.Query) > 512 {
		writeError(w, http.StatusBadRequest, "Enter a query of 1–512 characters")
		return
	}
	if input.TopK == 0 {
		input.TopK = 10
	}
	if input.TopK < 1 || input.TopK > 20 {
		writeError(w, http.StatusBadRequest, "top_k must be between 1 and 20")
		return
	}
	threshold := 0.30
	if input.MinScore != nil {
		threshold = *input.MinScore
	}
	if threshold < 0 || threshold > 1 {
		writeError(w, http.StatusBadRequest, "min_score must be between 0 and 1")
		return
	}
	encoded, _ := json.Marshal(map[string]string{"text": input.Query})
	vector, err := a.embed(r.Context(), "/embed/text", encoded, "application/json")
	if err != nil {
		log.Printf("text embedding: %v", err)
		writeError(w, http.StatusBadGateway, "Could not create the text embedding")
		return
	}
	results, err := a.searchVectors(r.Context(), strconv.FormatInt(user.ID, 10), vector, input.TopK, threshold)
	if err != nil {
		log.Printf("Qdrant text search: %v", err)
		writeError(w, http.StatusBadGateway, "Vector search failed")
		return
	}
	writeJSON(w, http.StatusOK, SearchResponse{Query: input.Query, Results: results})
}

func (a *API) searchImage(w http.ResponseWriter, r *http.Request) {
	user, err := a.authenticatedUser(r)
	if err != nil {
		writeError(w, http.StatusUnauthorized, err.Error())
		return
	}
	minScore, err := parseMinScore(r.URL.Query().Get("min_score"))
	if err != nil {
		writeError(w, http.StatusBadRequest, "min_score must be between 0 and 1")
		return
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 10<<20))
	if err != nil || len(body) == 0 {
		writeError(w, http.StatusBadRequest, "Choose an image smaller than 10 MB")
		return
	}
	vector, err := a.embed(r.Context(), "/embed/image", body, r.Header.Get("Content-Type"))
	if err != nil {
		writeError(w, http.StatusUnsupportedMediaType, "Could not embed that image")
		return
	}
	results, err := a.searchVectors(r.Context(), strconv.FormatInt(user.ID, 10), vector, 10, minScore)
	if err != nil {
		log.Printf("Qdrant image search: %v", err)
		writeError(w, http.StatusBadGateway, "Vector search failed")
		return
	}
	writeJSON(w, http.StatusOK, SearchResponse{Query: "similar images", Results: results})
}

func parseMinScore(raw string) (float64, error) {
	if raw == "" {
		return 0.30, nil
	}
	value, err := strconv.ParseFloat(raw, 64)
	if err != nil || value < 0 || value > 1 {
		return 0, fmt.Errorf("score must be between zero and one")
	}
	return value, nil
}

func extensionForImage(contentType string) (string, bool) {
	switch contentType {
	case "image/jpeg":
		return ".jpg", true
	case "image/png":
		return ".png", true
	case "image/webp":
		return ".webp", true
	default:
		return "", false
	}
}

func newUUID() (string, error) {
	value := make([]byte, 16)
	if _, err := rand.Read(value); err != nil {
		return "", err
	}
	value[6] = value[6]&0x0f | 0x40
	value[8] = value[8]&0x3f | 0x80
	encoded := hex.EncodeToString(value)
	return encoded[:8] + "-" + encoded[8:12] + "-" + encoded[12:16] + "-" + encoded[16:20] + "-" + encoded[20:], nil
}

func (a *API) upload(w http.ResponseWriter, r *http.Request) {
	user, err := a.authenticatedUser(r)
	if err != nil {
		writeError(w, http.StatusUnauthorized, err.Error())
		return
	}
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, 10<<20))
	if err != nil || len(body) == 0 {
		writeError(w, http.StatusBadRequest, "Choose an image smaller than 10 MB")
		return
	}
	contentType := http.DetectContentType(body)
	if len(body) > 512 && strings.HasPrefix(r.Header.Get("Content-Type"), "image/") {
		if detected := http.DetectContentType(body); detected != "application/octet-stream" {
			contentType = detected
		}
	}
	ext, ok := extensionForImage(contentType)
	if !ok {
		writeError(w, http.StatusUnsupportedMediaType, "Use a JPG, PNG, or WebP image")
		return
	}
	vector, err := a.embed(r.Context(), "/embed/image", body, contentType)
	if err != nil {
		writeError(w, http.StatusUnsupportedMediaType, "Could not read that image")
		return
	}
	id, err := newUUID()
	if err != nil {
		writeError(w, http.StatusInternalServerError, "Could not store image")
		return
	}
	filename := id + ext
	ownerID := strconv.FormatInt(user.ID, 10)
	directory := filepath.Join(a.uploadDir, ownerID)
	if err := os.MkdirAll(directory, 0o750); err != nil {
		writeError(w, http.StatusInternalServerError, "Could not create photo library")
		return
	}
	path := filepath.Join(directory, filename)
	if err := os.WriteFile(path, body, 0o600); err != nil {
		writeError(w, http.StatusInternalServerError, "Could not store image")
		return
	}
	_, err = a.db.ExecContext(r.Context(),
		"INSERT INTO images(id, owner_id, filename, content_type) VALUES($1, $2, $3, $4)",
		id, user.ID, filename, contentType,
	)
	if err == nil {
		err = a.upsertImageVector(r.Context(), id, ownerID, filename, vector)
	}
	if err != nil {
		_, _ = a.db.ExecContext(r.Context(), "DELETE FROM images WHERE id=$1", id)
		_ = os.Remove(path)
		log.Printf("store image vector: %v", err)
		writeError(w, http.StatusBadGateway, "Could not add image to the vector library")
		return
	}
	writeJSON(w, http.StatusCreated, map[string]string{
		"filename": filename, "image_url": "/api/images/" + filename,
	})
}

func (a *API) getImage(w http.ResponseWriter, r *http.Request) {
	user, err := a.authenticatedUser(r)
	if err != nil {
		writeError(w, http.StatusUnauthorized, err.Error())
		return
	}
	filename := filepath.Base(r.PathValue("filename"))
	if filename == "." || filename == string(filepath.Separator) {
		http.NotFound(w, r)
		return
	}
	var contentType string
	err = a.db.QueryRowContext(r.Context(),
		"SELECT content_type FROM images WHERE filename=$1 AND owner_id=$2", filename, user.ID,
	).Scan(&contentType)
	if err != nil {
		http.NotFound(w, r)
		return
	}
	path := filepath.Join(a.uploadDir, strconv.FormatInt(user.ID, 10), filename)
	w.Header().Set("Content-Type", contentType)
	http.ServeFile(w, r, path)
}
