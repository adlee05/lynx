package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

const defaultEmbeddingServiceURL = "http://127.0.0.1:8000"

var (
	embeddingHTTPClient = &http.Client{Timeout: 60 * time.Second}
	validImageFilename  = regexp.MustCompile(`^[0-9]{12}\.jpg$`)
)

type searchRequest struct {
	Query string `json:"query"`
	TopK  int    `json:"top_k"`
}

type searchResult struct {
	Rank     int     `json:"rank"`
	Filename string  `json:"filename"`
	ImageURL string  `json:"image_url"`
	Score    float64 `json:"score"`
}

type searchResponse struct {
	Query   string         `json:"query"`
	Results []searchResult `json:"results"`
}

func embeddingServiceURL() string {
	if configured := strings.TrimSpace(os.Getenv("EMBEDDING_SERVICE_URL")); configured != "" {
		return strings.TrimRight(configured, "/")
	}
	return defaultEmbeddingServiceURL
}

func searchAPI(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 1<<20)
	var input searchRequest
	if err := json.NewDecoder(r.Body).Decode(&input); err != nil {
		http.Error(w, "Invalid JSON search request", http.StatusBadRequest)
		return
	}

	input.Query = strings.TrimSpace(input.Query)
	if input.Query == "" {
		http.Error(w, "query is required", http.StatusBadRequest)
		return
	}
	if input.TopK == 0 {
		input.TopK = 5
	}
	if input.TopK < 1 || input.TopK > 20 {
		http.Error(w, "top_k must be between 1 and 20", http.StatusBadRequest)
		return
	}

	body, err := json.Marshal(input)
	if err != nil {
		http.Error(w, "Could not encode search request", http.StatusInternalServerError)
		return
	}
	upstream, err := http.NewRequestWithContext(
		r.Context(),
		http.MethodPost,
		embeddingServiceURL()+"/search",
		strings.NewReader(string(body)),
	)
	if err != nil {
		http.Error(w, "Could not create embedding-service request", http.StatusInternalServerError)
		return
	}
	upstream.Header.Set("Content-Type", "application/json")

	response, err := embeddingHTTPClient.Do(upstream)
	if err != nil {
		http.Error(w, "Embedding service is unavailable", http.StatusBadGateway)
		return
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		http.Error(w, "Embedding service search failed", http.StatusBadGateway)
		return
	}

	var result searchResponse
	if err := json.NewDecoder(io.LimitReader(response.Body, 2<<20)).Decode(&result); err != nil {
		http.Error(w, "Embedding service returned an invalid response", http.StatusBadGateway)
		return
	}
	for i := range result.Results {
		filename := result.Results[i].Filename
		if !validImageFilename.MatchString(filepath.Base(filename)) || filepath.Base(filename) != filename {
			http.Error(w, "Embedding service returned an invalid image name", http.StatusBadGateway)
			return
		}
		result.Results[i].ImageURL = "/api/images/" + url.PathEscape(filename)
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(result); err != nil {
		return
	}
}

func imageAPI(w http.ResponseWriter, r *http.Request) {
	filename := r.PathValue("filename")
	if !validImageFilename.MatchString(filename) || filepath.Base(filename) != filename {
		http.NotFound(w, r)
		return
	}

	upstreamURL := fmt.Sprintf("%s/images/%s", embeddingServiceURL(), url.PathEscape(filename))
	upstream, err := http.NewRequestWithContext(r.Context(), http.MethodGet, upstreamURL, nil)
	if err != nil {
		http.Error(w, "Could not create image request", http.StatusInternalServerError)
		return
	}
	response, err := embeddingHTTPClient.Do(upstream)
	if err != nil {
		http.Error(w, "Embedding service is unavailable", http.StatusBadGateway)
		return
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		http.NotFound(w, r)
		return
	}

	contentType := response.Header.Get("Content-Type")
	if contentType == "" {
		contentType = "image/jpeg"
	}
	w.Header().Set("Content-Type", contentType)
	w.WriteHeader(http.StatusOK)
	_, _ = io.Copy(w, response.Body)
}
