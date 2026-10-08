package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strings"
)

const collectionName = "lynx_images"

func (a *API) qdrantURL(path string) string { return a.qdrantBase + path }

func (a *API) qdrantRequest(ctx context.Context, method, path string, body any) (*http.Response, error) {
	var reader *bytes.Reader
	if body == nil {
		reader = bytes.NewReader(nil)
	} else {
		encoded, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		reader = bytes.NewReader(encoded)
	}
	req, err := http.NewRequestWithContext(ctx, method, a.qdrantURL(path), reader)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if key := os.Getenv("QDRANT_API_KEY"); key != "" {
		req.Header.Set("api-key", key)
	}
	return a.httpClient.Do(req)
}

func (a *API) ensureQdrantCollection(ctx context.Context) error {
	path := "/collections/" + url.PathEscape(collectionName)
	response, err := a.qdrantRequest(ctx, http.MethodGet, path, nil)
	if err != nil {
		return err
	}
	response.Body.Close()
	if response.StatusCode == http.StatusOK {
		return a.ensureOwnerPayloadIndex(ctx)
	}
	if response.StatusCode != http.StatusNotFound {
		return fmt.Errorf("Qdrant collection check returned %s", response.Status)
	}
	response, err = a.qdrantRequest(ctx, http.MethodPut, path, map[string]any{
		"vectors": map[string]any{"size": 512, "distance": "Cosine"},
	})
	if err != nil {
		return err
	}
	response.Body.Close()
	if response.StatusCode != http.StatusOK && response.StatusCode != http.StatusCreated {
		return fmt.Errorf("create Qdrant collection returned %s", response.Status)
	}
	return a.ensureOwnerPayloadIndex(ctx)
}

func (a *API) ensureOwnerPayloadIndex(ctx context.Context) error {
	response, err := a.qdrantRequest(ctx, http.MethodPut,
		"/collections/"+url.PathEscape(collectionName)+"/index",
		map[string]any{"field_name": "owner_id", "field_schema": "keyword"},
	)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return fmt.Errorf("create Qdrant owner index returned %s", response.Status)
	}
	return nil
}

func (a *API) upsertImageVector(ctx context.Context, id, ownerID, filename string, vector []float32) error {
	response, err := a.qdrantRequest(ctx, http.MethodPut,
		"/collections/"+url.PathEscape(collectionName)+"/points?wait=true",
		map[string]any{"points": []any{map[string]any{
			"id": id, "vector": vector,
			"payload": map[string]any{"owner_id": ownerID, "filename": filename},
		}}},
	)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return fmt.Errorf("Qdrant upsert returned %s", response.Status)
	}
	return nil
}

type qdrantQueryResponse struct {
	Result struct {
		Points []struct {
			Score   float64        `json:"score"`
			Payload map[string]any `json:"payload"`
		} `json:"points"`
	} `json:"result"`
}

func (a *API) searchVectors(ctx context.Context, ownerID string, vector []float32, topK int, threshold float64) ([]SearchResult, error) {
	response, err := a.qdrantRequest(ctx, http.MethodPost,
		"/collections/"+url.PathEscape(collectionName)+"/points/query",
		map[string]any{
			"query": vector, "limit": topK, "score_threshold": threshold,
			"with_payload": []string{"filename"},
			"filter": map[string]any{"must": []any{map[string]any{
				"key": "owner_id", "match": map[string]any{"value": ownerID},
			}}},
		},
	)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, fmt.Errorf("Qdrant search returned %s", response.Status)
	}
	var decoded qdrantQueryResponse
	if err := json.NewDecoder(response.Body).Decode(&decoded); err != nil {
		return nil, err
	}
	results := make([]SearchResult, 0, len(decoded.Result.Points))
	for _, point := range decoded.Result.Points {
		filename, _ := point.Payload["filename"].(string)
		if filename == "" || point.Score < threshold || strings.Contains(filename, "/") {
			continue
		}
		results = append(results, SearchResult{
			Filename: filename,
			ImageURL: "/api/images/" + url.PathEscape(filename),
			Score:    point.Score,
		})
	}
	for i := range results {
		results[i].Rank = i + 1
	}
	return results, nil
}
