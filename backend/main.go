package main

import (
	"context"
	"log"
	"net/http"
	"time"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	db, err := openDatabase(ctx)
	if err != nil {
		log.Fatalf("PostgreSQL startup failed: %v", err)
	}
	defer db.Close()

	api := newAPI(db)
	if err := api.initializeSchema(ctx); err != nil {
		log.Fatalf("PostgreSQL schema setup failed: %v", err)
	}
	if err := api.ensureQdrantCollection(ctx); err != nil {
		log.Fatalf("Qdrant startup failed: %v", err)
	}

	server := &http.Server{
		Addr:              ":8080",
		Handler:           api.routes(),
		ReadHeaderTimeout: 5 * time.Second,
	}
	log.Printf("Lynx API listening on %s", server.Addr)
	log.Fatal(server.ListenAndServe())
}
