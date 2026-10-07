package main

import (
	"net/http"
)

func main() {
	// use custom mux
	mux := http.NewServeMux()

	mux.HandleFunc("POST /api/search", searchAPI)
	mux.HandleFunc("GET /api/images/{filename}", imageAPI)

	server := http.Server{
		Addr : ":8080",
		Handler : mux,
	}	

	server.ListenAndServe()
}
