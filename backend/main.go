package main

import (
	"fmt"
	"net/http"
)

func hello(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "Hello from Go!")
}

func main() {
	http.HandleFunc("/hello", hello)

	fmt.Println("Server running on http://localhost:8080")

	http.ListenAndServe(":8080", nil)
}
