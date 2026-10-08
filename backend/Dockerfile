FROM golang:1.26.3-alpine AS build

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/lynx .

FROM alpine:3.22
RUN apk add --no-cache ca-certificates curl
WORKDIR /app
COPY --from=build /out/lynx /usr/local/bin/lynx
RUN mkdir -p /data/uploads
ENV LYNX_UPLOAD_DIR=/data/uploads
EXPOSE 8080
CMD ["/usr/local/bin/lynx"]
