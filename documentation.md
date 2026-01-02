# Video Embedding Server API Documentation

A REST API for searching video segments by text similarity using VideoPrism embeddings with optional tag filtering.

---

## Table of Contents

- [Overview](#overview)
- [Base URL](#base-url)
- [Endpoints](#endpoints)
  - [Health Check](#health-check)
  - [List Collections](#list-collections)
  - [List Tags](#list-tags)
  - [Search Videos](#search-videos)
- [Python Client Examples](#python-client-examples)
  - [Using requests](#using-requests)
  - [Full Client Class](#full-client-class)
  - [Command Line Usage](#command-line-usage)
- [Response Models](#response-models)

---

## Overview

This server provides a semantic video search API powered by [VideoPrism](https://arxiv.org/abs/2402.13217) embeddings stored in Qdrant. You can:

- Search video segments by natural language queries
- Filter results by tags (using AND/OR logic)
- List available collections and their tags

---

## Base URL

```
http://localhost:8003
```

Configure via environment variables:
- Server runs on port `8003` by default
- Qdrant host/port configurable via `QDRANT_HOST` and `QDRANT_PORT`

---

## Endpoints

### Health Check

Check server status and available collections.

```
GET /health
```

**Response:**

```json
{
  "status": "healthy",
  "model": "videoprism_lvt_public_v1_base",
  "qdrant_host": "localhost",
  "collections": ["my_videos", "archive"]
}
```

---

### List Collections

Get all available Qdrant collections.

```
GET /collections
```

**Response:**

```json
{
  "collections": ["my_videos", "archive"]
}
```

---

### List Tags

Get all unique tags in a specific collection.

```
GET /tags/{collection}
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `collection` | string | Name of the collection |

**Response:**

```json
{
  "tags": ["person:john", "location:beach", "activity:running"]
}
```

**Errors:**

| Status | Description |
|--------|-------------|
| 404 | Collection not found |

---

### Search Videos

Search for video segments matching a text query with optional tag filtering.

```
POST /search
```

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | ✅ | - | Natural language search query |
| `collection` | string | ✅ | - | Collection to search in |
| `top_k` | integer | ❌ | 5 | Number of results to return |
| `prompt_template` | string | ❌ | `"a video of {}."` | Template for query (use `{}` as placeholder) |
| `tags` | array[string] | ❌ | null | Filter by these tags |
| `tag_mode` | string | ❌ | `"all"` | `"all"` (AND) or `"any"` (OR) |

**Example Request:**

```json
{
  "query": "person dancing",
  "collection": "my_videos",
  "top_k": 10,
  "prompt_template": "a video of {}.",
  "tags": ["person:john", "location:studio"],
  "tag_mode": "all"
}
```

**Response:**

```json
{
  "results": [
    {
      "segment_id": "abc123",
      "video_id": "video_001",
      "segment_index": 3,
      "duration": 4.5,
      "path": "/videos/video_001.mp4",
      "tags": [
        {"name": "person:john", "start": 0.0, "end": 4.5},
        {"name": "location:studio", "start": 0.0, "end": 4.5}
      ],
      "similarity": 0.8542
    }
  ]
}
```

**Errors:**

| Status | Description |
|--------|-------------|
| 400 | Invalid `tag_mode` (must be `"all"` or `"any"`) |
| 404 | Collection not found |

---

## Python Client Examples

### Using requests

**Install dependency:**

```bash
pip install requests
```

**Basic search:**

```python
import requests

SERVER_URL = "http://localhost:8000"

# Simple search
response = requests.post(
    f"{SERVER_URL}/search",
    json={
        "query": "person running",
        "collection": "my_videos",
        "top_k": 5,
    }
)
response.raise_for_status()
results = response.json()

for result in results["results"]:
    print(f"Video: {result['video_id']}")
    print(f"  Path: {result['path']}")
    print(f"  Similarity: {result['similarity']:.4f}")
    print(f"  Tags: {[t['name'] for t in result['tags']]}")
    print()
```

**Search with tag filtering:**

```python
import requests

SERVER_URL = "http://localhost:8003"

# Search with tag filter (AND mode - all tags must match)
response = requests.post(
    f"{SERVER_URL}/search",
    json={
        "query": "talking",
        "collection": "my_videos",
        "top_k": 10,
        "tags": ["person:alice", "location:office"],
        "tag_mode": "all",  # Both tags must be present
    }
)
results = response.json()

# Search with tag filter (OR mode - any tag can match)
response = requests.post(
    f"{SERVER_URL}/search",
    json={
        "query": "laughing",
        "collection": "my_videos",
        "top_k": 10,
        "tags": ["person:alice", "person:bob"],
        "tag_mode": "any",  # Either Alice or Bob
    }
)
results = response.json()
```

**Health check and listing:**

```python
import requests

SERVER_URL = "http://localhost:8003"

# Health check
health = requests.get(f"{SERVER_URL}/health").json()
print(f"Status: {health['status']}")
print(f"Model: {health['model']}")
print(f"Collections: {health['collections']}")

# List collections
collections = requests.get(f"{SERVER_URL}/collections").json()
print(f"Available: {collections['collections']}")

# List tags in a collection
tags = requests.get(f"{SERVER_URL}/tags/my_videos").json()
print(f"Tags: {tags['tags']}")
```

---

### Full Client Class

A reusable client class for the API:

```python
"""Video Embedding Server Client"""

import requests
from typing import Optional


class VideoSearchClient:
    """Client for the Video Embedding Similarity Server."""

    def __init__(self, server_url: str = "http://localhost:8003"):
        self.server_url = server_url.rstrip("/")

    def health(self) -> dict:
        """Check server health status."""
        response = requests.get(f"{self.server_url}/health")
        response.raise_for_status()
        return response.json()

    def list_collections(self) -> list[str]:
        """List all available collections."""
        response = requests.get(f"{self.server_url}/collections")
        response.raise_for_status()
        return response.json()["collections"]

    def list_tags(self, collection: str) -> list[str]:
        """List all tags in a collection."""
        response = requests.get(f"{self.server_url}/tags/{collection}")
        response.raise_for_status()
        return response.json()["tags"]

    def search(
        self,
        query: str,
        collection: str,
        top_k: int = 5,
        prompt_template: str = "a video of {}.",
        tags: Optional[list[str]] = None,
        tag_mode: str = "all",
    ) -> list[dict]:
        """
        Search for videos matching a text query.

        Args:
            query: Natural language search query
            collection: Qdrant collection to search
            top_k: Number of results to return
            prompt_template: Template to wrap query (use {} as placeholder)
            tags: Optional list of tags to filter by
            tag_mode: "all" (AND) or "any" (OR) for tag filtering

        Returns:
            List of search results with video info and similarity scores
        """
        payload = {
            "query": query,
            "collection": collection,
            "top_k": top_k,
            "prompt_template": prompt_template,
        }
        if tags:
            payload["tags"] = tags
            payload["tag_mode"] = tag_mode

        response = requests.post(f"{self.server_url}/search", json=payload)
        response.raise_for_status()
        return response.json()["results"]


# Usage example
if __name__ == "__main__":
    client = VideoSearchClient()

    # Check health
    print("Server status:", client.health()["status"])

    # List collections
    collections = client.list_collections()
    print(f"Collections: {collections}")

    if collections:
        collection = collections[0]

        # List tags
        tags = client.list_tags(collection)
        print(f"Tags in {collection}: {tags[:5]}...")  # First 5 tags

        # Search
        results = client.search(
            query="person walking",
            collection=collection,
            top_k=3,
        )

        print(f"\nSearch results:")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['video_id']} (score: {r['similarity']:.4f})")
```

---

### Command Line Usage

The included `client.py` script provides CLI access:

```bash
# Health check
python client.py --health

# List all collections
python client.py --list-collections

# List tags in a collection
python client.py --list-tags --collection my_videos

# Basic search
python client.py "person dancing" --collection my_videos

# Search with options
python client.py "cooking food" \
    --collection my_videos \
    --top-k 10 \
    --template "a video showing {}."

# Search with tag filtering
python client.py "talking" \
    --collection my_videos \
    --tags person:alice person:bob \
    --tag-mode any

# Use different server
python client.py "running" \
    --collection my_videos \
    --server http://remote-server:8000
```

---

## Response Models

### SearchResult

| Field | Type | Description |
|-------|------|-------------|
| `segment_id` | string | Unique identifier for the video segment |
| `video_id` | string | Identifier of the source video |
| `segment_index` | integer | Index of this segment within the video |
| `duration` | float | Duration of the segment in seconds |
| `path` | string | File path to the source video |
| `tags` | array[TagInfo] | Tags associated with this segment |
| `similarity` | float | Cosine similarity score (0-1, higher is better) |

### TagInfo

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Tag name/label |
| `start` | float | Start time of the tag in seconds |
| `end` | float | End time of the tag in seconds |

---

## Error Handling

All endpoints return standard HTTP error codes:

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request (invalid parameters) |
| 404 | Not found (collection doesn't exist) |
| 500 | Internal server error |

Error responses include a `detail` field:

```json
{
  "detail": "Collection 'nonexistent' not found"
}
```

**Python error handling example:**

```python
import requests

try:
    response = requests.post(
        "http://localhost:8003/search",
        json={"query": "test", "collection": "my_videos"}
    )
    response.raise_for_status()
    results = response.json()
except requests.exceptions.ConnectionError:
    print("Could not connect to server")
except requests.exceptions.HTTPError as e:
    print(f"HTTP error {e.response.status_code}: {e.response.json()['detail']}")
```

