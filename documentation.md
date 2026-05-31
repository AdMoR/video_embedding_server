# Video Caption Search Server — API Documentation

A REST API for **caption-based video search**. Videos are captioned with
[Marlin-2B](https://huggingface.co/NemoStation/Marlin-2B); the caption text is
embedded with [EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m)
and stored in [Qdrant](https://qdrant.tech/). Searching embeds the query the same
way and returns the closest captions by cosine similarity. The source videos are
stored in MinIO (S3-compatible).

---

## Table of Contents

- [How it works](#how-it-works)
- [Configuration](#configuration)
- [Base URL](#base-url)
- [Endpoints](#endpoints)
  - [GET /health](#get-health)
  - [GET /collections](#get-collections)
  - [GET /tags/{collection}](#get-tagscollection)
  - [POST /search](#post-search)
  - [POST /upsert](#post-upsert)
  - [POST /caption](#post-caption)
  - [POST /find](#post-find)
  - [DELETE /videos/{collection}/{video_id}](#delete-videoscollectionvideo_id)
  - [DELETE /collections/{collection}/purge](#delete-collectionscollectionpurge)
- [Response models](#response-models)
- [Error handling](#error-handling)
- [Batch processing many videos](#batch-processing-many-videos)
- [Performance notes](#performance-notes)

---

## How it works

```
                 ┌──────────────┐      caption text      ┌────────────────┐
   video ───────▶│  Marlin-2B   │──────────────────────▶│ EmbeddingGemma │
                 │ (captioning) │   "Scene: ... Events"  │ (768-d vector) │
                 └──────────────┘                        └────────┬───────┘
                                                                   │
   video file ──────────────────────────────▶ MinIO               ▼
                                                              ┌─────────┐
   query text ─▶ EmbeddingGemma (768-d) ─── cosine search ──▶│ Qdrant  │
                                                              └─────────┘
```

- **Captioning (Marlin-2B):** generates a dense caption — a `Scene:` paragraph
  plus timed `Events:` — and can temporally locate an event (`find`).
- **Embedding (EmbeddingGemma):** turns caption text and queries into 768-d
  vectors. It uses **asymmetric prompts**: documents (captions) and queries are
  encoded with different instructions, handled automatically by the server.
- **Storage:** the caption vector + the caption/scene/events + tags are stored in
  a Qdrant point; the video file goes to MinIO under `videos/{collection}/`.

A Qdrant collection is created with a fixed vector size of **768** and cosine
distance. The caption/scene/events are kept in the point payload so they are
returned with search results.

---

## Configuration

All configuration is via environment variables.

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Port the API server listens on |
| `MARLIN_MODEL` | `NemoStation/Marlin-2B` | Captioning model (HF repo id) |
| `MARLIN_REVISION` | _(none)_ | Pin a specific Marlin commit/revision |
| `TEXT_EMBED_MODEL` | `google/embeddinggemma-300m` | Text embedding model |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `HF_TOKEN` | _(none)_ | HF token — required at **first startup** to download the gated EmbeddingGemma model; cached in the HF volume after that |
| `MINIO_ENDPOINT` | `http://minio:9000` | MinIO server URL |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `MINIO_BUCKET` | `video-embeddings` | MinIO bucket for video files |

Marlin's video decoding is tuned by these (set in the Docker image; override in
your shell before start if needed):

| Variable | Default | Description |
|----------|---------|-------------|
| `FORCE_QWENVL_VIDEO_READER` | `torchcodec` | Video decoder backend |
| `VIDEO_MAX_PIXELS` | `200704` | Max pixels per frame (~448×448) |
| `FPS` | `2.0` | Frame sampling rate |
| `FPS_MAX_FRAMES` | `240` | Frame cap (~2 min of video) |
| `FPS_MIN_FRAMES` | `4` | Frame floor for short videos |

> **GPU required.** Both models load onto CUDA (`device_map={"":"cuda"}`,
> bfloat16). There is no CPU-only mode.

---

## Base URL

```
http://localhost:8000
```

---

## Endpoints

### GET /health

Check server status, the loaded captioning model, and available collections.

**Response**

```json
{
  "status": "healthy",
  "model": "NemoStation/Marlin-2B",
  "qdrant_host": "localhost",
  "collections": ["my_videos", "archive"]
}
```

---

### GET /collections

List all Qdrant collections.

**Response**

```json
{ "collections": ["my_videos", "archive"] }
```

---

### GET /tags/{collection}

List all unique tag names in a collection.

**Path parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `collection` | string | Collection name |

**Response**

```json
{ "tags": ["alice", "bob", "beach"] }
```

**Errors:** `404` if the collection does not exist.

---

### POST /search

Search for videos whose captions match a natural-language query, with optional
tag filtering. The query is embedded with EmbeddingGemma's query prompt and
compared by cosine similarity against stored caption embeddings.

**Request body** (JSON)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | ✅ | – | Natural-language search query |
| `collection` | string | ✅ | – | Collection to search |
| `top_k` | integer | ❌ | `5` | Number of results |
| `tags` | array[string] | ❌ | `null` | Filter by these tag names |
| `tag_mode` | string | ❌ | `"all"` | `"all"` (AND) or `"any"` (OR) |

> The old `prompt_template` field has been removed — EmbeddingGemma applies its
> own query instruction, so pass the raw query.

**Example request**

```json
{
  "query": "a person walking a dog in the park",
  "collection": "my_videos",
  "top_k": 10,
  "tags": ["alice"],
  "tag_mode": "all"
}
```

**Response**

```json
{
  "results": [
    {
      "segment_id": "video_001_seg_0",
      "video_id": "video_001",
      "segment_index": 0,
      "duration": 12.5,
      "path": "minio://video-embeddings/videos/my_videos/video_001.mp4",
      "tags": [{ "name": "alice", "start": 0.0, "end": 12.5 }],
      "similarity": 0.7421,
      "caption": "Scene: A woman walks a golden retriever along a tree-lined path...",
      "scene": "A woman walks a golden retriever along a tree-lined path..."
    }
  ]
}
```

**Errors:** `400` invalid `tag_mode`; `404` collection not found.

---

### POST /upsert

Upload a video, caption it, embed the caption, and store everything (video → MinIO,
vector + caption + tags → Qdrant). This is the online path: captioning happens on
the server during the request.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | ✅ | Video file (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`) |
| `collection` | string | ✅ | Target collection (created if missing) |
| `tags` | string (JSON) | ❌ | JSON array of `{name, start, end}` tag objects |
| `embedding` | string (JSON) | ❌ | Precomputed 768-float vector (offline path — skips captioning) |
| `caption` | string | ❌ | Precomputed caption text, stored alongside a provided `embedding` |

Behavior:

- If `embedding` is **omitted**, the server captions the video with Marlin and
  embeds the caption. The caption/scene/events are stored automatically.
- If `embedding` is **provided** (the offline/batch path), it is used directly and
  captioning is skipped. Pass `caption` too so it shows up in search results.
- The `video_id` is the filename without extension. Re-uploading the same
  `video_id` replaces the previous video and its segments (old MinIO files deleted).

**Example (curl, online captioning)**

```bash
curl -F file=@video_001.mp4 \
     -F collection=my_videos \
     -F 'tags=[{"name":"alice","start":0,"end":12.5}]' \
     http://localhost:8000/upsert
```

**Response**

```json
{
  "segment_id": "video_001_seg_0",
  "minio_path": "minio://video-embeddings/videos/my_videos/video_001.mp4",
  "video_id": "video_001"
}
```

**Errors:** `400` invalid extension / malformed `embedding` (wrong length or JSON) / malformed `tags`.

---

### POST /caption

Generate a dense caption for a video **without** storing anything. Useful for
inspection or ad-hoc captioning.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | ✅ | Video file |

**Example**

```bash
curl -F file=@clip.mp4 http://localhost:8000/caption
```

**Response**

```json
{
  "caption": "Scene: Two people cook pasta in a bright kitchen. Events: <0.0 - 4.2> ...",
  "scene": "Two people cook pasta in a bright kitchen.",
  "events": [
    { "start": 0.0, "end": 4.2, "description": "A person fills a pot with water." },
    { "start": 4.2, "end": 9.8, "description": "Another person chops garlic." }
  ]
}
```

---

### POST /find

Temporally locate a natural-language event within a single video (Marlin's "find"
mode). Returns the time span in seconds.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | ✅ | Video file |
| `event` | string | ✅ | Event to locate, e.g. `"a person enters the room"` |

**Example**

```bash
curl -F file=@clip.mp4 \
     -F 'event=a person enters the room' \
     http://localhost:8000/find
```

**Response**

```json
{
  "raw": "From 14.3 to 18.2.",
  "span": [14.3, 18.2],
  "format_ok": true
}
```

`span` is `null` when the model output could not be parsed (`format_ok: false`).

---

### DELETE /videos/{collection}/{video_id}

Delete a video and all its segments from Qdrant and MinIO.

**Response**

```json
{ "video_id": "video_001", "deleted_segments": 1, "deleted_files": 1 }
```

**Errors:** `404` collection or video not found.

---

### DELETE /collections/{collection}/purge

Delete **all** data in a collection — every Qdrant point and every MinIO file under
`videos/{collection}/`. The empty collection is recreated.

**Response**

```json
{ "collection": "my_videos", "purged_segments": 142, "deleted_files": 142 }
```

**Errors:** `404` collection not found.

---

## Response models

### SearchResult

| Field | Type | Description |
|-------|------|-------------|
| `segment_id` | string | Unique segment identifier |
| `video_id` | string | Source video identifier (filename stem) |
| `segment_index` | integer | Segment index within the video (currently always `0`) |
| `duration` | float | Video duration in seconds |
| `path` | string | MinIO URI of the source video |
| `tags` | array[TagInfo] | Tags on this segment |
| `similarity` | float | Cosine similarity (0–1, higher is better) |
| `caption` | string \| null | Full generated caption |
| `scene` | string \| null | Parsed scene paragraph |

### TagInfo

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Tag name |
| `start` | float | Tag start time (seconds) |
| `end` | float | Tag end time (seconds) |

### CaptionResponse / event object

| Field | Type | Description |
|-------|------|-------------|
| `caption` | string | Full raw caption text |
| `scene` | string \| null | Scene paragraph |
| `events` | array | `{ "start": float, "end": float, "description": string }` |

---

## Error handling

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request (invalid parameters / file) |
| 404 | Not found (collection or video) |
| 500 | Internal server error |

Error responses include a `detail` field:

```json
{ "detail": "Collection 'nonexistent' not found" }
```

```python
import requests

try:
    r = requests.post("http://localhost:8000/search",
                      json={"query": "test", "collection": "my_videos"})
    r.raise_for_status()
except requests.exceptions.ConnectionError:
    print("Could not connect to server")
except requests.exceptions.HTTPError as e:
    print(f"HTTP {e.response.status_code}: {e.response.json()['detail']}")
```

---

## Batch processing many videos

Captioning is the expensive step: Marlin processes **one video at a time on the
GPU**, taking seconds per video. There are two ways to index a large folder of
videos. For anything beyond a handful, prefer the **offline pipeline**.

### Approach A — `batch_upsert.py` (simple, no local GPU needed)

`batch_upsert.py` sends a folder of videos to the server one by one. The server
captions each video with Marlin on its own GPU. Includes progress bar, retries,
duration filtering, and resume support.

```bash
# Basic upload
python batch_upsert.py /data/videos --collection my_videos

# With tags, duration cap, resume, and custom server
python batch_upsert.py /data/videos \
    --collection my_videos \
    --tags-file tags.json \
    --max-duration 60 \
    --resume \
    --server http://gpu-box:8000

# Preview without uploading
python batch_upsert.py /data/videos --collection my_videos --dry-run

# Wipe collection first
python batch_upsert.py /data/videos --collection my_videos --purge
```

All options:

| Flag | Default | Description |
|------|---------|-------------|
| `video_dir` | (required) | Folder of video files |
| `--collection` | (required) | Qdrant collection name |
| `--server` | `http://localhost:8000` | Server URL |
| `--tags-file` | — | JSON file: `{stem: [{name, start, end}]}` |
| `--max-duration` | — | Skip videos longer than N seconds (ffprobe) |
| `--timeout` | `300` | Per-video HTTP timeout (seconds) |
| `--retries` | `3` | Retry attempts per video on failure |
| `--resume` | off | Skip already-uploaded videos (state in `.batch_upsert_state.json`) |
| `--purge` | off | Purge the collection before uploading |
| `--dry-run` | off | List videos that would be uploaded, then exit |
| `-v` / `--verbose` | off | Show per-video details |

### Approach B — offline pipeline (recommended for large sets)

Two scripts decouple the slow GPU captioning from the network upload, and add
**resume** support and a duration filter:

1. **`preprocess.py`** — captions every video in a folder, embeds the caption, and
   writes one `<stem>.npy` (768-d vector) per video plus a `metadata.json` holding
   the caption/scene/events and source paths. Skips videos longer than
   `MAX_VIDEO_DURATION_SECONDS` (default **20s**) and can resume.
2. **`build_tags.py`** *(optional)* — generates a `tags.json` via face recognition
   (maps characters to timed tag spans). See the script's `--help`.
3. **`upsert.py`** — reads the `.npy` + `metadata.json` (+ optional `tags.json`) and
   pushes everything to `/upsert`, passing the precomputed `embedding` and
   `caption` so the server does **not** re-caption.

```bash
# 1. Caption + embed offline (writes ./embeddings/*.npy + metadata.json)
#    Re-run with --resume to continue an interrupted job.
python preprocess.py /data/videos ./embeddings
python preprocess.py /data/videos ./embeddings --resume

# 2. (optional) Generate tags via face recognition
python build_tags.py /data/videos tags.json --characters alice bob

# 3. Bulk upsert to the running server (no re-captioning)
python upsert.py ./embeddings \
    --collection my_videos \
    --tags-file tags.json \
    --server-url http://localhost:8000

# Wipe and re-index a collection in one go:
python upsert.py ./embeddings --collection my_videos --purge
```

`metadata.json` produced by `preprocess.py`:

```json
{
  "marlin_model": "NemoStation/Marlin-2B",
  "text_embed_model": "google/embeddinggemma-300m",
  "embedding_dim": 768,
  "videos": {
    "video_001": {
      "source_path": "/data/videos/video_001.mp4",
      "embedding_file": "video_001.npy",
      "caption": "Scene: ...",
      "scene": "...",
      "events": [{ "start": 0.0, "end": 4.2, "description": "..." }]
    }
  }
}
```

Why this is better for large jobs:

- **Resumable** — `--resume` skips videos that already have a `.npy`, so a crash or
  reboot doesn't lose hours of captioning.
- **Decoupled** — captioning runs once; you can re-upsert to different collections
  or after changing tags without re-running the GPU.
- **Filtered** — long videos are skipped up front (cheap parallel `ffprobe` check)
  to avoid GPU OOM.

### Tag format

Both `--tags` (in `/upsert`) and `build_tags.py` output use the same shape — a map
from video stem to a list of timed tags:

```json
{
  "video_001": [
    { "name": "alice", "start": 0.0, "end": 12.5 },
    { "name": "beach", "start": 3.0, "end": 8.0 }
  ]
}
```

Tags are filterable at search time via `tags` + `tag_mode`.

---

## Performance notes

- **Throughput is GPU-bound and sequential.** Marlin captions one video at a time;
  expect seconds per video. Do **not** try to parallelize captioning against a
  single GPU — it will contend for memory, not speed up.
- **Search is fast.** A query is a single short EmbeddingGemma encode plus a Qdrant
  cosine lookup.
- **Memory.** Marlin is ~5 GB (bf16) resident; EmbeddingGemma adds ~0.6 GB. Both
  share one GPU.
- **Scaling captioning.** To go faster, run multiple server/`preprocess.py`
  instances across multiple GPUs, each handling a shard of the video folder, then
  upsert all shards into the same Qdrant collection.
- **Re-indexing.** A collection's vectors are caption embeddings of a specific
  model. If you change `TEXT_EMBED_MODEL`, purge and re-index — vectors from
  different models are not comparable.
- **Duration cap.** Tune `MAX_VIDEO_DURATION_SECONDS` in `preprocess.py` (and
  Marlin's `FPS_MAX_FRAMES`) for longer videos; the defaults target clips up to a
  few minutes.
```
