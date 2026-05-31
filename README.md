# Video Caption Search Server

A GPU-accelerated FastAPI server for **caption-based video search**. Videos are
automatically captioned with [Marlin-2B](https://huggingface.co/NemoStation/Marlin-2B),
the caption text is embedded with
[EmbeddingGemma](https://huggingface.co/google/embeddinggemma-300m) (768-d), and the
vectors are stored in [Qdrant](https://qdrant.tech/) for cosine similarity search.
Video files are stored in [MinIO](https://min.io/) (S3-compatible).

---

## How it works

```
video ──▶ Marlin-2B ──▶ "Scene: ... Events: ..." ──▶ EmbeddingGemma ──▶ Qdrant
                                                         (768-d vector)

query ──▶ EmbeddingGemma ──▶ cosine search ──▶ results with captions
```

- **Captioning (Marlin-2B):** generates a `Scene:` paragraph + timed `Events:`.
  Also supports temporal grounding: given a natural-language event, returns the
  time span in seconds.
- **Text embedding (EmbeddingGemma 300M):** small, state-of-the-art Google model.
  Uses asymmetric query/document prompts automatically.
- **Search:** embeds the raw query and retrieves the closest captions by cosine
  similarity, with optional tag filtering (AND/OR).

---

## Requirements

- NVIDIA GPU with CUDA 12 and at least ~6 GB VRAM (Marlin ~5 GB + EmbeddingGemma ~0.6 GB, both bf16)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for Docker GPU access
- A Hugging Face account with the [EmbeddingGemma Gemma license accepted](https://huggingface.co/google/embeddinggemma-300m) and an HF token

---

## Quick start

### 1. Set your HF token

```bash
export HF_TOKEN=hf_...   # needs access to google/embeddinggemma-300m
```

### 2. Build and run

```bash
# Build (no model download at build time — weights are fetched at first startup)
docker build -t video-caption-server .

# Start Qdrant + the server
HF_TOKEN=hf_... docker compose up -d
```

The server starts on **port 8000**. Qdrant on 6333. MinIO must be available at
`MINIO_ENDPOINT` (default `http://minio:9000`).

On **first startup**, the server downloads Marlin-2B and EmbeddingGemma (~6 GB
total) into the `huggingface-cache` Docker volume. Subsequent restarts are instant
because the volume persists. `HF_TOKEN` is required once for the gated
EmbeddingGemma model; after that it can be omitted if the cache is warm.

### 3. Check health

```bash
curl http://localhost:8000/health
# {"status":"healthy","model":"NemoStation/Marlin-2B","qdrant_host":"qdrant","collections":[]}
```

### 4. Upload and index a video

```bash
curl -F file=@clip.mp4 -F collection=my_videos http://localhost:8000/upsert
```

### 5. Search

```bash
curl -s -XPOST http://localhost:8000/search \
  -H 'content-type: application/json' \
  -d '{"query":"a person cooking pasta","collection":"my_videos","top_k":5}' | python -m json.tool
```

---

## Batch indexing many videos

### Simple: `batch_upsert.py` (server does captioning)

Send a whole folder to the server — no local GPU or model download needed:

```bash
# Basic batch upload
python batch_upsert.py /data/videos --collection my_videos

# With optional tags, duration filter, and resume on crash
python batch_upsert.py /data/videos \
    --collection my_videos \
    --tags-file tags.json \
    --max-duration 60 \
    --resume

# Preview what would be sent without uploading
python batch_upsert.py /data/videos --collection my_videos --dry-run

# Wipe the collection first, then upload
python batch_upsert.py /data/videos --collection my_videos --purge
```

### Offline pipeline (recommended for hundreds of videos)

Decouple the slow GPU captioning from the upload; adds full resume support:

```bash
# Step 1 — caption + embed offline (writes ./embeddings/*.npy + metadata.json)
python preprocess.py /data/videos ./embeddings
python preprocess.py /data/videos ./embeddings --resume  # resume a crashed run

# Step 2 (optional) — face-recognition tags
python build_tags.py /data/videos tags.json --characters alice bob

# Step 3 — bulk push (no re-captioning on the server)
python upsert.py ./embeddings \
    --collection my_videos \
    --tags-file tags.json \
    --server-url http://localhost:8000
```

See [documentation.md](documentation.md) for the full batch-processing guide.

---

## API overview

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server status + loaded model |
| `GET` | `/collections` | List Qdrant collections |
| `GET` | `/tags/{collection}` | List tags in a collection |
| `POST` | `/search` | Search by text query |
| `POST` | `/upsert` | Upload + caption + index a video |
| `POST` | `/caption` | Caption a video (no storage) |
| `POST` | `/find` | Locate an event in a video (time span) |
| `DELETE` | `/videos/{collection}/{video_id}` | Delete a video |
| `DELETE` | `/collections/{collection}/purge` | Wipe a collection |

Full request/response documentation: [documentation.md](documentation.md).

---

## Configuration

Key environment variables (set in `docker-compose.yml` or your shell):

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | API server port |
| `MARLIN_MODEL` | `NemoStation/Marlin-2B` | Captioning model |
| `TEXT_EMBED_MODEL` | `google/embeddinggemma-300m` | Text embedding model |
| `HF_TOKEN` | — | Required at first startup to download gated EmbeddingGemma; cached in the HF volume after that |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `MINIO_ENDPOINT` | `http://minio:9000` | MinIO URL |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO credentials |
| `MINIO_SECRET_KEY` | `minioadmin` | |
| `MINIO_BUCKET` | `video-embeddings` | MinIO bucket |

---

## Project structure

```
server.py          FastAPI application (all endpoints)
captioner.py       Marlin + EmbeddingGemma model wrappers
storage.py         Qdrant vector store abstraction
minio_client.py    MinIO async client
batch_upsert.py    Batch upload a video folder via /upsert (server captions)
preprocess.py      Offline batch: caption + embed videos locally
upsert.py          Bulk push precomputed embeddings to server
build_tags.py      Face-recognition tag generation
client.py          CLI client for search / health / listing
Dockerfile
docker-compose.yml
documentation.md   Full API + batch-processing reference
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
