"""FastAPI server for video caption-based similarity search.

Videos are captioned with the configured backend (CHAI or Marlin-2B); the
caption text is embedded with EmbeddingGemma and stored in Qdrant. Search
embeds the query the same way and does cosine similarity over the embeddings.
"""

import json
import logging
import os
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

import captioner
import transcriber
from minio_client import MinIOClient
from storage import EMBEDDING_DIM, VectorStore

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant_server")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
PORT = int(os.getenv("PORT", "8004"))

DEFAULT_TOP_K = 5

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Global state
vector_store: VectorStore | None = None
minio_client: MinIOClient | None = None


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def extract_audio(video_path: str, audio_path: str) -> bool:
    """Extract mono 16 kHz WAV audio from a video for Whisper. Returns True on success."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                "-y", audio_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=120,
            check=False,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
        logger.debug(f"ffmpeg audio extraction error for {video_path}: {e}")
        return False


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            return float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, subprocess.SubprocessError) as e:
        logger.debug(f"ffprobe error for {video_path}: {e}")
    return 0.0


def _save_upload_to_temp(file: UploadFile) -> tuple[str, str]:
    """Validate extension and write an uploaded file to a temp path.

    Returns (temp_path, temp_dir). Caller is responsible for cleanup.
    """
    filename = Path(file.filename or "video.mp4").name  # basename only — avoids absolute-path injection via os.path.join
    file_ext = Path(filename).suffix.lower()
    if file_ext not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension '{file_ext}'. Allowed: {VIDEO_EXTENSIONS}",
        )
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, filename)
    return temp_path, temp_dir


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models, MinIO, and connect to Qdrant on startup."""
    global vector_store, minio_client

    logger.info(f"Loading models: backend={captioner.active_backend()} + EmbeddingGemma")
    captioner.load_models()
    logger.info("Models loaded successfully")

    # Connect to Qdrant
    logger.info(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    vector_store = VectorStore(host=QDRANT_HOST, port=QDRANT_PORT)
    collections = vector_store.list_collections()
    logger.info(f"Connected to Qdrant. Available collections: {collections}")

    # Initialize MinIO client
    logger.info("Initializing MinIO client")
    minio_client = MinIOClient()

    yield

    # Cleanup (if needed)
    logger.info("Shutting down...")


app = FastAPI(
    title="Video Caption Search Server",
    description="Search videos by caption similarity using EmbeddingGemma embeddings with tag filtering",
    lifespan=lifespan,
)


# Pydantic models
class TagInfo(BaseModel):
    name: str
    start: float
    end: float


class SearchRequest(BaseModel):
    query: str
    collection: str
    top_k: int = DEFAULT_TOP_K
    tags: list[str] | None = None
    tag_mode: str = "all"  # "all" (AND) or "any" (OR)


class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str


class SearchResult(BaseModel):
    segment_id: str
    video_id: str
    segment_index: int
    duration: float
    path: str
    tags: list[TagInfo]
    similarity: float
    caption: str | None = None
    scene: str | None = None
    transcription: list[TranscriptionSegment] | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]


class CaptionResponse(BaseModel):
    caption: str
    scene: str | None = None
    events: list[dict]


class CustomCaptionResponse(BaseModel):
    prompt: str
    caption: str


class FindResponse(BaseModel):
    raw: str
    span: list[float] | None = None
    format_ok: bool


class UpsertResponse(BaseModel):
    segment_id: str
    minio_path: str
    video_id: str


class DeleteVideoResponse(BaseModel):
    video_id: str
    deleted_segments: int
    deleted_files: int


class PurgeCollectionResponse(BaseModel):
    collection: str
    purged_segments: int
    deleted_files: int


@app.get("/health")
async def health():
    """Health check endpoint."""
    collections = vector_store.list_collections() if vector_store else []
    return {
        "status": "healthy",
        "model": captioner.active_backend(),
        "qdrant_host": QDRANT_HOST,
        "collections": collections,
    }


@app.get("/collections")
async def list_collections():
    """List all available collections."""
    return {"collections": vector_store.list_collections()}


@app.get("/tags/{collection}")
async def list_tags(collection: str):
    """List all unique tags in a collection."""
    if collection not in vector_store.list_collections():
        raise HTTPException(status_code=404, detail=f"Collection '{collection}' not found")
    return {"tags": vector_store.get_all_tags(collection)}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search for videos matching the text query with optional tag filtering."""
    # Validate collection exists
    if request.collection not in vector_store.list_collections():
        raise HTTPException(
            status_code=404, detail=f"Collection '{request.collection}' not found"
        )

    # Validate tag_mode
    if request.tag_mode not in ("all", "any"):
        raise HTTPException(
            status_code=400, detail="tag_mode must be 'all' or 'any'"
        )

    # Embed the query with EmbeddingGemma's query prompt
    query_embedding = captioner.embed_query(request.query)

    # Search with tag filtering
    results = vector_store.search(
        collection=request.collection,
        query_embedding=query_embedding,
        top_k=request.top_k,
        tags=request.tags,
        tag_mode=request.tag_mode,
    )

    return SearchResponse(
        results=[
            SearchResult(
                segment_id=str(r["id"]),
                video_id=r["video_id"],
                segment_index=r["segment_index"],
                duration=r["duration"],
                path=r["source_path"],
                tags=[TagInfo(**t) for t in r["tags"]],
                similarity=r["score"],
                caption=r.get("caption"),
                scene=r.get("scene"),
                transcription=[TranscriptionSegment(**s) for s in r.get("transcription") or []],
            )
            for r in results
        ]
    )


@app.post("/upsert", response_model=UpsertResponse)
async def upsert(
    file: UploadFile = File(..., description="Video file to upload"),
    collection: str = Form(..., description="Qdrant collection name"),
    tags: str | None = Form(None, description="JSON string of tags array"),
    embedding: str | None = Form(None, description=f"JSON string of embedding vector ({EMBEDDING_DIM} floats)"),
    caption: str | None = Form(None, description="Precomputed caption text (used with embedding)"),
):
    """
    Upload a video to MinIO and upsert its caption embedding to Qdrant.

    If embedding is provided, it is used directly (offline-precomputed path);
    pass the matching caption alongside it so it appears in search results.
    Otherwise the video is captioned with the active backend and the caption is embedded.
    """
    global minio_client

    # Validate file extension
    filename = Path(file.filename or "video.mp4").name
    file_ext = Path(filename).suffix.lower()
    if file_ext not in VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension '{file_ext}'. Allowed: {VIDEO_EXTENSIONS}",
        )

    # Save uploaded file to temp location
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, filename)

    try:
        # Write uploaded file to temp location
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # Transcribe audio with Whisper
        audio_path = os.path.join(temp_dir, "audio.wav")
        whisper_segments: list[dict] = []
        if extract_audio(temp_path, audio_path):
            try:
                logger.info(f"Transcribing audio for {filename}")
                whisper_segments = transcriber.transcribe_with_timestamps(audio_path)
                logger.info(f"Transcription done: {len(whisper_segments)} segments")
            except Exception as e:
                logger.warning(f"Transcription failed for {filename}: {e}")
            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
        else:
            logger.info(f"No audio track found in {filename}, skipping transcription")

        # Get or compute embedding
        caption_text = caption or ""
        scene_text = ""
        events: list[dict] = []
        if embedding:
            # Parse provided embedding (offline-precomputed path)
            try:
                embedding_vector = json.loads(embedding)
                if not isinstance(embedding_vector, list) or len(embedding_vector) != EMBEDDING_DIM:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Embedding must be a JSON array of {EMBEDDING_DIM} floats",
                    )
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid JSON in embedding: {e}",
                )
            logger.info(f"Using provided embedding for {filename}")
        else:
            # Caption the video and embed the caption text
            logger.info(f"Captioning {filename}")
            cap = captioner.caption(temp_path)
            caption_text = cap["caption"]
            scene_text = cap.get("scene") or ""
            events = cap.get("events") or []
            embedding_vector = captioner.embed_document(caption_text)

        # Parse tags if provided
        video_tags = []
        if tags:
            try:
                video_tags = json.loads(tags)
                if not isinstance(video_tags, list):
                    raise HTTPException(
                        status_code=400,
                        detail="Tags must be a JSON array",
                    )
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid JSON in tags: {e}",
                )

        # Ensure collection exists
        vector_store.ensure_collection(collection)

        # Generate video_id from filename (without extension)
        video_id = Path(filename).stem
        segment_id = f"{video_id}_seg_0"

        # Delete existing video if present (auto-cleanup)
        existing_paths = vector_store.delete_by_video_id(collection, video_id)
        if existing_paths:
            logger.info(f"Replacing existing video '{video_id}' ({len(existing_paths)} segments)")
            # Delete old files from MinIO
            for path in existing_paths:
                if path.startswith("minio://"):
                    parts = path.replace("minio://", "").split("/", 1)
                    if len(parts) == 2:
                        await minio_client.delete_object(parts[1])

        # Upload video to MinIO
        minio_key = f"videos/{collection}/{filename}"
        minio_path = await minio_client.upload_file(temp_path, minio_key)
        logger.info(f"Uploaded video to MinIO: {minio_path}")

        # Compute video duration
        duration = get_video_duration(temp_path)

        # Upsert to Qdrant with MinIO path
        vector_store.upsert(
            collection=collection,
            segment_id=segment_id,
            embedding=embedding_vector,
            video_id=video_id,
            tags=video_tags,
            duration=duration,
            segment_index=0,
            source_path=minio_path,
            caption=caption_text,
            scene=scene_text,
            events=events,
            transcription=whisper_segments,
        )
        logger.info(f"Upserted {segment_id} to collection {collection}")

        return UpsertResponse(
            segment_id=segment_id,
            minio_path=minio_path,
            video_id=video_id,
        )

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


@app.post("/caption", response_model=CaptionResponse)
async def caption_video(
    file: UploadFile = File(..., description="Video file to caption"),
):
    """Generate a dense caption (scene + timed events) for an uploaded video."""
    temp_path, temp_dir = _save_upload_to_temp(file)
    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        cap = captioner.caption(temp_path)
        return CaptionResponse(
            caption=cap["caption"],
            scene=cap.get("scene"),
            events=cap.get("events") or [],
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


@app.post("/caption/custom", response_model=CustomCaptionResponse)
async def caption_custom(
    file: UploadFile = File(..., description="Video file to caption"),
    prompt: str = Form(..., description="Custom prompt to pass to the model"),
    max_new_tokens: int = Form(512, description="Maximum tokens to generate"),
):
    """Caption a video with a custom user-supplied prompt (for prompt experimentation)."""
    temp_path, temp_dir = _save_upload_to_temp(file)
    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        result = captioner.caption_with_prompt(temp_path, prompt=prompt, max_new_tokens=max_new_tokens)
        return CustomCaptionResponse(prompt=prompt, caption=result)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


@app.post("/find", response_model=FindResponse)
async def find_event(
    file: UploadFile = File(..., description="Video file to search within"),
    event: str = Form(..., description="Natural-language event to locate"),
):
    """Temporally locate a natural-language event within an uploaded video."""
    temp_path, temp_dir = _save_upload_to_temp(file)
    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        res = captioner.find(temp_path, event=event)
        span = res.get("span")
        return FindResponse(
            raw=res.get("raw", ""),
            span=list(span) if span else None,
            format_ok=res.get("format_ok", False),
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)


@app.delete("/videos/{collection}/{video_id}", response_model=DeleteVideoResponse)
async def delete_video(collection: str, video_id: str):
    """
    Delete a video and all its segments from Qdrant and MinIO.

    Args:
        collection: Name of the collection
        video_id: ID of the video to delete
    """
    global minio_client

    # Validate collection exists
    if collection not in vector_store.list_collections():
        raise HTTPException(
            status_code=404, detail=f"Collection '{collection}' not found"
        )

    # Delete from Qdrant and get source paths
    source_paths = vector_store.delete_by_video_id(collection, video_id)

    if not source_paths:
        raise HTTPException(
            status_code=404, detail=f"Video '{video_id}' not found in collection '{collection}'"
        )

    # Delete files from MinIO
    deleted_files = 0
    for path in source_paths:
        # Extract key from minio:// URI
        if path.startswith("minio://"):
            # Format: minio://{bucket}/{key}
            parts = path.replace("minio://", "").split("/", 1)
            if len(parts) == 2:
                key = parts[1]
                await minio_client.delete_object(key)
                deleted_files += 1

    logger.info(f"Deleted video '{video_id}' from collection '{collection}': {len(source_paths)} segments, {deleted_files} files")

    return DeleteVideoResponse(
        video_id=video_id,
        deleted_segments=len(source_paths),
        deleted_files=deleted_files,
    )


@app.delete("/collections/{collection}/purge", response_model=PurgeCollectionResponse)
async def purge_collection(collection: str):
    """
    Purge all data from a collection (Qdrant + MinIO).

    This deletes all segments from the Qdrant collection and all video files
    under the videos/{collection}/ prefix in MinIO.
    """
    global minio_client

    # Validate collection exists
    if collection not in vector_store.list_collections():
        raise HTTPException(
            status_code=404, detail=f"Collection '{collection}' not found"
        )

    # Get count before purge
    count_before = vector_store.count(collection)

    # Purge Qdrant collection
    vector_store.purge_collection(collection)

    # Delete all files under videos/{collection}/ in MinIO
    minio_prefix = f"videos/{collection}/"
    deleted_files = await minio_client.delete_prefix(minio_prefix)

    logger.info(f"Purged collection '{collection}': {count_before} segments, {deleted_files} files")

    return PurgeCollectionResponse(
        collection=collection,
        purged_segments=count_before,
        deleted_files=deleted_files,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
