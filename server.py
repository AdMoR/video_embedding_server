"""FastAPI server for video embedding similarity queries."""

import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import jax
import jax.numpy as jnp
import mediapy
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from minio_client import MinIOClient
from storage import VectorStore
from videoprism import models as vp

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "videoprism_lvt_public_v1_base")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

NUM_FRAMES = 16
FRAME_SIZE = 288
USE_BFLOAT16 = False
DEFAULT_TOP_K = 5
DEFAULT_PROMPT_TEMPLATE = "a video of {}."

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Global state
flax_model = None
loaded_state = None
text_tokenizer = None
vector_store: VectorStore | None = None
minio_client: MinIOClient | None = None
cached_dummy_frames = None  # Cached dummy frames for text embedding computation


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@jax.jit
def forward_fn(inputs, text_token_ids, text_paddings, train=False):
    """JIT-compiled forward pass."""
    return flax_model.apply(
        loaded_state,
        inputs,
        text_token_ids,
        text_paddings,
        train=train,
    )


def create_dummy_frames():
    """Create and cache dummy frames for text embedding computation."""
    global cached_dummy_frames

    # Create random dummy frames (the video content doesn't affect text embeddings)
    dummy_frames = np.random.rand(NUM_FRAMES, FRAME_SIZE, FRAME_SIZE, 3).astype(
        np.float32
    )
    dummy_frames = jnp.asarray(dummy_frames[None, ...])  # Add batch dimension
    if USE_BFLOAT16:
        dummy_frames = dummy_frames.astype(jnp.bfloat16)

    cached_dummy_frames = dummy_frames
    logger.info("Created cached dummy frames for text embedding computation")


def read_and_preprocess_video(
    filename: str, target_num_frames: int, target_frame_size: tuple[int, int]
):
    """Reads and preprocesses a video for embedding computation."""
    frames = mediapy.read_video(filename)

    # Sample to target number of frames
    frame_indices = np.linspace(
        0, len(frames), num=target_num_frames, endpoint=False, dtype=np.int32
    )
    frames = np.array([frames[i] for i in frame_indices])

    # Resize to target size
    original_height, original_width = frames.shape[-3:-1]
    target_height, target_width = target_frame_size
    if original_height * target_width != original_width * target_height:
        # Center crop to target aspect ratio
        target_ratio = target_width / target_height
        original_ratio = original_width / original_height
        if original_ratio > target_ratio:
            # Crop width
            new_width = int(original_height * target_ratio)
            start = (original_width - new_width) // 2
            frames = frames[:, :, start : start + new_width, :]
        else:
            # Crop height
            new_height = int(original_width / target_ratio)
            start = (original_height - new_height) // 2
            frames = frames[:, start : start + new_height, :, :]

    frames = mediapy.resize_video(frames, shape=target_frame_size)

    # Normalize pixel values to [0.0, 1.0]
    frames = mediapy.to_float01(frames)

    return frames


def compute_video_embedding(video_path: str) -> np.ndarray:
    """Compute embedding for a single video file."""
    frames = read_and_preprocess_video(
        video_path,
        target_num_frames=NUM_FRAMES,
        target_frame_size=(FRAME_SIZE, FRAME_SIZE),
    )
    frames = jnp.asarray(frames[None, ...])  # Add batch dimension
    if USE_BFLOAT16:
        frames = frames.astype(jnp.bfloat16)

    # Compute video embedding (need dummy text for forward pass)
    dummy_text = ["dummy"]
    text_ids, text_paddings = vp.tokenize_texts(text_tokenizer, dummy_text)
    if USE_BFLOAT16:
        text_paddings = text_paddings.astype(jnp.bfloat16)

    video_embedding, _, _ = forward_fn(frames, text_ids, text_paddings)
    return np.array(video_embedding[0])  # Remove batch dim


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model, MinIO, and connect to Qdrant on startup."""
    global flax_model, loaded_state, text_tokenizer, vector_store, minio_client

    logger.info(f"Loading model: {MODEL_NAME}")
    fprop_dtype = jnp.bfloat16 if USE_BFLOAT16 else None
    flax_model = vp.get_model(MODEL_NAME, fprop_dtype=fprop_dtype)
    loaded_state = vp.load_pretrained_weights(MODEL_NAME)
    text_tokenizer = vp.load_text_tokenizer("c4_en")
    logger.info("Model loaded successfully")

    # Create dummy frames for text embedding computation
    create_dummy_frames()

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
    title="Video Embedding Similarity Server",
    description="Search videos by text similarity using VideoPrism embeddings with tag filtering",
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
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    tags: list[str] | None = None
    tag_mode: str = "all"  # "all" (AND) or "any" (OR)


class SearchResult(BaseModel):
    segment_id: str
    video_id: str
    segment_index: int
    duration: float
    path: str
    tags: list[TagInfo]
    similarity: float


class SearchResponse(BaseModel):
    results: list[SearchResult]


class UpsertResponse(BaseModel):
    segment_id: str
    minio_path: str
    video_id: str


@app.get("/health")
async def health():
    """Health check endpoint."""
    collections = vector_store.list_collections() if vector_store else []
    return {
        "status": "healthy",
        "model": MODEL_NAME,
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
    global cached_dummy_frames

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

    # Apply prompt template
    text_query = request.prompt_template.format(request.query)

    # Tokenize the text query
    text_ids, text_paddings = vp.tokenize_texts(text_tokenizer, [text_query])
    if USE_BFLOAT16:
        text_paddings = text_paddings.astype(jnp.bfloat16)

    # Get text embedding using dummy frames
    _, text_embedding, _ = forward_fn(cached_dummy_frames, text_ids, text_paddings)
    text_embedding = np.array(text_embedding[0])  # Remove batch dim

    # Search with tag filtering
    results = vector_store.search(
        collection=request.collection,
        query_embedding=text_embedding.tolist(),
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
            )
            for r in results
        ]
    )


@app.post("/upsert", response_model=UpsertResponse)
async def upsert(
    file: UploadFile = File(..., description="Video file to upload"),
    collection: str = Form(..., description="Qdrant collection name"),
    tags: str | None = Form(None, description="JSON string of tags array"),
    embedding: str | None = Form(None, description="JSON string of embedding vector (768 floats)"),
):
    """
    Upload a video to MinIO and upsert its embedding to Qdrant.

    If embedding is provided, it will be used directly.
    Otherwise, the embedding will be computed from the video.
    """
    global minio_client

    # Validate file extension
    filename = file.filename or "video.mp4"
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

        # Get or compute embedding
        if embedding:
            # Parse provided embedding
            try:
                embedding_vector = json.loads(embedding)
                if not isinstance(embedding_vector, list) or len(embedding_vector) != 768:
                    raise HTTPException(
                        status_code=400,
                        detail="Embedding must be a JSON array of 768 floats",
                    )
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid JSON in embedding: {e}",
                )
            logger.info(f"Using provided embedding for {filename}")
        else:
            # Compute embedding from video
            logger.info(f"Computing embedding for {filename}")
            embedding_array = compute_video_embedding(temp_path)
            embedding_vector = embedding_array.tolist()

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

        # Upload video to MinIO
        minio_key = f"videos/{collection}/{filename}"
        minio_path = await minio_client.upload_file(temp_path, minio_key)
        logger.info(f"Uploaded video to MinIO: {minio_path}")

        # Ensure collection exists
        vector_store.ensure_collection(collection)

        # Generate video_id from filename (without extension)
        video_id = Path(filename).stem
        segment_id = f"{video_id}_seg_0"

        # Upsert to Qdrant with MinIO path
        vector_store.upsert(
            collection=collection,
            segment_id=segment_id,
            embedding=embedding_vector,
            video_id=video_id,
            tags=video_tags,
            duration=0.0,  # Could be computed from video if needed
            segment_index=0,
            source_path=minio_path,
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8004)
