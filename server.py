"""FastAPI server for video embedding similarity queries."""

import logging
import os
from contextlib import asynccontextmanager

import jax
import jax.numpy as jnp
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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

# Global state
flax_model = None
loaded_state = None
text_tokenizer = None
vector_store: VectorStore | None = None
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model and connect to Qdrant on startup."""
    global flax_model, loaded_state, text_tokenizer, vector_store

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
