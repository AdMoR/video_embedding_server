"""FastAPI server for video embedding similarity queries."""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import jax
import jax.numpy as jnp
import mediapy
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from videoprism import models as vp

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "videoprism_lvt_public_v1_base")
VIDEO_FOLDER = os.getenv("VIDEO_FOLDER", "")
EMBEDDINGS_FOLDER = os.getenv("EMBEDDINGS_FOLDER", "")  # If set, load pre-computed embeddings

print(VIDEO_FOLDER)
print(EMBEDDINGS_FOLDER)

NUM_FRAMES = 16
FRAME_SIZE = 288
USE_BFLOAT16 = False
DEFAULT_TOP_K = 5
DEFAULT_PROMPT_TEMPLATE = "a video of {}."

# Global state
flax_model = None
loaded_state = None
text_tokenizer = None
video_registry: dict[str, np.ndarray] = {}  # video_id -> embedding
cached_dummy_frames = None  # Cached dummy frames for text embedding computation


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def read_and_preprocess_video(
    filename: str, target_num_frames: int, target_frame_size: tuple[int, int]
):
    """Reads and preprocesses a video."""
    frames = mediapy.read_video(filename)

    # Sample to target number of frames.
    frame_indices = np.linspace(
        0, len(frames), num=target_num_frames, endpoint=False, dtype=np.int32
    )
    frames = np.array([frames[i] for i in frame_indices])

    # Resize to target size.
    original_height, original_width = frames.shape[-3:-1]
    target_height, target_width = target_frame_size
    assert (
        original_height * target_width == original_width * target_height
    ), "Currently does not support aspect ratio mismatch."
    frames = mediapy.resize_video(frames, shape=target_frame_size)

    # Normalize pixel values to [0.0, 1.0].
    frames = mediapy.to_float01(frames)

    return frames


def compute_similarity_matrix(
    video_embeddings,
    text_embeddings,
    temperature: float,
    apply_softmax: str | None = None,
) -> np.ndarray:
    """Computes cosine similarity matrix."""
    assert apply_softmax in [None, "over_texts", "over_videos"]
    emb_dim = video_embeddings[0].shape[-1]
    assert emb_dim == text_embeddings[0].shape[-1]

    video_embeddings = np.array(video_embeddings).reshape(-1, emb_dim)
    text_embeddings = np.array(text_embeddings).reshape(-1, emb_dim)
    similarity_matrix = np.dot(video_embeddings, text_embeddings.T)

    if temperature is not None:
        similarity_matrix /= temperature

    if apply_softmax == "over_videos":
        similarity_matrix = np.exp(similarity_matrix)
        similarity_matrix = similarity_matrix / np.sum(
            similarity_matrix, axis=0, keepdims=True
        )
    elif apply_softmax == "over_texts":
        similarity_matrix = np.exp(similarity_matrix)
        similarity_matrix = similarity_matrix / np.sum(
            similarity_matrix, axis=1, keepdims=True
        )

    return similarity_matrix


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


def load_embeddings_from_folder():
    """Load pre-computed embeddings from a folder."""
    global video_registry

    embeddings_path = Path(EMBEDDINGS_FOLDER)
    metadata_file = embeddings_path / "metadata.json"

    if not metadata_file.exists():
        logger.info(f"Warning: metadata.json not found in {EMBEDDINGS_FOLDER}")
        return False

    with open(metadata_file) as f:
        metadata = json.load(f)

    logger.info(f"Loading embeddings from: {EMBEDDINGS_FOLDER}")
    logger.info(f"  Model: {metadata.get('model_name', 'unknown')}")

    videos_metadata = metadata.get("videos", {})
    for video_name, video_info in videos_metadata.items():
        embedding_file = embeddings_path / video_info["embedding_file"]
        if embedding_file.exists():
            embedding = np.load(embedding_file)
            video_registry[video_name] = {
                "path": video_info["source_path"],
                "embedding": embedding,
                "name": video_name,
            }
        else:
            logger.info(f"  Warning: Embedding file not found: {embedding_file}")

    logger.info(f"Loaded {len(video_registry)} embeddings from pre-computed folder")
    return len(video_registry) > 0


def load_videos_from_folder():
    """Load videos from folder and compute their embeddings."""
    global video_registry

    # For demo: use the same video multiple times with different IDs
    demo_video_path = os.path.join(VIDEO_FOLDER, "water_bottle_drumming.mp4")

    if os.path.exists(demo_video_path):
        # Load and preprocess the video
        frames = read_and_preprocess_video(
            demo_video_path,
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
        video_embedding = np.array(video_embedding[0])  # Remove batch dim

        # Register the same video multiple times for demo
        for i in range(5):
            video_id = f"video_{i+1}"
            video_registry[video_id] = {
                "path": demo_video_path,
                "embedding": video_embedding,
                "name": f"water_bottle_drumming_{i+1}",
            }
        logger.info(f"Loaded {len(video_registry)} videos into registry")
    else:
        logger.info(f"Warning: Demo video not found at {demo_video_path}")


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


def load_videos():
    """Load videos - either from pre-computed embeddings or by computing them."""
    if EMBEDDINGS_FOLDER and os.path.isdir(EMBEDDINGS_FOLDER):
        if load_embeddings_from_folder():
            logger.info("Success for embedding loading")
            create_dummy_frames()
            return
        logger.info("Falling back to computing embeddings from videos...")

    load_videos_from_folder()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model and load videos on startup."""
    global flax_model, loaded_state, text_tokenizer

    logger.info(f"Loading model: {MODEL_NAME}")
    fprop_dtype = jnp.bfloat16 if USE_BFLOAT16 else None
    flax_model = vp.get_model(MODEL_NAME, fprop_dtype=fprop_dtype)
    loaded_state = vp.load_pretrained_weights(MODEL_NAME)
    text_tokenizer = vp.load_text_tokenizer("c4_en")
    logger.info("Model loaded successfully")

    logger.info("Loading videos...")
    load_videos()

    yield

    # Cleanup (if needed)
    logger.info("Shutting down...")


app = FastAPI(
    title="Video Embedding Similarity Server",
    description="Search videos by text similarity using VideoPrism embeddings",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
    query: str
    top_k: int = DEFAULT_TOP_K
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE


class SearchResult(BaseModel):
    video_id: str
    path: str
    name: str
    similarity: float


class SearchResponse(BaseModel):
    results: list[SearchResult]


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "videos_loaded": len(video_registry),
    }


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search for videos matching the text query."""
    global cached_dummy_frames

    # Apply prompt template
    text_query = request.prompt_template.format(request.query)

    # Tokenize the text query
    text_ids, text_paddings = vp.tokenize_texts(text_tokenizer, [text_query])
    if USE_BFLOAT16:
        text_paddings = text_paddings.astype(jnp.bfloat16)

    # Get text embedding (need dummy video for forward pass)
    if cached_dummy_frames is not None:
        dummy_frames = cached_dummy_frames
    else:
        # Use any video's frames - we only care about text embedding here
        first_video = next(iter(video_registry.values()))
        dummy_frames = read_and_preprocess_video(
            first_video["path"],
            target_num_frames=NUM_FRAMES,
            target_frame_size=(FRAME_SIZE, FRAME_SIZE),
        )
        dummy_frames = jnp.asarray(dummy_frames[None, ...])
        if USE_BFLOAT16:
            dummy_frames = dummy_frames.astype(jnp.bfloat16)
        # Cache for future use
        cached_dummy_frames = dummy_frames

    _, text_embedding, _ = forward_fn(dummy_frames, text_ids, text_paddings)
    text_embedding = np.array(text_embedding[0])  # Remove batch dim

    # Compute similarity with all videos
    video_ids = list(video_registry.keys())
    logger.info(video_ids)
    video_embeddings = [video_registry[vid]["embedding"] for vid in video_ids]

    similarity_matrix = compute_similarity_matrix(
        video_embeddings,
        [text_embedding],  # Pass as list of 1 item
        temperature=0.01,
        apply_softmax="over_videos",
    )

    # Get similarities for the single text query (column 0)
    similarities = similarity_matrix[:, 0]

    # Sort by similarity and get top-k
    top_indices = np.argsort(similarities)[::-1][: request.top_k]

    results = []
    for idx in top_indices:
        video_id = video_ids[idx]
        video_info = video_registry[video_id]
        results.append(
            SearchResult(
                video_id=video_id,
                path=video_info["path"],
                name=video_info["name"],
                similarity=float(similarities[idx]),
            )
        )

    return SearchResponse(results=results)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

