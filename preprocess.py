#!/usr/bin/env python3
"""Preprocess videos to generate embeddings for faster server startup."""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import jax
import jax.numpy as jnp
import mediapy
import numpy as np

from videoprism import models as vp

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "videoprism_lvt_public_v1_base")
NUM_FRAMES = 16
FRAME_SIZE = 288
USE_BFLOAT16 = False
MAX_VIDEO_DURATION_SECONDS = 30  # Skip videos longer than this to avoid OOM

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def get_video_duration(filename: str) -> float:
    """Get video duration in seconds."""
    info = mediapy.read_video_info(filename)
    return info.num_images / info.fps if info.fps > 0 else 0


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

    # Normalize pixel values to [0.0, 1.0].
    frames = mediapy.to_float01(frames)

    return frames


def preprocess_video_task(video_file: Path) -> tuple[str, np.ndarray | None, Path, str | None]:
    """
    Worker function to preprocess a single video (CPU-bound).
    
    Returns:
        Tuple of (video_name, preprocessed_frames, video_path, error_message)
        If successful, error_message is None. If failed, frames is None.
    """
    video_name = video_file.stem
    try:
        frames = read_and_preprocess_video(
            str(video_file),
            target_num_frames=NUM_FRAMES,
            target_frame_size=(FRAME_SIZE, FRAME_SIZE),
        )
        return (video_name, frames, video_file, None)
    except Exception as e:
        return (video_name, None, video_file, str(e))


def load_model():
    """Load the VideoPrism model."""
    print(f"Loading model: {MODEL_NAME}")
    fprop_dtype = jnp.bfloat16 if USE_BFLOAT16 else None
    flax_model = vp.get_model(MODEL_NAME, fprop_dtype=fprop_dtype)
    loaded_state = vp.load_pretrained_weights(MODEL_NAME)
    text_tokenizer = vp.load_text_tokenizer("c4_en")
    print("Model loaded successfully")
    return flax_model, loaded_state, text_tokenizer


def compute_video_embedding(
    video_path: str,
    flax_model,
    loaded_state,
    text_tokenizer,
    forward_fn,
) -> np.ndarray:
    """Compute embedding for a single video."""
    # Load and preprocess video
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


def preprocess_folder(
    input_folder: str,
    output_folder: str,
    flax_model,
    loaded_state,
    text_tokenizer,
    batch_size: int = 1,
):
    """Process all videos in a folder and save embeddings.
    
    Uses parallel CPU preprocessing with batched GPU inference to maximize
    throughput while staying within VRAM limits.
    
    Args:
        batch_size: Number of videos to process in a single GPU forward pass.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create JIT-compiled forward function
    @jax.jit
    def forward_fn(inputs, text_token_ids, text_paddings, train=False):
        return flax_model.apply(
            loaded_state,
            inputs,
            text_token_ids,
            text_paddings,
            train=train,
        )

    # Prepare dummy text inputs once (reused for all videos)
    dummy_text = ["dummy"]
    text_ids, text_paddings = vp.tokenize_texts(text_tokenizer, dummy_text)
    if USE_BFLOAT16:
        text_paddings = text_paddings.astype(jnp.bfloat16)

    # Find all video files
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(input_path.glob(f"*{ext}"))
        video_files.extend(input_path.glob(f"*{ext.upper()}"))

    if not video_files:
        print(f"No video files found in {input_folder}")
        return

    # Filter out videos that already have embeddings or are too long
    videos_to_process = []
    skipped_too_long: list[tuple[Path, float]] = []  # (path, duration)
    metadata = {}
    
    for video_file in video_files:
        video_name = video_file.stem
        embedding_file = output_path / f"{video_name}.npy"
        
        if embedding_file.exists():
            # Already processed, just add to metadata
            metadata[video_name] = {
                "source_path": str(video_file.absolute()),
                "embedding_file": str(embedding_file.name),
            }
            print(f"Skipping (already exists): {video_file.name}")
        else:
            # Check video duration
            try:
                duration = get_video_duration(str(video_file))
                if duration > MAX_VIDEO_DURATION_SECONDS:
                    skipped_too_long.append((video_file, duration))
                    print(f"Skipping (too long: {duration:.1f}s > {MAX_VIDEO_DURATION_SECONDS}s): {video_file.name}")
                else:
                    videos_to_process.append(video_file)
            except Exception as e:
                logging.error(f"Error reading video info for {video_file.name}: {e}")
                skipped_too_long.append((video_file, -1))

    if not videos_to_process:
        print("No videos to process")
        if skipped_too_long:
            print(f"\n{len(skipped_too_long)} video(s) were skipped due to duration > {MAX_VIDEO_DURATION_SECONDS}s")
        return

    print(f"Found {len(video_files)} video(s), {len(videos_to_process)} to process, {len(skipped_too_long)} skipped (too long)")

    # Determine number of workers (conservative to avoid memory pressure)
    num_workers = max(2, (os.cpu_count() or 4) // 2)
    # Limit preloaded videos to avoid RAM saturation
    max_preloaded = batch_size * 2
    print(f"Using {num_workers} CPU workers for parallel preprocessing")
    print(f"GPU batch size: {batch_size}, max preloaded: {max_preloaded}")

    def run_batched_inference(batch_items: list[tuple[str, np.ndarray, Path]]):
        """Run GPU inference on a batch of preprocessed videos."""
        if not batch_items:
            return
        
        video_names = [item[0] for item in batch_items]
        frames_list = [item[1] for item in batch_items]
        video_files = [item[2] for item in batch_items]
        
        try:
            # Stack frames into a batch
            batched_frames = np.stack(frames_list, axis=0)
            jax_frames = jnp.asarray(batched_frames)
            if USE_BFLOAT16:
                jax_frames = jax_frames.astype(jnp.bfloat16)

            # Tile text inputs to match batch size
            batch_text_ids = jnp.tile(text_ids, (len(batch_items), 1))
            batch_text_paddings = jnp.tile(text_paddings, (len(batch_items), 1))

            # Run batched inference
            video_embeddings, _, _ = forward_fn(jax_frames, batch_text_ids, batch_text_paddings)
            
            # Save each embedding
            for i, (video_name, video_file) in enumerate(zip(video_names, video_files)):
                embedding = np.array(video_embeddings[i])
                embedding_file = output_path / f"{video_name}.npy"
                np.save(embedding_file, embedding)
                
                metadata[video_name] = {
                    "source_path": str(video_file.absolute()),
                    "embedding_file": str(embedding_file.name),
                }
                print(f"  Saved: {embedding_file.name}")
                
        except Exception as e:
            logging.error(f"  Error during batched GPU inference: {e}", exc_info=True)

    # Parallel CPU preprocessing with batched GPU inference
    # Limit in-flight tasks to max_preloaded to avoid RAM saturation
    processed_count = 0
    batch_buffer: list[tuple[str, np.ndarray, Path]] = []
    video_iter = iter(videos_to_process)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit initial batch of tasks (up to max_preloaded)
        pending_futures: dict = {}
        for video_file in video_iter:
            future = executor.submit(preprocess_video_task, video_file)
            pending_futures[future] = video_file
            if len(pending_futures) >= max_preloaded:
                break

        # Process as completed, submitting new tasks to maintain max_preloaded
        while pending_futures:
            # Wait for at least one future to complete
            done_futures = set()
            for future in as_completed(pending_futures):
                done_futures.add(future)
                video_name, frames, video_file, error = future.result()
                processed_count += 1
                
                if error is not None:
                    logging.error(f"[{processed_count}/{len(videos_to_process)}] Error preprocessing {video_file.name}: {error}")
                else:
                    print(f"[{processed_count}/{len(videos_to_process)}] Preprocessed: {video_file.name}")
                    batch_buffer.append((video_name, frames, video_file))
                    
                    # Run inference when batch is full
                    if len(batch_buffer) >= batch_size:
                        print(f"  Running GPU inference on batch of {len(batch_buffer)} videos...")
                        run_batched_inference(batch_buffer)
                        batch_buffer = []
                
                # Submit new task if there are more videos
                try:
                    next_video = next(video_iter)
                    new_future = executor.submit(preprocess_video_task, next_video)
                    pending_futures[new_future] = next_video
                except StopIteration:
                    pass
                
                # Remove completed future
                del pending_futures[future]
                break  # Process one at a time to maintain order

    # Process remaining items in buffer
    if batch_buffer:
        print(f"  Running GPU inference on final batch of {len(batch_buffer)} videos...")
        run_batched_inference(batch_buffer)

    # Save metadata
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "num_frames": NUM_FRAMES,
                "frame_size": FRAME_SIZE,
                "videos": metadata,
            },
            f,
            indent=2,
        )

    print(f"\nDone! Processed {len(metadata)} videos")
    print(f"Embeddings saved to: {output_path}")
    print(f"Metadata saved to: {metadata_file}")

    # Report skipped videos
    if skipped_too_long:
        print(f"\n⚠️  {len(skipped_too_long)} video(s) were skipped (duration > {MAX_VIDEO_DURATION_SECONDS}s):")
        for video_file, duration in skipped_too_long:
            if duration > 0:
                print(f"  - {video_file.name} ({duration:.1f}s)")
            else:
                print(f"  - {video_file.name} (unable to read duration)")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess videos to generate embeddings"
    )
    parser.add_argument(
        "input_folder",
        help="Folder containing video files",
    )
    parser.add_argument(
        "output_folder",
        help="Folder to save embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of videos to process in a single GPU forward pass (default: 1)",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder does not exist: {args.input_folder}")
        sys.exit(1)

    if args.batch_size < 1:
        print(f"Error: Batch size must be at least 1")
        sys.exit(1)

    # Load model
    flax_model, loaded_state, text_tokenizer = load_model()

    # Process videos
    preprocess_folder(
        args.input_folder,
        args.output_folder,
        flax_model,
        loaded_state,
        text_tokenizer,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

