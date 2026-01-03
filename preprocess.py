#!/usr/bin/env python3
"""Preprocess videos to generate embeddings for faster server startup."""

import argparse
import json
import logging
import os
import sys
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

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


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
):
    """Process all videos in a folder and save embeddings."""
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

    # Find all video files
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(input_path.glob(f"*{ext}"))
        video_files.extend(input_path.glob(f"*{ext.upper()}"))

    if not video_files:
        print(f"No video files found in {input_folder}")
        return

    print(f"Found {len(video_files)} video(s) to process")

    # Process each video
    metadata = {}
    for i, video_file in enumerate(video_files, 1):
        video_name = video_file.stem
        print(f"[{i}/{len(video_files)}] Processing: {video_file.name}")
        embedding_file = output_path / f"{video_name}.npy"
        try:
            if not embedding_file.exists():
                embedding = compute_video_embedding(
                    str(video_file),
                    flax_model,
                    loaded_state,
                    text_tokenizer,
                    forward_fn,
                )
                #print(embedding)
                # Save embedding as .npy file
                np.save(embedding_file, embedding)

            # Track metadata
            metadata[video_name] = {
                "source_path": str(video_file.absolute()),
                "embedding_file": str(embedding_file.name),
            }

            print(f"  Saved: {embedding_file.name} ")

        except Exception as e:
            logging.error(f"  Error processing {video_file.name}: {e}", exc_info=True)
            continue

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

    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder does not exist: {args.input_folder}")
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
    )


if __name__ == "__main__":
    main()

