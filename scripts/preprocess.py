#!/usr/bin/env python3
"""Preprocess videos: caption with Marlin-2B and embed captions for faster indexing.

For each video this saves a 768-d caption embedding (.npy) plus the caption text
to metadata.json, which `upsert.py` then pushes to the server. Captioning runs on
the GPU one video at a time (Marlin is not batched here), so this is a sequential
loop; only the ffprobe duration pre-check is parallelised.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import captioner

# Configuration
MARLIN_MODEL = os.getenv("MARLIN_MODEL", "NemoStation/Marlin-2B")
TEXT_EMBED_MODEL = os.getenv("TEXT_EMBED_MODEL", "google/embeddinggemma-300m")
MAX_VIDEO_DURATION_SECONDS = 20  # Skip videos longer than this to avoid OOM

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def get_video_duration_ffprobe(filename: str) -> float:
    """Get video duration in seconds using ffprobe (much faster than decoding)."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                filename,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,  # Safety timeout
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            return float(result.stdout.strip())
        return 0.0
    except (subprocess.TimeoutExpired, ValueError, subprocess.SubprocessError) as e:
        logging.debug(f"ffprobe error for {filename}: {e}")
        return 0.0


def check_video_duration_task(video_file: Path) -> tuple[Path, float | None, str | None]:
    """Worker function for duration checking with ffprobe (runs in separate process)."""
    try:
        duration = get_video_duration_ffprobe(str(video_file))
        if duration <= 0:
            return (video_file, None, "Invalid duration")
        return (video_file, duration, None)
    except Exception as e:
        return (video_file, None, str(e))


def save_job_state(
    output_path: Path,
    videos_to_process: list[Path],
    processed_videos: set[str],
    skipped_too_long: list[tuple[Path, float]],
    metadata: dict,
) -> None:
    """Save job state for resuming."""
    state_file = output_path / ".preprocess_state.json"
    state = {
        "videos_to_process": [str(v.absolute()) for v in videos_to_process],
        "processed_videos": list(processed_videos),
        "skipped_too_long": [(str(v.absolute()), d) for v, d in skipped_too_long],
        "metadata": metadata,
    }
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def load_job_state(output_path: Path) -> tuple[list[Path], set[str], list[tuple[Path, float]], dict] | None:
    """Load job state for resuming. Returns None if no state exists."""
    state_file = output_path / ".preprocess_state.json"
    if not state_file.exists():
        return None

    try:
        with open(state_file, "r") as f:
            state = json.load(f)

        videos_to_process = [Path(v) for v in state["videos_to_process"]]
        processed_videos = set(state["processed_videos"])
        skipped_too_long = [(Path(v), d) for v, d in state["skipped_too_long"]]
        metadata = state["metadata"]

        return videos_to_process, processed_videos, skipped_too_long, metadata
    except Exception as e:
        logging.warning(f"Error loading job state: {e}")
        return None


def preprocess_folder(
    input_folder: str,
    output_folder: str,
    resume: bool = False,
):
    """Caption all videos in a folder and save caption embeddings.

    Args:
        resume: If True, resume from saved state if available.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Try to resume from saved state
    processed_videos: set[str] = set()
    if resume:
        state = load_job_state(output_path)
        if state:
            _, processed_videos, skipped_too_long_loaded, metadata_loaded = state
            print(f"Resuming: Found {len(processed_videos)} already processed videos")
            metadata = metadata_loaded
            skipped_too_long = skipped_too_long_loaded
        else:
            metadata = {}
            skipped_too_long: list[tuple[Path, float]] = []
    else:
        metadata = {}
        skipped_too_long: list[tuple[Path, float]] = []

    # Also check for existing embedding files
    for video_file in input_path.glob("*"):
        if video_file.suffix.lower() in VIDEO_EXTENSIONS:
            video_name = video_file.stem
            embedding_file = output_path / f"{video_name}.npy"
            if embedding_file.exists() and video_name not in processed_videos:
                processed_videos.add(video_name)
                if video_name not in metadata:
                    metadata[video_name] = {
                        "source_path": str(video_file.absolute()),
                        "embedding_file": str(embedding_file.name),
                    }

    # Find all video files
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        video_files.extend(input_path.glob(f"*{ext}"))
        video_files.extend(input_path.glob(f"*{ext.upper()}"))

    if not video_files:
        print(f"No video files found in {input_folder}")
        return

    # Filter out videos that already have embeddings
    videos_to_check = []
    for video_file in video_files:
        video_name = video_file.stem
        embedding_file = output_path / f"{video_name}.npy"

        if embedding_file.exists() or video_name in processed_videos:
            # Already processed, just add to metadata
            if video_name not in metadata:
                metadata[video_name] = {
                    "source_path": str(video_file.absolute()),
                    "embedding_file": str(embedding_file.name),
                }
        else:
            videos_to_check.append(video_file)

    if not videos_to_check:
        print("No videos to process")
        if skipped_too_long:
            print(f"\n{len(skipped_too_long)} video(s) were skipped due to duration > {MAX_VIDEO_DURATION_SECONDS}s")
        return

    # Parallel duration checking with progress bar (pure ffprobe, no GPU)
    print(f"Checking duration for {len(videos_to_check)} videos using ffprobe...")
    videos_to_process = []
    duration_workers = min(64, (os.cpu_count() or 4) * 8)

    with ProcessPoolExecutor(max_workers=duration_workers) as executor:
        futures = {
            executor.submit(check_video_duration_task, video_file): video_file
            for video_file in videos_to_check
        }

        with tqdm(total=len(videos_to_check), desc="Checking durations") as pbar:
            for future in as_completed(futures):
                video_file, duration, error = future.result()
                pbar.update(1)

                if error:
                    logging.error(f"Error reading video info for {video_file.name}: {error}")
                    skipped_too_long.append((video_file, -1))
                elif duration and duration > MAX_VIDEO_DURATION_SECONDS:
                    skipped_too_long.append((video_file, duration))
                else:
                    videos_to_process.append(video_file)

    # Filter out already processed videos
    videos_to_process = [v for v in videos_to_process if v.stem not in processed_videos]

    if not videos_to_process:
        print("No videos to process after duration checks")
        if skipped_too_long:
            print(f"\n{len(skipped_too_long)} video(s) were skipped due to duration > {MAX_VIDEO_DURATION_SECONDS}s")
        return

    print(f"Found {len(video_files)} video(s), {len(videos_to_process)} to process, {len(skipped_too_long)} skipped (too long)")

    # Load models once (Marlin + EmbeddingGemma on GPU)
    print(f"Loading models: {MARLIN_MODEL} + {TEXT_EMBED_MODEL}")
    captioner.load_models()
    print("Models loaded successfully")

    # Sequential captioning + embedding (one video at a time on the GPU)
    pbar = tqdm(total=len(videos_to_process), desc="Captioning videos", unit="video")
    for video_file in videos_to_process:
        video_name = video_file.stem
        try:
            cap = captioner.caption(str(video_file))
            caption_text = cap["caption"]
            vec = captioner.embed_document(caption_text)

            embedding_file = output_path / f"{video_name}.npy"
            np.save(embedding_file, np.asarray(vec, dtype=np.float32))

            metadata[video_name] = {
                "source_path": str(video_file.absolute()),
                "embedding_file": str(embedding_file.name),
                "caption": caption_text,
                "scene": cap.get("scene"),
                "events": cap.get("events") or [],
            }
        except Exception as e:
            logging.error(f"Error captioning {video_file.name}: {e}", exc_info=True)
        finally:
            processed_videos.add(video_name)
            pbar.update(1)

        # Save state periodically
        if len(processed_videos) % 50 == 0:
            remaining_list = [v for v in videos_to_process if v.stem not in processed_videos]
            save_job_state(output_path, remaining_list, processed_videos, skipped_too_long, metadata)

    pbar.close()

    # Final state save
    save_job_state(output_path, [], processed_videos, skipped_too_long, metadata)

    # Save metadata
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(
            {
                "marlin_model": MARLIN_MODEL,
                "text_embed_model": TEXT_EMBED_MODEL,
                "embedding_dim": captioner.TEXT_EMBED_DIM,
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
        description="Caption videos and generate caption embeddings"
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
        "--resume",
        action="store_true",
        help="Resume from saved state if available",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print(f"Error: Input folder does not exist: {args.input_folder}")
        sys.exit(1)

    preprocess_folder(
        args.input_folder,
        args.output_folder,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
