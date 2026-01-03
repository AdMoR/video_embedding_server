#!/usr/bin/env python3
"""Generate tags for videos using face recognition and video understanding.

This script processes videos to detect known faces and generates tags
in the format expected by upsert.py.

Output format (tags.json):
{
    "video_001": [
        {"name": "fred", "start": 0.0, "end": 30.5},
        {"name": "jamy", "start": 10.0, "end": 25.0}
    ]
}

Identity configuration (identities.json):
{
    "fred": [
        "/path/to/fred_1.jpeg",
        "/path/to/fred_2.jpeg"
    ],
    "jamy": [
        "/path/to/jamy_1.jpeg",
        "/path/to/jamy_2.jpeg"
    ]
}

Usage:
    python build_tags.py ./videos tags.json --identities identities.json
    python build_tags.py ./videos tags.json --identities identities.json --workers 8
"""

import argparse
import json
import multiprocessing as mp
import os
import subprocess
from functools import partial
from pathlib import Path

import cv2
import face_recognition
import numpy as np

# Video extensions to look for
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def load_identities(identities_file: str) -> tuple[list, list[str]]:
    """Load face embeddings from identities configuration file.

    Args:
        identities_file: Path to JSON file mapping names to image paths.

    Returns:
        Tuple of (list of face encodings, list of names).
    """
    with open(identities_file) as f:
        identities = json.load(f)

    all_encodings = []
    all_names = []

    for name, image_paths in identities.items():
        for image_path in image_paths:
            if not os.path.exists(image_path):
                print(f"  Warning: Image not found: {image_path}")
                continue

            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if encodings:
                all_encodings.append(encodings[0])
                all_names.append(name)
            else:
                print(f"  Warning: No face found in: {image_path}")

    return all_encodings, all_names


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe.

    Args:
        video_path: Path to video file.

    Returns:
        Duration in seconds.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return float(result.stdout)
    except Exception:
        return 0.0


def detect_faces_in_video(
    video_path: str,
    known_face_encodings: list,
    known_face_names: list[str],
    speed_up_factor: int = 2,
) -> list[tuple[str, int, tuple]]:
    """Detect known faces in a video.

    Args:
        video_path: Path to video file.
        known_face_encodings: List of face encodings to match against.
        known_face_names: List of names corresponding to encodings.
        speed_up_factor: Process every N seconds (default: 2).

    Returns:
        List of (name, frame_index, face_location) tuples.
    """
    video_stream = cv2.VideoCapture(video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30  # Default fallback

    index = 0
    results = []

    while True:
        index += 1
        still_reading, frame = video_stream.read()

        if not still_reading:
            video_stream.release()
            break

        # Process every N seconds
        if index % int(fps * speed_up_factor) != 1:
            continue

        # Detect faces in frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for i, face_encoding in enumerate(face_encodings):
            # Compare with known faces
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding
            )

            if not any(matches):
                continue

            # Find best match
            face_distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                results.append((name, index, face_locations[i]))

    return results


def faces_to_tags(
    face_detections: list[tuple[str, int, tuple]],
    fps: float,
    duration: float,
) -> list[dict]:
    """Convert face detections to tag format with timestamps.

    Groups consecutive detections of the same person into time ranges.

    Args:
        face_detections: List of (name, frame_index, location) tuples.
        fps: Video frames per second.
        duration: Total video duration in seconds.

    Returns:
        List of tag dicts with name, start, end.
    """
    if not face_detections:
        return []

    # Group detections by person
    person_frames: dict[str, list[int]] = {}
    for name, frame_idx, _ in face_detections:
        if name not in person_frames:
            person_frames[name] = []
        person_frames[name].append(frame_idx)

    tags = []
    for name, frames in person_frames.items():
        frames = sorted(frames)

        # Convert frame indices to timestamps
        # For simplicity, use first and last appearance
        start_time = frames[0] / fps if fps > 0 else 0.0
        end_time = frames[-1] / fps if fps > 0 else duration

        # Clamp to video duration
        start_time = max(0.0, min(start_time, duration))
        end_time = max(start_time, min(end_time, duration))

        tags.append({
            "name": name,
            "start": round(start_time, 2),
            "end": round(end_time, 2),
        })

    return tags


def process_single_video(
    video_path: Path,
    known_face_encodings: list,
    known_face_names: list[str],
    speed_up_factor: int = 2,
) -> list[dict]:
    """Process a single video and generate tags.

    Args:
        video_path: Path to video file.
        known_face_encodings: List of face encodings.
        known_face_names: List of names for encodings.
        speed_up_factor: Process every N seconds.

    Returns:
        List of tag dicts for upsert.py format.
    """
    video_stream = cv2.VideoCapture(str(video_path))
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    video_stream.release()

    if fps <= 0:
        fps = 30

    duration = get_video_duration(str(video_path))

    # Detect faces
    face_detections = detect_faces_in_video(
        str(video_path),
        known_face_encodings,
        known_face_names,
        speed_up_factor=speed_up_factor,
    )

    # Convert to tags format
    tags = faces_to_tags(face_detections, fps, duration)

    return tags


def process_video_worker(args: tuple) -> tuple[str, list[dict], str | None]:
    """Worker function for multiprocessing.

    Args:
        args: Tuple of (video_path, known_encodings, known_names, speed_up_factor)

    Returns:
        Tuple of (video_name, tags, error_message or None)
    """
    video_path, known_encodings, known_names, speed_up_factor = args
    video_name = Path(video_path).stem

    try:
        tags = process_single_video(
            Path(video_path),
            known_encodings,
            known_names,
            speed_up_factor=speed_up_factor,
        )
        return (video_name, tags, None)
    except Exception as e:
        return (video_name, [], str(e))


def process_videos_parallel(
    video_files: list[Path],
    known_encodings: list,
    known_names: list[str],
    speed_up_factor: int,
    num_workers: int,
    min_duration: float,
) -> tuple[dict, int]:
    """Process videos in parallel using multiprocessing.

    Args:
        video_files: List of video file paths.
        known_encodings: Face encodings to match against.
        known_names: Names for the encodings.
        speed_up_factor: Process every N seconds.
        num_workers: Number of parallel workers.
        min_duration: Minimum video duration to process.

    Returns:
        Tuple of (tags dict, skipped count)
    """
    # Filter videos by duration and prepare work items
    work_items = []
    skipped = 0

    print(f"\nChecking video durations...")
    for video_file in sorted(video_files):
        if min_duration > 0:
            duration = get_video_duration(str(video_file))
            if duration < min_duration:
                skipped += 1
                continue
        work_items.append((
            str(video_file),
            known_encodings,
            known_names,
            speed_up_factor,
        ))

    if not work_items:
        return {}, skipped

    print(f"Processing {len(work_items)} videos with {num_workers} workers...")
    print()

    tags = {}
    completed = 0
    total = len(work_items)

    # Use spawn to avoid issues with fork and numpy/opencv
    ctx = mp.get_context("spawn")

    with ctx.Pool(processes=num_workers) as pool:
        for video_name, video_tags, error in pool.imap_unordered(
            process_video_worker, work_items
        ):
            completed += 1

            if error:
                print(f"[{completed}/{total}] {video_name}: Error - {error}")
                tags[video_name] = []
            elif video_tags:
                tag_names = [t["name"] for t in video_tags]
                print(f"[{completed}/{total}] {video_name}: Found {tag_names}")
                tags[video_name] = video_tags
            else:
                print(f"[{completed}/{total}] {video_name}: No faces detected")
                tags[video_name] = []

    return tags, skipped


def process_videos_sequential(
    video_files: list[Path],
    known_encodings: list,
    known_names: list[str],
    speed_up_factor: int,
    min_duration: float,
) -> tuple[dict, int]:
    """Process videos sequentially (single-threaded).

    Args:
        video_files: List of video file paths.
        known_encodings: Face encodings to match against.
        known_names: Names for the encodings.
        speed_up_factor: Process every N seconds.
        min_duration: Minimum video duration to process.

    Returns:
        Tuple of (tags dict, skipped count)
    """
    tags = {}
    skipped = 0
    total = len(video_files)

    for i, video_file in enumerate(sorted(video_files), 1):
        video_name = video_file.stem

        # Check duration
        duration = get_video_duration(str(video_file))
        if duration < min_duration:
            print(f"[{i}/{total}] Skipping {video_name} (too short: {duration:.1f}s)")
            skipped += 1
            continue

        print(f"[{i}/{total}] Processing: {video_name} ({duration:.1f}s)")

        try:
            video_tags = process_single_video(
                video_file,
                known_encodings,
                known_names,
                speed_up_factor=speed_up_factor,
            )
            tags[video_name] = video_tags

            if video_tags:
                tag_names = [t["name"] for t in video_tags]
                print(f"  Found: {tag_names}")
            else:
                print("  No faces detected")

        except Exception as e:
            print(f"  Error: {e}")
            tags[video_name] = []

    return tags, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Generate tags for videos using face recognition"
    )
    parser.add_argument(
        "videos_folder",
        help="Folder containing video files",
    )
    parser.add_argument(
        "output_file",
        help="Output JSON file for tags (format for upsert.py)",
    )
    parser.add_argument(
        "--identities",
        "-i",
        required=True,
        help="JSON file mapping identity names to reference images",
    )
    parser.add_argument(
        "--speed-up",
        "-s",
        type=int,
        default=2,
        help="Process every N seconds of video (default: 2)",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Search for videos recursively",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        help="Minimum video duration in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=0,
        help="Number of parallel workers (default: 1, use 0 for CPU count)",
    )

    args = parser.parse_args()

    videos_path = Path(args.videos_folder)
    if not videos_path.exists():
        print(f"Error: Videos folder does not exist: {videos_path}")
        return

    if not os.path.exists(args.identities):
        print(f"Error: Identities file does not exist: {args.identities}")
        return

    # Load known identities
    print(f"Loading identities from: {args.identities}")
    known_encodings, known_names = load_identities(args.identities)

    if not known_encodings:
        print("Error: No valid face encodings loaded")
        return

    unique_identities = set(known_names)
    print(f"Loaded {len(known_encodings)} face encodings for {len(unique_identities)} identities:")
    for identity in unique_identities:
        count = known_names.count(identity)
        print(f"  - {identity}: {count} reference image(s)")

    # Find all video files
    video_files = []
    for ext in VIDEO_EXTENSIONS:
        if args.recursive:
            video_files.extend(videos_path.rglob(f"*{ext}"))
            video_files.extend(videos_path.rglob(f"*{ext.upper()}"))
        else:
            video_files.extend(videos_path.glob(f"*{ext}"))
            video_files.extend(videos_path.glob(f"*{ext.upper()}"))

    if not video_files:
        print(f"No video files found in {videos_path}")
        return

    print(f"\nFound {len(video_files)} video(s) to process")

    # Determine number of workers
    num_workers = args.workers
    if num_workers == 0:
        num_workers = mp.cpu_count()
    elif num_workers < 0:
        num_workers = max(1, mp.cpu_count() + num_workers)

    # Process videos
    if num_workers > 1:
        print(f"Using {num_workers} parallel workers")
        tags, skipped = process_videos_parallel(
            video_files,
            known_encodings,
            known_names,
            args.speed_up,
            num_workers,
            args.min_duration,
        )
    else:
        tags, skipped = process_videos_sequential(
            video_files,
            known_encodings,
            known_names,
            args.speed_up,
            args.min_duration,
        )

    # Write output
    with open(args.output_file, "w") as f:
        json.dump(tags, f, indent=2)

    # Summary
    total_tags = sum(len(t) for t in tags.values())
    videos_with_tags = sum(1 for t in tags.values() if t)

    print()
    print(f"Generated tags for {len(tags)} videos -> {args.output_file}")
    print(f"  Processed: {len(tags)}")
    print(f"  Skipped (too short): {skipped}")
    print(f"  Videos with faces: {videos_with_tags}")
    print(f"  Total tags: {total_tags}")


if __name__ == "__main__":
    main()
