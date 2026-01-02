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
"""

import argparse
import json
import os
import subprocess
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


def process_video(
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
        default=1.0,
        help="Minimum video duration in seconds (default: 1.0)",
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

    # Process each video
    tags = {}
    skipped = 0

    for i, video_file in enumerate(sorted(video_files), 1):
        video_name = video_file.stem

        # Check duration
        duration = get_video_duration(str(video_file))
        if duration < args.min_duration:
            print(f"[{i}/{len(video_files)}] Skipping {video_name} (too short: {duration:.1f}s)")
            skipped += 1
            continue

        print(f"[{i}/{len(video_files)}] Processing: {video_name} ({duration:.1f}s)")

        try:
            video_tags = process_video(
                video_file,
                known_encodings,
                known_names,
                speed_up_factor=args.speed_up,
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
