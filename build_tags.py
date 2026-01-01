#!/usr/bin/env python3
"""Example script for generating tags. Customize for your use case.

This is a template that developers should adapt to their specific tagging needs.
The output format should be a JSON file compatible with upsert.py.

Output format (tags.json):
{
    "video_001": [
        {"name": "Character_A", "start": 0.0, "end": 30.5},
        {"name": "walking", "start": 10.0, "end": 25.0}
    ],
    "video_002": [
        {"name": "Character_B", "start": 0.0, "end": 60.0}
    ]
}

Usage:
    python build_tags.py ./videos tags.json
"""

import argparse
import json
from pathlib import Path

# Video extensions to look for
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def generate_tags_for_video(video_path: Path) -> list[dict]:
    """Generate tags for a single video.

    This is where you implement your tagging logic. Examples:
    - Parse filename for character/scene info
    - Run object detection or face recognition model
    - Read from external annotation tool exports
    - Query a database or API
    - Use an LLM to describe video content

    Args:
        video_path: Path to the video file.

    Returns:
        List of tag dicts with keys: name, start, end
    """
    video_name = video_path.stem
    tags = []

    # ==========================================================
    # TODO: Implement your tagging logic here
    # ==========================================================

    # Example 1: Parse filename patterns
    # Assumes filenames like "scene01_character_a_walking.mp4"
    name_lower = video_name.lower()

    if "character_a" in name_lower:
        tags.append({"name": "Character_A", "start": 0.0, "end": 0.0})

    if "character_b" in name_lower:
        tags.append({"name": "Character_B", "start": 0.0, "end": 0.0})

    if "walking" in name_lower:
        tags.append({"name": "walking", "start": 0.0, "end": 0.0})

    if "running" in name_lower:
        tags.append({"name": "running", "start": 0.0, "end": 0.0})

    # Example 2: Tag by directory structure
    # parent_dir = video_path.parent.name
    # if parent_dir:
    #     tags.append({"name": f"scene_{parent_dir}", "start": 0.0, "end": 0.0})

    # Example 3: Read from sidecar JSON file
    # sidecar_file = video_path.with_suffix(".json")
    # if sidecar_file.exists():
    #     with open(sidecar_file) as f:
    #         sidecar_data = json.load(f)
    #         tags.extend(sidecar_data.get("tags", []))

    # Example 4: Use video duration (requires mediapy)
    # import mediapy
    # try:
    #     video = mediapy.read_video(str(video_path))
    #     duration = len(video) / 30.0  # assuming 30 fps
    #     tags.append({"name": "has_video", "start": 0.0, "end": duration})
    # except Exception:
    #     pass

    return tags


def main():
    parser = argparse.ArgumentParser(
        description="Generate tags for videos (template - customize for your use case)"
    )
    parser.add_argument(
        "videos_folder",
        help="Folder containing video files",
    )
    parser.add_argument(
        "output_file",
        help="Output JSON file for tags",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Search for videos recursively",
    )

    args = parser.parse_args()

    videos_path = Path(args.videos_folder)
    if not videos_path.exists():
        print(f"Error: Videos folder does not exist: {videos_path}")
        return

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

    print(f"Found {len(video_files)} video(s)")

    # Generate tags for each video
    tags = {}
    for video_file in sorted(video_files):
        video_name = video_file.stem
        video_tags = generate_tags_for_video(video_file)
        tags[video_name] = video_tags

        tag_names = [t["name"] for t in video_tags]
        if tag_names:
            print(f"  {video_name}: {tag_names}")
        else:
            print(f"  {video_name}: (no tags)")

    # Write output
    with open(args.output_file, "w") as f:
        json.dump(tags, f, indent=2)

    # Summary
    total_tags = sum(len(t) for t in tags.values())
    videos_with_tags = sum(1 for t in tags.values() if t)

    print()
    print(f"Generated tags for {len(tags)} videos -> {args.output_file}")
    print(f"  Videos with tags: {videos_with_tags}")
    print(f"  Total tags: {total_tags}")


if __name__ == "__main__":
    main()

