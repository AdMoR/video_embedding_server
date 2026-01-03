#!/usr/bin/env python3
"""Upsert embeddings and tags into Qdrant.

This script combines pre-computed embeddings (from preprocess.py) with tags
(from build_tags.py or another source) and uploads them to Qdrant.

Usage:
    python upsert.py ./embeddings --collection my_project --tags-file tags.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from storage import VectorStore


def main():
    parser = argparse.ArgumentParser(
        description="Upsert embeddings and tags into Qdrant"
    )
    parser.add_argument(
        "embeddings_folder",
        help="Folder containing .npy embedding files and metadata.json",
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--tags-file",
        help="JSON file with tags per video: {'video1': [{'name': 'X', 'start': 0, 'end': 10}]}",
    )
    parser.add_argument(
        "--qdrant-host",
        default="localhost",
        help="Qdrant server hostname (default: localhost)",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Qdrant server port (default: 6333)",
    )

    args = parser.parse_args()

    embeddings_path = Path(args.embeddings_folder)
    metadata_file = embeddings_path / "metadata.json"

    if not embeddings_path.exists():
        print(f"Error: Embeddings folder does not exist: {embeddings_path}")
        sys.exit(1)

    if not metadata_file.exists():
        print(f"Error: metadata.json not found in {embeddings_path}")
        sys.exit(1)

    # Load embeddings metadata
    print(f"Loading embeddings from: {embeddings_path}")
    with open(metadata_file) as f:
        metadata = json.load(f)

    videos_metadata = metadata.get("videos", {})
    if not videos_metadata:
        print("Error: No videos found in metadata.json")
        sys.exit(1)

    print(f"Found {len(videos_metadata)} videos in metadata")

    # Load tags if provided
    tags_mapping = {}
    if args.tags_file:
        tags_path = Path(args.tags_file)
        if not tags_path.exists():
            print(f"Error: Tags file does not exist: {tags_path}")
            sys.exit(1)

        with open(tags_path) as f:
            tags_mapping = json.load(f)
        print(f"Loaded tags for {len(tags_mapping)} videos")

    # Connect to Qdrant
    print(f"Connecting to Qdrant at {args.qdrant_host}:{args.qdrant_port}")
    store = VectorStore(host=args.qdrant_host, port=args.qdrant_port)
    store.ensure_collection(args.collection)
    print(f"Using collection: {args.collection}")

    # Upsert each video
    success_count = 0
    error_count = 0

    for video_name, video_info in videos_metadata.items():
        embedding_file = embeddings_path / video_info["embedding_file"]
        print(embedding_file)
        if not embedding_file.exists():
            print(f"  Warning: Embedding file not found: {embedding_file}")
            error_count += 1
            continue


        # Load embedding
        embedding = np.load(embedding_file)
        print(embedding)

        # Get tags for this video (empty list if not provided)
        video_tags = tags_mapping.get(video_name, [])

        # Get duration if available
        duration = video_info.get("duration", 0.0)

        # Get source path
        source_path = video_info.get("source_path", "")

        # Upsert to Qdrant
        segment_id = f"{video_name}_seg_0"
        store.upsert(
            collection=args.collection,
            segment_id=segment_id,
            embedding=embedding.tolist(),
            video_id=video_name,
            tags=video_tags,
            duration=duration,
            segment_index=0,
            source_path=source_path,
        )

        tag_count = len(video_tags)
        print(f"  Upserted: {video_name} ({tag_count} tags)")
        success_count += 1


    # Summary
    print()
    print(f"Total segments in collection: {store.count(args.collection)}")


if __name__ == "__main__":
    main()

