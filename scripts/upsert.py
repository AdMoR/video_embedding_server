#!/usr/bin/env python3
"""Upsert embeddings and tags into the video embedding server.

This script reads pre-computed embeddings (from preprocess.py) and tags
(from build_tags.py or another source) and uploads them via the server's
/upsert endpoint, which handles MinIO storage and Qdrant insertion.

Usage:
    python upsert.py ./embeddings --collection my_project --tags-file tags.json
    python upsert.py ./embeddings --collection my_project --purge  # Purge before upsert
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import requests


def main():
    parser = argparse.ArgumentParser(
        description="Upsert embeddings and tags via the video embedding server"
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
        "--server-url",
        default="http://localhost:8000",
        help="Video embedding server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Purge the collection (Qdrant + MinIO) before upserting",
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

    # Upsert each video via the server
    server_url = args.server_url.rstrip("/")
    upsert_url = f"{server_url}/upsert"
    print(f"Using server: {server_url}")

    # Purge collection if requested
    if args.purge:
        purge_url = f"{server_url}/collections/{args.collection}/purge"
        print(f"Purging collection '{args.collection}'...")
        try:
            response = requests.delete(purge_url)
            if response.status_code == 200:
                result = response.json()
                print(f"  Purged {result['purged_segments']} segments, {result['deleted_files']} files")
            elif response.status_code == 404:
                print(f"  Collection '{args.collection}' does not exist yet, skipping purge")
            else:
                print(f"  Warning: Purge failed: {response.status_code} - {response.text}")
        except requests.RequestException as e:
            print(f"  Warning: Purge request failed: {e}")

    success_count = 0
    error_count = 0

    for video_name, video_info in videos_metadata.items():
        embedding_file = embeddings_path / video_info["embedding_file"]
        source_path = video_info.get("source_path", "")

        if not embedding_file.exists():
            print(f"  Warning: Embedding file not found: {embedding_file}")
            error_count += 1
            continue

        if not source_path or not Path(source_path).exists():
            print(f"  Warning: Source video not found: {source_path}")
            error_count += 1
            continue

        # Load embedding
        embedding = np.load(embedding_file)
        embedding_list = embedding.tolist()

        # Get tags for this video (empty list if not provided)
        video_tags = tags_mapping.get(video_name, [])

        # Prepare multipart form data
        video_path = Path(source_path)
        
        try:
            with open(video_path, "rb") as video_file:
                files = {
                    "file": (video_path.name, video_file, "video/mp4"),
                }
                data = {
                    "collection": args.collection,
                    "embedding": json.dumps(embedding_list),
                    "tags": json.dumps(video_tags),
                }
                # Carry the precomputed caption so it shows up in search results
                caption_text = video_info.get("caption")
                if caption_text:
                    data["caption"] = caption_text

                response = requests.post(upsert_url, files=files, data=data)

            if response.status_code == 200:
                result = response.json()
                tag_count = len(video_tags)
                print(f"  Upserted: {video_name} ({tag_count} tags) -> {result['minio_path']}")
                success_count += 1
            else:
                print(f"  Error uploading {video_name}: {response.status_code} - {response.text}")
                error_count += 1

        except requests.RequestException as e:
            print(f"  Error uploading {video_name}: {e}")
            error_count += 1
        except Exception as e:
            print(f"  Error processing {video_name}: {e}")
            error_count += 1

    # Summary
    print()
    print(f"Successfully upserted: {success_count}")
    print(f"Errors: {error_count}")


if __name__ == "__main__":
    main()
