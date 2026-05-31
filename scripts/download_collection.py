"""Download all video segments and captions for a given collection from MinIO/Qdrant."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from caption_formatters import FORMATTERS

import boto3
from botocore.client import Config
from qdrant_client import QdrantClient

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant_server")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "video-embeddings")


def scroll_collection(client, collection, limit=None):
    segments = []
    offset = None
    while True:
        batch = min(1000, limit - len(segments)) if limit else 1000
        results, offset = client.scroll(
            collection_name=collection,
            limit=batch,
            offset=offset,
            with_payload=True,
        )
        segments.extend(results)
        if offset is None or (limit is not None and len(segments) >= limit):
            break
    return segments[:limit] if limit else segments


def parse_minio_uri(uri):
    rest = uri[len("minio://"):]
    bucket, _, key = rest.partition("/")
    return bucket, key


def main():
    parser = argparse.ArgumentParser(
        description="Download video segments and captions for a collection"
    )
    parser.add_argument("collection", help="Qdrant collection name")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Download only the top N segments")
    parser.add_argument("--output", "-o", default=None, help="Output directory (default: ./<collection>)")
    parser.add_argument(
        "--output-format",
        choices=["raw", "ai-toolkit"],
        default="raw",
        help="Output format: 'raw' (captions.json) or 'ai-toolkit' (per-file .txt sidecars)",
    )
    parser.add_argument(
        "--caption-module",
        default="narrative",
        help="Caption formatter for ai-toolkit mode (default: narrative)",
    )
    args = parser.parse_args()

    formatter = None
    if args.output_format == "ai-toolkit":
        if args.caption_module not in FORMATTERS:
            available = ", ".join(sorted(FORMATTERS))
            print(f"Unknown caption module '{args.caption_module}'. Available: {available}", file=sys.stderr)
            sys.exit(1)
        formatter = FORMATTERS[args.caption_module]()

    output_dir = Path(args.output) if args.output else Path(args.collection)
    output_dir.mkdir(parents=True, exist_ok=True)

    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    s3 = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

    collections = [c.name for c in qdrant.get_collections().collections]
    if args.collection not in collections:
        print(f"Collection '{args.collection}' not found. Available: {collections}", file=sys.stderr)
        sys.exit(1)

    limit_msg = f" (limit {args.limit})" if args.limit else ""
    print(f"Fetching segments from '{args.collection}'{limit_msg}...")
    segments = scroll_collection(qdrant, args.collection, limit=args.limit)
    print(f"Found {len(segments)} segments")

    captions = []
    for i, point in enumerate(segments):
        payload = point.payload
        video_id = payload.get("video_id", str(point.id))
        source_path = payload.get("source_path", "")

        captions.append({
            "id": str(point.id),
            "video_id": video_id,
            "segment_index": payload.get("segment_index", 0),
            "caption": payload.get("caption", ""),
            "scene": payload.get("scene", ""),
            "tags": payload.get("tags", []),
            "duration": payload.get("duration", 0.0),
            "source_path": source_path,
        })

        if source_path.startswith("minio://"):
            filename = Path(source_path).name
            stem = Path(filename).stem
        else:
            filename = None
            stem = video_id

        if formatter is not None:
            txt_path = output_dir / f"{stem}.txt"
            try:
                txt_path.write_text(formatter.format(point.payload), encoding="utf-8")
            except Exception as e:
                print(f"  Warning: caption write failed for {stem}: {e}", file=sys.stderr)

        if not source_path.startswith("minio://"):
            print(f"[{i+1}/{len(segments)}] No MinIO path for {video_id}, skipping")
            continue

        dest = output_dir / filename
        if dest.exists():
            print(f"[{i+1}/{len(segments)}] Already exists: {filename}")
            continue

        print(f"[{i+1}/{len(segments)}] Downloading {filename}...")
        try:
            bucket, key = parse_minio_uri(source_path)
            s3.download_file(bucket, key, str(dest))
        except Exception as e:
            print(f"  Error: {e}", file=sys.stderr)

    if args.output_format == "raw":
        captions_path = output_dir / "captions.json"
        with open(captions_path, "w") as f:
            json.dump(captions, f, indent=2)
        print(f"Saved {len(captions)} captions to {captions_path}")
    else:
        print(f"Wrote {len(segments)} caption files to {output_dir}")


if __name__ == "__main__":
    main()
