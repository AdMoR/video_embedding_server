#!/usr/bin/env python3
"""Preprocess videos: scene-split, filter by duration, and upsert to server.

Pipeline:
  1. Split each source video into scenes with PySceneDetect (parallel, resumable).
  2. Filter out scene chunks longer than --max-duration.
  3. Upsert remaining chunks to the server in parallel (server handles captioning).

Usage:
    python scripts/preprocess.py /data/videos /data/chunks --collection my_col
    python scripts/preprocess.py /data/videos /data/chunks --collection my_col --dry-run
    python scripts/preprocess.py /data/videos /data/chunks --collection my_col --purge --workers 8
"""

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SCENE_PATTERN = re.compile(r"^(.+)-Scene-\d+\.mp4$")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_video_duration(path: Path) -> float:
    """Return video duration in seconds via ffprobe, or 0 on failure."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            return float(result.stdout.strip())
    except Exception:
        pass
    return 0.0


def get_processed_video_names(output_path: Path) -> set[str]:
    """Return stems of source videos that already have scene chunks in output_path."""
    processed = set()
    for f in output_path.iterdir():
        if f.is_file():
            m = SCENE_PATTERN.match(f.name)
            if m:
                processed.add(m.group(1))
    return processed


def sanitize_stem(stem: str) -> str:
    """Replace characters that confuse ffmpeg/scenedetect (dots, spaces, etc.) with underscores."""
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', stem)


def source_stem(chunk_stem: str) -> str:
    """Extract source video stem from a scene chunk stem (strip -Scene-NNN suffix)."""
    m = re.match(r"^(.+)-Scene-\d+$", chunk_stem)
    return m.group(1) if m else chunk_stem


def split_video(
    video_file: Path,
    output_path: Path,
    min_scene_len: float,
) -> tuple[Path, bool, str, str | None]:
    """Run scenedetect on a single video.

    Returns (video_file, ok, message, temp_dir_to_cleanup). The caller is
    responsible for deleting temp_dir_to_cleanup after ALL workers have
    finished so that any ffmpeg child processes scenedetect may have spawned
    can still access the file until they exit.
    """
    safe_stem = sanitize_stem(video_file.stem)
    temp_dir: str | None = None
    try:
        if safe_stem != video_file.stem:
            temp_dir = tempfile.mkdtemp()
            input_path = Path(temp_dir) / f"{safe_stem}{video_file.suffix}"
            shutil.copy2(video_file, input_path)
        else:
            input_path = video_file

        result = subprocess.run(
            [
                "scenedetect",
                "-i", str(input_path),
                "--min-scene-len", str(min_scene_len),
                "--merge-last-scene",
                "split-video",
                "-o", str(output_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            return (video_file, True, "ok", temp_dir)
        error_msg = result.stderr[:500] if result.stderr else "unknown error"
        return (video_file, False, f"scenedetect error: {error_msg}", temp_dir)
    except FileNotFoundError:
        return (video_file, False, "scenedetect not found — install with: pip install scenedetect[opencv]", temp_dir)
    except Exception as e:
        return (video_file, False, str(e), temp_dir)


def check_server(server_url: str) -> bool:
    try:
        r = requests.get(f"{server_url}/health", timeout=10)
        return r.status_code == 200
    except requests.RequestException:
        return False


def purge_collection(server_url: str, collection: str) -> None:
    url = f"{server_url}/collections/{collection}/purge"
    try:
        r = requests.delete(url, timeout=30)
        if r.status_code == 200:
            d = r.json()
            print(f"Purged collection '{collection}': "
                  f"{d['purged_segments']} segments, {d['deleted_files']} files")
        elif r.status_code == 404:
            print(f"Collection '{collection}' does not exist yet — nothing to purge")
        else:
            print(f"Warning: purge returned {r.status_code}: {r.text}")
    except requests.RequestException as e:
        print(f"Warning: purge request failed: {e}")


def upsert_video(
    server_url: str,
    video_path: Path,
    collection: str,
    tags: list[dict],
    timeout: int,
    retries: int,
) -> dict | None:
    """POST a video chunk to /upsert. Returns response JSON on success, None on failure."""
    url = f"{server_url}/upsert"
    data = {"collection": collection, "tags": json.dumps(tags)}
    for attempt in range(1, retries + 1):
        try:
            with open(video_path, "rb") as fh:
                r = requests.post(
                    url,
                    files={"file": (video_path.name, fh, "video/mp4")},
                    data=data,
                    timeout=timeout,
                )
            if r.status_code == 200:
                return r.json()
            log.warning(
                "Attempt %d/%d — server %d for %s: %s",
                attempt, retries, r.status_code, video_path.name, r.text[:200],
            )
        except requests.Timeout:
            log.warning("Attempt %d/%d — timeout for %s", attempt, retries, video_path.name)
        except requests.RequestException as e:
            log.warning("Attempt %d/%d — request error for %s: %s", attempt, retries, video_path.name, e)
        if attempt < retries:
            time.sleep(5 * attempt)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scene-split videos, filter by duration, and upsert to server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_dir", type=Path, help="Source folder of original videos")
    parser.add_argument("output_dir", type=Path, help="Folder where scene chunks are written and kept")
    parser.add_argument("--collection", required=True, help="Qdrant collection name")
    parser.add_argument("--server", default="http://localhost:8004", metavar="URL",
                        help="Server base URL (default: http://localhost:8004)")
    parser.add_argument("--workers", type=int, default=4, metavar="N",
                        help="Parallel workers for scene splitting (default: 4)")
    parser.add_argument("--upsert-workers", type=int, default=4, metavar="N",
                        help="Parallel workers for HTTP upsert (default: 4)")
    parser.add_argument("--min-scene-len", type=float, default=1.0, metavar="SECS",
                        help="Minimum scene length in seconds (default: 1.0)")
    parser.add_argument("--max-duration", type=float, default=20.0, metavar="SECS",
                        help="Drop chunks longer than this many seconds (default: 20.0)")
    parser.add_argument("--tags-file", metavar="FILE",
                        help="JSON file mapping source video stem → list of {name, start, end} tag objects")
    parser.add_argument("--timeout", type=int, default=300, metavar="SECS",
                        help="Per-chunk HTTP timeout in seconds (default: 300)")
    parser.add_argument("--retries", type=int, default=3, metavar="N",
                        help="Retry attempts per chunk on failure (default: 3)")
    parser.add_argument("--purge", action="store_true",
                        help="Purge the collection before upserting")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Print plan without executing")
    parser.add_argument("--pattern", default="**/*.mp4",
                        help="Glob pattern for input videos (default: **/*.mp4)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    if not args.input_dir.is_dir():
        print(f"Error: input directory does not exist: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    server_url = args.server.rstrip("/")

    # -- Load tags ------------------------------------------------------------
    tags_mapping: dict[str, list[dict]] = {}
    if args.tags_file:
        tags_path = Path(args.tags_file)
        if not tags_path.exists():
            print(f"Error: tags file not found: {tags_path}", file=sys.stderr)
            sys.exit(1)
        tags_mapping = json.loads(tags_path.read_text())
        print(f"Loaded tags for {len(tags_mapping)} video(s)")

    # -----------------------------------------------------------------------
    # Step 1 — Scene detection (parallel, resumable)
    # -----------------------------------------------------------------------
    source_videos = sorted(args.input_dir.glob(args.pattern))
    source_videos = [v for v in source_videos if v.suffix.lower() in VIDEO_EXTENSIONS]
    if not source_videos:
        print(f"No video files matching '{args.pattern}' in {args.input_dir}")
        sys.exit(0)

    # Map sanitized stem → original stem for tags lookups after chunking
    sanitized_to_original = {sanitize_stem(v.stem): v.stem for v in source_videos}

    already_split = get_processed_video_names(args.output_dir)
    to_split = [v for v in source_videos if sanitize_stem(v.stem) not in already_split]
    print(f"Found {len(source_videos)} source video(s): "
          f"{len(already_split)} already split, {len(to_split)} to process")

    if to_split and not args.dry_run:
        print(f"Splitting {len(to_split)} video(s) with {args.workers} worker(s) "
              f"(min-scene-len={args.min_scene_len}s) …")
        split_errors: list[str] = []
        temp_dirs_to_cleanup: list[str] = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(split_video, v, args.output_dir, args.min_scene_len): v
                for v in to_split
            }
            for i, future in enumerate(as_completed(futures), 1):
                video_file, ok, msg, temp_dir = future.result()
                if temp_dir:
                    temp_dirs_to_cleanup.append(temp_dir)
                status = "✓" if ok else "✗"
                print(f"  [{i}/{len(to_split)}] {status} {video_file.name}: {msg}")
                if not ok:
                    split_errors.append(video_file.name)
        # All workers finished — safe to remove temp copies now
        for d in temp_dirs_to_cleanup:
            shutil.rmtree(d, ignore_errors=True)
        if split_errors:
            print(f"Warning: {len(split_errors)} video(s) failed to split")

    # -----------------------------------------------------------------------
    # Step 2 — Duration filter
    # -----------------------------------------------------------------------
    all_chunks = sorted(args.output_dir.glob("*-Scene-*.mp4"))
    if not all_chunks:
        print("No scene chunks found in output directory.")
        sys.exit(0)

    print(f"\nChecking duration of {len(all_chunks)} chunk(s) (max {args.max_duration}s) …")
    kept: list[Path] = []
    dropped: list[str] = []
    for chunk in tqdm(all_chunks, unit="chunk", leave=False, disable=args.dry_run):
        dur = get_video_duration(chunk)
        if dur > args.max_duration:
            dropped.append(f"{chunk.name} ({dur:.1f}s)")
        else:
            kept.append(chunk)

    if dropped:
        print(f"Dropped {len(dropped)} chunk(s) over {args.max_duration}s: "
              + ", ".join(dropped[:5]) + (" …" if len(dropped) > 5 else ""))
    print(f"Chunks to upsert: {len(kept)}")

    # -----------------------------------------------------------------------
    # Dry run
    # -----------------------------------------------------------------------
    if args.dry_run:
        print(f"\nDry run — would upsert {len(kept)} chunk(s) to '{args.collection}':")
        for chunk in kept[:20]:
            src = sanitized_to_original.get(source_stem(chunk.stem), source_stem(chunk.stem))
            tags = tags_mapping.get(src, [])
            tag_str = f"  [{', '.join(t['name'] for t in tags)}]" if tags else ""
            print(f"  {chunk.name}{tag_str}")
        if len(kept) > 20:
            print(f"  … and {len(kept) - 20} more")
        sys.exit(0)

    if not kept:
        print("Nothing to upsert.")
        sys.exit(0)

    # -----------------------------------------------------------------------
    # Step 3 — Parallel upsert
    # -----------------------------------------------------------------------
    print(f"\nChecking server at {server_url} …")
    if not check_server(server_url):
        print(f"Error: server not reachable at {server_url}", file=sys.stderr)
        sys.exit(1)
    print("Server is healthy.")

    if args.purge:
        purge_collection(server_url, args.collection)

    print(f"\nUpserting {len(kept)} chunk(s) to collection '{args.collection}' "
          f"with {args.upsert_workers} worker(s) …")

    success_count = 0
    error_count = 0
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=args.upsert_workers) as executor:
        futures = {
            executor.submit(
                upsert_video,
                server_url,
                chunk,
                args.collection,
                tags_mapping.get(sanitized_to_original.get(source_stem(chunk.stem), source_stem(chunk.stem)), []),
                args.timeout,
                args.retries,
            ): chunk
            for chunk in kept
        }
        pbar = tqdm(as_completed(futures), total=len(kept), unit="chunk")
        for future in pbar:
            chunk = futures[future]
            pbar.set_description(chunk.stem[:40])
            result = future.result()
            if result:
                success_count += 1
                log.info("OK  %s → %s", chunk.name, result.get("minio_path", ""))
            else:
                error_count += 1
                errors.append(chunk.name)
                tqdm.write(f"FAILED  {chunk.name}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("Done.")
    print(f"  Upserted: {success_count}")
    print(f"  Failed:   {error_count}")
    if dropped:
        print(f"  Dropped (too long): {len(dropped)}")
    if errors:
        print("\nFailed chunks:")
        for name in errors:
            print(f"  {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
