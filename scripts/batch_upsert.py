#!/usr/bin/env python3
"""Batch upload a folder of videos to the server's /upsert endpoint.

The server captions each video with Marlin-2B on its own GPU — this script just
feeds videos to it one at a time and tracks progress.  It is the simplest path:
no local GPU or model download required.  For large sets (hundreds of videos) the
offline pipeline (preprocess.py → upsert.py) is more efficient because it keeps a
resumable state file and skips the network round-trip per video.

Usage examples:
    python batch_upsert.py /data/videos --collection my_videos
    python batch_upsert.py /data/videos --collection my_videos --tags-file tags.json
    python batch_upsert.py /data/videos --collection my_videos --resume
    python batch_upsert.py /data/videos --collection my_videos --purge --max-duration 30
    python batch_upsert.py /data/videos --collection my_videos --server http://gpu-box:8000
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
STATE_FILENAME = ".batch_upsert_state.json"

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


def load_state(video_dir: Path) -> set[str]:
    """Load the set of already-processed video stems from the state file."""
    state_file = video_dir / STATE_FILENAME
    if state_file.exists():
        try:
            data = json.loads(state_file.read_text())
            return set(data.get("done", []))
        except Exception:
            pass
    return set()


def save_state(video_dir: Path, done: set[str]) -> None:
    """Persist the set of processed video stems."""
    state_file = video_dir / STATE_FILENAME
    state_file.write_text(json.dumps({"done": sorted(done)}, indent=2))


def check_server(server_url: str) -> bool:
    """Return True if the server is reachable and healthy."""
    try:
        r = requests.get(f"{server_url}/health", timeout=10)
        return r.status_code == 200
    except requests.RequestException:
        return False


def purge_collection(server_url: str, collection: str) -> None:
    """Purge a collection (Qdrant + MinIO), or skip if it doesn't exist yet."""
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
    """POST a video to /upsert.  Returns the response JSON on success, None on failure."""
    url = f"{server_url}/upsert"
    data = {
        "collection": collection,
        "tags": json.dumps(tags),
    }

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
                "Attempt %d/%d — server returned %d for %s: %s",
                attempt, retries, r.status_code, video_path.name, r.text[:200],
            )
        except requests.Timeout:
            log.warning(
                "Attempt %d/%d — timeout after %ds for %s",
                attempt, retries, timeout, video_path.name,
            )
        except requests.RequestException as e:
            log.warning("Attempt %d/%d — request error for %s: %s", attempt, retries, video_path.name, e)

        if attempt < retries:
            time.sleep(5 * attempt)  # back off before retry

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch upload videos to the caption search server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video_dir", help="Folder containing video files")
    parser.add_argument("--collection", required=True, help="Qdrant collection name")
    parser.add_argument(
        "--server",
        default="http://localhost:8004",
        metavar="URL",
        help="Server base URL (default: http://localhost:8004)",
    )
    parser.add_argument(
        "--tags-file",
        metavar="FILE",
        help="JSON file mapping video stem → list of {name, start, end} tag objects",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        metavar="SECS",
        help="Skip videos longer than this many seconds (uses ffprobe)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        metavar="SECS",
        help="Per-video HTTP timeout in seconds (default: 300). "
             "Increase for long videos — captioning takes time.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        metavar="N",
        help="Number of retry attempts per video on failure (default: 3)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip videos that were successfully uploaded in a previous run "
             f"(tracked in {STATE_FILENAME} inside the video folder)",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Purge the collection (Qdrant + MinIO) before uploading",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List videos that would be uploaded without sending anything",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    video_dir = Path(args.video_dir)
    if not video_dir.is_dir():
        print(f"Error: '{video_dir}' is not a directory")
        sys.exit(1)

    server_url = args.server.rstrip("/")

    # -- Discover videos ------------------------------------------------------
    videos = sorted(
        p for p in video_dir.iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        print(f"No video files found in {video_dir}")
        sys.exit(0)

    print(f"Found {len(videos)} video file(s) in {video_dir}")

    # -- Load tags ------------------------------------------------------------
    tags_mapping: dict[str, list[dict]] = {}
    if args.tags_file:
        tags_path = Path(args.tags_file)
        if not tags_path.exists():
            print(f"Error: tags file not found: {tags_path}")
            sys.exit(1)
        tags_mapping = json.loads(tags_path.read_text())
        print(f"Loaded tags for {len(tags_mapping)} video(s)")

    # -- Duration filter ------------------------------------------------------
    skipped_duration: list[str] = []
    if args.max_duration is not None:
        print(f"Checking durations (max {args.max_duration}s) …")
        filtered: list[Path] = []
        for v in tqdm(videos, unit="video", leave=False):
            dur = get_video_duration(v)
            if dur > args.max_duration:
                skipped_duration.append(v.name)
            else:
                filtered.append(v)
        videos = filtered
        if skipped_duration:
            print(f"Skipped {len(skipped_duration)} video(s) over {args.max_duration}s: "
                  + ", ".join(skipped_duration[:5])
                  + (" …" if len(skipped_duration) > 5 else ""))

    # -- Resume filter --------------------------------------------------------
    done: set[str] = set()
    if args.resume:
        done = load_state(video_dir)
        before = len(videos)
        videos = [v for v in videos if v.stem not in done]
        skipped_resume = before - len(videos)
        if skipped_resume:
            print(f"Resuming: skipping {skipped_resume} already-processed video(s)")

    if not videos:
        print("Nothing to upload.")
        sys.exit(0)

    # -- Dry run --------------------------------------------------------------
    if args.dry_run:
        print(f"\nDry run — would upload {len(videos)} video(s) to '{args.collection}':")
        for v in videos:
            tags = tags_mapping.get(v.stem, [])
            tag_str = f"  [{', '.join(t['name'] for t in tags)}]" if tags else ""
            print(f"  {v.name}{tag_str}")
        sys.exit(0)

    # -- Server check ---------------------------------------------------------
    print(f"Checking server at {server_url} …")
    if not check_server(server_url):
        print(f"Error: server is not reachable at {server_url}")
        sys.exit(1)
    print("Server is healthy.")

    # -- Purge ----------------------------------------------------------------
    if args.purge:
        purge_collection(server_url, args.collection)

    # -- Upload ---------------------------------------------------------------
    print(f"\nUploading {len(videos)} video(s) to collection '{args.collection}' …")
    print(f"  timeout={args.timeout}s  retries={args.retries}")
    print()

    success_count = 0
    error_count = 0
    errors: list[str] = []

    pbar = tqdm(videos, unit="video")
    for video_path in pbar:
        pbar.set_description(video_path.stem[:40])

        tags = tags_mapping.get(video_path.stem, [])
        result = upsert_video(
            server_url, video_path, args.collection, tags, args.timeout, args.retries
        )

        if result:
            success_count += 1
            done.add(video_path.stem)
            if args.resume:
                save_state(video_dir, done)
            log.info("OK  %s → %s", video_path.name, result.get("minio_path", ""))
        else:
            error_count += 1
            errors.append(video_path.name)
            tqdm.write(f"FAILED  {video_path.name}")

    pbar.close()

    # -- Summary --------------------------------------------------------------
    print()
    print(f"Done.")
    print(f"  Uploaded:  {success_count}")
    print(f"  Failed:    {error_count}")
    if skipped_duration:
        print(f"  Too long:  {len(skipped_duration)}")
    if args.resume and done:
        print(f"  (state saved to {video_dir / STATE_FILENAME})")
    if errors:
        print(f"\nFailed videos:")
        for name in errors:
            print(f"  {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
