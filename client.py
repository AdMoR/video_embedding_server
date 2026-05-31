#!/usr/bin/env python3
"""Client script to query the video embedding similarity server."""

import argparse
import sys

import requests


def search(
    query: str,
    collection: str,
    server_url: str = "http://localhost:8000",
    top_k: int = 5,
    tags: list[str] | None = None,
    tag_mode: str = "all",
) -> dict:
    """Search for videos matching the text query.

    Args:
        query: The text query to search for.
        collection: The Qdrant collection to search in.
        server_url: The base URL of the server.
        top_k: Number of top results to return.
        tags: Optional list of tags to filter by.
        tag_mode: Tag filter mode - "all" (AND) or "any" (OR).

    Returns:
        The search response as a dictionary.
    """
    payload = {
        "query": query,
        "collection": collection,
        "top_k": top_k,
    }
    if tags:
        payload["tags"] = tags
        payload["tag_mode"] = tag_mode

    response = requests.post(f"{server_url}/search", json=payload)
    response.raise_for_status()
    return response.json()


def health_check(server_url: str = "http://localhost:8000") -> dict:
    """Check the server health status.

    Args:
        server_url: The base URL of the server.

    Returns:
        The health status as a dictionary.
    """
    response = requests.get(f"{server_url}/health")
    response.raise_for_status()
    return response.json()


def list_collections(server_url: str = "http://localhost:8000") -> dict:
    """List all available collections.

    Args:
        server_url: The base URL of the server.

    Returns:
        Dictionary with list of collection names.
    """
    response = requests.get(f"{server_url}/collections")
    response.raise_for_status()
    return response.json()


def list_tags(collection: str, server_url: str = "http://localhost:8000") -> dict:
    """List all tags in a collection.

    Args:
        collection: The collection to get tags from.
        server_url: The base URL of the server.

    Returns:
        Dictionary with list of tag names.
    """
    response = requests.get(f"{server_url}/tags/{collection}")
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(
        description="Query the video embedding similarity server"
    )
    parser.add_argument("query", nargs="?", help="Text query to search for")
    parser.add_argument(
        "--collection",
        "-c",
        help="Collection to search in (required for search)",
    )
    parser.add_argument(
        "--server",
        "-s",
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        help="Filter by tags (space-separated)",
    )
    parser.add_argument(
        "--tag-mode",
        choices=["all", "any"],
        default="all",
        help="Tag filter mode: 'all' (AND) or 'any' (OR) (default: all)",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Check server health",
    )
    parser.add_argument(
        "--list-collections",
        action="store_true",
        help="List all collections",
    )
    parser.add_argument(
        "--list-tags",
        action="store_true",
        help="List all tags in a collection (requires --collection)",
    )

    args = parser.parse_args()

    try:
        if args.health:
            result = health_check(args.server)
            print("Server Health:")
            print(f"  Status: {result['status']}")
            print(f"  Model: {result['model']}")
            print(f"  Qdrant: {result['qdrant_host']}")
            print(f"  Collections: {result['collections']}")

        elif args.list_collections:
            result = list_collections(args.server)
            print("Available collections:")
            for coll in result["collections"]:
                print(f"  - {coll}")

        elif args.list_tags:
            if not args.collection:
                print("Error: --collection is required for --list-tags")
                sys.exit(1)
            result = list_tags(args.collection, args.server)
            print(f"Tags in collection '{args.collection}':")
            for tag in result["tags"]:
                print(f"  - {tag}")

        elif args.query:
            if not args.collection:
                print("Error: --collection is required for search")
                sys.exit(1)
            result = search(
                query=args.query,
                collection=args.collection,
                server_url=args.server,
                top_k=args.top_k,
                tags=args.tags,
                tag_mode=args.tag_mode,
            )
            print(f"Search results for: '{args.query}'")
            if args.tags:
                print(f"Filtered by tags ({args.tag_mode}): {args.tags}")
            print("-" * 50)
            for i, r in enumerate(result["results"], 1):
                print(f"{i}. {r['video_id']}")
                print(f"   Segment: {r['segment_id']}")
                print(f"   Path: {r['path']}")
                print(f"   Duration: {r['duration']:.1f}s")
                print(f"   Tags: {[t['name'] for t in r['tags']]}")
                print(f"   Similarity: {r['similarity']:.4f}")
                print()
        else:
            parser.print_help()
            sys.exit(1)

    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to server at {args.server}")
        print("Make sure the server is running (docker compose up)")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"Error: Server returned {e.response.status_code}")
        print(e.response.text)
        sys.exit(1)


if __name__ == "__main__":
    main()
