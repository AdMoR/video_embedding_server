#!/usr/bin/env python3
"""Client script to query the video embedding similarity server."""

import argparse
import sys

import requests


def search(
    query: str,
    server_url: str = "http://localhost:8000",
    top_k: int = 5,
    prompt_template: str = "a video of {}.",
) -> dict:
    """Search for videos matching the text query.

    Args:
        query: The text query to search for.
        server_url: The base URL of the server.
        top_k: Number of top results to return.
        prompt_template: Template to wrap the query (use {} as placeholder).

    Returns:
        The search response as a dictionary.
    """
    response = requests.post(
        f"{server_url}/search",
        json={
            "query": query,
            "top_k": top_k,
            "prompt_template": prompt_template,
        },
    )
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


def main():
    parser = argparse.ArgumentParser(
        description="Query the video embedding similarity server"
    )
    parser.add_argument("query", nargs="?", help="Text query to search for")
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
        "--template",
        "-t",
        default="a video of {}.",
        help="Prompt template (default: 'a video of {}.')",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Check server health instead of searching",
    )

    args = parser.parse_args()

    try:
        if args.health:
            result = health_check(args.server)
            print("Server Health:")
            print(f"  Status: {result['status']}")
            print(f"  Model: {result['model']}")
            print(f"  Videos loaded: {result['videos_loaded']}")
        elif args.query:
            result = search(
                query=args.query,
                server_url=args.server,
                top_k=args.top_k,
                prompt_template=args.template,
            )
            print(f"Search results for: '{args.query}'")
            print("-" * 50)
            for i, r in enumerate(result["results"], 1):
                print(f"{i}. {r['name']}")
                print(f"   Video ID: {r['video_id']}")
                print(f"   Path: {r['path']}")
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

