"""Qdrant vector store abstraction for video embeddings."""
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

EMBEDDING_DIM = 768


class VectorStore:
    """Abstraction layer for Qdrant vector database operations."""

    def __init__(self, host: str = "localhost", port: int = 6333):
        """Initialize connection to Qdrant.

        Args:
            host: Qdrant server hostname.
            port: Qdrant server port.
        """
        self.client = QdrantClient(host=host, port=port)

    def ensure_collection(self, collection: str) -> None:
        """Create collection if it doesn't exist.

        Args:
            collection: Name of the collection to create.
        """
        collections = [c.name for c in self.client.get_collections().collections]
        if collection not in collections:
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM, distance=Distance.COSINE
                ),
            )

    def upsert(
        self,
        collection: str,
        segment_id: str,
        embedding: list[float],
        video_id: str,
        tags: list[dict],
        duration: float = 0.0,
        segment_index: int = 0,
        source_path: str = "",
        caption: str = "",
        scene: str = "",
        events: list[dict] | None = None,
    ) -> None:
        """Insert or update a segment.

        Args:
            collection: Name of the collection.
            segment_id: Unique identifier for the segment.
            embedding: Vector embedding for the segment.
            video_id: ID of the parent video.
            tags: List of tag dicts with keys: name, start, end.
            duration: Total video duration in seconds.
            segment_index: Index of this segment within the video.
            source_path: Path to the source video file.
            caption: Generated caption text for the video.
            scene: Parsed scene paragraph from the caption.
            events: List of dense event dicts with keys: start, end, description.
        """
        tag_names = [t["name"] for t in tags]
        self.client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=uuid.uuid4(),
                    vector=embedding,
                    payload={
                        "video_id": video_id,
                        "segment_index": segment_index,
                        "duration": duration,
                        "source_path": source_path,
                        "tags": tags,
                        "tag_names": tag_names,
                        "caption": caption,
                        "scene": scene,
                        "events": events or [],
                    },
                )
            ],
        )

    def search(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 10,
        tags: list[str] | None = None,
        tag_mode: str = "all",
    ) -> list[dict]:
        """Search for similar segments with optional tag filtering.

        Args:
            collection: Name of the collection to search.
            query_embedding: Query vector embedding.
            top_k: Number of results to return.
            tags: List of tag names to filter by.
            tag_mode: Filter mode - "all" (AND) or "any" (OR).

        Returns:
            List of matching segments with scores and payloads.
        """
        query_filter = None
        if tags:
            conditions = [
                FieldCondition(key="tag_names", match=MatchValue(value=t))
                for t in tags
            ]
            if tag_mode == "all":
                query_filter = Filter(must=conditions)
            else:
                query_filter = Filter(should=conditions)

        results = self.client.query_points(
            collection_name=collection,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
        )
        return [{"id": r.id, "score": r.score, **r.payload} for r in results.points]

    def get_all_tags(self, collection: str) -> list[str]:
        """Get all unique tag names in the collection.

        Args:
            collection: Name of the collection.

        Returns:
            Sorted list of unique tag names.
        """
        tags = set()
        offset = None
        while True:
            results, offset = self.client.scroll(
                collection_name=collection,
                limit=1000,
                offset=offset,
                with_payload=["tag_names"],
            )
            for point in results:
                tags.update(point.payload.get("tag_names", []))
            if offset is None:
                break
        return sorted(tags)

    def list_collections(self) -> list[str]:
        """Get list of all collection names.

        Returns:
            List of collection names.
        """
        return [c.name for c in self.client.get_collections().collections]

    def count(self, collection: str) -> int:
        """Get total number of segments in a collection.

        Args:
            collection: Name of the collection.

        Returns:
            Number of segments.
        """
        return self.client.count(collection_name=collection).count

    def delete_by_video_id(self, collection: str, video_id: str) -> list[str]:
        """Delete all segments for a specific video.

        Args:
            collection: Name of the collection.
            video_id: ID of the video to delete.

        Returns:
            List of source_path values (MinIO URIs) for cleanup.
        """
        # Find all points matching video_id
        results, _ = self.client.scroll(
            collection_name=collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="video_id", match=MatchValue(value=video_id))]
            ),
            limit=1000,
            with_payload=["source_path"],
        )

        if not results:
            return []

        # Collect source paths for MinIO cleanup
        source_paths = [
            p.payload.get("source_path", "")
            for p in results
            if p.payload.get("source_path")
        ]

        # Delete points from Qdrant
        point_ids = [p.id for p in results]
        self.client.delete(
            collection_name=collection,
            points_selector=point_ids,
        )

        return source_paths

    def purge_collection(self, collection: str) -> None:
        """Delete all points in a collection (recreates the collection).

        Args:
            collection: Name of the collection to purge.
        """
        self.client.delete_collection(collection_name=collection)
        self.ensure_collection(collection)

