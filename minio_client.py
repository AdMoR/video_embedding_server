"""MinIO Storage Client for Video Embedding Server.

This module implements an async MinIO (S3-compatible) storage client
using boto3 with a custom endpoint URL.
"""

import asyncio
import os
from functools import partial
from pathlib import Path
from typing import Optional

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError


class MinIOClient:
    """
    Async storage client using MinIO (S3-compatible object storage).

    Configuration via environment variables:
    - MINIO_ENDPOINT: MinIO server URL (default: http://minio:9000)
    - MINIO_ACCESS_KEY: Access key (default: minioadmin)
    - MINIO_SECRET_KEY: Secret key (default: minioadmin)
    - MINIO_BUCKET: Bucket name (default: video-embeddings)
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        bucket: Optional[str] = None,
    ):
        """
        Initialize MinIO client.

        Args:
            endpoint: MinIO server URL (overrides env var)
            access_key: Access key (overrides env var)
            secret_key: Secret key (overrides env var)
            bucket: Bucket name (overrides env var)
        """
        self.endpoint = endpoint or os.environ.get("MINIO_ENDPOINT", "http://minio:9000")
        self.access_key = access_key or os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
        self.secret_key = secret_key or os.environ.get("MINIO_SECRET_KEY", "minioadmin")
        self.bucket = bucket or os.environ.get("MINIO_BUCKET", "video-embeddings")

        # Create boto3 S3 client with MinIO endpoint
        self._client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",  # MinIO doesn't care, but boto3 needs it
        )

        self._bucket_ensured = False
        print(f"Initialized MinIOClient: endpoint={self.endpoint}, bucket={self.bucket}")

    async def _ensure_bucket(self) -> None:
        """Create bucket if it doesn't exist."""
        if self._bucket_ensured:
            return

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                partial(self._client.head_bucket, Bucket=self.bucket)
            )
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code")
            if error_code in ("404", "NoSuchBucket"):
                await loop.run_in_executor(
                    None,
                    partial(self._client.create_bucket, Bucket=self.bucket)
                )
                print(f"Created bucket: {self.bucket}")
            else:
                raise

        self._bucket_ensured = True

    async def upload_file(self, local_path: str, key: str) -> str:
        """
        Upload a local file to MinIO storage.

        Args:
            local_path: Path to the local file to upload
            key: Object key (path) in MinIO

        Returns:
            The MinIO URI: minio://{bucket}/{key}
        """
        await self._ensure_bucket()

        loop = asyncio.get_event_loop()

        # Determine content type based on file extension
        content_type = self._guess_content_type(local_path)

        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        await loop.run_in_executor(
            None,
            partial(
                self._client.upload_file,
                local_path,
                self.bucket,
                key,
                ExtraArgs=extra_args if extra_args else None,
            )
        )

        minio_uri = f"minio://{self.bucket}/{key}"
        print(f"Uploaded: {local_path} -> {minio_uri}")
        return minio_uri

    async def put_object(self, key: str, data: bytes, content_type: Optional[str] = None) -> str:
        """
        Store binary data with the given key.

        Args:
            key: Object key (path) in MinIO
            data: Binary data to store
            content_type: Optional MIME type

        Returns:
            The MinIO URI: minio://{bucket}/{key}
        """
        await self._ensure_bucket()

        loop = asyncio.get_event_loop()
        kwargs = {
            "Bucket": self.bucket,
            "Key": key,
            "Body": data,
        }
        if content_type:
            kwargs["ContentType"] = content_type

        await loop.run_in_executor(
            None,
            partial(self._client.put_object, **kwargs)
        )

        minio_uri = f"minio://{self.bucket}/{key}"
        print(f"Stored object: {minio_uri}")
        return minio_uri

    async def object_exists(self, key: str) -> bool:
        """
        Check if an object exists.

        Args:
            key: Object key to check

        Returns:
            True if object exists, False otherwise
        """
        await self._ensure_bucket()

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                partial(self._client.head_object, Bucket=self.bucket, Key=key)
            )
            return True
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                return False
            raise

    def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """
        Generate a presigned URL for accessing an object.

        Args:
            key: Object key
            expires_in: URL expiration time in seconds (default: 1 hour)

        Returns:
            Presigned URL string
        """
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    async def delete_object(self, key: str) -> None:
        """
        Delete a single object from MinIO.

        Args:
            key: Object key to delete
        """
        await self._ensure_bucket()

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                partial(self._client.delete_object, Bucket=self.bucket, Key=key)
            )
            print(f"Deleted object: {key}")
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") != "NoSuchKey":
                raise

    async def delete_prefix(self, prefix: str) -> int:
        """
        Delete all objects under a prefix.

        Args:
            prefix: Prefix to match (e.g., "videos/my_collection/")

        Returns:
            Number of objects deleted
        """
        await self._ensure_bucket()

        loop = asyncio.get_event_loop()

        # List all objects with prefix
        def _list_objects():
            keys = []
            paginator = self._client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        keys.append(obj["Key"])
            return keys

        keys = await loop.run_in_executor(None, _list_objects)

        if not keys:
            return 0

        # Delete objects in batches of 1000 (S3 limit)
        deleted_count = 0
        for i in range(0, len(keys), 1000):
            batch = keys[i:i + 1000]
            delete_request = {
                "Objects": [{"Key": k} for k in batch],
                "Quiet": True,
            }
            await loop.run_in_executor(
                None,
                partial(
                    self._client.delete_objects,
                    Bucket=self.bucket,
                    Delete=delete_request,
                )
            )
            deleted_count += len(batch)

        print(f"Deleted {deleted_count} objects with prefix: {prefix}")
        return deleted_count

    @staticmethod
    def _guess_content_type(filepath: str) -> Optional[str]:
        """Guess content type based on file extension."""
        ext = Path(filepath).suffix.lower()
        content_types = {
            ".mp4": "video/mp4",
            ".avi": "video/x-msvideo",
            ".mov": "video/quicktime",
            ".mkv": "video/x-matroska",
            ".webm": "video/webm",
            ".json": "application/json",
            ".npy": "application/octet-stream",
        }
        return content_types.get(ext)


# Global instance
_minio_client: Optional[MinIOClient] = None


def get_minio_client() -> MinIOClient:
    """
    Get or create the global MinIO client instance.

    Returns:
        MinIOClient instance
    """
    global _minio_client

    if _minio_client is None:
        _minio_client = MinIOClient()

    return _minio_client


def reset_minio_client() -> None:
    """Reset the global MinIO client instance (useful for testing)."""
    global _minio_client
    _minio_client = None

