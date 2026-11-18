import shutil
from pathlib import Path
from typing import BinaryIO, Protocol

from app.core.config import settings


class Storage(Protocol):
    def save(self, key: str, fileobj: BinaryIO) -> None: ...
    def read(self, key: str) -> bytes: ...
    def delete(self, key: str) -> None: ...


class LocalStorage:
    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root not in path.parents:  # blocks "../" traversal
            raise ValueError("invalid storage key")
        return path

    def save(self, key: str, fileobj: BinaryIO) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as out:
            shutil.copyfileobj(fileobj, out)

    def read(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)


class S3Storage:
    def __init__(self, bucket: str):
        import boto3

        self.bucket = bucket
        self.client = boto3.client("s3")

    def save(self, key: str, fileobj: BinaryIO) -> None:
        self.client.upload_fileobj(fileobj, self.bucket, key)

    def read(self, key: str) -> bytes:
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()

    def delete(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket, Key=key)


_storage: Storage | None = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        if settings.storage_backend == "s3":
            _storage = S3Storage(settings.s3_bucket)
        else:
            _storage = LocalStorage(settings.upload_dir)
    return _storage
