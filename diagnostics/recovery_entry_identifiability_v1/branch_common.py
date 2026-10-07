"""Shared, dependency-light helpers for the recovery-entry branch audit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_semantic_hash(payload: Mapping[str, Any], path: Path) -> None:
    claimed = payload.get("content_sha256")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    if not isinstance(claimed, str) or canonical_hash(body) != claimed:
        raise RuntimeError((str(path), "semantic content hash mismatch"))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True, allow_nan=False)
                if isinstance(value, (list, tuple, dict)) else value
                for key, value in row.items()
            })
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


__all__ = [
    "atomic_csv", "atomic_json", "atomic_text", "canonical_hash", "file_hash",
    "read_csv", "verify_semantic_hash",
]
