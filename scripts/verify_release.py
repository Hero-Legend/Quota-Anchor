from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "release_manifest.json"
FORBIDDEN_JSON_FIELDS = {"title", "abstract", "text", "document_text"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_jsonl(path: Path) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            bad = FORBIDDEN_JSON_FIELDS.intersection(row)
            if bad:
                raise ValueError(f"{path}:{line_number} contains forbidden fields: {sorted(bad)}")


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = ROOT / item["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != item["sha256"]:
            raise ValueError(f"checksum mismatch for {item['path']}: {actual}")
        if ".jsonl" in path.name:
            audit_jsonl(path)
    print(f"Verified {len(manifest['files'])} dataset files; no article-text fields detected.")


if __name__ == "__main__":
    main()

