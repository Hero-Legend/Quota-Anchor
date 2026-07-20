from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = DATA / "release_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_terms(relative: str) -> str:
    if relative.startswith("data/civic/"):
        return "CIViC CC0 1.0; PubMed IDs only outside the frozen CIViC source TSV"
    if relative.startswith("data/universes/"):
        return "Identifier inventory; upstream source terms apply"
    if relative.startswith("data/historical_firewall/"):
        return "Derived CTD/PheKnowLator benchmark annotations; upstream terms apply"
    return "Derived CTD benchmark annotations; CTD custom non-commercial terms apply"


def main() -> None:
    files = []
    for path in sorted(DATA.rglob("*")):
        if not path.is_file() or path in {OUTPUT, DATA / "README.md"}:
            continue
        relative = path.relative_to(ROOT).as_posix()
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "source_terms": source_terms(relative),
            }
        )
    payload = {
        "release": "TempProv-Audit v1.0.0",
        "generated": "2026-07-20",
        "article_text_redistributed": False,
        "experimental_results_included": False,
        "files": files,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} with {len(files)} files")


if __name__ == "__main__":
    main()

