from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import requests


URL = "https://civicdb.org/api/graphql"
QUERY = """
query {
  dataReleases {
    name
    acceptedEvidenceTsv { filename path }
    acceptedAndSubmittedEvidenceTsv { filename path }
  }
}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--release", default="nightly")
    args = parser.parse_args()
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)

    response = requests.post(URL, json={"query": QUERY}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if payload.get("errors"):
        raise RuntimeError(payload["errors"])
    releases = payload["data"]["dataReleases"]
    (output / "release_index.json").write_text(
        json.dumps(releases, indent=2) + "\n", encoding="utf-8"
    )
    matches = [item for item in releases if item["name"] == args.release]
    if not matches:
        raise ValueError(f"Unknown release {args.release!r}; available: {[item['name'] for item in releases]}")
    release = matches[0]
    file_info = release["acceptedEvidenceTsv"] or release["acceptedAndSubmittedEvidenceTsv"]
    if file_info is None:
        raise ValueError(f"Release {release['name']} does not expose an evidence TSV.")
    path = file_info["path"]
    download_url = path if path.startswith("http") else f"https://civicdb.org{path}"
    target = output / file_info["filename"]
    with requests.get(download_url, stream=True, timeout=120) as download:
        download.raise_for_status()
        with target.open("wb") as handle:
            for chunk in download.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = {
        "api": URL,
        "release": release["name"],
        "download_url": download_url,
        "filename": target.name,
        "bytes": target.stat().st_size,
        "sha256": digest,
        "license": "CC0 1.0 Universal",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
