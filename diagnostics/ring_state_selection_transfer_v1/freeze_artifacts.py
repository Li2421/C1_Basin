"""Hash the completed derived research delivery; never mutates source evidence."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "ARTIFACT_MANIFEST.json"


def main():
    assert json.loads((ROOT / "FINAL_VALIDATION.json").read_text())["status"] == "PASS"
    for name in ("REPORT.md", "HANDOFF.md", "PREREGISTRATION.md", "results.json"):
        assert (ROOT / name).stat().st_size > 0, name
    entries = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == DEST or "__pycache__" in path.parts:
            continue
        if path.suffix in (".tmp", ".pyc"):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        entries.append({"path": str(path.relative_to(ROOT)),
                        "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    manifest = {"utc": datetime.now(timezone.utc).isoformat(),
                "root": str(ROOT), "algorithm": "sha256", "files": entries,
                "excluded": ["ARTIFACT_MANIFEST.json", "__pycache__", "*.pyc", "*.tmp"]}
    DEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"files": len(entries), "bytes": sum(x["bytes"] for x in entries),
                      "manifest": str(DEST)}))


if __name__ == "__main__":
    main()
