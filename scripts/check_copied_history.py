"""Check that copied generation metadata is rebound to its current artifact root.

This is a read-only diagnostic. Run it from the repository root with
``python scripts/check_copied_history.py <artifact-root>`` after installing the
package or adding ``src`` to ``PYTHONPATH``.
"""

import argparse
import json
import logging
from pathlib import Path

from conductor_core.storage import FilesystemArtifactStore


def check_history(artifact_root: Path) -> bool:
    """Print stored and reconstructed paths, returning whether all entries are usable."""
    root = artifact_root.resolve()
    store = FilesystemArtifactStore(root)
    valid = True

    generation_dirs = sorted(path for path in root.glob("gen_*") if path.is_dir())
    if not generation_dirs:
        print(f"No generation directories found under {root}")
        return False

    print(f"Checking {len(generation_dirs)} generation director(ies) under {root}\n")
    for generation_dir in generation_dirs:
        generation_id = generation_dir.name.removeprefix("gen_")
        metadata_path = generation_dir / "metadata.json"
        print(f"{generation_dir.name}:")

        try:
            raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  FAIL: cannot read metadata.json: {exc}\n")
            valid = False
            continue

        loaded = store.get_generation(generation_id)
        if loaded is None:
            print("  FAIL: FilesystemArtifactStore rejected this entry")
            print(f"  directory ID: {generation_id!r}")
            print(f"  metadata ID:  {raw_metadata.get('id')!r}\n")
            valid = False
            continue

        expected_paths = {
            "midi_path": generation_dir / "loop.mid",
            "audio_path": generation_dir / "loop.mp3",
            "messages_path": generation_dir / "messages.json",
        }
        for field_name, expected_path in expected_paths.items():
            stored_path = raw_metadata.get(field_name)
            loaded_path = getattr(loaded, field_name)
            expected_value = str(expected_path) if field_name == "midi_path" else None
            if field_name != "midi_path" and expected_path.is_file():
                expected_value = str(expected_path)

            matches = loaded_path == expected_value
            exists = expected_path.is_file()
            status = "PASS" if matches and (exists or field_name != "midi_path") else "FAIL"
            print(f"  {field_name}:")
            print(f"    stored: {stored_path!r}")
            print(f"    loaded: {loaded_path!r}")
            print(f"    local file exists: {exists} [{status}]")
            if status == "FAIL":
                valid = False
        print()

    return valid


def main() -> int:
    """Parse the artifact root and return a process-friendly status code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path, help="Directory containing gen_* folders")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if not args.artifact_root.is_dir():
        parser.error(f"artifact root does not exist or is not a directory: {args.artifact_root}")

    return 0 if check_history(args.artifact_root) else 1


if __name__ == "__main__":
    raise SystemExit(main())
