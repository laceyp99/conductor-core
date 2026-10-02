"""Check a consumer against the built wheel without editable-source resolution.

Run with the locked development environment after ``uv build``. The temporary
consumer environment installs the bare wheel, not provider or typing extras.
No generation functions or live services are called.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    wheel = args.wheel.resolve(strict=True)
    repo = Path(__file__).resolve().parents[1]
    ty = shutil.which("ty")
    if ty is None:
        raise SystemExit("Run through uv run --locked --all-extras to provide Ty")

    with tempfile.TemporaryDirectory(prefix="conductor-wheel-typing-") as directory:
        root = Path(directory)
        environment = root / "venv"
        subprocess.run(
            ["uv", "venv", "--python", sys.executable, str(environment)], check=True
        )
        python = environment / (
            "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
        )
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), str(wheel)], check=True
        )
        sample = root / "consumer.py"
        shutil.copyfile(repo / "tests/typing/consumer.py", sample)
        config = root / "ty.toml"
        config.write_text(
            '[environment]\npython-version = "3.10"\nroot = ["."]\n'
            "[terminal]\nerror-on-warning = false\n",
            encoding="utf-8",
        )
        # Running outside the repository prevents fallback to src/ or .venv.
        command = [
            ty,
            "check",
            "--project",
            str(root),
            "--python",
            str(python),
            "--config-file",
            str(config),
        ]
        subprocess.run([*command, str(sample)], cwd=root, check=True)
        subprocess.run([str(python), "-I", str(sample)], cwd=root, check=True)

        # Ensure this checker actually rejects a wrong exported request type.
        invalid = root / "invalid_consumer.py"
        invalid.write_text(
            "from conductor_core import GenerationRequest\n"
            'GenerationRequest(key=123, scale="major", description="melody", model="gpt-6-sol")\n',
            encoding="utf-8",
        )
        result = subprocess.run(
            [*command, "--output-format", "concise", str(invalid)],
            cwd=root,
            capture_output=True,
            text=True,
        )
        if (
            result.returncode != 1
            or "invalid-argument-type" not in result.stdout + result.stderr
        ):
            raise SystemExit("Ty failed to reject the invalid installed-wheel consumer")
        print("Installed-wheel typing and py.typed smoke checks passed")


if __name__ == "__main__":
    main()
