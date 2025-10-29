#!/usr/bin/env python3
"""Generate a .pypirc file with TestPyPI and/or PyPI tokens from environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def main():
    """Create or update the ~/.pypirc file with credentials from environment variables."""
    test_token = os.getenv("TESTPYPI_TOKEN", "").strip()
    pypi_token = os.getenv("PYPI_TOKEN", "").strip()

    if not test_token and not pypi_token:
        raise SystemExit(
            "No tokens provided. Set TESTPYPI_TOKEN and/or PYPI_TOKEN in environment."
        )

    pypirc_path = Path.home() / ".pypirc"
    lines = ["[distutils]\n", "index-servers =\n"]

    if test_token:
        lines.append("    testpypi\n")
    if pypi_token:
        lines.append("    pypi\n")
    lines.append("\n")

    if test_token:
        lines.extend(
            [
                "[testpypi]\n",
                "repository = https://test.pypi.org/legacy/\n",
                "username = __token__\n",
                f"password = {test_token}\n\n",
            ]
        )

    if pypi_token:
        lines.extend(
            [
                "[pypi]\n",
                "repository = https://upload.pypi.org/legacy/\n",
                "username = __token__\n",
                f"password = {pypi_token}\n",
            ]
        )

    pypirc_path.write_text("".join(lines), encoding="utf-8")
    os.chmod(pypirc_path, 0o600)
    print(f"Wrote {pypirc_path}")


if __name__ == "__main__":
    main()
