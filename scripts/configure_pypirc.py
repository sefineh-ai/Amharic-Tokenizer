#!/usr/bin/env python3
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def main():
    test_token = os.getenv("TESTPYPI_TOKEN", "").strip()
    pypi_token = os.getenv("PYPI_TOKEN", "").strip()

    if not test_token and not pypi_token:
        raise SystemExit("No tokens provided. Set TESTPYPI_TOKEN and/or PYPI_TOKEN in environment.")

    pypirc = Path.home() / ".pypirc"
    lines = []
    lines.append("[distutils]\n")
    lines.append("index-servers =\n")
    if test_token:
        lines.append("    testpypi\n")
    if pypi_token:
        lines.append("    pypi\n")
    lines.append("\n")

    if test_token:
        lines.append("[testpypi]\n")
        lines.append("repository = https://test.pypi.org/legacy/\n")
        lines.append("username = __token__\n")
        lines.append(f"password = {test_token}\n\n")

    if pypi_token:
        lines.append("[pypi]\n")
        lines.append("repository = https://upload.pypi.org/legacy/\n")
        lines.append("username = __token__\n")
        lines.append(f"password = {pypi_token}\n")

    pypirc.write_text("".join(lines), encoding="utf-8")
    os.chmod(pypirc, 0o600)
    print(f"Wrote {pypirc}")

if __name__ == "__main__":
    main()
