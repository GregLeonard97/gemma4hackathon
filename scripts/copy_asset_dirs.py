#!/usr/bin/env python3
"""Copy table PNGs and guideline PDFs into iOS Resources."""

from __future__ import annotations

import shutil
from pathlib import Path


RESOURCES = Path("GuidelineAssistant/GuidelineAssistant/Resources")

SRC_TABLES = Path("data/extracted_tables")
DST_TABLES = RESOURCES / "extracted_tables"

SRC_GUIDELINES = Path("data/guidelines")
DST_GUIDELINES = RESOURCES / "Guidelines"


def _dir_file_count(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file())


def _dir_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _copy_dir(src: Path, dst: Path, ignore=None) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing source directory: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)


def main() -> None:
    RESOURCES.mkdir(parents=True, exist_ok=True)

    _copy_dir(SRC_TABLES, DST_TABLES)
    _copy_dir(
        SRC_GUIDELINES,
        DST_GUIDELINES,
        ignore=shutil.ignore_patterns("*.gitkeep", ".DS_Store"),
    )

    tables_count = _dir_file_count(DST_TABLES)
    tables_size_mb = _dir_size_bytes(DST_TABLES) / (1024 * 1024)
    guidelines_count = _dir_file_count(DST_GUIDELINES)
    guidelines_size_mb = _dir_size_bytes(DST_GUIDELINES) / (1024 * 1024)

    print("Asset copy complete")
    print(f"  extracted_tables files: {tables_count}")
    print(f"  extracted_tables size: {tables_size_mb:.2f} MB")
    print(f"  Guidelines files: {guidelines_count}")
    print(f"  Guidelines size: {guidelines_size_mb:.2f} MB")


if __name__ == "__main__":
    main()
