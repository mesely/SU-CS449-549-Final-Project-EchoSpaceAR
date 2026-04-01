"""Tiny `.env` loader used by the pipeline scripts."""

from __future__ import annotations

import os


def load_env(path: str = ".env") -> None:
    """Populate `os.environ` with values from a simple KEY=VALUE file."""
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
