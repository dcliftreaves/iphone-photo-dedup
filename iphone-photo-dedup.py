#!/usr/bin/env python3
"""Compatibility entry point. Install with pip install -e . first."""
import sys

from photo_dedup.cli import main

if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments and arguments[0] not in {"scan", "export", "doctor", "ui", "-h", "--help", "--version"}:
        arguments.insert(0, "scan")
    raise SystemExit(main(arguments))
