"""PyInstaller entry point for the bundled engine."""
from photo_dedup.worker import main

if __name__ == '__main__':
    raise SystemExit(main())
