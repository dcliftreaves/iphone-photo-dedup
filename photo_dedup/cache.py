"""A versioned cache of inspection data, never of matching decisions."""

import json
import sqlite3
from pathlib import Path

VERSION = "1"


class Cache:
    def __init__(self, path: Path | None, engine: str):
        self.engine = engine
        self.hits = 0
        self.pending = 0
        self.db = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(path)
            self.db.execute("CREATE TABLE IF NOT EXISTS inspections "
                            "(path TEXT PRIMARY KEY, stamp TEXT, payload TEXT)")

    def stamp(self, stat):
        return json.dumps([VERSION, self.engine, stat.st_dev, stat.st_ino,
                           stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns])

    def get(self, path, stat):
        if self.db is None:
            return None
        row = self.db.execute("SELECT payload FROM inspections WHERE path=? AND stamp=?",
                              (str(path), self.stamp(stat))).fetchone()
        if row:
            self.hits += 1
            return json.loads(row[0])
        return None

    def put(self, path, stat, payload):
        if self.db is not None:
            self.db.execute("INSERT OR REPLACE INTO inspections VALUES (?, ?, ?)",
                            (str(path), self.stamp(stat), json.dumps(payload)))
            self.pending += 1
            if self.pending >= 128:
                self.db.commit()
                self.pending = 0

    def close(self):
        if self.db is not None:
            self.db.commit()
            self.db.close()
