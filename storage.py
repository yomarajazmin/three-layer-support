"""Versioned storage for the snapshots each layer writes (gate, scan, analysis).

Snapshots are regenerable caches, so they stay as pickles (DataFrames and nested
results round-trip exactly). What made raw pickles fragile is that a file
written by an older version of a result class loads "successfully" with missing
fields. Every file is therefore wrapped in an envelope carrying a schema
fingerprint - a hash of the field names/types of the classes it contains (and
any extra layout details the caller passes). When the code's fingerprint
differs, or the file can't be unpickled, `load` returns None and the caller
rebuilds instead of crashing or showing half-populated data.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import pickle
from pathlib import Path

from pydantic import BaseModel


def fingerprint(*types: type, extra: object = None) -> str:
    parts = []
    for t in types:
        if dataclasses.is_dataclass(t):
            parts.append([t.__qualname__, [(f.name, str(f.type)) for f in dataclasses.fields(t)]])
        elif isinstance(t, type) and issubclass(t, BaseModel):
            parts.append([t.__qualname__, t.model_json_schema()])
        else:
            raise TypeError(f"fingerprint() needs dataclasses or pydantic models, got {t!r}")
    blob = json.dumps({"types": parts, "extra": extra}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def save(obj: object, path: str, schema: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("wb") as fh:   # write-then-rename so a crash never leaves a truncated file
        pickle.dump({"schema": schema, "saved_at": dt.datetime.now(), "obj": obj}, fh)
    tmp.replace(p)
    return p


def load(path: str, schema: str, expected: type) -> object | None:
    """The stored object, or None if missing, unreadable, from another schema, or the wrong type."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        with p.open("rb") as fh:
            env = pickle.load(fh)
    except Exception:
        return None
    if not isinstance(env, dict) or env.get("schema") != schema or not isinstance(env.get("obj"), expected):
        return None
    return env["obj"]
