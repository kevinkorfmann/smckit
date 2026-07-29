from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml


source = Path(sys.argv[1])
target = Path(sys.argv[2])
raw = source.read_bytes()
payload = yaml.safe_load(raw)
payload["config_sha256"] = hashlib.sha256(raw).hexdigest()
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
