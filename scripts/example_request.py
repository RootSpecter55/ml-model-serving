"""Prints a POST /predict body built from real rows of the bundled dataset.

uv run python -m scripts.example_request 2 > body.json
"""

import json
import sys

from app.modules.training.trainer import load_dataset

count = int(sys.argv[1]) if len(sys.argv) > 1 else 1
features, _, _ = load_dataset()
instances = [{k: float(v) for k, v in row.items()} for _, row in features.head(count).iterrows()]
print(json.dumps({"instances": instances}, indent=2))
