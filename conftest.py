"""
Root conftest.py — runs before any test collection.

Adds shared/generated (protobuf stubs) to sys.path so that
`import cogniboiler_pb2` works in all test modules.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
GENERATED = REPO_ROOT / "shared" / "generated"

if str(GENERATED) not in sys.path:
    sys.path.insert(0, str(GENERATED))
