"""
conftest.py
-----------
Ensures the project root (containing the `src` package) is on sys.path
regardless of the working directory pytest is invoked from.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
