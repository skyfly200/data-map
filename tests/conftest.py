"""Put the pipeline scripts on sys.path so `pytest tests/` works from the repo root.

The pipeline modules live in `scripts/` and import each other by bare name
(`import species_store as store`), so the directory itself has to be importable
rather than the package. Without this the whole test module fails to collect.
"""
import os
import sys

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts')

if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
