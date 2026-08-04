"""Marks every test under integration/ as requiring network access.

Doing this here rather than decorating each test means new files in this directory are covered automatically. Note that
pytest_collection_modifyitems is a global hook even when defined in a subdirectory conftest, so items must be filtered
by path.
"""

from pathlib import Path

import pytest

_INTEGRATION_DIR = Path(__file__).parent


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.path is not None and item.path.is_relative_to(_INTEGRATION_DIR):
            item.add_marker(pytest.mark.integration)
