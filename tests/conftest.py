# Empty conftest for pytest-asyncio configuration
import pytest

def pytest_collection_modifyitems(config, items):
    # Auto-mark async tests
    for item in items:
        if "asyncio" in item.keywords:
            continue
        if item.get_closest_marker("asyncio") is None:
            # pytest-asyncio >= 0.21 auto-detects async, but be explicit
            pass