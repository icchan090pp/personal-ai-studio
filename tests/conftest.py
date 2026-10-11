import pytest
from fastapi.testclient import TestClient

from studio.app import create_app
from studio.hardware import inspect_system


@pytest.fixture
def studio(tmp_path):
    inspect_system.cache_clear()
    try:
        with TestClient(create_app(tmp_path, run_workers=False)) as client:
            yield client, tmp_path
    finally:
        inspect_system.cache_clear()
