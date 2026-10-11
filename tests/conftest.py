import pytest
from fastapi.testclient import TestClient

from studio.app import create_app


@pytest.fixture
def studio(tmp_path):
    with TestClient(create_app(tmp_path, run_workers=False)) as client:
        yield client, tmp_path
