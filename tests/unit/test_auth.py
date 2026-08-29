from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.api.auth import require_api_key
from src.config import settings

_app = FastAPI()


@_app.get("/protected", dependencies=[Depends(require_api_key)])
def _protected():
    return {"ok": True}


client = TestClient(_app)


def test_missing_key_is_401():
    assert client.get("/protected").status_code == 401


def test_wrong_key_is_401():
    assert client.get("/protected", headers={"X-API-Key": "nope"}).status_code == 401


def test_correct_key_is_200():
    resp = client.get("/protected", headers={"X-API-Key": settings.api_key})
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
