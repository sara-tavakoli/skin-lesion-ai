from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

fastapi_testclient = pytest.importorskip("fastapi.testclient")


@pytest.fixture
def client(monkeypatch, tmp_path):
    """API client with the heavy predictor/explainer swapped for fakes."""
    from skinlesion.serve import api

    class _FakePred:
        eval_tf = staticmethod(lambda image: {"image": np.zeros((3, 8, 8), dtype=np.float32)})
        device = "cpu"
        model = object()

        def predict(self, arr, mc_dropout_samples=0):
            from skinlesion.serve.inference import Prediction

            probs = dict.fromkeys(["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"], 1 / 7)
            return Prediction(
                label="nv",
                label_name="Melanocytic nevus",
                probabilities=probs,
                malignant_probability=3 / 7,
                entropy=1.9,
                warnings=[],
            )

    api.get_predictor.cache_clear()
    fake = _FakePred()
    api.app.dependency_overrides[api.get_predictor] = lambda: fake
    try:
        yield fastapi_testclient.TestClient(api.app)
    finally:
        api.app.dependency_overrides.clear()


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.fromarray((np.random.rand(32, 32, 3) * 255).astype("uint8")).save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_model_card_lists_all_classes(client):
    r = client.get("/model-card")
    assert r.status_code == 200
    assert len(r.json()["classes"]) == 7
    assert "disclaimer" in r.json()


def test_predict_returns_probabilities(client):
    r = client.post("/predict", files={"file": ("x.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert set(body["probabilities"]) == {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
    assert "disclaimer" in body


def test_predict_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400
