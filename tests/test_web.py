import io

import cv2
import numpy as np

from photo_enhance.web import app


def _jpeg_bytes() -> bytes:
    img = np.full((16, 16, 3), 130, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_index_lists_all_presets():
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    text = resp.get_data(as_text=True)
    for name in ("Warm Film", "Cool Moody", "High Contrast B&amp;W", "Faded Vintage"):
        assert name in text


def test_upload_without_file_returns_400():
    client = app.test_client()
    resp = client.post("/upload", data={})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_oversized_upload_returns_friendly_json_error():
    client = app.test_client()
    original_limit = app.config["MAX_CONTENT_LENGTH"]
    app.config["MAX_CONTENT_LENGTH"] = 10
    try:
        resp = client.post(
            "/upload",
            data={"photo": (io.BytesIO(_jpeg_bytes()), "photo.jpg")},
            content_type="multipart/form-data",
        )
    finally:
        app.config["MAX_CONTENT_LENGTH"] = original_limit

    assert resp.status_code == 413
    assert "maximum upload size" in resp.get_json()["error"].lower()


def test_upload_rejects_excessive_decoded_pixels(monkeypatch):
    from photo_enhance import web

    monkeypatch.setattr(web, "MAX_DECODED_PIXELS", 100)
    client = app.test_client()
    resp = client.post(
        "/upload",
        data={"photo": (io.BytesIO(_jpeg_bytes()), "photo.jpg")},
        content_type="multipart/form-data",
    )

    assert resp.status_code == 400
    assert "decoded pixels" in resp.get_json()["error"].lower()


def test_upload_returns_session_id_and_images():
    client = app.test_client()
    data = {"photo": (io.BytesIO(_jpeg_bytes()), "photo.jpg")}
    resp = client.post("/upload", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["session_id"]
    assert body["before"].startswith("data:image/jpeg;base64,")
    assert body["after"].startswith("data:image/jpeg;base64,")


def test_upload_enhancement_failure_returns_friendly_error(monkeypatch):
    from photo_enhance import web

    def fail_enhancement(_image):
        raise ValueError("simulated pipeline failure")

    monkeypatch.setattr(web, "auto_enhance", fail_enhancement)
    client = app.test_client()
    resp = client.post(
        "/upload",
        data={"photo": (io.BytesIO(_jpeg_bytes()), "photo.jpg")},
        content_type="multipart/form-data",
    )

    assert resp.status_code == 422
    assert resp.get_json()["error"] == "Could not enhance that image."


def test_upload_encoding_failure_returns_friendly_error(monkeypatch):
    from photo_enhance import web

    def fail_encoding(_image):
        raise ValueError("simulated encoding failure")

    monkeypatch.setattr(web, "_bgr_to_data_uri", fail_encoding)
    client = app.test_client()
    resp = client.post(
        "/upload",
        data={"photo": (io.BytesIO(_jpeg_bytes()), "photo.jpg")},
        content_type="multipart/form-data",
    )

    assert resp.status_code == 500
    assert resp.get_json()["error"] == "Could not encode the enhanced preview."


def test_apply_without_prior_upload_returns_400():
    client = app.test_client()
    resp = client.post("/apply", json={"session_id": "does-not-exist", "preset": "warm_film", "intensity": 100})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_apply_with_valid_session_and_preset_returns_image():
    client = app.test_client()
    data = {"photo": (io.BytesIO(_jpeg_bytes()), "photo.jpg")}
    upload_resp = client.post("/upload", data=data, content_type="multipart/form-data")
    session_id = upload_resp.get_json()["session_id"]

    resp = client.post("/apply", json={"session_id": session_id, "preset": "warm_film", "intensity": 50})
    assert resp.status_code == 200
    assert resp.get_json()["after"].startswith("data:image/jpeg;base64,")


def test_apply_with_unknown_preset_returns_400():
    client = app.test_client()
    data = {"photo": (io.BytesIO(_jpeg_bytes()), "photo.jpg")}
    upload_resp = client.post("/upload", data=data, content_type="multipart/form-data")
    session_id = upload_resp.get_json()["session_id"]

    resp = client.post("/apply", json={"session_id": session_id, "preset": "not_a_real_preset", "intensity": 100})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_apply_without_preset_returns_base_enhanced_image():
    client = app.test_client()
    data = {"photo": (io.BytesIO(_jpeg_bytes()), "photo.jpg")}
    upload_resp = client.post("/upload", data=data, content_type="multipart/form-data")
    session_id = upload_resp.get_json()["session_id"]

    resp = client.post("/apply", json={"session_id": session_id, "preset": "", "intensity": 100})
    assert resp.status_code == 200
    assert resp.get_json()["after"].startswith("data:image/jpeg;base64,")
