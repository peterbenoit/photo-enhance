"""Minimal local web UI: upload a photo once, then tweak preset/intensity live.

Runs entirely on localhost, no data leaves the machine. Thin wrapper around
the same auto_enhance/apply_preset functions the CLI uses. The decoded image
and its base auto-enhanced version are kept server-side in memory, keyed by
an id handed back to the browser, so changing the preset or intensity re-runs
only the preset step (via fetch) instead of re-uploading the file.
"""

import base64
import os
import threading
import uuid
from collections import OrderedDict

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from photo_enhance.auto_levels import auto_enhance
from photo_enhance.presets import apply_preset_blended, list_preset_choices, load_preset

app = Flask(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20MB
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

MAX_SESSIONS = 20
_sessions: "OrderedDict[str, dict]" = OrderedDict()
_sessions_lock = threading.Lock()


def _store_session(original: np.ndarray, enhanced: np.ndarray) -> str:
    session_id = uuid.uuid4().hex
    with _sessions_lock:
        _sessions[session_id] = {"original": original, "enhanced": enhanced}
        _sessions.move_to_end(session_id)
        while len(_sessions) > MAX_SESSIONS:
            _sessions.popitem(last=False)
    return session_id


def _get_session(session_id: str) -> dict | None:
    with _sessions_lock:
        session = _sessions.get(session_id)
        if session is not None:
            _sessions.move_to_end(session_id)
        return session


def _bgr_to_data_uri(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("Failed to encode image")
    encoded = base64.b64encode(buf).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", presets=list_preset_choices())


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("photo")
    if file is None or file.filename == "":
        return jsonify(error="Choose a photo first."), 400

    file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(error="Could not read that file as an image."), 400

    enhanced = auto_enhance(img)
    session_id = _store_session(img, enhanced)

    return jsonify(
        session_id=session_id,
        before=_bgr_to_data_uri(img),
        after=_bgr_to_data_uri(enhanced),
    )


@app.route("/apply", methods=["POST"])
def apply():
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")
    preset_name = data.get("preset") or None
    intensity = data.get("intensity", 100)

    session = _get_session(session_id) if session_id else None
    if session is None:
        return jsonify(error="Session expired or not found. Please re-upload the photo."), 400

    try:
        intensity_fraction = max(0, min(100, int(intensity))) / 100.0
    except (TypeError, ValueError):
        return jsonify(error="Invalid intensity value."), 400

    enhanced = session["enhanced"]
    if preset_name:
        try:
            preset = load_preset(preset_name)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        result = apply_preset_blended(enhanced, preset, intensity_fraction)
    else:
        result = enhanced

    return jsonify(after=_bgr_to_data_uri(result))


def main():
    debug = os.environ.get("PHOTO_ENHANCE_DEBUG", "").strip().lower() in ("1", "true", "yes")
    app.run(host="127.0.0.1", port=5050, debug=debug)


if __name__ == "__main__":
    main()
