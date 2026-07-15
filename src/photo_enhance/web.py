"""Minimal local web UI: upload a photo, pick a preset, see before/after.

Runs entirely on localhost, no data leaves the machine. Thin wrapper around
the same auto_enhance/apply_preset functions the CLI uses.
"""

import base64

import cv2
import numpy as np
from flask import Flask, render_template, request

from photo_enhance.auto_levels import auto_enhance
from photo_enhance.presets import apply_preset, list_presets, load_preset

app = Flask(__name__)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20MB
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def _bgr_to_data_uri(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("Failed to encode image")
    encoded = base64.b64encode(buf).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", presets=list_presets(), result=None, error=None)


@app.route("/enhance", methods=["POST"])
def enhance():
    presets = list_presets()
    file = request.files.get("photo")
    preset_name = request.form.get("preset") or None

    if file is None or file.filename == "":
        return render_template("index.html", presets=presets, result=None,
                                error="Choose a photo first.")

    file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return render_template("index.html", presets=presets, result=None,
                                error="Could not read that file as an image.")

    result = auto_enhance(img)
    if preset_name:
        result = apply_preset(result, load_preset(preset_name))

    return render_template(
        "index.html",
        presets=presets,
        result={
            "before": _bgr_to_data_uri(img),
            "after": _bgr_to_data_uri(result),
            "preset": preset_name,
        },
        error=None,
    )


def main():
    app.run(debug=True, port=5050)


if __name__ == "__main__":
    main()
