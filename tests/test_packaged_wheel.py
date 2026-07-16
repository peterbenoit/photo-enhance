import os
import shutil
import subprocess
from pathlib import Path


def _run(command: list[str], *, cwd: Path, env: dict[str, str] | None = None):
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_built_wheel_runs_both_console_entry_points(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    uv = shutil.which("uv")
    assert uv is not None, "uv is required to build and smoke-test the wheel"

    dist_dir = tmp_path / "dist"
    _run(
        [uv, "build", "--wheel", "--out-dir", str(dist_dir)],
        cwd=project_root,
    )
    wheels = list(dist_dir.glob("photo_enhance-*.whl"))
    assert len(wheels) == 1

    venv_dir = tmp_path / "venv"
    _run([uv, "venv", str(venv_dir)], cwd=tmp_path)
    python = venv_dir / "bin" / "python"
    _run(
        [uv, "pip", "install", "--python", str(python), f"{wheels[0]}[web]"],
        cwd=tmp_path,
    )

    clean_env = os.environ.copy()
    clean_env.pop("PYTHONPATH", None)
    clean_env["PYTHONNOUSERSITE"] = "1"

    cli_result = _run(
        [str(venv_dir / "bin" / "photo-enhance"), "--help"],
        cwd=tmp_path,
        env=clean_env,
    )
    assert "Auto-enhance a photo" in cli_result.stdout
    assert "--preset" in cli_result.stdout

    patch_dir = tmp_path / "startup-patch"
    patch_dir.mkdir()
    (patch_dir / "sitecustomize.py").write_text(
        """from flask import Flask


def smoke_run(self, *, host, port, debug):
    response = self.test_client().get('/')
    assert response.status_code == 200
    assert b'Photo Auto-Enhance' in response.data
    assert host == '127.0.0.1'
    assert port == 5050
    assert debug is False
    print('WEB_ENTRY_POINT_OK')


Flask.run = smoke_run
"""
    )
    web_env = clean_env | {"PYTHONPATH": str(patch_dir), "PHOTO_ENHANCE_DEBUG": ""}
    web_result = _run(
        [str(venv_dir / "bin" / "photo-enhance-web")],
        cwd=tmp_path,
        env=web_env,
    )
    assert "WEB_ENTRY_POINT_OK" in web_result.stdout
