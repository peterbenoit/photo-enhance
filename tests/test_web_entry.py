import importlib

import pytest

from photo_enhance import web_entry


def test_web_entry_explains_how_to_install_optional_dependency(monkeypatch, capsys):
    def missing_flask(_name):
        raise ModuleNotFoundError("No module named 'flask'", name="flask")

    monkeypatch.setattr(importlib, "import_module", missing_flask)

    with pytest.raises(SystemExit) as error:
        web_entry.main()

    assert error.value.code == 2
    assert "photo-enhance[web]" in capsys.readouterr().err
