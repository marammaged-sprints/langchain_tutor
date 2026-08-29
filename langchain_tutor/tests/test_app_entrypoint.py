from pathlib import Path

from streamlit.testing.v1 import AppTest

from langchain_tutor.app import main


def test_package_app_exposes_main():
    assert callable(main)


def test_root_streamlit_entrypoint_starts_without_error():
    entrypoint = Path(__file__).parents[2] / "streamlit_app.py"

    app = AppTest.from_file(entrypoint).run(timeout=30)

    assert not app.exception
