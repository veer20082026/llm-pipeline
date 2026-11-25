"""
Test Vercel serverless function entrypoint (api/index.py).
"""

from io import BytesIO
import json
from unittest.mock import MagicMock, patch
from api.index import app, handler


def test_api_entrypoint_exports_app_and_handler():
    assert app is not None
    assert handler is not None
    assert app == handler


def test_api_entrypoint_get_health():
    mock_request = MagicMock()
    mock_request.makefile.return_value = BytesIO(b"")

    h = handler(mock_request, ("127.0.0.1", 8080), None)  # type: ignore
    assert hasattr(h, "do_GET")
    assert hasattr(h, "do_POST")
