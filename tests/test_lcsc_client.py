"""Tests for LCSC stock lookup client."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from kicad_pipeline.production.lcsc_client import (
    LCSCStockInfo,
    _parse_lcsc_response,
    fetch_lcsc_stock,
    fetch_lcsc_stock_batch,
)


def _mock_api_response(
    stock: int = 5000,
    price: float = 0.005,
    desc: str = "10kΩ ±1%",
    package: str = "0805",
) -> dict[str, object]:
    """Build a mock LCSC API response dict."""
    return {
        "result": {
            "stockNumber": stock,
            "productPriceList": [{"productPrice": price, "ladder": 1}],
            "productDescEn": desc,
            "encapStandard": package,
        }
    }


def test_lcsc_stock_info_frozen() -> None:
    info = LCSCStockInfo(
        lcsc="C17414", in_stock=True, stock_qty=5000,
        unit_price_usd=0.005, description="10k", package="0805",
    )
    with pytest.raises(AttributeError):
        info.lcsc = "other"  # type: ignore[misc]


def test_parse_lcsc_response_valid() -> None:
    data = _mock_api_response()
    result = _parse_lcsc_response("C17414", data)
    assert result is not None
    assert result.lcsc == "C17414"
    assert result.in_stock is True
    assert result.stock_qty == 5000
    assert result.unit_price_usd == 0.005
    assert result.package == "0805"


def test_parse_lcsc_response_no_result() -> None:
    result = _parse_lcsc_response("C17414", {})
    assert result is None


def test_parse_lcsc_response_zero_stock() -> None:
    data = _mock_api_response(stock=0)
    result = _parse_lcsc_response("C17414", data)
    assert result is not None
    assert result.in_stock is False
    assert result.stock_qty == 0


def test_parse_lcsc_response_no_price_list() -> None:
    data: dict[str, object] = {
        "result": {
            "stockNumber": 100,
            "productPriceList": [],
            "productDescEn": "part",
            "encapStandard": "0402",
        }
    }
    result = _parse_lcsc_response("C1234", data)
    assert result is not None
    assert result.unit_price_usd is None


def test_fetch_lcsc_stock_invalid_code() -> None:
    assert fetch_lcsc_stock("") is None
    assert fetch_lcsc_stock("NOTLCSC") is None


@patch("kicad_pipeline.production.lcsc_client.urllib.request.urlopen")
def test_fetch_lcsc_stock_success(mock_urlopen: MagicMock) -> None:
    resp_data = _mock_api_response()
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(resp_data).encode("utf-8")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = mock_resp

    result = fetch_lcsc_stock("C17414", timeout=5.0)
    assert result is not None
    assert result.in_stock is True
    assert result.lcsc == "C17414"


@patch("kicad_pipeline.production.lcsc_client.urllib.request.urlopen")
def test_fetch_lcsc_stock_network_error(mock_urlopen: MagicMock) -> None:
    mock_urlopen.side_effect = TimeoutError("timeout")
    result = fetch_lcsc_stock("C17414", timeout=1.0)
    assert result is None


@patch("kicad_pipeline.production.lcsc_client.urllib.request.urlopen")
def test_fetch_lcsc_stock_batch(mock_urlopen: MagicMock) -> None:
    resp_data = _mock_api_response()
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(resp_data).encode("utf-8")
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_urlopen.return_value = mock_resp

    results = fetch_lcsc_stock_batch(("C17414", "C25804"), timeout=5.0)
    assert len(results) == 2
    assert "C17414" in results
    assert "C25804" in results


@patch("kicad_pipeline.production.lcsc_client.urllib.request.urlopen")
def test_fetch_batch_partial_failure(mock_urlopen: MagicMock) -> None:
    call_count = 0

    def side_effect(*args: object, **kwargs: object) -> MagicMock:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            resp = MagicMock()
            resp.read.return_value = json.dumps(_mock_api_response()).encode("utf-8")
            resp.__enter__ = lambda s: s
            resp.__exit__ = MagicMock(return_value=False)
            return resp
        raise TimeoutError("timeout")

    mock_urlopen.side_effect = side_effect
    results = fetch_lcsc_stock_batch(("C17414", "C99999"), timeout=1.0)
    assert len(results) == 1
    assert "C17414" in results
