"""Unit tests for the mock payer FastAPI + TwiML tree.

Uses fastapi.testclient.TestClient — no live server, no network.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mock_payer.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _body(client: TestClient, path: str, **form) -> str:
    response = client.post(path, data=form)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/xml")
    return response.text


# --- /voice ----------------------------------------------------------------


def test_voice_offers_three_options(client: TestClient):
    body = _body(client, "/voice").lower()
    assert "<gather" in body
    assert 'numdigits="1"' in body
    assert 'action="/menu/main"' in body
    assert "press 1 for eligibility" in body
    assert "press 2 for claims" in body
    assert "press 3 for prior authorization" in body


# --- /menu/main ------------------------------------------------------------


def test_menu_main_digit_1_advances_to_member_id(client: TestClient):
    body = _body(client, "/menu/main", Digits="1").lower()
    assert "enter the member id" in body
    assert 'finishonkey="#"' in body
    assert 'action="/auth/dob"' in body


def test_menu_main_digit_2_says_claims_unavailable(client: TestClient):
    body = _body(client, "/menu/main", Digits="2").lower()
    assert "claims is not available" in body
    assert "<hangup" in body


def test_menu_main_digit_3_says_prior_auth_unavailable(client: TestClient):
    body = _body(client, "/menu/main", Digits="3").lower()
    assert "prior authorization is not available" in body
    assert "<hangup" in body


def test_menu_main_unknown_digit_repeats_entry(client: TestClient):
    body = _body(client, "/menu/main", Digits="9").lower()
    assert "thanks for calling acme health" in body
    assert 'action="/menu/main"' in body


def test_menu_main_no_digits_repeats_entry(client: TestClient):
    body = _body(client, "/menu/main").lower()
    assert "thanks for calling acme health" in body


# --- /auth/member_id ------------------------------------------------------


def test_member_id_prompt(client: TestClient):
    body = _body(client, "/auth/member_id").lower()
    assert "enter the member id" in body
    assert 'finishonkey="#"' in body
    assert 'action="/auth/dob"' in body


# --- /auth/dob -----------------------------------------------------------


def test_dob_prompt_carries_member_id_forward(client: TestClient):
    body = _body(client, "/auth/dob", Digits="123456").lower()
    assert "date of birth" in body
    assert 'numdigits="8"' in body
    assert "member_id=123456" in body


def test_dob_prompt_redirect_recovers_member_id_from_query(client: TestClient):
    # No-input timeout: Twilio follows the <Redirect> URL with no Digits in the
    # form body. The handler must fall back to the query string so the next
    # prompt doesn't render with an empty member_id.
    response = client.post("/auth/dob?member_id=123456")
    body = response.text.lower()
    assert "date of birth" in body
    assert "member_id=123456" in body


# --- /auth/confirm -------------------------------------------------------


def test_confirm_echoes_both_values(client: TestClient):
    response = client.post(
        "/auth/confirm?member_id=123456",
        data={"Digits": "05121980"},
    )
    body = response.text.lower()
    assert "you entered 123456 and 05121980" in body
    assert "press 1 to confirm" in body
    assert "press 2 to re-enter" in body
    # Forward both values through to the handler.
    assert "member_id=123456" in body
    assert "dob=05121980" in body


# --- /auth/confirm/handle ------------------------------------------------


def test_confirm_handle_digit_1_advances_to_benefits(client: TestClient):
    response = client.post(
        "/auth/confirm/handle?member_id=123456&dob=05121980",
        data={"Digits": "1"},
    )
    body = response.text.lower()
    assert "connecting you to a representative" in body


def test_confirm_handle_digit_2_returns_to_member_id(client: TestClient):
    response = client.post(
        "/auth/confirm/handle?member_id=123456&dob=05121980",
        data={"Digits": "2"},
    )
    body = response.text.lower()
    assert "enter the member id" in body
    assert 'finishonkey="#"' in body


def test_confirm_handle_unknown_digit_repeats_confirm(client: TestClient):
    response = client.post(
        "/auth/confirm/handle?member_id=123456&dob=05121980",
        data={"Digits": "9"},
    )
    body = response.text.lower()
    assert "you entered 123456 and 05121980" in body


# --- /rep/dialog ---------------------------------------------------------


def test_rep_dialog_full_script(client: TestClient):
    body = _body(client, "/rep/dialog").lower()
    assert "this is sam at acme health" in body
    assert "coverage is active" in body
    assert "deductible remaining is two hundred fifty dollars" in body
    assert "copay is thirty dollars" in body
    assert "coinsurance is twenty percent" in body
    assert "out of network coverage is not included" in body
    assert "<pause" in body
    assert "thank you for calling" in body
    assert "<hangup" in body
