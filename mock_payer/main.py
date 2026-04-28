"""Mock payer FastAPI app.

Simulates an Aetna-style insurance IVR using plain TwiML. Twilio webhooks
POST form-encoded data; we route on the `Digits` field. Captured values
forward through query-string parameters on each Gather `action` URL —
the server stays stateless across requests.
"""

from __future__ import annotations

from fastapi import FastAPI, Form, Response

from mock_payer import ivr_tree

app = FastAPI()

# Twilio sends form fields with capitalized names ("Digits"); we alias to keep
# Python params snake-case while matching the wire format.
_DigitsForm = Form(default="", alias="Digits")


def _twiml(response: ivr_tree.VoiceResponse) -> Response:
    return Response(content=str(response), media_type="application/xml")


@app.post("/voice")
async def voice() -> Response:
    return _twiml(ivr_tree.entry())


@app.post("/menu/main")
async def menu_main(digits: str = _DigitsForm) -> Response:
    if digits == "1":
        return _twiml(ivr_tree.member_id_prompt())
    if digits == "2":
        return _twiml(ivr_tree.menu_unavailable("Claims"))
    if digits == "3":
        return _twiml(ivr_tree.menu_unavailable("Prior authorization"))
    return _twiml(ivr_tree.entry())


@app.post("/auth/member_id")
async def auth_member_id() -> Response:
    return _twiml(ivr_tree.member_id_prompt())


@app.post("/auth/dob")
async def auth_dob(digits: str = _DigitsForm) -> Response:
    return _twiml(ivr_tree.dob_prompt(member_id=digits))


@app.post("/auth/confirm")
async def auth_confirm(member_id: str = "", digits: str = _DigitsForm) -> Response:
    return _twiml(ivr_tree.confirm_prompt(member_id=member_id, dob=digits))


@app.post("/auth/confirm/handle")
async def auth_confirm_handle(
    member_id: str = "",
    dob: str = "",
    digits: str = _DigitsForm,
) -> Response:
    if digits == "1":
        return _twiml(ivr_tree.benefits_intro())
    if digits == "2":
        return _twiml(ivr_tree.member_id_prompt())
    return _twiml(ivr_tree.confirm_prompt(member_id=member_id, dob=dob))


@app.post("/rep/dialog")
async def rep_dialog() -> Response:
    return _twiml(ivr_tree.rep_dialog())
