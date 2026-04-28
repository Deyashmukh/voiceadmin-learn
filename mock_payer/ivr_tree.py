"""Pure TwiML builders for the mock payer IVR tree.

Each function corresponds to one Twilio webhook step. Twilio re-fetches the
webhook on every <Gather>, so the IVR is encoded as separate routes — no
in-process state. Captured digits flow forward through query-string
parameters on the Gather `action` URL.

The Gather classes are constructed directly and appended (rather than
`response.gather(...).say(...)`) because the Twilio library types
`response.gather()` as the parent `TwiML | str` union, hiding `.say()`.
"""

from __future__ import annotations

from twilio.twiml.voice_response import Gather, VoiceResponse


def entry() -> VoiceResponse:
    """Top of the menu."""
    response = VoiceResponse()
    gather = Gather(num_digits=1, action="/menu/main", method="POST")
    gather.say(
        "Thanks for calling Acme Health. "
        "Press 1 for eligibility and benefits, "
        "press 2 for claims, "
        "press 3 for prior authorization."
    )
    response.append(gather)
    # Reprompt if no digit arrives.
    response.redirect("/voice", method="POST")
    return response


def menu_unavailable(label: str) -> VoiceResponse:
    response = VoiceResponse()
    response.say(f"{label} is not available in this demo.")
    response.hangup()
    return response


def member_id_prompt() -> VoiceResponse:
    response = VoiceResponse()
    gather = Gather(finish_on_key="#", action="/auth/dob", method="POST")
    gather.say("Please enter the member ID followed by the pound sign.")
    response.append(gather)
    response.redirect("/auth/member_id", method="POST")
    return response


def dob_prompt(member_id: str) -> VoiceResponse:
    response = VoiceResponse()
    action = f"/auth/confirm?member_id={member_id}"
    gather = Gather(num_digits=8, action=action, method="POST")
    gather.say(
        "Please enter the patient's date of birth as eight digits, "
        "month month day day year year year year."
    )
    response.append(gather)
    response.redirect(f"/auth/dob?member_id={member_id}", method="POST")
    return response


def confirm_prompt(member_id: str, dob: str) -> VoiceResponse:
    response = VoiceResponse()
    action = f"/auth/confirm/handle?member_id={member_id}&dob={dob}"
    gather = Gather(num_digits=1, action=action, method="POST")
    gather.say(f"You entered {member_id} and {dob}. Press 1 to confirm, press 2 to re-enter.")
    response.append(gather)
    response.redirect(action, method="POST")
    return response


def benefits_intro() -> VoiceResponse:
    response = VoiceResponse()
    response.say("Connecting you to a representative.")
    response.redirect("/rep/dialog", method="POST")
    return response


def rep_dialog() -> VoiceResponse:
    """Scripted rep dialog. M5 keeps this deterministic; M7c may upgrade to LLM."""
    response = VoiceResponse()
    response.say(
        "Hi, this is Sam at Acme Health. "
        "The patient's coverage is active. "
        "Their deductible remaining is two hundred fifty dollars. "
        "Their copay is thirty dollars. "
        "Their coinsurance is twenty percent. "
        "Out of network coverage is not included. "
        "Is there anything else?"
    )
    response.pause(length=2)
    response.say("Thank you for calling. Goodbye.")
    response.hangup()
    return response
