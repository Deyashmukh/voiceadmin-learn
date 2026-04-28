"""Unit tests for the rule-based ivr_nav classifier."""

from __future__ import annotations

from agent.classifier import RuleBasedClassifier
from agent.schemas import PatientInfo


def _classify(transcript: str, patient: PatientInfo | None = None):
    return RuleBasedClassifier().classify(transcript, patient)


# --- Shape (a): menu prompts ------------------------------------------------


def test_menu_forward_eligibility_at_position_1():
    r = _classify("Press 1 for eligibility, press 2 for claims, press 3 for prior auth.")
    assert r.outcome == "dtmf"
    assert r.dtmf == "1"


def test_menu_forward_eligibility_at_position_2():
    r = _classify("Press 1 for claims, press 2 for eligibility, press 3 for prior auth.")
    assert r.outcome == "dtmf"
    assert r.dtmf == "2"


def test_menu_reverse_phrasing():
    r = _classify("If you are calling about eligibility, press 1.")
    assert r.outcome == "dtmf"
    assert r.dtmf == "1"


def test_menu_benefits_keyword_also_matches():
    r = _classify("For benefits and verification, press 4.")
    assert r.outcome == "dtmf"
    assert r.dtmf == "4"


def test_menu_no_eligibility_option_is_unknown():
    r = _classify("Press 1 for claims, press 2 for prior auth.")
    assert r.outcome == "unknown"


# --- Shape (b): data-entry prompts -----------------------------------------


def test_data_entry_member_id(patient: PatientInfo):
    r = _classify("Please enter the member ID followed by the pound sign.", patient)
    assert r.outcome == "dtmf"
    # patient.member_id == "M123456" → numeric portion only, plus pound suffix.
    assert r.dtmf == "123456#"


def test_data_entry_member_id_no_pound(patient: PatientInfo):
    r = _classify("Please enter your member ID.", patient)
    assert r.outcome == "dtmf"
    assert r.dtmf == "123456"


def test_data_entry_dob_iso_to_mmddyyyy(patient: PatientInfo):
    r = _classify("Please enter the patient's date of birth as eight digits.", patient)
    assert r.outcome == "dtmf"
    # patient.dob == "1980-05-12" → MMDDYYYY.
    assert r.dtmf == "05121980"


def test_data_entry_dob_abbreviation(patient: PatientInfo):
    r = _classify("Enter the DOB now.", patient)
    assert r.outcome == "dtmf"
    assert r.dtmf == "05121980"


def test_data_entry_without_patient_is_unknown():
    r = _classify("Please enter the member ID followed by the pound sign.")
    assert r.outcome == "unknown"


def test_data_entry_member_id_with_no_digits_is_unknown():
    alpha_only = PatientInfo(
        member_id="ABCDEF",
        first_name="Alice",
        last_name="Example",
        dob="1980-05-12",
    )
    r = _classify("Please enter your member ID.", alpha_only)
    assert r.outcome == "unknown"


# --- Shape (c): transitions ------------------------------------------------


def test_transition_handoff_to_rep_wait():
    r = _classify("Connecting you to a representative.")
    assert r.outcome == "transition"
    assert r.transition_to == "rep_wait"


def test_transition_please_hold_to_rep_wait():
    r = _classify("Please hold while I transfer you.")
    assert r.outcome == "transition"
    assert r.transition_to == "rep_wait"


def test_transition_benefits_stream_to_ivr_extract():
    r = _classify("Your benefits are as follows.")
    assert r.outcome == "transition"
    assert r.transition_to == "ivr_extract"


def test_transition_here_are_your_benefits():
    r = _classify("Here are your benefits for the policy year.")
    assert r.outcome == "transition"
    assert r.transition_to == "ivr_extract"


# --- Shape (d): confirmations ----------------------------------------------


def test_confirmation_press_1_to_confirm():
    r = _classify("You entered 123. Press 1 to confirm, press 2 to re-enter.")
    assert r.outcome == "dtmf"
    assert r.dtmf == "1"


def test_confirmation_press_1_if_correct():
    r = _classify("Press 1 if this is correct.")
    assert r.outcome == "dtmf"
    assert r.dtmf == "1"


# --- Unknown / edge cases --------------------------------------------------


def test_unknown_random_sentence():
    r = _classify("The weather today is rather pleasant.")
    assert r.outcome == "unknown"


def test_unknown_empty_transcript():
    r = _classify("")
    assert r.outcome == "unknown"
