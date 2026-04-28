"""Rule-based IVR keyword classifier — the deterministic ivr_nav decision tree.

Given an IVR transcript (and optionally the caller's PatientInfo), pick exactly
one action: send DTMF, transition the graph to a different node, or report
unknown so the outer state machine can fall back. Zero LLM, zero I/O.

The four prompt shapes the classifier recognizes (checked in this order; first
match wins):

1. Confirmations    — "press 1 to confirm" → DTMF 1
2. Transitions      — "connecting you to a representative" → transition rep_wait
                      "your benefits are..."               → transition ivr_extract
3. Data-entry       — "enter the member ID followed by #"  → DTMF from patient
4. Menu prompts     — "press 1 for eligibility, ..."       → DTMF for eligibility
"""

from __future__ import annotations

import re
from collections.abc import Callable
from itertools import chain

from agent.schemas import ClassifierResult, PatientInfo

_ELIGIBILITY_KEYWORDS = ("eligibility", "benefits")

# "press N to confirm" / "press N if (this is) correct"
_CONFIRM_RE = re.compile(
    r"press\s+(\d)\s+(?:to\s+confirm|if\s+(?:this\s+is\s+)?correct)",
    re.IGNORECASE,
)

# Handoff to a human rep.
_HANDOFF_RE = re.compile(
    r"\b(?:connecting\s+you|transferring\s+you|transfer\s+you|please\s+hold)\b",
    re.IGNORECASE,
)

# Payer about to read the benefits stream.
_BENEFITS_STREAM_RE = re.compile(
    r"\b(?:your\s+benefits\s+are|here\s+(?:are|is)\s+your\s+benefits)\b",
    re.IGNORECASE,
)

_MEMBER_ID_RE = re.compile(r"\b(?:enter|input|key\s+in).{0,40}\bmember\s*id\b", re.IGNORECASE)
_DOB_RE = re.compile(
    r"\b(?:enter|input|key\s+in).{0,40}\b(?:date\s+of\s+birth|dob)\b",
    re.IGNORECASE,
)
_POUND_SUFFIX_RE = re.compile(r"\bfollowed\s+by\s+(?:the\s+)?pound\b", re.IGNORECASE)

# Forward direction: "press 1 for eligibility" / "enter 2 for claims".
_MENU_FORWARD_RE = re.compile(
    r"(?:press|enter)\s+(\d)\s+(?:for|to)\s+([^.,\n]+?)(?=[.,\n]|\s+(?:press|enter)\s+\d|$)",
    re.IGNORECASE,
)
# Reverse direction: "for eligibility, press 1" / "if you are calling about eligibility, press 1".
_MENU_REVERSE_RE = re.compile(
    r"(?:for|about)\s+([^.,\n]+?),?\s+(?:press|enter)\s+(\d)",
    re.IGNORECASE,
)


def _matches_eligibility(label: str) -> bool:
    label = label.lower()
    return any(kw in label for kw in _ELIGIBILITY_KEYWORDS)


class RuleBasedClassifier:
    def classify(
        self,
        transcript: str,
        patient: PatientInfo | None = None,
    ) -> ClassifierResult:
        if not transcript:
            return ClassifierResult(outcome="unknown")

        if m := _CONFIRM_RE.search(transcript):
            return ClassifierResult(outcome="dtmf", dtmf=m.group(1), confidence=0.95)

        if _HANDOFF_RE.search(transcript):
            return ClassifierResult(outcome="transition", transition_to="rep_wait", confidence=0.9)
        if _BENEFITS_STREAM_RE.search(transcript):
            return ClassifierResult(
                outcome="transition", transition_to="ivr_extract", confidence=0.9
            )

        if _MEMBER_ID_RE.search(transcript):
            return self._data_entry(patient, _member_id_digits, transcript)
        if _DOB_RE.search(transcript):
            return self._data_entry(patient, _dob_digits, transcript)

        return self._menu(transcript)

    @staticmethod
    def _data_entry(
        patient: PatientInfo | None,
        to_digits: Callable[[PatientInfo], str | None],
        transcript: str,
    ) -> ClassifierResult:
        if patient is None:
            return ClassifierResult(outcome="unknown")
        digits = to_digits(patient)
        if not digits:
            return ClassifierResult(outcome="unknown")
        if _POUND_SUFFIX_RE.search(transcript):
            digits += "#"
        return ClassifierResult(outcome="dtmf", dtmf=digits, confidence=0.9)

    @staticmethod
    def _menu(transcript: str) -> ClassifierResult:
        forward = ((m.group(1), m.group(2)) for m in _MENU_FORWARD_RE.finditer(transcript))
        reverse = ((m.group(2), m.group(1)) for m in _MENU_REVERSE_RE.finditer(transcript))
        for digit, label in chain(forward, reverse):
            if _matches_eligibility(label):
                return ClassifierResult(outcome="dtmf", dtmf=digit, confidence=0.85)
        return ClassifierResult(outcome="unknown")


def _dob_digits(patient: PatientInfo) -> str:
    # ISO YYYY-MM-DD → MMDDYYYY DTMF.
    year, month, day = patient.dob.split("-")
    return f"{month}{day}{year}"


def _member_id_digits(patient: PatientInfo) -> str | None:
    # Member IDs may be alphanumeric (e.g. "M123456"); DTMF only carries 0-9*#,
    # so we send the numeric portion. Alpha-only IDs surface as unknown.
    digits = re.sub(r"\D", "", patient.member_id)
    return digits or None
