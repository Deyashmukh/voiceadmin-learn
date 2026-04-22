"""Rule-based IVR keyword classifier.

M4 ships only the "always speak" stub — enough to exercise the pipeline end-to-end.
The real keyword tree lands at M5 alongside the mock payer.
"""

from __future__ import annotations

from agent.schemas import ClassifierResult


class RuleBasedClassifier:
    def classify(self, transcript: str) -> ClassifierResult:
        return ClassifierResult(outcome="speak", confidence=1.0)
