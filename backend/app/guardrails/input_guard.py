import re
from typing import Tuple, Optional
from pydantic import BaseModel

class InputGuardResult(BaseModel):
    passed: bool
    reason: Optional[str] = None
    sanitized_query: str

class InputGuard:
    """
    Ultra-low latency Pre-Retrieval Guardrail (<1ms).
    Filters out malicious injections, empty inputs, and blatantly off-topic attacks.
    """
    # Common prompt injection patterns
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"disregard\s+(all\s+)?instructions",
        r"system\s*:\s*override",
        r"you\s+are\s+now\s+(DAN|unfiltered|jailbroken)",
        r"reveal\s+your\s+(system\s+prompt|instructions|secret)",
        r"drop\s+database",
        r"<script.*?>.*?</script>"
    ]

    def __init__(self):
        self.compiled_injections = [re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS]

    def validate(self, query: str) -> InputGuardResult:
        sanitized = query.strip()

        # 1. Length checks
        if not sanitized:
            return InputGuardResult(passed=False, reason="Query is empty.", sanitized_query="")
        
        if len(sanitized) < 3:
            return InputGuardResult(passed=False, reason="Query is too short.", sanitized_query=sanitized)

        if len(sanitized) > 1000:
            return InputGuardResult(passed=False, reason="Query exceeds maximum allowed length of 1000 characters.", sanitized_query=sanitized[:1000])

        # 2. Injection & Safety detection
        for pattern in self.compiled_injections:
            if pattern.search(sanitized):
                return InputGuardResult(
                    passed=False,
                    reason="Security violation: Detected potential prompt injection or unauthorized instruction override.",
                    sanitized_query=sanitized
                )

        return InputGuardResult(passed=True, reason=None, sanitized_query=sanitized)

input_guard = InputGuard()
