import re
from app.observability.langfuse_client import trace_guardrail

INJECTION_PATTERNS = [
    r"(?i)ignore\s+(all\s+)?previous\s+instructions",
    r"(?i)ignore\s+(all\s+)?above\s+instructions",
    r"(?i)disregard\s+(all\s+)?previous",
    r"(?i)forget\s+(all\s+)?previous",
    r"(?i)you\s+are\s+now\s+(a|an)\s+",
    r"(?i)act\s+as\s+(a|an)\s+",
    r"(?i)pretend\s+(you\s+are|to\s+be)\s+",
    r"(?i)new\s+instructions?\s*:",
    r"(?i)system\s*prompt\s*:",
    r"(?i)reveal\s+(your\s+)?(system\s+)?prompt",
    r"(?i)show\s+(me\s+)?(your\s+)?(system\s+)?prompt",
    r"(?i)what\s+(is|are)\s+(your\s+)?(system\s+)?instructions",
    r"(?i)override\s+(safety|content)\s+polic",
    r"(?i)\[INST\]",
    r"(?i)<\|im_start\|>",
    r"(?i)```\s*system",
    r"(?i)jailbreak",
    r"(?i)DAN\s+mode",
]

_compiled = [re.compile(p) for p in INJECTION_PATTERNS]
_blocked_count = 0


def check_prompt_injection(query: str, trace_id: str | None = None) -> tuple[bool, str]:
    global _blocked_count
    for pattern in _compiled:
        if pattern.search(query):
            _blocked_count += 1
            trace_guardrail(trace_id, query="[REDACTED]", blocked=True)
            return True, "Query blocked: potential prompt injection detected."
    trace_guardrail(trace_id, query=query, blocked=False)
    return False, ""


def get_blocked_count() -> int:
    return _blocked_count
