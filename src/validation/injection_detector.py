
import re

INJECTION_PATTERNS = [
    # Instruction override
    r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above|earlier)\s+instructions?",
    r"disregard\s+(all\s+)?(the\s+)?(previous|prior|above|earlier)\s+instructions?",
    r"forget\s+(all\s+)?(your\s+)?(previous|prior|original)\s+instructions?",
    r"override\s+(all\s+)?(your\s+)?instructions?",
    r"do\s+not\s+follow\s+(your\s+)?(previous|original)\s+instructions?",

    # System prompt extraction
    r"(reveal|show|display|print|output|expose)\s+(me\s+)?(your\s+)?system\s+prompt",
    r"(reveal|show|print|output|expose)\s+(your\s+)?hidden\s+instructions",
    r"what\s+are\s+your\s+(system\s+)?instructions",
    r"repeat\s+(the\s+)?(system\s+prompt|instructions)\s+above",

    # Role manipulation
    r"you\s+are\s+now\s+(an?\s+)?(unrestricted|unfiltered|different)\s+",
    r"act\s+as\s+(an?\s+)?(unrestricted|unfiltered|different)\s+",
    r"pretend\s+(that\s+)?you\s+have\s+no\s+restrictions",
    r"switch\s+to\s+(developer|admin|root)\s+mode",
    r"enter\s+(developer|admin|god)\s+mode",

    # Security bypass
    r"bypass\s+(all\s+)?(security|permission|access|safety)\s+(checks|restrictions|controls|rules)?",
    r"disable\s+(your\s+)?(security|safety|permission)\s+(checks|rules|filters)",
    r"ignore\s+(all\s+)?(security|access|permission)\s+(restrictions|controls|rules)",
    r"remove\s+(all\s+)?(restrictions|limitations|safeguards)",

    # Jailbreak attempts
    r"developer\s+mode\s+(enabled|activated)",
    r"do\s+anything\s+now",
    r"dan\s+mode",
    r"no\s+ethical\s+guidelines",
    r"without\s+(any\s+)?restrictions",

    # Database-specific malicious instructions
    r"ignore\s+(the\s+)?database\s+permissions",
    r"bypass\s+(the\s+)?(database|sql)\s+(permissions|restrictions|security)",
    r"access\s+(all\s+)?(restricted|confidential|unauthorized)\s+(data|tables|columns)",
    r"execute\s+(any\s+)?sql\s+(without|regardless\s+of)\s+(permissions|restrictions)",
]


def detect_injection(text: str) -> bool:

    normalized_text = " ".join(text.lower().split())

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, normalized_text, re.IGNORECASE):
            return True

    return False
