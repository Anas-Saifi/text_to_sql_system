from .input_validator import UserQuery
from .injection_detector import detect_injection


def validate_user_input(question):

    # 1. Basic input validation
    try:
        query = UserQuery(question=question)
    except Exception as e:
        return {
            "valid": False,
            "reason": str(e)
        }

    question = query.question

    # 2. Injection detection
    if detect_injection(question):
        return {
            "valid": False,
            "reason": "Potential prompt injection detected"
        }

    return {
        "valid": True,
        "question": question
    }