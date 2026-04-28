import re


def redact_surname(text: str, surname: str) -> str:
    return re.sub(r"\b" + re.escape(surname) + r"\b", "[NAME]", text)


def redact_surnames(text: str, surnames: list[str]) -> str:
    for s in surnames:
        text = redact_surname(text, s)
    return text
