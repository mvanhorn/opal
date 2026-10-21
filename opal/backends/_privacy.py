"""Credential-shaped text is masked before diagnostics become OPAL records."""

import json
import re


_SECRET = re.compile(
    r"(?:sk-ant-|sk-|ghp_|github_pat_|hf_)[A-Za-z0-9_\-]{8,}|"
    r"AKIA[A-Z0-9]{16,}|Bearer\s+[^\s\"'<>\\,;{}\[\]]+",
    re.IGNORECASE,
)


def redact_secrets(text):
    """Preserve diagnostic text and JSON quoting while masking keys and bearer tokens."""
    return _SECRET.sub("<hidden>", str(text or ""))


def redact_record(value):
    """Decode JSON before filtering so escaped whitespace or key prefixes cannot bypass redaction."""
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        return {redact_secrets(k) if isinstance(k, str) else k: redact_record(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_record(v) for v in value]
    return value


def redact_stream(source, destination):
    """Filter JSONL and plain diagnostics a line at a time, including a bearer split over two plain lines."""
    bearer = False
    for line in source:
        try:
            value = json.loads(line)
        except ValueError:
            if bearer and line.strip():
                line = re.sub(r"^\s*\S+", "<hidden>", line, count=1)
                bearer = False
            bearer = bearer or bool(re.search(r"\bBearer\s*$", line, re.IGNORECASE))
            destination.write(redact_secrets(line))
        else:
            bearer = False
            destination.write(json.dumps(redact_record(value)) + "\n")
