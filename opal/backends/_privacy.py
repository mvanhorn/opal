"""Credential-shaped text is masked before diagnostics become OPAL records."""

import json
import re


# Prefixes are case sensitive and must start a token, not a suffix of a model or function name.
_SECRET = re.compile(
    r"(?<![A-Za-z0-9_])(?:sk-ant-|sk-|ghp_|github_pat_)[A-Za-z0-9_\-]{16,}|"
    r"(?<![A-Za-z0-9_])hf_[A-Za-z0-9]{16,}(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])AKIA[A-Z0-9]{16}(?![A-Za-z0-9_])|"
    r"(?i:\bBearer)\s+[^\s\"'<>\\,;{}\[\]&]+|"
    r"(?i:\baccess_token)=([^\s&\"'<>]+)",
)
_ESCAPE = re.compile(r'\\(?:u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|x[0-9a-fA-F]{2}|[\\/"\'abfnrtv])|%[0-9a-fA-F]{2}')
_STRING_JOIN = re.compile(r"""(?:\"{3}|'{3}|(?<!')'(?!'')|(?<!\")\"(?!\"\"))\s*(?:[bBuUrR]{0,2})(?:\"{3}|'{3}|'(?!')|\"(?!\"))""")
_ESCAPE = re.compile(_ESCAPE.pattern + "|" + _STRING_JOIN.pattern + r"|(?i:(?<=Bearer))\+")
_SIMPLE = {'a': '\a', 'b': '\b', 'f': '\f', 'n': '\n', 'r': '\r', 't': '\t', 'v': '\v'}
_DECODE_DEPTH = 8
_DECODE_LENGTH = 65_536


def _decoded_spans(text):
    """Find secrets in bounded decoded views and map them back to spans in the original text.

    JSON strings and exception reprs can add several layers of backslashes. Decode escapes only for
    matching, never rewrite the diagnostic itself. Windows bound both the index map and decode work;
    overlap retains credentials straddling a window. Direct matching also covers arbitrarily long keys.
    """
    spans = [(m.start(), m.end()) for m in _SECRET.finditer(text)]
    overlap = 4096
    for offset in range(0, len(text), _DECODE_LENGTH - overlap):
        decoded = text[offset:offset + _DECODE_LENGTH]
        if '\\' not in decoded and '%' not in decoded and '+' not in decoded and not _STRING_JOIN.search(decoded):
            continue
        positions = [(offset + i, offset + i + 1) for i in range(len(decoded))]
        for _ in range(_DECODE_DEPTH):
            pieces, mapped, end = [], [], 0
            for match in _ESCAPE.finditer(decoded):
                pieces.append(decoded[end:match.start()])
                mapped.extend(positions[end:match.start()])
                escape = match[0]
                if escape[0] in "\"'":
                    char = ""  # adjacent Python string literals, as printed in diagnostics/reprs
                elif escape == "+":
                    char = " "  # application/x-www-form-urlencoded bearer separator
                elif escape[0] == '%':
                    char = chr(int(escape[1:], 16))
                elif escape[1] in 'uUx':
                    value = int(escape[2:], 16)
                    char = chr(value) if value <= 0x10ffff else escape
                else:
                    char = _SIMPLE.get(escape[1], escape[1])
                pieces.append(char)
                mapped.extend([(positions[match.start()][0], positions[match.end() - 1][1])] * len(char))
                end = match.end()
            if end == 0:
                break
            pieces.append(decoded[end:])
            mapped.extend(positions[end:])
            decoded, positions = ''.join(pieces), mapped
            for match in _SECRET.finditer(decoded):
                start, stop = positions[match.start()]
                for left, right in positions[match.start() + 1:match.end()]:
                    if left <= stop:
                        stop = max(stop, right)
                    else:
                        # Keep the quotes between adjacent literals, masking each credential fragment.
                        spans.append((start, stop))
                        start, stop = left, right
                spans.append((start, stop))
    return spans


def redact_secrets(text):
    """One filter for records, subsequent prompts and HTML; preserve all noncredential source spans."""
    text = str(text or "")
    spans = sorted(_decoded_spans(text))
    pieces, end = [], 0
    for start, stop in spans:
        if start > end:
            pieces.append(text[end:start])
        if stop > end:
            if start >= end:
                pieces.append('<hidden>')
            end = stop
    pieces.append(text[end:])
    return ''.join(pieces)


def redact_record(value):
    """Filter strings at every level of JSON diagnostics, including nested encoded strings."""
    if isinstance(value, str):
        return redact_secrets(value)
    if isinstance(value, dict):
        return {redact_secrets(k) if isinstance(k, str) else k: redact_record(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_record(v) for v in value]
    return value


def redact_stream(source, destination):
    """Filter JSONL and plain diagnostics, including a bearer split over two plain lines."""
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
