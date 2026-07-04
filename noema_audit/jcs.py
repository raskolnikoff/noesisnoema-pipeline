"""
RFC 8785 JSON Canonicalization Scheme (JCS) — the cross-language contract for
audit event hashing (spec-audit-pipeline.md §2).

Numbers-as-strings rule
------------------------
RFC 8785 requires ES6 (``Number.prototype.toString``) formatting for JSON
numbers, whose shortest-round-trip float algorithm is the classic source of
Python/Swift byte divergence (spec-audit-pipeline.md §2: "number normalization
... [is a] usual failure point"). Rather than reimplement that algorithm
twice, audit events forbid floats outright: scores/thresholds/recall values
are emitted as decimal *strings* (e.g. ``"0.87"``), never as JSON numbers.
``canonicalize`` therefore raises on any ``float`` it encounters — this is the
"reject any float in event payloads" rule enforced at the emitter level
(see ``emitter.py``).  Plain integers ARE permitted (counts, k, timestamps as
epoch ints if ever needed); JCS integer formatting is unambiguous and does not
raise the shortest-round-trip problem floats do.
"""

from __future__ import annotations

from typing import Any

#: ES6 Number.MAX_SAFE_INTEGER / MIN_SAFE_INTEGER — the range JCS integer
#: formatting can serialize unambiguously across a JS-compatible reader.
_MAX_SAFE_INTEGER = 2**53 - 1
_MIN_SAFE_INTEGER = -(2**53 - 1)


class CanonicalizationError(ValueError):
    """Raised when a value cannot be canonicalized under the audit JCS rules."""


def _encode_string(s: str) -> str:
    """Minimal JSON string escaping (matches ECMAScript JSON.stringify)."""
    out = ['"']
    for ch in s:
        cp = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\b":
            out.append("\\b")
        elif ch == "\f":
            out.append("\\f")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif cp < 0x20:
            out.append(f"\\u{cp:04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _utf16_sort_key(s: str) -> bytes:
    """
    RFC 8785 §3.2.3: object member names are sorted by their UTF-16 code unit
    sequence. Encoding to big-endian UTF-16 and comparing bytes lexically
    reproduces that ordering exactly (each code unit is 2 bytes, big-endian,
    so byte-wise comparison equals numeric code-unit comparison), including
    for supplementary-plane characters represented as surrogate pairs.
    """
    return s.encode("utf-16-be")


def _encode(obj: Any) -> str:
    if obj is None:
        return "null"
    if obj is True:
        return "true"
    if obj is False:
        return "false"
    if isinstance(obj, float):
        raise CanonicalizationError(
            "floats are forbidden in audit event payloads (numbers-as-strings "
            f"rule, spec-audit-pipeline.md §2); got {obj!r}. Emit as a string, "
            'e.g. str(0.87) -> "0.87".'
        )
    if isinstance(obj, int):
        if not (_MIN_SAFE_INTEGER <= obj <= _MAX_SAFE_INTEGER):
            raise CanonicalizationError(
                f"integer {obj} exceeds the JS safe-integer range "
                f"[{_MIN_SAFE_INTEGER}, {_MAX_SAFE_INTEGER}]; emit as a string instead"
            )
        return str(obj)
    if isinstance(obj, str):
        return _encode_string(obj)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_encode(v) for v in obj) + "]"
    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: _utf16_sort_key(kv[0]))
        return "{" + ",".join(f"{_encode_string(k)}:{_encode(v)}" for k, v in items) + "}"
    raise CanonicalizationError(f"non-JSON-serializable type: {type(obj).__name__}")


def canonicalize(obj: Any) -> bytes:
    """
    Return the RFC 8785 JCS canonical form of ``obj`` as UTF-8 bytes.

    Raises:
        CanonicalizationError: on a float, an out-of-range integer, or a
            non-JSON-serializable value anywhere in ``obj``.
    """
    return _encode(obj).encode("utf-8")
