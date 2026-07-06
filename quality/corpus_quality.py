"""
Corpus quality gates for source chunks before RAGpack generation.

The rules here are intentionally heuristic and document-agnostic.  They catch
common extraction failures and end-of-book publisher material without changing
retrieval, embedding, or query behavior.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


QUALITY_REPORT_VERSION = "1.0"
QUALITY_REPORT_FILE = "quality_report.json"

PASS = "pass"
WARNING = "warning"
REJECT = "reject"

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*|\d+[A-Za-z]+|[A-Za-z]+\d+")
_REPEATED_PUNCT_RE = re.compile(r"([^\w\s])\1{3,}")
_VOWEL_RE = re.compile(r"[aeiouyAEIOUY]")
_STAMP_KEYWORDS = frozenset({
    "archive",
    "archives",
    "department",
    "library",
    "property",
    "university",
    "college",
    "institute",
    "accession",
    "stamp",
    "withdrawn",
    "duplicate",
})
_BACK_MATTER_PHRASES = (
    "publisher's catalogue",
    "publishers' catalogue",
    "publisher catalogue",
    "catalogue of books",
    "catalog of books",
    "books by the same author",
    "also by the author",
    "also by",
    "recent publications",
    "new publications",
    "now ready",
    "just published",
    "forthcoming",
    "advertisement",
    "advertisements",
    "selected titles",
    "list of titles",
    "complete list",
    "edition de luxe",
)
_BACK_MATTER_TERMS = frozenset({
    "catalogue",
    "catalog",
    "advertisement",
    "advertisements",
    "publisher",
    "publishers",
    "published",
    "publication",
    "publications",
    "books",
    "titles",
    "volumes",
    "volume",
    "edition",
    "editions",
    "cloth",
    "paper",
    "paperback",
    "price",
    "net",
    "series",
    "forthcoming",
})


@dataclass(frozen=True)
class ChunkValidation:
    """Validation outcome for a single chunk."""

    ordinal: int
    status: str
    warnings: tuple[str, ...]
    rejection_reasons: tuple[str, ...]
    metadata: dict[str, Any]

    @property
    def accepted(self) -> bool:
        return self.status != REJECT

    def to_report_entry(self) -> dict[str, Any]:
        entry = {
            "ordinal": self.ordinal,
            "status": self.status,
            "warnings": list(self.warnings),
            "rejection_reasons": list(self.rejection_reasons),
        }
        entry.update(self.metadata)
        return entry


@dataclass(frozen=True)
class CorpusQualityReport:
    """Machine-readable summary of corpus validation."""

    report_version: str
    validation_timestamp: str
    total_chunks: int
    passed_chunks: int
    warning_chunks: int
    rejected_chunks: int
    accepted_chunks: int
    warnings: dict[str, int]
    rejection_reasons: dict[str, int]
    checks: list[str]
    chunks: list[ChunkValidation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_report_version": self.report_version,
            "validation_timestamp": self.validation_timestamp,
            "total_chunks": self.total_chunks,
            "passed_chunks": self.passed_chunks,
            "warning_chunks": self.warning_chunks,
            "rejected_chunks": self.rejected_chunks,
            "accepted_chunks": self.accepted_chunks,
            "warnings": dict(self.warnings),
            "rejection_reasons": dict(self.rejection_reasons),
            "checks": list(self.checks),
            "chunks": [c.to_report_entry() for c in self.chunks],
        }

    def manifest_metadata(self) -> dict[str, Any]:
        """Return a compact, backward-compatible manifest metadata block."""
        return {
            "quality_report_version": self.report_version,
            "validation_timestamp": self.validation_timestamp,
            "corpus_quality": {
                "total_chunks": self.total_chunks,
                "accepted_chunks": self.accepted_chunks,
                "passed_chunks": self.passed_chunks,
                "warning_chunks": self.warning_chunks,
                "rejected_chunks": self.rejected_chunks,
                "quality_report": QUALITY_REPORT_FILE,
            },
        }


def validate_corpus_chunks(
    chunks: Sequence[Any],
    *,
    validation_timestamp: str,
) -> tuple[list[Any], CorpusQualityReport]:
    """
    Validate chunks and return accepted chunks plus a quality report.

    ``chunks`` may contain app-facing dict chunks with a ``text`` field or
    ``ChunkRecord`` objects whose ``text_snippet`` is available.
    """
    validations: list[ChunkValidation] = []
    accepted: list[Any] = []

    for ordinal, chunk in enumerate(chunks):
        text = _chunk_text(chunk)
        validation = _validate_text(text, ordinal=ordinal, chunk=chunk)
        validations.append(validation)
        if validation.accepted:
            accepted.append(chunk)

    warning_counts = Counter(
        warning
        for result in validations
        for warning in result.warnings
    )
    rejection_counts = Counter(
        reason
        for result in validations
        for reason in result.rejection_reasons
    )
    passed = sum(1 for v in validations if v.status == PASS)
    warning = sum(1 for v in validations if v.status == WARNING)
    rejected = sum(1 for v in validations if v.status == REJECT)

    report = CorpusQualityReport(
        report_version=QUALITY_REPORT_VERSION,
        validation_timestamp=validation_timestamp,
        total_chunks=len(chunks),
        passed_chunks=passed,
        warning_chunks=warning,
        rejected_chunks=rejected,
        accepted_chunks=len(accepted),
        warnings=dict(sorted(warning_counts.items())),
        rejection_reasons=dict(sorted(rejection_counts.items())),
        checks=[
            "ocr_symbol_density",
            "ocr_repeated_punctuation",
            "ocr_token_repetition",
            "ocr_uppercase_stamp",
            "ocr_alphabetic_ratio",
            "ocr_noise_tokens",
            "publisher_back_matter",
        ],
        chunks=validations,
    )
    return accepted, report


def write_quality_report(report: CorpusQualityReport, output_dir: Path) -> Path:
    """Write quality_report.json to ``output_dir`` and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / QUALITY_REPORT_FILE
    path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _validate_text(text: str, *, ordinal: int, chunk: Any) -> ChunkValidation:
    warnings: list[str] = []
    rejections: list[str] = []
    normalized = " ".join(text.split())

    metrics = _quality_metrics(normalized)
    if metrics["nonspace_chars"] >= 40:
        if metrics["symbol_density"] > 0.18:
            rejections.append("excessive_symbol_density")
        elif metrics["symbol_density"] > 0.10:
            warnings.append("elevated_symbol_density")

        if metrics["alphabetic_ratio"] < 0.35:
            rejections.append("very_low_alphabetic_ratio")
        elif metrics["alphabetic_ratio"] < 0.50:
            warnings.append("low_alphabetic_ratio")

        if metrics["ocr_noise_token_ratio"] > 0.35 and metrics["token_count"] >= 8:
            rejections.append("suspicious_ocr_noise")
        elif metrics["ocr_noise_token_ratio"] > 0.20 and metrics["token_count"] >= 8:
            warnings.append("possible_ocr_noise")

    repeated_punct = _REPEATED_PUNCT_RE.search(normalized)
    if repeated_punct:
        if len(repeated_punct.group(0)) >= 5:
            rejections.append("repeated_punctuation")
        else:
            warnings.append("repeated_punctuation")

    if metrics["token_count"] >= 8:
        if metrics["max_token_repetition"] >= 0.45 and metrics["max_token_count"] >= 6:
            rejections.append("abnormal_token_repetition")
        elif metrics["max_token_repetition"] >= 0.32 and metrics["max_token_count"] >= 4:
            warnings.append("repeated_tokens")

    if _looks_like_uppercase_stamp(text, metrics):
        rejections.append("uppercase_stamp_fragment")

    back_matter_score = _publisher_back_matter_score(text, metrics)
    if back_matter_score >= 4:
        rejections.append("publisher_back_matter")
    elif back_matter_score >= 3:
        warnings.append("possible_publisher_back_matter")

    status = REJECT if rejections else WARNING if warnings else PASS
    return ChunkValidation(
        ordinal=ordinal,
        status=status,
        warnings=tuple(sorted(set(warnings))),
        rejection_reasons=tuple(sorted(set(rejections))),
        metadata=_chunk_metadata(chunk),
    )


def _quality_metrics(text: str) -> dict[str, Any]:
    nonspace = [c for c in text if not c.isspace()]
    nonspace_count = len(nonspace)
    alpha_count = sum(1 for c in nonspace if c.isalpha())
    digit_count = sum(1 for c in nonspace if c.isdigit())
    alnum_count = sum(1 for c in nonspace if c.isalnum())
    symbol_count = nonspace_count - alnum_count
    uppercase_alpha = sum(1 for c in nonspace if c.isalpha() and c.isupper())
    words = [w.lower() for w in _WORD_RE.findall(text)]
    repeated = Counter(words)
    max_token_count = max(repeated.values(), default=0)
    noise_tokens = sum(1 for w in words if _is_ocr_noise_token(w))
    title_like_lines = sum(1 for line in _nonempty_lines(text) if _is_title_list_line(line))

    return {
        "nonspace_chars": nonspace_count,
        "alphabetic_ratio": alpha_count / nonspace_count if nonspace_count else 0.0,
        "symbol_density": symbol_count / nonspace_count if nonspace_count else 0.0,
        "uppercase_alpha_ratio": uppercase_alpha / alpha_count if alpha_count else 0.0,
        "digit_ratio": digit_count / nonspace_count if nonspace_count else 0.0,
        "token_count": len(words),
        "max_token_count": max_token_count,
        "max_token_repetition": max_token_count / len(words) if words else 0.0,
        "ocr_noise_token_ratio": noise_tokens / len(words) if words else 0.0,
        "title_like_lines": title_like_lines,
        "line_count": len(_nonempty_lines(text)),
    }


def _is_ocr_noise_token(token: str) -> bool:
    if len(token) >= 7 and not _VOWEL_RE.search(token):
        return True
    has_alpha = any(c.isalpha() for c in token)
    has_digit = any(c.isdigit() for c in token)
    if has_alpha and has_digit and len(token) >= 4:
        return True
    transitions = sum(
        1
        for a, b in zip(token, token[1:])
        if (a.isdigit() and b.isalpha()) or (a.isalpha() and b.isdigit())
    )
    return transitions >= 2


def _looks_like_uppercase_stamp(text: str, metrics: dict[str, Any]) -> bool:
    words = [w.lower() for w in _WORD_RE.findall(text)]
    if not words:
        return False
    keyword_hits = sum(1 for word in words if word in _STAMP_KEYWORDS)
    compact = " ".join(text.split())
    if len(compact) > 160 or metrics["token_count"] > 16:
        return False
    if keyword_hits >= 1 and metrics["uppercase_alpha_ratio"] >= 0.65:
        return True
    return (
        metrics["uppercase_alpha_ratio"] >= 0.85
        and 2 <= metrics["token_count"] <= 10
        and metrics["line_count"] <= 3
        and keyword_hits >= 1
    )


def _publisher_back_matter_score(text: str, metrics: dict[str, Any]) -> int:
    lower = text.lower()
    words = [w.lower() for w in _WORD_RE.findall(text)]
    term_hits = sum(1 for w in words if w in _BACK_MATTER_TERMS)
    phrase_hits = sum(1 for phrase in _BACK_MATTER_PHRASES if phrase in lower)
    score = phrase_hits * 2
    if term_hits >= 4:
        score += 2
    elif term_hits >= 2:
        score += 1
    if metrics["title_like_lines"] >= 4:
        score += 2
    elif metrics["title_like_lines"] >= 2:
        score += 1
    if re.search(r"\b(?:\$|£|s\.|d\.|cents?|shillings?|net)\b", lower):
        score += 1
    return score


def _is_title_list_line(line: str) -> bool:
    stripped = line.strip(" \t-•*")
    if len(stripped) < 8 or len(stripped) > 100:
        return False
    words = _WORD_RE.findall(stripped)
    if len(words) < 2 or len(words) > 12:
        return False
    titlecase_words = sum(1 for word in words if word[:1].isupper())
    has_listing_marker = bool(re.search(r"\b(?:vol|volume|no|series|by|price)\b", stripped, re.I))
    return titlecase_words / len(words) >= 0.55 or has_listing_marker


def _chunk_text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return str(chunk.get("text", ""))
    return str(getattr(chunk, "text_snippet", ""))


def _chunk_metadata(chunk: Any) -> dict[str, Any]:
    if isinstance(chunk, dict):
        return {
            "chunk_index": chunk.get("chunk_index", chunk.get("chunk_id")),
            "doc_id": chunk.get("doc_id"),
            "source_path": chunk.get("source_path"),
            "source_hash": chunk.get("source_hash"),
            "char_start": chunk.get("char_start", chunk.get("start_char")),
            "char_end": chunk.get("char_end", chunk.get("end_char")),
        }
    return {
        "chunk_id": getattr(chunk, "chunk_id", None),
        "chunk_index": getattr(chunk, "chunk_index", None),
        "source_id": getattr(chunk, "source_id", None),
        "source_path": getattr(chunk, "source_path", None),
        "source_hash": getattr(chunk, "source_hash", None),
        "char_start": getattr(chunk, "char_start", None),
        "char_end": getattr(chunk, "char_end", None),
    }


def _nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]
