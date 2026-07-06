"""
Manifest schema validation — dispatches a manifest dict to its versioned
JSON Schema (schemas/manifest_v1_1.json, manifest_v1_2.json,
ragpack-manifest-1.3.schema.json) and raises on nonconformance.

Reading v1.2 (and v1.1) manifests is untouched by v1.3: this module only adds
a dispatch table entry, it does not change how earlier versions are read or
validated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

#: Directory containing the versioned manifest schemas.
_SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"

#: manifest "version" field value -> schema filename.
_SCHEMA_FILENAMES: dict[str, str] = {
    "1.1": "manifest_v1_1.json",
    "1.2": "manifest_v1_2.json",
    "1.3": "ragpack-manifest-1.3.schema.json",
}

#: manifest version -> the manifest field that carries it ("pack_version" for
#: v1.1/v1.2, "ragpack_version" for v1.3).
_VERSION_FIELD: dict[str, str] = {
    "1.1": "pack_version",
    "1.2": "pack_version",
    "1.3": "ragpack_version",
}


class ManifestValidationError(ValueError):
    """Raised when a manifest fails schema validation."""


def _load_schema(version: str) -> dict[str, Any]:
    filename = _SCHEMA_FILENAMES.get(version)
    if filename is None:
        raise ManifestValidationError(
            f"no schema registered for manifest version {version!r}; "
            f"known versions: {sorted(_SCHEMA_FILENAMES)}"
        )
    schema_path = _SCHEMAS_DIR / filename
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_manifest(manifest: dict[str, Any], version: str | None = None) -> None:
    """
    Validate ``manifest`` against its versioned JSON Schema.

    Args:
        manifest: The parsed manifest dict (e.g. from ``manifest.json``).
        version:  Schema version to validate against ("1.1", "1.2", "1.3").
                  When omitted, it is read from the manifest's own version
                  field (``ragpack_version`` for v1.3, ``pack_version`` for
                  v1.1/v1.2).

    Raises:
        ManifestValidationError: if the version cannot be determined/resolved,
            or the manifest does not conform to its schema.
    """
    if version is None:
        version = manifest.get("ragpack_version") or manifest.get("pack_version")
    if not version:
        raise ManifestValidationError(
            "could not determine manifest version "
            "(no 'ragpack_version' or 'pack_version' field)"
        )

    schema = _load_schema(version)
    try:
        jsonschema.validate(instance=manifest, schema=schema)
    except jsonschema.ValidationError as exc:
        raise ManifestValidationError(
            f"manifest failed v{version} schema validation: {exc.message}"
        ) from exc


def validate_manifest_file(path: str | Path, version: str | None = None) -> dict[str, Any]:
    """
    Load and validate a manifest.json file from disk.

    Returns the parsed manifest dict on success (so callers can reuse it
    without re-parsing).
    """
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    validate_manifest(manifest, version=version)
    return manifest
