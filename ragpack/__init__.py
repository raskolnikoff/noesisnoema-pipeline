"""
noesisnoema-pipeline ragpack module.

Provides deterministic RAGPack artifact generation (EPIC3 Stage 3).

Public API
----------
    from ragpack import ManifestBuilder, Ragpack, RagpackBuilder, RagpackWriter
"""

from .manifest_builder import ManifestBuilder, build_manifest_v1_2, build_manifest_v1_3
from .manifest_validator import ManifestValidationError, validate_manifest, validate_manifest_file
from .ragpack_builder import Ragpack, RagpackBuilder
from .ragpack_writer import RagpackWriter

__all__ = [
    "ManifestBuilder",
    "Ragpack",
    "RagpackBuilder",
    "RagpackWriter",
    "build_manifest_v1_2",
    "build_manifest_v1_3",
    "ManifestValidationError",
    "validate_manifest",
    "validate_manifest_file",
]

