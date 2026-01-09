"""Memory extraction pipeline.

Extracts actionable memories from conversation transcripts.
"""

from __future__ import annotations

from memory_layer.extraction.extractor import (
    ConflictRelationship,
    ConflictResult,
    ExtractedMemory,
    ExtractionConfig,
    ExtractionResult,
    MemoryExtractor,
    detect_entities,
    detect_pii,
    extract_from_transcript,
    filter_pii,
)

__all__ = [
    # Main class
    "MemoryExtractor",
    # Config
    "ExtractionConfig",
    # Results
    "ExtractionResult",
    "ExtractedMemory",
    "ConflictResult",
    "ConflictRelationship",
    # Convenience functions
    "extract_from_transcript",
    "detect_entities",
    "detect_pii",
    "filter_pii",
]
