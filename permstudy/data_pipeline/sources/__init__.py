"""Source acquisition stages that turn official data into ``QuestionRecord`` streams.

``SourceSnapshot`` is the fixed cross-source result type for every acquisition
stage: the validated records plus the private manifest that was written beside
them. ``sources/reclor.py`` and the later ``sources/math.py`` both return it.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ..schema import QuestionRecord, Source


QUESTION_RECORD_SCHEMA_VERSION = "question_record_v1"


@dataclass(frozen=True)
class SourceSnapshot:
    """Validated source records together with their private snapshot manifest."""

    source: Source
    source_revision: str
    source_snapshot_id: str
    questions: tuple[QuestionRecord, ...]
    private_manifest: Mapping[str, object]

    def __post_init__(self) -> None:
        # Copy incoming collections so callers cannot mutate frozen records indirectly.
        object.__setattr__(self, "questions", tuple(self.questions))
        object.__setattr__(self, "private_manifest", MappingProxyType(dict(self.private_manifest)))

    def validate(self) -> None:
        """Fail unless the records and their provenance belong to this snapshot."""
        if not isinstance(self.source, Source):
            raise ValueError("SourceSnapshot source must be a Source")
        for name in ("source_revision", "source_snapshot_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"SourceSnapshot {name} must be nonempty text")
        if not isinstance(self.private_manifest, Mapping):
            raise ValueError("SourceSnapshot private_manifest must be a mapping")
        for record in self.questions:
            if not isinstance(record, QuestionRecord):
                raise ValueError("SourceSnapshot questions must be QuestionRecord values")
            if record.source is not self.source:
                raise ValueError("SourceSnapshot question source does not match its snapshot")
            if record.source_snapshot_id != self.source_snapshot_id or record.source_revision != self.source_revision:
                raise ValueError("SourceSnapshot question provenance does not match its snapshot")
