"""Runner layer: the resumable (id-keyed) generation loop and JSONL result store.

M2 adds the record schema + store + single-condition loop; M3 adds factorial matrix
expansion and run manifests on top of the same loop.
"""

from __future__ import annotations

from .loop import Condition, RunSummary, run_conditions
from .records import GenerationRecord, RecordStore, make_record_id

__all__ = [
    "Condition",
    "GenerationRecord",
    "RecordStore",
    "RunSummary",
    "make_record_id",
    "run_conditions",
]
