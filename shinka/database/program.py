from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np


def clean_nan_values(obj: Any) -> Any:
    """Recursively replace NaN/Inf values so JSON serialization is stable."""
    if isinstance(obj, dict):
        return {key: clean_nan_values(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(clean_nan_values(item) for item in obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, np.floating) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if hasattr(obj, "dtype") and np.issubdtype(obj.dtype, np.floating):
        if np.isscalar(obj):
            return None if np.isnan(obj) or np.isinf(obj) else float(obj)
        return clean_nan_values(obj.tolist())
    return obj


@dataclass
class Program:
    """Primary evolved artifact domain object."""

    id: str
    code: str
    language: str = "python"
    parent_id: Optional[str] = None
    archive_inspiration_ids: List[str] = field(default_factory=list)
    top_k_inspiration_ids: List[str] = field(default_factory=list)
    island_idx: Optional[int] = None
    generation: int = 0
    timestamp: float = field(default_factory=time.time)
    code_diff: Optional[str] = None
    combined_score: float = 0.0
    public_metrics: Dict[str, Any] = field(default_factory=dict)
    private_metrics: Dict[str, Any] = field(default_factory=dict)
    text_feedback: Union[str, List[str]] = ""
    correct: bool = False
    children_count: int = 0
    complexity: float = 0.0
    embedding: List[float] = field(default_factory=list)
    embedding_pca_2d: List[float] = field(default_factory=list)
    embedding_pca_3d: List[float] = field(default_factory=list)
    embedding_cluster_id: Optional[int] = None
    migration_history: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    in_archive: bool = False
    system_prompt_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return clean_nan_values(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Program":
        data = dict(data)
        data["public_metrics"] = (
            data.get("public_metrics")
            if isinstance(data.get("public_metrics"), dict)
            else {}
        )
        data["private_metrics"] = (
            data.get("private_metrics")
            if isinstance(data.get("private_metrics"), dict)
            else {}
        )
        data["metadata"] = (
            data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        )
        data["archive_inspiration_ids"] = (
            data.get("archive_inspiration_ids")
            if isinstance(data.get("archive_inspiration_ids"), list)
            else []
        )
        data["top_k_inspiration_ids"] = (
            data.get("top_k_inspiration_ids")
            if isinstance(data.get("top_k_inspiration_ids"), list)
            else []
        )
        data["embedding"] = data.get("embedding") if isinstance(data.get("embedding"), list) else []
        data["embedding_pca_2d"] = (
            data.get("embedding_pca_2d")
            if isinstance(data.get("embedding_pca_2d"), list)
            else []
        )
        data["embedding_pca_3d"] = (
            data.get("embedding_pca_3d")
            if isinstance(data.get("embedding_pca_3d"), list)
            else []
        )
        data["migration_history"] = (
            data.get("migration_history")
            if isinstance(data.get("migration_history"), list)
            else []
        )
        program_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in program_fields}
        return cls(**filtered_data)
