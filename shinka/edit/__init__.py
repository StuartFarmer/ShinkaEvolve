"""Lazy exports for edit helpers."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "redact_immutable",
    "apply_diff_patch",
    "apply_full_patch",
    "summarize_diff",
]


def __getattr__(name: str):
    if name in {"redact_immutable", "apply_diff_patch"}:
        return getattr(import_module("shinka.edit.apply_diff"), name)
    if name == "apply_full_patch":
        return getattr(import_module("shinka.edit.apply_full"), name)
    if name == "summarize_diff":
        return getattr(import_module("shinka.edit.summary"), name)
    raise AttributeError(f"module 'shinka.edit' has no attribute {name!r}")
