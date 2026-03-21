"""Lazy exports for core components."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "PromptSampler",
    "MetaSummarizer",
    "NoveltyJudge",
    "AsyncNoveltyJudge",
    "ContextSampler",
    "AsyncContextSampler",
    "SampledContext",
    "ShinkaEvolveRunner",
    "EvolutionConfig",
    "run_shinka_eval",
    "SystemPromptEvolver",
    "SystemPromptSampler",
    "AsyncSystemPromptEvolver",
]


def __getattr__(name: str):
    if name == "EvolutionConfig":
        return getattr(import_module("shinka.core.config"), name)
    if name == "ShinkaEvolveRunner":
        return getattr(import_module("shinka.core.async_runner"), name)
    if name == "PromptSampler":
        return getattr(import_module("shinka.core.sampler"), name)
    if name == "MetaSummarizer":
        return getattr(import_module("shinka.core.summarizer"), name)
    if name == "NoveltyJudge":
        return getattr(import_module("shinka.core.novelty_judge"), name)
    if name == "AsyncNoveltyJudge":
        return getattr(import_module("shinka.core.async_novelty_judge"), name)
    if name in {"ContextSampler", "AsyncContextSampler", "SampledContext"}:
        return getattr(import_module("shinka.core.context_sampler"), name)
    if name == "run_shinka_eval":
        return getattr(import_module("shinka.core.wrap_eval"), name)
    if name in {
        "SystemPromptEvolver",
        "SystemPromptSampler",
        "AsyncSystemPromptEvolver",
    }:
        return getattr(import_module("shinka.core.prompt_evolver"), name)
    raise AttributeError(f"module 'shinka.core' has no attribute {name!r}")
