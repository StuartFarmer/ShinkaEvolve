"""Runtime and lifecycle package."""

__all__ = [
    "AsyncContextSampler",
    "AsyncProgramDatabase",
    "ContextSampler",
    "SampledContext",
    "ShinkaEvolveRunner",
    "run_shinka_eval",
]


def __getattr__(name):
    if name == "AsyncProgramDatabase":
        from .async_store import AsyncProgramDatabase

        return AsyncProgramDatabase
    if name in {"AsyncContextSampler", "ContextSampler", "SampledContext"}:
        from .context import AsyncContextSampler, ContextSampler, SampledContext

        return {
            "AsyncContextSampler": AsyncContextSampler,
            "ContextSampler": ContextSampler,
            "SampledContext": SampledContext,
        }[name]
    if name == "run_shinka_eval":
        from .evaluation import run_shinka_eval

        return run_shinka_eval
    if name == "ShinkaEvolveRunner":
        from .runner import ShinkaEvolveRunner

        return ShinkaEvolveRunner
    raise AttributeError(name)
