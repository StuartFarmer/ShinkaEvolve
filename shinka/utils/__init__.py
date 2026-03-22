"""Lazy exports for utility helpers."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "load_programs_to_df",
    "get_path_to_best_node",
    "store_best_path",
    "parse_time_to_seconds",
    "load_results",
    "build_cfgs_from_python",
    "add_evolve_markers",
    "chdir_to_function_dir",
    "wrap_object",
    "load_hydra_config",
    "load_configs_from_yaml",
    "get_language_extension",
    "load_prompts_to_df",
]


def __getattr__(name: str):
    if name in {
        "load_programs_to_df",
        "get_path_to_best_node",
        "store_best_path",
        "load_prompts_to_df",
    }:
        return getattr(import_module("shinka.reporting.dataframe"), name)
    if name in {"parse_time_to_seconds", "load_results", "load_configs_from_yaml"}:
        if name == "load_configs_from_yaml":
            return getattr(import_module("shinka.configuration.hydra"), name)
        return getattr(import_module("shinka.launch.support"), name)
    if name in {
        "build_cfgs_from_python",
        "add_evolve_markers",
        "chdir_to_function_dir",
        "wrap_object",
        "load_hydra_config",
    }:
        return getattr(import_module("shinka.configuration.hydra"), name)
    if name == "get_language_extension":
        return getattr(import_module("shinka.common.languages"), name)
    raise AttributeError(f"module 'shinka.utils' has no attribute {name!r}")
