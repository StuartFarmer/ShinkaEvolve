__all__ = [
    "Program",
    "Database",
    "DatabaseConfig",
    "AsyncProgramDatabase",
    "program_reads",
    "program_writes",
    "InspirationUse",
    "ProgramCountSnapshot",
    "RunMetadataSnapshot",
    "Island",
    "ArchivePolicy",
    "FitnessArchivePolicy",
    "CrowdingArchivePolicy",
    "create_archive_policy",
    "SimilarityService",
    "cosine_similarity",
    "SystemPromptDatabase",
    "SystemPrompt",
    "SystemPromptConfig",
    "create_system_prompt",
]


def __getattr__(name):
    if name == "Program":
        from shinka.programs.model import Program

        return Program
    if name == "Database":
        from .connection import Database

        return Database
    if name == "DatabaseConfig":
        from .config import DatabaseConfig

        return DatabaseConfig
    if name == "AsyncProgramDatabase":
        from shinka.runtime.async_store import AsyncProgramDatabase

        return AsyncProgramDatabase
    if name == "program_reads":
        from shinka.programs import reads as program_reads

        return program_reads
    if name == "program_writes":
        from shinka.programs import writes as program_writes

        return program_writes
    if name == "ProgramCountSnapshot":
        from .types import ProgramCountSnapshot

        return ProgramCountSnapshot
    if name == "InspirationUse":
        from .types import InspirationUse

        return InspirationUse
    if name == "RunMetadataSnapshot":
        from .types import RunMetadataSnapshot

        return RunMetadataSnapshot
    if name == "Island":
        from .types import Island

        return Island
    if name in {
        "ArchivePolicy",
        "FitnessArchivePolicy",
        "CrowdingArchivePolicy",
        "create_archive_policy",
    }:
        from shinka.programs.archive import (
            ArchivePolicy,
            FitnessArchivePolicy,
            CrowdingArchivePolicy,
            create_archive_policy,
        )

        return {
            "ArchivePolicy": ArchivePolicy,
            "FitnessArchivePolicy": FitnessArchivePolicy,
            "CrowdingArchivePolicy": CrowdingArchivePolicy,
            "create_archive_policy": create_archive_policy,
        }[name]
    if name in {"SimilarityService", "cosine_similarity"}:
        from shinka.programs.similarity import SimilarityService, cosine_similarity

        return {
            "SimilarityService": SimilarityService,
            "cosine_similarity": cosine_similarity,
        }[name]
    if name in {
        "SystemPromptDatabase",
        "SystemPrompt",
        "SystemPromptConfig",
        "create_system_prompt",
    }:
        from .prompt_dbase import (
            SystemPromptDatabase,
            SystemPrompt,
            SystemPromptConfig,
            create_system_prompt,
        )

        return {
            "SystemPromptDatabase": SystemPromptDatabase,
            "SystemPrompt": SystemPrompt,
            "SystemPromptConfig": SystemPromptConfig,
            "create_system_prompt": create_system_prompt,
        }[name]
    raise AttributeError(name)
