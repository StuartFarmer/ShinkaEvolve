__all__ = [
    "Program",
    "DatabaseConfig",
    "AsyncProgramDatabase",
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
        from .program import Program

        return Program
    if name == "DatabaseConfig":
        from .config import DatabaseConfig

        return DatabaseConfig
    if name == "AsyncProgramDatabase":
        from .async_dbase import AsyncProgramDatabase

        return AsyncProgramDatabase
    if name == "ProgramCountSnapshot":
        from shinka.controllers.types import ProgramCountSnapshot

        return ProgramCountSnapshot
    if name == "InspirationUse":
        from shinka.controllers.types import InspirationUse

        return InspirationUse
    if name == "RunMetadataSnapshot":
        from shinka.controllers.types import RunMetadataSnapshot

        return RunMetadataSnapshot
    if name == "Island":
        from shinka.controllers.types import Island

        return Island
    if name in {
        "ArchivePolicy",
        "FitnessArchivePolicy",
        "CrowdingArchivePolicy",
        "create_archive_policy",
    }:
        from .archive_policy import (
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
        from .similarity_service import SimilarityService, cosine_similarity

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
