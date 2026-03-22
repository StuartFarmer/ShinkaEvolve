__all__ = [
    "Program",
    "DatabaseConfig",
    "DatabaseConnector",
    "AsyncProgramDatabase",
    "ProgramRepository",
    "RepositoryBundle",
    "InspirationRepository",
    "InspirationUse",
    "ProgramCountSnapshot",
    "MetadataRepository",
    "RunMetadataSnapshot",
    "Island",
    "IslandRepository",
    "ArchivePolicy",
    "FitnessArchivePolicy",
    "CrowdingArchivePolicy",
    "create_archive_policy",
    "SimilarityService",
    "cosine_similarity",
    "EmbeddingFeatureService",
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
    if name == "DatabaseConnector":
        from .connector import DatabaseConnector

        return DatabaseConnector
    if name == "AsyncProgramDatabase":
        from .async_dbase import AsyncProgramDatabase

        return AsyncProgramDatabase
    if name in {"ProgramRepository", "ProgramCountSnapshot"}:
        from .repository import ProgramCountSnapshot, ProgramRepository

        return {
            "ProgramRepository": ProgramRepository,
            "ProgramCountSnapshot": ProgramCountSnapshot,
        }[name]
    if name == "RepositoryBundle":
        from .repository_bundle import RepositoryBundle

        return RepositoryBundle
    if name in {"InspirationRepository", "InspirationUse"}:
        from .inspiration_repository import InspirationRepository, InspirationUse

        return {
            "InspirationRepository": InspirationRepository,
            "InspirationUse": InspirationUse,
        }[name]
    if name in {"MetadataRepository", "RunMetadataSnapshot"}:
        from .metadata_repository import MetadataRepository, RunMetadataSnapshot

        return {
            "MetadataRepository": MetadataRepository,
            "RunMetadataSnapshot": RunMetadataSnapshot,
        }[name]
    if name in {"Island", "IslandRepository"}:
        from .island_repository import Island, IslandRepository

        return {
            "Island": Island,
            "IslandRepository": IslandRepository,
        }[name]
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
    if name == "EmbeddingFeatureService":
        from .embedding_feature_service import EmbeddingFeatureService

        return EmbeddingFeatureService
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
