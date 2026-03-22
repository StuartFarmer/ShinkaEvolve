__all__ = [
    "DatabaseController",
    "ProgramController",
    "MetadataController",
    "InspirationController",
    "EmbeddingController",
    "IslandController",
    "RunStateController",
    "InspirationUse",
    "Island",
    "ProgramCountSnapshot",
    "RunMetadataSnapshot",
]


def __getattr__(name):
    if name == "DatabaseController":
        from .database_controller import DatabaseController

        return DatabaseController
    if name == "ProgramController":
        from .program_controller import ProgramController

        return ProgramController
    if name == "MetadataController":
        from .metadata_controller import MetadataController

        return MetadataController
    if name == "InspirationController":
        from .inspiration_controller import InspirationController

        return InspirationController
    if name == "EmbeddingController":
        from .embedding_controller import EmbeddingController

        return EmbeddingController
    if name == "IslandController":
        from .island_controller import IslandController

        return IslandController
    if name == "RunStateController":
        from .run_state_controller import RunStateController

        return RunStateController
    if name in {"InspirationUse", "Island", "ProgramCountSnapshot", "RunMetadataSnapshot"}:
        from .types import InspirationUse, Island, ProgramCountSnapshot, RunMetadataSnapshot

        return {
            "InspirationUse": InspirationUse,
            "Island": Island,
            "ProgramCountSnapshot": ProgramCountSnapshot,
            "RunMetadataSnapshot": RunMetadataSnapshot,
        }[name]
    raise AttributeError(name)
