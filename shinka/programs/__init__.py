"""Program-domain package."""

__all__ = [
    "ArchivePolicy",
    "CrowdingArchivePolicy",
    "FitnessArchivePolicy",
    "Program",
    "ProgramWriteResult",
    "ProgramWriteService",
    "create_archive_policy",
    "pick_random_archive_program",
]


def __getattr__(name):
    if name in {
        "ArchivePolicy",
        "CrowdingArchivePolicy",
        "FitnessArchivePolicy",
        "create_archive_policy",
        "pick_random_archive_program",
    }:
        from .archive import (
            ArchivePolicy,
            CrowdingArchivePolicy,
            FitnessArchivePolicy,
            create_archive_policy,
            pick_random_archive_program,
        )

        return {
            "ArchivePolicy": ArchivePolicy,
            "CrowdingArchivePolicy": CrowdingArchivePolicy,
            "FitnessArchivePolicy": FitnessArchivePolicy,
            "create_archive_policy": create_archive_policy,
            "pick_random_archive_program": pick_random_archive_program,
        }[name]
    if name == "Program":
        from .model import Program

        return Program
    if name in {"ProgramWriteResult", "ProgramWriteService"}:
        from .service import ProgramWriteResult, ProgramWriteService

        return {
            "ProgramWriteResult": ProgramWriteResult,
            "ProgramWriteService": ProgramWriteService,
        }[name]
    raise AttributeError(name)
