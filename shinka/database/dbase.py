import json
import logging
import sqlite3
import time
import warnings
from dataclasses import asdict, dataclass, field
from functools import wraps
from pathlib import Path
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import math
from .islands import CombinedIslandManager
from .island_sampler import create_island_sampler, IslandSampler
from .island_repository import IslandRepository
from .metadata_repository import MetadataRepository
from .embedding_feature_service import EmbeddingFeatureService
from .program_write_service import ProgramWriteService
from .repository_bundle import RepositoryBundle
from .similarity_service import SimilarityService
from .display import DatabaseDisplay
from shinka.embed import EmbeddingClient
from shinka.defaults import default_archive_criteria

logger = logging.getLogger(__name__)

def clean_nan_values(obj: Any) -> Any:
    """
    Recursively clean NaN values from a data structure, replacing them with
    None. This ensures JSON serialization works correctly.
    """
    if isinstance(obj, dict):
        return {key: clean_nan_values(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(clean_nan_values(item) for item in obj)
    elif isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    elif isinstance(obj, np.floating) and (np.isnan(obj) or np.isinf(obj)):
        return None
    elif hasattr(obj, "dtype") and np.issubdtype(obj.dtype, np.floating):
        # Handle numpy arrays and scalars
        if np.isscalar(obj):
            if np.isnan(obj) or np.isinf(obj):
                return None
            else:
                return float(obj)
        else:
            # For numpy arrays, convert to list and clean recursively
            return clean_nan_values(obj.tolist())
    else:
        return obj


@dataclass
class DatabaseConfig:
    """
    Search-memory and lineage-structure knobs.

    These are not evaluator/task settings. They control how Shinka stores
    programs and samples from history.

    Most important levers:
    - `num_islands`: number of semi-independent search populations
    - `archive_size`: how many high-value programs are retained for reuse
    - `num_archive_inspirations`, `num_top_k_inspirations`: prompt context size
    - `parent_selection_strategy`: weighted / beam / winner-take-all, etc.
    - `migration_interval`, `migration_rate`: how ideas move across islands
    - `enforce_island_separation`: whether parents/inspirations stay local to an
      island except via explicit migration
    """
    db_path: Optional[str] = None  # Path to SQLite database file
    num_islands: int = 2
    archive_size: int = 40

    # Inspiration parameters
    elite_selection_ratio: float = 0.3  # Prop of elites inspirations
    num_archive_inspirations: int = 1  # No. inspiration programs
    num_top_k_inspirations: int = 1  # No. top-k inspiration programs

    # Island model/migration parameters
    migration_interval: int = 10  # Migrate every N generations
    migration_rate: float = 0.0  # Prop. of island pop. to migrate
    island_elitism: bool = True  # Keep best prog on their islands
    enforce_island_separation: bool = (
        True  # Enforce full island separation for inspirations
    )
    island_selection_strategy: str = "uniform"  # Island sampling strategy: "uniform"/"equal"/"proportional"/"weighted"

    # Dynamic island spawning parameters (stagnation-based)
    enable_dynamic_islands: bool = False  # Enable stagnation-based island spawning
    stagnation_threshold: int = 100  # Gens without improvement to trigger spawn
    island_spawn_strategy: str = (
        "initial"  # How to seed new islands: "initial", "best", "archive_random"
    )
    island_spawn_subtree_size: int = 1  # Max programs to copy (1=single, >1=subtree)

    # Parent selection parameters
    parent_selection_strategy: str = (
        "weighted"  # "weighted"/"power_law"/"beam_search"/"winner_take_all"
    )

    # Power-law parent selection parameters
    exploitation_alpha: float = 1.0  # 0=uniform, 1=power-law
    exploitation_ratio: float = 0.2  # Chance to pick from archive

    # Weighted tree parent selection parameters
    parent_selection_lambda: float = 10.0  # >0 sharpness of sigmoid

    # Beam search parent selection parameters
    num_beams: int = 5

    # Archive selection parameters
    archive_selection_strategy: str = "fitness"  # "fitness" or "crowding"
    # Criteria weights for archive selection (sign indicates direction):
    #   Positive weight = higher is better (e.g., combined_score)
    #   Negative weight = lower is better (e.g., loc, complexity)
    # Weights represent relative importance after rank normalization
    archive_criteria: Dict[str, float] = field(default_factory=default_archive_criteria)

def db_retry(max_retries=5, initial_delay=0.1, backoff_factor=2):
    """
    A decorator to retry database operations on specific SQLite errors.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (
                    sqlite3.OperationalError,
                    sqlite3.DatabaseError,
                    sqlite3.IntegrityError,
                ) as e:
                    if i == max_retries - 1:
                        logger.error(
                            f"DB operation {func.__name__} failed after "
                            f"{max_retries} retries: {e}"
                        )
                        raise
                    logger.warning(
                        f"DB operation {func.__name__} failed with "
                        f"{type(e).__name__}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                    delay *= backoff_factor
            # This part should not be reachable if max_retries > 0
            raise RuntimeError(
                f"DB retry logic failed for function {func.__name__} without "
                "raising an exception."
            )

        return wrapper

    return decorator


@dataclass
class Program:
    """Represents a program in the database"""

    # Program identification
    id: str
    code: str
    language: str = "python"

    # Evolution information
    parent_id: Optional[str] = None
    archive_inspiration_ids: List[str] = field(
        default_factory=list
    )  # IDs of programs used as archive inspiration
    top_k_inspiration_ids: List[str] = field(
        default_factory=list
    )  # IDs of programs used as top-k inspiration
    island_idx: Optional[int] = None
    generation: int = 0
    timestamp: float = field(default_factory=time.time)
    code_diff: Optional[str] = None

    # Performance metrics
    combined_score: float = 0.0
    public_metrics: Dict[str, Any] = field(default_factory=dict)
    private_metrics: Dict[str, Any] = field(default_factory=dict)
    text_feedback: Union[str, List[str]] = ""
    correct: bool = False  # Whether the program is functionally correct
    children_count: int = 0

    # Derived features
    complexity: float = 0.0  # Calculated based on code or other features
    embedding: List[float] = field(default_factory=list)
    embedding_pca_2d: List[float] = field(default_factory=list)
    embedding_pca_3d: List[float] = field(default_factory=list)
    embedding_cluster_id: Optional[int] = None

    # Migration history
    migration_history: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Archive status
    in_archive: bool = False

    # Meta-prompt evolution: track which system prompt generated this program
    system_prompt_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict representation, cleaning NaN values for JSON."""
        data = asdict(self)
        return clean_nan_values(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Program":
        """Create from dictionary representation, ensuring correct types for
        nested dicts."""
        # Ensure metrics and metadata are dictionaries, even if None/empty from
        # DB or input
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
        # Ensure inspiration_ids is a list
        archive_ids_val = data.get("archive_inspiration_ids")
        if isinstance(archive_ids_val, list):
            data["archive_inspiration_ids"] = archive_ids_val
        else:
            data["archive_inspiration_ids"] = []

        top_k_ids_val = data.get("top_k_inspiration_ids")
        if isinstance(top_k_ids_val, list):
            data["top_k_inspiration_ids"] = top_k_ids_val
        else:
            data["top_k_inspiration_ids"] = []

        # Ensure embedding is a list
        embedding_val = data.get("embedding")
        if isinstance(embedding_val, list):
            data["embedding"] = embedding_val
        else:
            data["embedding"] = []

        embedding_pca_2d_val = data.get("embedding_pca_2d")
        if isinstance(embedding_pca_2d_val, list):
            data["embedding_pca_2d"] = embedding_pca_2d_val
        else:
            data["embedding_pca_2d"] = []

        embedding_pca_3d_val = data.get("embedding_pca_3d")
        if isinstance(embedding_pca_3d_val, list):
            data["embedding_pca_3d"] = embedding_pca_3d_val
        else:
            data["embedding_pca_3d"] = []

        # Ensure migration_history is a list
        migration_history_val = data.get("migration_history")
        if isinstance(migration_history_val, list):
            data["migration_history"] = migration_history_val
        else:
            data["migration_history"] = []

        # Filter out keys not in Program fields to avoid TypeError with **data
        program_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in program_fields}

        return cls(**filtered_data)


class ProgramDatabase:
    """
    SQLite-backed database for storing and managing programs during an
    evolutionary process.
    Supports MAP-Elites style feature-based organization, island
    populations, and an archive of elites.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        num_islands: int = 2,
        archive_size: int = 40,
        migration_interval: int = 10,
        migration_rate: float = 0.0,
        island_elitism: bool = True,
        island_selection_strategy: str = "uniform",
        enable_dynamic_islands: bool = False,
        stagnation_threshold: int = 100,
        island_spawn_strategy: str = "initial",
        island_spawn_subtree_size: int = 1,
        parent_selection_strategy: str = "weighted",
        exploitation_alpha: float = 1.0,
        exploitation_ratio: float = 0.2,
        parent_selection_lambda: float = 10.0,
        num_beams: int = 5,
        archive_selection_strategy: str = "fitness",
        archive_criteria: Optional[Dict[str, float]] = None,
        elite_selection_ratio: float = 0.3,
        num_archive_inspirations: int = 1,
        num_top_k_inspirations: int = 1,
        enforce_island_separation: bool = True,
        embedding_model: str = "text-embedding-3-small",
        read_only: bool = False,
    ):
        self.db_path = db_path
        self.num_islands = num_islands
        self.archive_size = archive_size
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.island_elitism = island_elitism
        self.island_selection_strategy = island_selection_strategy
        self.enable_dynamic_islands = enable_dynamic_islands
        self.stagnation_threshold = stagnation_threshold
        self.island_spawn_strategy = island_spawn_strategy
        self.island_spawn_subtree_size = island_spawn_subtree_size
        self.parent_selection_strategy = parent_selection_strategy
        self.exploitation_alpha = exploitation_alpha
        self.exploitation_ratio = exploitation_ratio
        self.parent_selection_lambda = parent_selection_lambda
        self.num_beams = num_beams
        self.archive_selection_strategy = archive_selection_strategy
        self.archive_criteria = archive_criteria or default_archive_criteria()
        self.elite_selection_ratio = elite_selection_ratio
        self.num_archive_inspirations = num_archive_inspirations
        self.num_top_k_inspirations = num_top_k_inspirations
        self.enforce_island_separation = enforce_island_separation
        self.embedding_model = embedding_model
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
        self.read_only = read_only
        self.display_console: Optional[Any] = None

        # Lazy-init embedding client to avoid requiring API credentials for
        # database-only operations and tests that do not compute embeddings.
        self.embedding_client: Optional[EmbeddingClient] = None
        self._embedding_client_init_failed = False
        self.metadata_repo: Optional[MetadataRepository] = None
        self.island_repo: Optional[IslandRepository] = None

        self.last_iteration: int = 0
        self.best_program_id: Optional[str] = None
        self.beam_search_parent_id: Optional[str] = None
        self.initial_program_count_adjustment: int = 0
        # For deferring expensive operations
        self._schedule_migration: bool = False

        # Stagnation tracking for dynamic island spawning
        self.best_score_generation: int = 0  # Generation when best score was found
        self.best_score_ever: Optional[float] = None  # Track best score for comparison

        # Initialize island manager (will be set after db connection)
        self.island_manager: Optional[CombinedIslandManager] = None
        # Initialize island sampler (will be set after db connection)
        self.island_sampler: Optional[IslandSampler] = None
        self.repository_bundle = RepositoryBundle.open(
            db_path=self.db_path,
            num_islands=self.num_islands,
            read_only=self.read_only,
        )
        self.conn = self.repository_bundle.conn
        self.cursor = self.repository_bundle.cursor
        self.program_repository = self.repository_bundle.programs
        self.metadata_repo = self.repository_bundle.metadata
        self.island_repo = self.repository_bundle.islands
        self._load_metadata_from_db()

        from .archive_policy import create_archive_policy

        self.archive_policy = create_archive_policy(
            archive_selection_strategy=self.archive_selection_strategy,
            archive_size=self.archive_size,
            archive_criteria=self.archive_criteria,
        )
        self._initialize_runtime_services()

        count = self._count_programs_in_db()
        logger.debug(f"DB initialized with {count} programs.")
        logger.debug(
            f"Last iter: {self.last_iteration}. Best ID: {self.best_program_id}"
        )

    def _set_repository_bundle(self, bundle: RepositoryBundle) -> None:
        self.repository_bundle = bundle
        self.conn = bundle.conn
        self.cursor = bundle.cursor
        self.program_repository = bundle.programs
        self.metadata_repo = bundle.metadata
        self.island_repo = bundle.islands

    def _initialize_runtime_services(self) -> None:
        self.island_manager = CombinedIslandManager(
            num_islands=self.num_islands,
            migration_interval=self.migration_interval,
            migration_rate=self.migration_rate,
            island_elitism=self.island_elitism,
            island_spawn_strategy=self.island_spawn_strategy,
            island_spawn_subtree_size=self.island_spawn_subtree_size,
            program_repository=self.program_repository,
            island_repository=self.island_repo,
            archive_policy=self.archive_policy,
        )
        self.similarity_service = SimilarityService(self.program_repository)
        self.embedding_feature_service = EmbeddingFeatureService(
            self.program_repository,
            embedding_client_factory=self._ensure_embedding_client,
            read_only=self.read_only,
        )
        self.program_write_service = ProgramWriteService(
            program_repository=self.program_repository,
            island_manager=self.island_manager,
            update_best_program=self._update_best_program,
            update_metadata=self._update_metadata_in_db,
            recompute_embeddings=self.embedding_feature_service.recompute,
            print_program_summary=self._print_program_summary,
            maybe_spawn_island=self.check_and_spawn_island_if_stagnant,
        )
        self.island_sampler = create_island_sampler(
            program_repository=self.program_repository,
            strategy=self.island_selection_strategy,
        )
        if hasattr(self, "_database_display"):
            delattr(self, "_database_display")

    def _get_database_display(self) -> DatabaseDisplay:
        if not hasattr(self, "_database_display"):
            self._database_display = DatabaseDisplay(
                program_repository=self.program_repository,
                archive_size=self.archive_size,
                num_islands=self.num_islands,
                island_manager=self.island_manager,
                archive_policy=self.archive_policy,
                default_console=self.display_console,
            )
        return self._database_display

    def _ensure_embedding_client(self) -> Optional[EmbeddingClient]:
        """Create embedding client on demand.

        Returns:
            EmbeddingClient if available, otherwise None.
        """
        if self.read_only or not self.embedding_model:
            return None
        if self.embedding_client is not None:
            return self.embedding_client
        if self._embedding_client_init_failed:
            return None

        try:
            self.embedding_client = EmbeddingClient(model_name=self.embedding_model)
        except Exception as e:
            self._embedding_client_init_failed = True
            logger.warning(
                "Embedding client init failed for model '%s'; "
                "continuing without embedding recomputation: %s",
                self.embedding_model,
                e,
            )
            return None

        return self.embedding_client

    @db_retry()
    def _load_metadata_from_db(self):
        if self.metadata_repo is None:
            raise ConnectionError("Metadata repository not initialized.")

        snapshot = self.metadata_repo.load_snapshot()
        self.last_iteration = snapshot.last_iteration
        self.best_program_id = snapshot.best_program_id
        self.beam_search_parent_id = snapshot.beam_search_parent_id
        self.best_score_generation = snapshot.best_score_generation
        self.best_score_ever = snapshot.best_score_ever
        self.initial_program_count_adjustment = snapshot.initial_program_count_adjustment

    @db_retry()
    def _update_metadata_in_db(self, key: str, value: Optional[str]):
        if self.metadata_repo is None:
            raise ConnectionError("Metadata repository not initialized.")
        self.metadata_repo.set(key, value)

    @db_retry()
    def _count_programs_in_db(self) -> int:
        if not getattr(self, "program_repository", None):
            return 0
        return self.program_repository.get_count_snapshot().count

    @db_retry()
    def set_initial_program_count_adjustment(self, adjustment: int):
        if self.read_only:
            raise PermissionError("Cannot update metadata in read-only mode.")
        self.initial_program_count_adjustment = max(int(adjustment), 0)
        self._update_metadata_in_db(
            "initial_program_count_adjustment",
            str(self.initial_program_count_adjustment),
        )

    @db_retry()
    def add(self, program: Program, verbose: bool = False) -> str:
        """
        Add a program to the database with optimized performance.

        This method uses batched transactions and defers expensive operations
        to improve performance with large databases. After adding a program,
        you should call check_scheduled_operations() to run any deferred
        operations like migrations.

        Example:
            db.add(program)  # Fast add
            db.check_scheduled_operations()  # Run deferred operations

        Args:
            program: The Program object to add

        Returns:
            str: The ID of the added program
        """
        if self.read_only:
            raise PermissionError("Cannot add program in read-only mode.")
        if not hasattr(self, "program_write_service"):
            raise ConnectionError("DB not connected.")
        result = self.program_write_service.add(
            program,
            verbose=verbose,
            current_last_iteration=self.last_iteration,
        )
        self.last_iteration = result.last_iteration
        self._schedule_migration = False
        return result.program_id

    def save(self, path: Optional[str] = None) -> None:
        if not self.conn or not self.cursor:
            logger.warning("No DB connection, skipping save.")
            return

        # Main purpose here is to save/commit metadata like last_iteration.
        current_db_file_path_str = self.db_path
        if path and current_db_file_path_str:
            if Path(path).resolve() != Path(current_db_file_path_str).resolve():
                logger.warning(
                    f"Save path '{path}' differs from connected DB "
                    f"'{current_db_file_path_str}'. Metadata saved to "
                    "connected DB."
                )
        elif path and not current_db_file_path_str:
            logger.warning(
                f"Attempting to save with path '{path}' but current "
                "database is in-memory. Metadata will be committed to the "
                "in-memory instance."
            )

        self._update_metadata_in_db("last_iteration", str(self.last_iteration))

        self.conn.commit()  # Commit any pending transactions
        logger.info(
            f"Database state committed. Last iteration: "
            f"{self.last_iteration}. Best: {self.best_program_id}"
        )

    def load(self, path: str) -> None:
        logger.info(f"Loading database from '{path}'...")
        if self.repository_bundle:
            db_display_name = self.db_path or ":memory:"
            logger.info(f"Closing existing connection to '{db_display_name}'.")
            self.repository_bundle.close()

        self.db_path = str(Path(path).resolve())
        self._set_repository_bundle(
            RepositoryBundle.open(
                db_path=self.db_path,
                num_islands=self.num_islands,
                read_only=self.read_only,
            )
        )
        self._load_metadata_from_db()
        self._initialize_runtime_services()

        count = self._count_programs_in_db()
        logger.info(
            f"Loaded DB from '{self.db_path}'. {count} programs. "
            f"Last iter: {self.last_iteration}."
        )

    def _is_better(
        self,
        program1: Program,
        program2: Program,
        archive_programs: Optional[List[Program]] = None,
    ) -> bool:
        """
        Compare two programs to determine if program1 is better than program2.

        Args:
            program1: First program to compare
            program2: Second program to compare
            archive_programs: Optional archive context for rank-based scoring.
                If provided and archive_criteria has multiple criteria,
                uses rank-based normalization for scale-invariant comparison.

        Returns:
            True if program1 is better than program2
        """
        return self.archive_policy._is_better(
            program1,
            program2,
            archive_programs=archive_programs,
        )

    @db_retry()
    def _update_archive(self, program: Program) -> None:
        """Deprecated no-op. Archive membership is now computed on demand."""
        logger.debug(
            "ProgramDatabase._update_archive(%s) is deprecated; archive persistence is disabled.",
            program.id,
        )
        return

    def _update_archive_fitness(self, program: Program) -> None:
        logger.debug(
            "ProgramDatabase._update_archive_fitness(%s) is deprecated and no longer used.",
            program.id,
        )
        return

    def _update_archive_crowding(self, program: Program) -> None:
        logger.debug(
            "ProgramDatabase._update_archive_crowding(%s) is deprecated and no longer used.",
            program.id,
        )
        return

    @db_retry()
    def _update_best_program(self, program: Program) -> None:
        # Only consider correct programs for best program tracking
        if not program.correct:
            logger.debug(f"Program {program.id} not considered for best (not correct).")
            return

        current_best_p = None
        if self.best_program_id and getattr(self, "program_repository", None):
            current_best_p = self.program_repository.get(self.best_program_id)

        if current_best_p is None or self._is_better(program, current_best_p):
            self.best_program_id = program.id
            self._update_metadata_in_db("best_program_id", self.best_program_id)

            # Update stagnation tracking - new best found
            program_score = program.combined_score or 0.0
            if self.best_score_ever is None or program_score > self.best_score_ever:
                self.best_score_ever = program_score
                self.best_score_generation = program.generation
                self._update_metadata_in_db(
                    "best_score_generation", str(self.best_score_generation)
                )
                self._update_metadata_in_db(
                    "best_score_ever", str(self.best_score_ever)
                )

            log_msg = f"New best program: {program.id}"
            if current_best_p:
                p1_score = program.combined_score or 0.0
                p2_score = current_best_p.combined_score or 0.0
                log_msg += (
                    f" (gen: {current_best_p.generation} → {program.generation}, "
                    f"score: {p2_score:.4f} → {p1_score:.4f}, "
                    f"island: {current_best_p.island_idx} → {program.island_idx})"
                )
            else:
                score = program.combined_score or 0.0
                log_msg += (
                    f" (gen: {program.generation}, score: {score:.4f}, initialized "
                    f"island: {program.island_idx})."
                )
            logger.info(log_msg)

    def print_summary(self, console=None) -> None:
        """Print a summary of the database contents using DatabaseDisplay."""
        if not hasattr(self, "_database_display"):
            self._database_display = self._get_database_display()
            self._database_display.set_last_iteration(self.last_iteration)

        if hasattr(self._database_display, "set_default_console"):
            self._database_display.set_default_console(self.display_console)
        self._database_display.print_summary(console)

    def _print_program_summary(self, program) -> None:
        """Print a rich summary of a newly added program using DatabaseDisplay."""
        if not hasattr(self, "_database_display"):
            self._database_display = self._get_database_display()

        if hasattr(self._database_display, "set_default_console"):
            self._database_display.set_default_console(self.display_console)
        self._database_display.print_program_summary(
            program, console=self.display_console
        )

    def set_display_console(self, console: Optional[Any]) -> None:
        """Set shared console used for rich DB summaries."""
        self.display_console = console
        if hasattr(self, "_database_display") and hasattr(
            self._database_display, "set_default_console"
        ):
            self._database_display.set_default_console(console)

    def check_scheduled_operations(self):
        """Run any operations that were scheduled during add but deferred for performance."""
        if self._schedule_migration:
            logger.info("Running scheduled migration operation")
            self.island_manager.perform_migration(self.last_iteration)
            self._schedule_migration = False

    def is_stagnant(self, current_generation: int) -> bool:
        """Check if evolution is stagnant based on generations without improvement.

        Args:
            current_generation: The current generation number

        Returns:
            True if stagnant (no improvement for stagnation_threshold generations)
        """
        if not self.enable_dynamic_islands:
            return False

        threshold = self.stagnation_threshold
        gens_since_improvement = current_generation - self.best_score_generation

        return gens_since_improvement >= threshold

    def check_and_spawn_island_if_stagnant(self, current_generation: int) -> bool:
        """Check for stagnation and spawn a new island if needed.

        Args:
            current_generation: The current generation number

        Returns:
            True if a new island was spawned, False otherwise
        """
        if not self.is_stagnant(current_generation):
            return False

        if not self.island_manager:
            logger.warning("Cannot spawn island: no island manager configured")
            return False

        # Spawn new island
        spawned = self.island_manager.spawn_new_island()

        if spawned:
            # Reset stagnation tracking
            self.best_score_generation = current_generation
            self._update_metadata_in_db(
                "best_score_generation", str(self.best_score_generation)
            )
            gens_stagnant = current_generation - self.best_score_generation
            logger.info(
                f"🏝️ Spawned new island due to stagnation "
                f"(no improvement for {gens_stagnant} generations)"
            )

        return spawned

    def close(self):
        """Closes the database connection."""
        if self.repository_bundle:
            self.repository_bundle.close()
            self.repository_bundle = None
        self.conn = None
        self.cursor = None
