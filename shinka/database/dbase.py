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
from .complexity import analyze_code_metrics
from .islands import CombinedIslandManager
from .island_sampler import create_island_sampler, IslandSampler
from .island_repository import IslandRepository
from .metadata_repository import MetadataRepository
from .display import DatabaseDisplay
from shinka.embed import EmbeddingClient
from shinka.defaults import default_archive_criteria

logger = logging.getLogger(__name__)


def _warn_repository_deprecation(method_name: str) -> None:
    warnings.warn(
        f"ProgramDatabase.{method_name}() is deprecated; use ProgramRepository instead.",
        DeprecationWarning,
        stacklevel=2,
    )


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
        config: DatabaseConfig,
        embedding_model: str = "text-embedding-3-small",
        read_only: bool = False,
    ):
        self.config = config
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

        db_path_str = getattr(self.config, "db_path", None)

        if db_path_str:
            db_file = Path(db_path_str).resolve()
            if not read_only:
                # Robustness check for unclean shutdown with WAL
                db_wal_file = Path(f"{db_file}-wal")
                db_shm_file = Path(f"{db_file}-shm")
                if (
                    db_file.exists()
                    and db_file.stat().st_size == 0
                    and (db_wal_file.exists() or db_shm_file.exists())
                ):
                    logger.warning(
                        f"Database file {db_file} is empty but WAL/SHM files "
                        "exist. This may indicate an unclean shutdown. "
                        "Removing WAL/SHM files to attempt recovery."
                    )
                    if db_wal_file.exists():
                        db_wal_file.unlink()
                    if db_shm_file.exists():
                        db_shm_file.unlink()
                db_file.parent.mkdir(parents=True, exist_ok=True)
                self.conn = sqlite3.connect(str(db_file), timeout=30.0)
                logger.debug(f"Connected to SQLite database: {db_file}")
            else:
                if not db_file.exists():
                    raise FileNotFoundError(
                        f"Database file not found for read-only connection: {db_file}"
                    )
                db_uri = f"file:{db_file}?mode=ro"
                self.conn = sqlite3.connect(db_uri, uri=True, timeout=30.0)
                logger.debug(
                    "Connected to SQLite database in read-only mode: %s",
                    db_file,
                )
        else:
            self.conn = sqlite3.connect(":memory:")
            logger.info("Initialized in-memory SQLite database.")

        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self.metadata_repo = MetadataRepository(
            conn=self.conn,
            cursor=self.cursor,
            read_only=self.read_only,
        )
        self.island_repo = IslandRepository(
            conn=self.conn,
            cursor=self.cursor,
            config=self.config,
        )
        if not self.read_only:
            self._create_tables()
        self._load_metadata_from_db()

        # Initialize island manager now that database is ready
        self.island_manager = CombinedIslandManager(
            cursor=self.cursor,
            conn=self.conn,
            config=self.config,
        )

        # Initialize island sampler with configured strategy
        island_selection_strategy = getattr(
            self.config, "island_selection_strategy", "uniform"
        )
        self.island_sampler = create_island_sampler(
            cursor=self.cursor,
            conn=self.conn,
            config=self.config,
            strategy=island_selection_strategy,
        )

        count = self._count_programs_in_db()
        logger.debug(f"DB initialized with {count} programs.")
        logger.debug(
            f"Last iter: {self.last_iteration}. Best ID: {self.best_program_id}"
        )

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

    def _create_tables(self):
        if not self.cursor or not self.conn:
            raise ConnectionError("DB not connected.")

        # Set SQLite pragmas for better performance and stability
        # Use WAL mode for better concurrency support and reduced locking
        self.cursor.execute("PRAGMA journal_mode = WAL;")
        self.cursor.execute("PRAGMA busy_timeout = 30000;")  # 30 second busy timeout
        self.cursor.execute(
            "PRAGMA wal_autocheckpoint = 1000;"
        )  # Checkpoint every 1000 pages
        self.cursor.execute("PRAGMA synchronous = NORMAL;")  # Safer, faster
        self.cursor.execute("PRAGMA cache_size = -64000;")  # 64MB cache
        self.cursor.execute("PRAGMA temp_store = MEMORY;")
        self.cursor.execute("PRAGMA foreign_keys = ON;")  # For data integrity

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS programs (
                id TEXT PRIMARY KEY,
                code TEXT NOT NULL,
                language TEXT NOT NULL,
                parent_id TEXT,
                archive_inspiration_ids TEXT,  -- JSON serialized List[str]
                top_k_inspiration_ids TEXT,    -- JSON serialized List[str]
                generation INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                code_diff TEXT,     -- Stores edit difference
                combined_score REAL,
                public_metrics TEXT, -- JSON serialized Dict[str, Any]
                private_metrics TEXT, -- JSON serialized Dict[str, Any]
                text_feedback TEXT, -- Text feedback for the program
                complexity REAL,   -- Calculated complexity metric
                embedding TEXT,    -- JSON serialized List[float]
                embedding_pca_2d TEXT, -- JSON serialized List[float]
                embedding_pca_3d TEXT, -- JSON serialized List[float]
                embedding_cluster_id INTEGER,
                correct BOOLEAN DEFAULT 0,  -- Correct (0=False, 1=True)
                children_count INTEGER NOT NULL DEFAULT 0,
                metadata TEXT,      -- JSON serialized Dict[str, Any]
                migration_history TEXT, -- JSON of migration events
                island_idx INTEGER,  -- Add island_idx to the schema
                system_prompt_id TEXT  -- ID of system prompt that generated this program
            )
            """
        )

        # Add indices for common query patterns
        idx_cmds = [
            "CREATE INDEX IF NOT EXISTS idx_programs_generation ON "
            "programs(generation)",
            "CREATE INDEX IF NOT EXISTS idx_programs_timestamp ON programs(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_programs_complexity ON "
            "programs(complexity)",
            "CREATE INDEX IF NOT EXISTS idx_programs_parent_id ON programs(parent_id)",
            "CREATE INDEX IF NOT EXISTS idx_programs_children_count ON "
            "programs(children_count)",
            "CREATE INDEX IF NOT EXISTS idx_programs_island_idx ON "
            "programs(island_idx)",
            "CREATE INDEX IF NOT EXISTS idx_programs_system_prompt_id ON "
            "programs(system_prompt_id)",
        ]
        for cmd in idx_cmds:
            self.cursor.execute(cmd)

        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS archive (
                program_id TEXT PRIMARY KEY,
                FOREIGN KEY (program_id) REFERENCES programs(id)
                    ON DELETE CASCADE
            )
            """
        )

        if self.metadata_repo is None:
            raise ConnectionError("Metadata repository not initialized.")
        self.metadata_repo.ensure_schema()

        self.conn.commit()

        # Run any necessary migrations
        self._run_migrations()

        logger.debug("Database tables and indices ensured to exist.")

    def _run_migrations(self):
        """Run database migrations for schema changes."""
        if not self.cursor or not self.conn:
            raise ConnectionError("DB not connected.")

        # Get current columns
        self.cursor.execute("PRAGMA table_info(programs)")
        columns = [row[1] for row in self.cursor.fetchall()]

        # Migration 1: Add text_feedback column if it doesn't exist
        try:
            if "text_feedback" not in columns:
                logger.info("Adding text_feedback column to programs table")
                self.cursor.execute(
                    "ALTER TABLE programs ADD COLUMN text_feedback TEXT DEFAULT ''"
                )
                self.conn.commit()
                logger.info("Successfully added text_feedback column")
        except sqlite3.Error as e:
            logger.error(f"Error during text_feedback migration: {e}")
            # Don't raise - this is not critical for existing functionality

        # Migration 2: Add system_prompt_id column if it doesn't exist
        try:
            if "system_prompt_id" not in columns:
                logger.info("Adding system_prompt_id column to programs table")
                self.cursor.execute(
                    "ALTER TABLE programs ADD COLUMN system_prompt_id TEXT"
                )
                self.conn.commit()
                logger.info("Successfully added system_prompt_id column")
        except sqlite3.Error as e:
            logger.error(f"Error during system_prompt_id migration: {e}")

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
        if not self.cursor:
            return 0
        self.cursor.execute("SELECT COUNT(*) FROM programs")
        return (self.cursor.fetchone() or {"COUNT(*)": 0})["COUNT(*)"]

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
        if not self.cursor or not self.conn:
            raise ConnectionError("DB not connected.")

        self.island_manager.assign_island(program)

        # Calculate complexity if not pre-set (or if default 0.0)
        if program.complexity == 0.0:
            try:
                code_metrics = analyze_code_metrics(program.code, program.language)
                program.complexity = code_metrics.get("complexity_score", 0.0)
                if program.metadata is None:
                    program.metadata = {}
                program.metadata["code_analysis_metrics"] = code_metrics
            except Exception as e:
                logger.warning(
                    f"Could not calculate complexity for program {program.id}: {e}"
                )
                program.complexity = float(len(program.code))  # Fallback to length

        # Embedding is expected to be provided by the user.
        # Ensure program.embedding is a list, even if empty.
        if not isinstance(program.embedding, list):
            logger.warning(
                f"Program {program.id} embedding is not a list, "
                "defaulting to empty list."
            )
            program.embedding = []

        # Pre-serialize all JSON data once
        public_metrics_json = json.dumps(program.public_metrics or {})
        private_metrics_json = json.dumps(program.private_metrics or {})
        metadata_json = json.dumps(program.metadata or {})
        archive_insp_ids_json = json.dumps(program.archive_inspiration_ids or [])
        top_k_insp_ids_json = json.dumps(program.top_k_inspiration_ids or [])
        embedding_json = json.dumps(program.embedding)  # Serialize embedding
        embedding_pca_2d_json = json.dumps(program.embedding_pca_2d or [])
        embedding_pca_3d_json = json.dumps(program.embedding_pca_3d or [])
        migration_history_json = json.dumps(program.migration_history or [])

        # Handle text_feedback - convert to string if it's a list
        text_feedback_str = program.text_feedback
        if isinstance(text_feedback_str, list):
            # Join list items with newlines for readability
            text_feedback_str = "\n".join(str(item) for item in text_feedback_str)
        elif text_feedback_str is None:
            text_feedback_str = ""
        else:
            text_feedback_str = str(text_feedback_str)

        # Begin transaction - this improves performance by batching operations
        self.conn.execute("BEGIN TRANSACTION")

        try:
            # Insert the program in a single operation
            self.cursor.execute(
                """
                INSERT INTO programs
                   (id, code, language, parent_id, archive_inspiration_ids,
                    top_k_inspiration_ids, generation, timestamp, code_diff,
                    combined_score, public_metrics, private_metrics,
                    text_feedback, complexity, embedding, embedding_pca_2d,
                    embedding_pca_3d, embedding_cluster_id, correct,
                    children_count, metadata, island_idx, migration_history,
                    system_prompt_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    program.id,
                    program.code,
                    program.language,
                    program.parent_id,
                    archive_insp_ids_json,
                    top_k_insp_ids_json,
                    program.generation,
                    program.timestamp,
                    program.code_diff,
                    program.combined_score,
                    public_metrics_json,
                    private_metrics_json,
                    text_feedback_str,
                    program.complexity,
                    embedding_json,  # Use serialized embedding
                    embedding_pca_2d_json,
                    embedding_pca_3d_json,
                    program.embedding_cluster_id,
                    program.correct,
                    program.children_count,
                    metadata_json,
                    program.island_idx,
                    migration_history_json,
                    program.system_prompt_id,
                ),
            )

            # Increment parent's children_count
            if program.parent_id:
                self.cursor.execute(
                    "UPDATE programs SET children_count = children_count + 1 "
                    "WHERE id = ?",
                    (program.parent_id,),
                )

            # Commit the main program insertion and related operations
            self.conn.commit()
            logger.info(
                "Program %s added to DB - score: %s.",
                program.id,
                program.combined_score,
            )

        except sqlite3.IntegrityError as e:
            self.conn.rollback()
            logger.error(f"IntegrityError for program {program.id}: {e}")
            raise
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error adding program {program.id}: {e}")
            raise

        # Update best program tracking
        self._update_best_program(program)

        # Recompute embeddings and clusters for all programs
        self._recompute_embeddings_and_clusters()

        # Update generation tracking
        if program.generation > self.last_iteration:
            self.last_iteration = program.generation
            self._update_metadata_in_db("last_iteration", str(self.last_iteration))

        # Print verbose summary if requested
        if verbose:
            self._print_program_summary(program)

        # Check if this program needs to be copied to other islands
        if self.island_manager.needs_island_copies(program):
            logger.info(
                f"Creating copies of initial program {program.id} for all islands"
            )
            self.island_manager.copy_program_to_islands(program)
            # Remove the flag from the original program's metadata
            if program.metadata:
                program.metadata.pop("_needs_island_copies", None)
                metadata_json = json.dumps(program.metadata)
                self.cursor.execute(
                    "UPDATE programs SET metadata = ? WHERE id = ?",
                    (metadata_json, program.id),
                )
                self.conn.commit()

        # Check if migration should be scheduled
        if self.island_manager.should_schedule_migration(program):
            self._schedule_migration = True

        # Check for stagnation and spawn new island if needed
        self.check_and_spawn_island_if_stagnant(program.generation)

        self.check_scheduled_operations()
        return program.id

    def _program_from_row(self, row: sqlite3.Row) -> Optional[Program]:
        """Helper to create a Program object from a database row."""
        if not row:
            return None

        program_data = dict(row)

        # Use faster json loads
        public_metrics_text = program_data.get("public_metrics")
        if public_metrics_text:
            try:
                program_data["public_metrics"] = json.loads(public_metrics_text)
            except json.JSONDecodeError:
                program_data["public_metrics"] = {}
        else:
            program_data["public_metrics"] = {}

        private_metrics_text = program_data.get("private_metrics")
        if private_metrics_text:
            try:
                program_data["private_metrics"] = json.loads(private_metrics_text)
            except json.JSONDecodeError:
                program_data["private_metrics"] = {}
        else:
            program_data["private_metrics"] = {}

        # Same for metadata
        metadata_text = program_data.get("metadata")
        if metadata_text:
            try:
                program_data["metadata"] = json.loads(metadata_text)
            except json.JSONDecodeError:
                program_data["metadata"] = {}
        else:
            program_data["metadata"] = {}

        # Handle text_feedback (simple string field)
        if "text_feedback" not in program_data or program_data["text_feedback"] is None:
            program_data["text_feedback"] = ""

        # Handle inspiration_ids
        archive_insp_ids_text = program_data.get("archive_inspiration_ids")
        if archive_insp_ids_text:
            try:
                program_data["archive_inspiration_ids"] = json.loads(
                    archive_insp_ids_text
                )
            except json.JSONDecodeError:
                program_data["archive_inspiration_ids"] = []
        else:
            program_data["archive_inspiration_ids"] = []

        top_k_insp_ids_text = program_data.get("top_k_inspiration_ids")
        if top_k_insp_ids_text:
            try:
                program_data["top_k_inspiration_ids"] = json.loads(top_k_insp_ids_text)
            except json.JSONDecodeError:
                logger.warning(
                    "Could not decode top_k_inspiration_ids for "
                    f"program {program_data.get('id')}. "
                    "Defaulting to empty list."
                )
                program_data["top_k_inspiration_ids"] = []
        else:
            program_data["top_k_inspiration_ids"] = []

        # Handle embedding
        embedding_text = program_data.get("embedding")
        if embedding_text:
            try:
                program_data["embedding"] = json.loads(embedding_text)
            except json.JSONDecodeError:
                logger.warning(
                    f"Could not decode embedding for program "
                    f"{program_data.get('id')}. Defaulting to empty list."
                )
                program_data["embedding"] = []
        else:
            program_data["embedding"] = []

        embedding_pca_2d_text = program_data.get("embedding_pca_2d")
        if embedding_pca_2d_text:
            try:
                program_data["embedding_pca_2d"] = json.loads(embedding_pca_2d_text)
            except json.JSONDecodeError:
                program_data["embedding_pca_2d"] = []
        else:
            program_data["embedding_pca_2d"] = []

        embedding_pca_3d_text = program_data.get("embedding_pca_3d")
        if embedding_pca_3d_text:
            try:
                program_data["embedding_pca_3d"] = json.loads(embedding_pca_3d_text)
            except json.JSONDecodeError:
                program_data["embedding_pca_3d"] = []
        else:
            program_data["embedding_pca_3d"] = []

        # Handle migration_history
        migration_history_text = program_data.get("migration_history")
        if migration_history_text:
            try:
                program_data["migration_history"] = json.loads(migration_history_text)
            except json.JSONDecodeError:
                logger.warning(
                    f"Could not decode migration_history for program "
                    f"{program_data.get('id')}. Defaulting to empty list."
                )
                program_data["migration_history"] = []
        else:
            program_data["migration_history"] = []

        # Handle archive status
        program_data["in_archive"] = bool(program_data.get("in_archive", 0))

        return Program.from_dict(program_data)

    def _get_program_internal(self, program_id: str) -> Optional[Program]:
        if not self.cursor:
            raise ConnectionError("DB not connected.")
        self.cursor.execute("SELECT * FROM programs WHERE id = ?", (program_id,))
        row = self.cursor.fetchone()
        return self._program_from_row(row)

    def _get_ancestry_internal(
        self, program_id: str, max_ancestors: int = 10
    ) -> List[Program]:
        if not self.cursor:
            raise ConnectionError("DB not connected.")

        ancestors: List[Program] = []
        current_id = program_id

        for _ in range(max_ancestors):
            self.cursor.execute(
                "SELECT parent_id FROM programs WHERE id = ?", (current_id,)
            )
            row = self.cursor.fetchone()
            if not row or not row["parent_id"]:
                break

            parent_id = row["parent_id"]
            parent = self._get_program_internal(parent_id)
            if parent is None:
                break
            ancestors.append(parent)
            current_id = parent_id

        ancestors.reverse()
        return ancestors

    @db_retry()
    def get(self, program_id: str) -> Optional[Program]:
        """Get a program by its ID."""
        _warn_repository_deprecation("get")
        return self._get_program_internal(program_id)

    @db_retry()
    def get_ancestry(self, program_id: str, max_ancestors: int = 10) -> List[Program]:
        """
        Get the ancestry (lineage) of a program by walking up the parent chain.

        Args:
            program_id: ID of the program to get ancestry for
            max_ancestors: Maximum number of ancestors to retrieve

        Returns:
            List of ancestor programs, sorted chronologically (oldest first)
        """
        _warn_repository_deprecation("get_ancestry")
        ancestors = self._get_ancestry_internal(
            program_id=program_id,
            max_ancestors=max_ancestors,
        )

        if ancestors:
            logger.info(
                f"Retrieved {len(ancestors)} ancestors for program {program_id} "
                f"(generations: {[p.generation for p in ancestors]})"
            )

        return ancestors

    @db_retry()
    def sample(
        self,
        target_generation=None,
        novelty_attempt=None,
        max_novelty_attempts=None,
        resample_attempt=None,
        max_resample_attempts=None,
    ) -> Tuple[Program, List[Program], List[Program]]:
        """
        Main sampling entry point used by the runner.

        The sequence is:
        1. choose an island
        2. choose a parent within that island using the configured strategy
        3. choose archive inspirations
        4. choose top-k inspirations

        This means "search policy" is distributed across:
        - island sampling
        - parent selection
        - inspiration selection

        If you want finer control, this method is a good place to split into
        smaller public building blocks.
        """
        logger.warning(
            "ProgramDatabase.sample() is deprecated. Using repository-backed context sampling."
        )
        context = self._sample_context_via_repository(
            target_generation=target_generation,
            novelty_attempt=novelty_attempt,
            max_novelty_attempts=max_novelty_attempts,
            resample_attempt=resample_attempt,
            max_resample_attempts=max_resample_attempts,
            with_fix_mode=False,
        )
        self._print_sampling_summary_helper(
            context.parent,
            context.archive_inspirations,
            context.top_k_inspirations,
            target_generation,
            novelty_attempt,
            max_novelty_attempts,
            resample_attempt,
            max_resample_attempts,
        )
        return (
            context.parent,
            context.archive_inspirations,
            context.top_k_inspirations,
        )

    @db_retry()
    def sample_with_fix_mode(
        self,
        target_generation=None,
        novelty_attempt=None,
        max_novelty_attempts=None,
        resample_attempt=None,
        max_resample_attempts=None,
    ) -> Tuple[Program, List[Program], List[Program], bool]:
        """
        Sample a parent program, returning fix mode indicator if no correct
        programs exist.

        Returns:
            Tuple of (parent, archive_inspirations, top_k_inspirations, needs_fix)
            where needs_fix is True if no correct programs exist and fix mode
            should be used.
        """
        logger.warning(
            "ProgramDatabase.sample_with_fix_mode() is deprecated. Using repository-backed context sampling."
        )
        context = self._sample_context_via_repository(
            target_generation=target_generation,
            novelty_attempt=novelty_attempt,
            max_novelty_attempts=max_novelty_attempts,
            resample_attempt=resample_attempt,
            max_resample_attempts=max_resample_attempts,
            with_fix_mode=True,
        )
        if context.needs_fix:
            self._print_sampling_summary_helper(
                context.parent,
                [],
                [],
                target_generation,
                novelty_attempt,
                max_novelty_attempts,
                resample_attempt,
                max_resample_attempts,
                ancestor_inspirations=context.archive_inspirations,
                is_fix_mode=True,
            )
        else:
            self._print_sampling_summary_helper(
                context.parent,
                context.archive_inspirations,
                context.top_k_inspirations,
                target_generation,
                novelty_attempt,
                max_novelty_attempts,
                resample_attempt,
                max_resample_attempts,
            )
        return (
            context.parent,
            context.archive_inspirations,
            context.top_k_inspirations,
            context.needs_fix,
        )

    def _print_sampling_summary_helper(
        self,
        parent,
        archive_inspirations,
        top_k_inspirations,
        target_generation=None,
        novelty_attempt=None,
        max_novelty_attempts=None,
        resample_attempt=None,
        max_resample_attempts=None,
        ancestor_inspirations=None,
        is_fix_mode=False,
    ):
        """Helper method to print sampling summary."""
        if not hasattr(self, "_database_display"):
            self._database_display = DatabaseDisplay(
                cursor=self.cursor,
                conn=self.conn,
                config=self.config,
                island_manager=self.island_manager,
                count_programs_func=self._count_programs_in_db,
                get_best_program_func=self._get_best_program_internal,
                default_console=self.display_console,
            )

        self._database_display.print_sampling_summary(
            parent,
            archive_inspirations,
            top_k_inspirations,
            target_generation,
            novelty_attempt,
            max_novelty_attempts,
            resample_attempt,
            max_resample_attempts,
            ancestor_inspirations,
            is_fix_mode,
            console=self.display_console,
        )

    def _sample_context_via_repository(
        self,
        *,
        target_generation=None,
        novelty_attempt=None,
        max_novelty_attempts=None,
        resample_attempt=None,
        max_resample_attempts=None,
        with_fix_mode: bool,
    ):
        if not self.config.db_path:
            raise RuntimeError(
                "Repository-backed context sampling requires config.db_path."
            )

        from .repository import ProgramRepository
        from shinka.core.context_sampler import ContextSampler

        repository = ProgramRepository.from_config(
            self.config,
            embedding_model=self.embedding_model,
            read_only=True,
        )
        try:
            sampler = ContextSampler(repository)
            return sampler.sample(
                target_generation=target_generation,
                novelty_attempt=novelty_attempt,
                max_novelty_attempts=max_novelty_attempts,
                resample_attempt=resample_attempt,
                max_resample_attempts=max_resample_attempts,
                with_fix_mode=with_fix_mode,
            )
        finally:
            repository.close()

    def _get_best_program_internal(self, metric: Optional[str] = None) -> Optional[Program]:
        from .repository import ProgramRepository

        repository = ProgramRepository.from_config(
            self.config,
            embedding_model=self.embedding_model,
            read_only=True,
        )
        try:
            best = repository.get_best(metric=metric)
            if best is not None and self.best_program_id != best.id:
                self.best_program_id = best.id
            return best
        finally:
            repository.close()

    @db_retry()
    def get_best_program(self, metric: Optional[str] = None) -> Optional[Program]:
        _warn_repository_deprecation("get_best_program")
        return self._get_best_program_internal(metric=metric)

    @db_retry()
    def get_all_programs(self) -> List[Program]:
        """Get all programs from the database."""
        _warn_repository_deprecation("get_all_programs")
        from .repository import ProgramRepository

        repository = ProgramRepository.from_config(
            self.config,
            embedding_model=self.embedding_model,
            read_only=True,
        )
        try:
            return repository.list_all()
        finally:
            repository.close()

    @db_retry()
    def get_programs_summary(self) -> List[Dict[str, Any]]:
        """
        Get lightweight summary of all programs for visualization.
        Excludes heavy fields like code, embeddings, and large metadata.
        Returns raw dicts instead of Program objects for efficiency.
        """
        _warn_repository_deprecation("get_programs_summary")
        from .repository import ProgramRepository

        repository = ProgramRepository.from_config(
            self.config,
            embedding_model=self.embedding_model,
            read_only=True,
        )
        try:
            return repository.get_summaries()
        finally:
            repository.close()

    @db_retry()
    def get_program_count_and_timestamp(self) -> Dict[str, Any]:
        """
        Get program count and max timestamp for efficient change detection.
        Used by auto-refresh to check if data has changed without loading all programs.
        """
        _warn_repository_deprecation("get_program_count_and_timestamp")
        from .repository import ProgramRepository

        repository = ProgramRepository.from_config(
            self.config,
            embedding_model=self.embedding_model,
            read_only=True,
        )
        try:
            snapshot = repository.get_count_snapshot()
            return {"count": snapshot.count, "max_timestamp": snapshot.max_timestamp}
        finally:
            repository.close()

    @db_retry()
    def get_programs_by_generation(self, generation: int) -> List[Program]:
        """Get all programs from a specific generation."""
        _warn_repository_deprecation("get_programs_by_generation")
        from .repository import ProgramRepository

        repository = ProgramRepository.from_config(
            self.config,
            embedding_model=self.embedding_model,
            read_only=True,
        )
        try:
            return repository.list_by_generation(generation)
        finally:
            repository.close()

    @db_retry()
    def get_top_programs(
        self,
        n: int = 10,
        metric: Optional[str] = "combined_score",
        correct_only: bool = False,
    ) -> List[Program]:
        """Get top programs, using SQL for sorting when possible."""
        _warn_repository_deprecation("get_top_programs")
        from .repository import ProgramRepository

        repository = ProgramRepository.from_config(
            self.config,
            embedding_model=self.embedding_model,
            read_only=True,
        )
        try:
            return repository.list_top(n=n, metric=metric, correct_only=correct_only)
        finally:
            repository.close()

    def save(self, path: Optional[str] = None) -> None:
        if not self.conn or not self.cursor:
            logger.warning("No DB connection, skipping save.")
            return

        # Main purpose here is to save/commit metadata like last_iteration.
        current_db_file_path_str = self.config.db_path
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
        if self.conn:
            db_display_name = self.config.db_path or ":memory:"
            logger.info(f"Closing existing connection to '{db_display_name}'.")
            self.conn.close()

        db_path_obj = Path(path).resolve()
        # Robustness check for unclean shutdown with WAL
        db_wal_file = Path(f"{db_path_obj}-wal")
        db_shm_file = Path(f"{db_path_obj}-shm")
        if (
            db_path_obj.exists()
            and db_path_obj.stat().st_size == 0
            and (db_wal_file.exists() or db_shm_file.exists())
        ):
            logger.warning(
                f"Database file {db_path_obj} is empty but WAL/SHM files "
                "exist. This may indicate an unclean shutdown. Removing "
                "WAL/SHM files to attempt recovery.",
                db_path_obj,
            )
            if db_wal_file.exists():
                db_wal_file.unlink()
            if db_shm_file.exists():
                db_shm_file.unlink()

        self.config.db_path = str(db_path_obj)  # Update config

        if not db_path_obj.exists():
            logger.warning(
                f"DB file '{db_path_obj}' not found. New DB created if writes occur."
            )
            db_path_obj.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(db_path_obj), timeout=30.0)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self._create_tables()
        self._load_metadata_from_db()

        count = self._count_programs_in_db()
        logger.info(
            f"Loaded DB from '{db_path_obj}'. {count} programs. "
            f"Last iter: {self.last_iteration}."
        )

    def _get_criterion_value(self, program: Program, criterion: str) -> float:
        """
        Get the value of a specific criterion for a program.

        Supported criteria:
            - combined_score: The program's combined fitness score
            - loc: Lines of code
            - lloc: Logical lines of code
            - complexity: Cyclomatic complexity
            - maintainability: Maintainability index
            - nesting: Maximum nesting depth
        """
        if criterion == "combined_score":
            return program.combined_score or 0.0

        # Get code analysis metrics from metadata
        metrics = {}
        if program.metadata:
            metrics = program.metadata.get("code_analysis_metrics", {})

        if criterion == "loc":
            return metrics.get(
                "lines_of_code", len(program.code.split("\n")) if program.code else 1
            )
        elif criterion == "lloc":
            return metrics.get(
                "logical_lines_of_code",
                len(program.code.split("\n")) if program.code else 1,
            )
        elif criterion == "complexity":
            return metrics.get("cyclomatic_complexity", 1.0)
        elif criterion == "maintainability":
            return metrics.get("maintainability_index", 100.0)
        elif criterion == "nesting":
            return metrics.get("max_nesting_depth", 1)

        # Unknown criterion - return 0
        logger.warning(f"Unknown archive criterion: {criterion}")
        return 0.0

    def _compute_archive_score_ranked(
        self, program: Program, archive_programs: List[Program]
    ) -> float:
        """
        Compute score using rank-based normalization for scale-invariant comparison.

        Each criterion is converted to a percentile rank (0-1) based on the
        archive population, then weighted according to archive_criteria config.

        Args:
            program: The program to score
            archive_programs: Current archive programs for rank computation

        Returns:
            Combined score where higher is always better
        """
        criteria = getattr(self.config, "archive_criteria", {"combined_score": 1.0})

        if not archive_programs:
            # No archive yet - just use the primary criterion
            primary_criterion = next(iter(criteria.keys()), "combined_score")
            primary_weight = criteria.get(primary_criterion, 1.0)
            value = self._get_criterion_value(program, primary_criterion)
            return value if primary_weight > 0 else -value

        all_programs = archive_programs + [program]

        score = 0.0
        for criterion, weight in criteria.items():
            # Get values for all programs
            values = [self._get_criterion_value(p, criterion) for p in all_programs]
            program_value = values[-1]  # The new program's value

            # Compute percentile rank (0 = worst, 1 = best)
            if weight > 0:
                # Higher is better: count how many are strictly worse
                rank = sum(1 for v in values if v < program_value) / len(values)
            else:
                # Lower is better: count how many are strictly worse (i.e., higher)
                rank = sum(1 for v in values if v > program_value) / len(values)
                weight = abs(weight)  # Use absolute weight after handling direction

            score += weight * rank

        return score

    def _get_archive_programs(self) -> List[Program]:
        """Deprecated persisted archive view. Computes archive from programs."""
        logger.debug(
            "ProgramDatabase._get_archive_programs() is deprecated; computing archive from persisted programs."
        )
        from .archive_policy import create_archive_policy
        from .repository import ProgramRepository

        policy = create_archive_policy(self.config)
        repository = ProgramRepository.from_config(
            self.config,
            embedding_model=self.embedding_model,
            read_only=True,
        )
        try:
            return policy.compute(repository.list_all())
        finally:
            repository.close()

    def _find_most_similar_in_archive(
        self, embedding: List[float]
    ) -> Optional[Program]:
        """
        Find the most similar program in the archive by embedding cosine similarity.
        Used for crowding-based archive selection.

        Args:
            embedding: The embedding vector to compare against

        Returns:
            The most similar program, or None if no valid comparisons possible
        """
        if not embedding:
            return None

        archive_programs = self._get_archive_programs()
        if not archive_programs:
            return None

        best_similarity = -float("inf")
        most_similar = None

        embedding_arr = np.array(embedding)
        embedding_norm = np.linalg.norm(embedding_arr)

        if embedding_norm < 1e-8:
            return None

        for prog in archive_programs:
            if not prog or not prog.embedding:
                continue

            prog_embedding = np.array(prog.embedding)
            prog_norm = np.linalg.norm(prog_embedding)

            if prog_norm < 1e-8:
                continue

            # Cosine similarity
            similarity = np.dot(embedding_arr, prog_embedding) / (
                embedding_norm * prog_norm
            )

            if similarity > best_similarity:
                best_similarity = similarity
                most_similar = prog

        return most_similar

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
        # First prioritize correctness (always)
        if program1.correct and not program2.correct:
            return True
        if program2.correct and not program1.correct:
            return False

        # Check if we should use multi-criteria ranked scoring
        criteria = getattr(self.config, "archive_criteria", {"combined_score": 1.0})
        use_ranked = archive_programs is not None and len(criteria) > 1

        if use_ranked:
            # Use rank-based scoring with archive context
            # Include both programs for fair ranking
            context = [
                p for p in archive_programs if p.id not in (program1.id, program2.id)
            ]
            s1 = self._compute_archive_score_ranked(program1, context)
            s2 = self._compute_archive_score_ranked(program2, context)

            if s1 != s2:
                return s1 > s2
        else:
            # Simple single-criterion comparison (original behavior)
            s1 = program1.combined_score
            s2 = program2.combined_score

            if s1 is not None and s2 is not None:
                if s1 != s2:
                    return s1 > s2
            elif s1 is not None:
                return True  # p1 has score, p2 doesn't
            elif s2 is not None:
                return False  # p2 has score, p1 doesn't

            # Fallback to average public metrics
            try:
                avg1 = (
                    sum(program1.public_metrics.values()) / len(program1.public_metrics)
                    if program1.public_metrics
                    else -float("inf")
                )
                avg2 = (
                    sum(program2.public_metrics.values()) / len(program2.public_metrics)
                    if program2.public_metrics
                    else -float("inf")
                )
                if avg1 != avg2:
                    return avg1 > avg2
            except Exception:
                pass

        # Tie-breaker: prefer newer programs
        return program1.timestamp > program2.timestamp

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
        if self.best_program_id:
            current_best_p = self._get_program_internal(self.best_program_id)

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
            self._database_display = DatabaseDisplay(
                cursor=self.cursor,
                conn=self.conn,
                config=self.config,
                island_manager=self.island_manager,
                count_programs_func=self._count_programs_in_db,
                get_best_program_func=self._get_best_program_internal,
                default_console=self.display_console,
            )
            self._database_display.set_last_iteration(self.last_iteration)

        if hasattr(self._database_display, "set_default_console"):
            self._database_display.set_default_console(self.display_console)
        self._database_display.print_summary(console)

    def _print_program_summary(self, program) -> None:
        """Print a rich summary of a newly added program using DatabaseDisplay."""
        if not hasattr(self, "_database_display"):
            self._database_display = DatabaseDisplay(
                cursor=self.cursor,
                conn=self.conn,
                config=self.config,
                island_manager=self.island_manager,
                count_programs_func=self._count_programs_in_db,
                get_best_program_func=self._get_best_program_internal,
                default_console=self.display_console,
            )

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
        if not getattr(self.config, "enable_dynamic_islands", False):
            return False

        threshold = getattr(self.config, "stagnation_threshold", 100)
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
        if self.conn:
            self.conn.close()

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        arr1 = np.array(vec1, dtype=np.float32)
        arr2 = np.array(vec2, dtype=np.float32)

        norm_a = np.linalg.norm(arr1)
        norm_b = np.linalg.norm(arr2)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = np.dot(arr1, arr2) / (norm_a * norm_b)
        return float(similarity)

    @db_retry()
    def compute_similarity_thread_safe(
        self, vec: List[float], island_idx: int
    ) -> List[float]:
        """
        Thread-safe version of similarity computation. Creates its own DB connection.
        """
        conn = None
        try:
            # Create a new connection for this thread
            conn = sqlite3.connect(
                self.config.db_path, check_same_thread=False, timeout=60.0
            )
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                "SELECT embedding FROM programs WHERE island_idx = ? AND embedding IS NOT NULL AND embedding != '[]'",
                (island_idx,),
            )
            rows = cursor.fetchall()

            if not rows:
                return []

            similarities = []
            for row in rows:
                db_embedding = json.loads(row["embedding"])
                if db_embedding:
                    sim = self._cosine_similarity(vec, db_embedding)
                    similarities.append(sim)
            return similarities

        except Exception as e:
            logger.error(f"Thread-safe similarity computation failed: {e}")
            raise
        finally:
            if conn:
                conn.close()

    @db_retry()
    def compute_similarity(
        self, code_embedding: List[float], island_idx: int
    ) -> List[float]:
        """
        Compute cosine similarity against stored code embeddings in one island.

        This is the primary novelty test used by `NoveltyJudge`.
        Nothing semantic happens here; it is purely an embedding-nearest-neighbor
        lookup. The novelty gate later compares `max(similarities)` against
        `EvolutionConfig.code_embed_sim_threshold`.

        Args:
            code_embedding: The embedding to compare against
            island_idx: The island index to constrain the search to

        Returns:
            List of similarity scores (cosine similarity between 0 and 1)
        """
        if not self.cursor:
            raise ConnectionError("DB not connected.")

        if not code_embedding:
            logger.warning("Empty code embedding provided to compute_similarity")
            return []

        # Novelty is intentionally local to the island, not global to the whole
        # run. That allows different islands to explore similar ideas without
        # immediately rejecting each other as duplicates.
        self.cursor.execute(
            """
            SELECT id, embedding FROM programs 
            WHERE island_idx = ? AND embedding IS NOT NULL AND embedding != '[]'
            """,
            (island_idx,),
        )
        rows = self.cursor.fetchall()

        if not rows:
            logger.debug(f"No programs with embeddings found in island {island_idx}")
            return []

        # Extract embeddings and compute similarities
        similarity_scores = []
        for row in rows:
            try:
                embedding = json.loads(row["embedding"])
                if embedding:  # Skip empty embeddings
                    similarity = self._cosine_similarity(code_embedding, embedding)
                    similarity_scores.append(similarity)
                else:
                    similarity_scores.append(0.0)
            except json.JSONDecodeError:
                logger.warning(f"Could not decode embedding for program {row['id']}")
                similarity_scores.append(0.0)
                continue

        logger.debug(
            f"Computed {len(similarity_scores)} similarity scores for "
            f"island {island_idx}"
        )
        return similarity_scores

    @db_retry()
    def get_most_similar_program(
        self, code_embedding: List[float], island_idx: int
    ) -> Optional[Program]:
        """
        Get the nearest embedded neighbor in the current island.

        This is used only when embedding similarity already says "too close" and
        the optional novelty LLM needs an exact comparison target.

        Args:
            code_embedding: The embedding to compare against
            island_idx: The island index to constrain the search to

        Returns:
            The most similar Program object, or None if no programs found
        """
        if not self.cursor:
            raise ConnectionError("DB not connected.")

        if not code_embedding:
            logger.warning("Empty code embedding provided to get_most_similar_program")
            return None

        # Get all programs in the specified island that have embeddings
        self.cursor.execute(
            """
            SELECT id, embedding FROM programs 
            WHERE island_idx = ? AND embedding IS NOT NULL AND embedding != '[]'
            """,
            (island_idx,),
        )
        rows = self.cursor.fetchall()

        if not rows:
            logger.debug(f"No programs with embeddings found in island {island_idx}")
            return None

        # Find the program with highest similarity
        max_similarity = -1.0
        most_similar_id = None

        for row in rows:
            try:
                embedding = json.loads(row["embedding"])
                if embedding:  # Skip empty embeddings
                    similarity = self._cosine_similarity(code_embedding, embedding)
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_id = row["id"]
            except json.JSONDecodeError:
                logger.warning(f"Could not decode embedding for program {row['id']}")
                continue

        if most_similar_id:
            return self._get_program_internal(most_similar_id)
        return None

    @db_retry()
    def get_most_similar_program_thread_safe(
        self, code_embedding: List[float], island_idx: int
    ) -> Optional[Program]:
        """
        Thread-safe version of get_most_similar_program that creates its own DB connection.

        Args:
            code_embedding: The embedding to compare against
            island_idx: The island index to constrain the search to

        Returns:
            The most similar Program object, or None if not found
        """
        if not code_embedding:
            logger.warning(
                "Empty code embedding provided to get_most_similar_program_thread_safe"
            )
            return None

        conn = None
        try:
            # Create a new connection for this thread
            conn = sqlite3.connect(
                self.config.db_path, check_same_thread=False, timeout=60.0
            )
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get all programs in the specified island that have embeddings
            cursor.execute(
                """
                SELECT id, embedding FROM programs 
                WHERE island_idx = ? AND embedding IS NOT NULL AND embedding != '[]'
                """,
                (island_idx,),
            )

            rows = cursor.fetchall()
            if not rows:
                return None

            # Compute similarities
            import numpy as np

            similarities = []
            program_ids = []

            for row in rows:
                try:
                    embedding = json.loads(row["embedding"])
                    if embedding:  # Check if embedding is not empty
                        similarity = np.dot(code_embedding, embedding) / (
                            np.linalg.norm(code_embedding) * np.linalg.norm(embedding)
                        )
                        similarities.append(similarity)
                        program_ids.append(row["id"])
                except (json.JSONDecodeError, ValueError, ZeroDivisionError) as e:
                    logger.warning(
                        f"Error computing similarity for program {row['id']}: {e}"
                    )
                    continue

            if not similarities:
                return None

            # Find the most similar program
            max_similarity_idx = np.argmax(similarities)
            most_similar_id = program_ids[max_similarity_idx]

            # Get the full program data
            cursor.execute("SELECT * FROM programs WHERE id = ?", (most_similar_id,))
            row = cursor.fetchone()

            if row:
                return self._program_from_row(row)
            return None

        except Exception as e:
            logger.error(f"Error in get_most_similar_program_thread_safe: {e}")
            return None
        finally:
            if conn:
                conn.close()

    @db_retry()
    def _recompute_embeddings_and_clusters(self, num_clusters: int = 4):
        if self.read_only:
            return
        if not self.cursor or not self.conn:
            raise ConnectionError("DB not connected.")

        self.cursor.execute(
            "SELECT id, embedding FROM programs "
            "WHERE embedding IS NOT NULL AND embedding != '[]'"
        )
        rows = self.cursor.fetchall()

        if len(rows) < num_clusters:
            logger.info(
                f"Not enough programs with embeddings ({len(rows)}) to "
                f"perform clustering. Need at least {num_clusters}."
            )
            return

        program_ids = [row["id"] for row in rows]
        embeddings = [json.loads(row["embedding"]) for row in rows]
        embedding_client = self._ensure_embedding_client()
        if embedding_client is None:
            return

        # Use EmbeddingClient for dim reduction and clustering
        try:
            logger.info(
                "Recomputing PCA-reduced embedding features for %s programs.",
                len(program_ids),
            )
            reduced_2d = embedding_client.get_dim_reduction(
                embeddings, method="pca", dims=2
            )
            reduced_3d = embedding_client.get_dim_reduction(
                embeddings, method="pca", dims=3
            )
            cluster_ids = embedding_client.get_embedding_clusters(
                embeddings, num_clusters=num_clusters
            )
        except Exception as e:
            logger.error(f"Failed to recompute embedding features: {e}")
            return

        # Update all programs in a single transaction
        self.conn.execute("BEGIN TRANSACTION")
        try:
            for i, program_id in enumerate(program_ids):
                embedding_pca_2d_json = json.dumps(reduced_2d[i].tolist())
                embedding_pca_3d_json = json.dumps(reduced_3d[i].tolist())
                cluster_id = int(cluster_ids[i])

                self.cursor.execute(
                    """
                    UPDATE programs
                    SET embedding_pca_2d = ?,
                        embedding_pca_3d = ?,
                        embedding_cluster_id = ?
                    WHERE id = ?
                    """,
                    (
                        embedding_pca_2d_json,
                        embedding_pca_3d_json,
                        cluster_id,
                        program_id,
                    ),
                )
            self.conn.commit()
            logger.info(
                "Successfully updated embedding features for %s programs.",
                len(program_ids),
            )
        except Exception as e:
            self.conn.rollback()
            logger.error("Failed to update programs with new embedding features: %s", e)

    @db_retry()
    def _recompute_embeddings_and_clusters_thread_safe(self, num_clusters: int = 4):
        """
        Thread-safe version of embedding recomputation. Creates its own DB connection.
        """
        if self.read_only:
            return

        conn = None
        try:
            # Create a new connection for this thread
            conn = sqlite3.connect(
                self.config.db_path, check_same_thread=False, timeout=60.0
            )
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                "SELECT id, embedding FROM programs "
                "WHERE embedding IS NOT NULL AND embedding != '[]'"
            )
            rows = cursor.fetchall()

            if len(rows) < num_clusters:
                if len(rows) > 0:
                    logger.info(
                        f"Not enough programs with embeddings ({len(rows)}) to "
                        f"perform clustering. Need at least {num_clusters}."
                    )
                return

            program_ids = [row["id"] for row in rows]
            embeddings = [json.loads(row["embedding"]) for row in rows]
            embedding_client = self._ensure_embedding_client()
            if embedding_client is None:
                return

            # Use EmbeddingClient for dim reduction and clustering
            try:
                logger.info(
                    "Recomputing PCA-reduced embedding features for %s programs.",
                    len(program_ids),
                )

                logger.info("Computing 2D PCA reduction...")
                reduced_2d = embedding_client.get_dim_reduction(
                    embeddings, method="pca", dims=2
                )
                logger.info("2D PCA reduction completed")

                logger.info("Computing 3D PCA reduction...")
                reduced_3d = embedding_client.get_dim_reduction(
                    embeddings, method="pca", dims=3
                )
                logger.info("3D PCA reduction completed")

                logger.info(f"Computing GMM clustering with {num_clusters} clusters...")
                cluster_ids = embedding_client.get_embedding_clusters(
                    embeddings, num_clusters=num_clusters
                )
                logger.info("GMM clustering completed")
            except Exception as e:
                logger.error(f"Failed to recompute embedding features: {e}")
                return

            # Update all programs in a single transaction
            conn.execute("BEGIN TRANSACTION")
            try:
                for i, program_id in enumerate(program_ids):
                    embedding_pca_2d_json = json.dumps(reduced_2d[i].tolist())
                    embedding_pca_3d_json = json.dumps(reduced_3d[i].tolist())
                    cluster_id = int(cluster_ids[i])

                    cursor.execute(
                        """
                        UPDATE programs
                        SET embedding_pca_2d = ?,
                            embedding_pca_3d = ?,
                            embedding_cluster_id = ?
                        WHERE id = ?
                        """,
                        (
                            embedding_pca_2d_json,
                            embedding_pca_3d_json,
                            cluster_id,
                            program_id,
                        ),
                    )
                conn.commit()
                logger.info(
                    "Successfully updated embedding features for %s programs.",
                    len(program_ids),
                )
            except Exception as e:
                conn.rollback()
                logger.error(
                    "Failed to update programs with new embedding features: %s", e
                )
                raise  # Re-raise exception

        except Exception as e:
            logger.error(f"Thread-safe embedding recomputation failed: {e}")
            raise  # Re-raise exception

        finally:
            if conn:
                conn.close()

    @db_retry()
    def get_programs_by_generation_thread_safe(self, generation: int) -> List[Program]:
        """Thread-safe version of get_programs_by_generation."""
        _warn_repository_deprecation("get_programs_by_generation_thread_safe")
        from .repository import ProgramRepository

        repository = ProgramRepository.from_config(
            self.config,
            embedding_model=self.embedding_model,
            read_only=True,
        )
        try:
            return repository.list_by_generation(generation)
        finally:
            repository.close()

    @db_retry()
    def get_top_programs_thread_safe(
        self,
        n: int = 10,
        correct_only: bool = True,
    ) -> List[Program]:
        """Thread-safe version of get_top_programs."""
        _warn_repository_deprecation("get_top_programs_thread_safe")
        from .repository import ProgramRepository

        repository = ProgramRepository.from_config(
            self.config,
            embedding_model=self.embedding_model,
            read_only=True,
        )
        try:
            return repository.list_top(n=n, correct_only=correct_only)
        finally:
            repository.close()

    def _get_programs_for_island(self, island_idx: int) -> List[Program]:
        """
        Get all programs for a specific island.
        """
