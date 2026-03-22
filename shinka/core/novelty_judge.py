"""
Novelty filtering for proposed programs.

This module is intentionally small, but it sits on an important decision
boundary in Shinka:

1. Proposal code is generated.
2. The code is embedded.
3. Embedding similarity is compared against prior programs in the parent's island.
4. If the proposal is "too similar", an optional LLM breaks the tie by deciding
   whether the proposal is still meaningfully novel.

The result is not a score adjustment. It is a gate:
- accept the proposal and continue to evaluation, or
- reject it and resample a new proposal context.

The main levers exposed here are:
- `similarity_threshold`: higher means stricter embedding-based novelty filtering
- `max_novelty_attempts`: how many rejection-sampling retries to allow
- `novelty_llm_client`: optional semantic judge for near-duplicates

If you want custom novelty behavior, this module is the cleanest place to swap
in another strategy:
- different similarity metric
- different candidate comparison scope
- different semantic judge prompt / parser
- richer acceptance policy than binary accept/reject
"""

from typing import Optional, Tuple, List
import logging
from pathlib import Path
from shinka.database import Program
from shinka.llm import LLMClient
from shinka.prompts import NOVELTY_SYSTEM_MSG, NOVELTY_USER_MSG

logger = logging.getLogger(__name__)


class NoveltyJudge:
    """
    Two-stage novelty gate.

    Stage 1 is cheap and deterministic:
    compare the candidate embedding against programs already stored in the
    relevant island.

    Stage 2 is optional and semantic:
    if the embedding test says "too similar", ask an LLM whether the change is
    still meaningful enough to keep.

    This class does not generate new candidates itself. It only decides whether
    a generated candidate is novel enough to proceed.
    """

    def __init__(
        self,
        novelty_llm_client: Optional[LLMClient] = None,
        language: str = "python",
        similarity_threshold: float = 1.0,
        max_novelty_attempts: int = 3,
    ):
        self.novelty_llm_client = novelty_llm_client
        self.language = language
        self.similarity_threshold = similarity_threshold
        self.max_novelty_attempts = max_novelty_attempts

    def should_check_novelty(
        self,
        code_embedding: List[float],
        generation: int,
        parent_program: Optional[Program],
        database,
    ) -> bool:
        """
        Check if novelty assessment should be performed.

        Args:
            code_embedding: Embedding vector of the proposed code
            generation: Current generation number
            parent_program: Parent program
            database: Database instance for similarity computation

        Returns:
            Boolean indicating if novelty check should be performed
        """
        # Novelty checking only makes sense once we have:
        # - an embedding for the candidate,
        # - a non-initial generation,
        # - and a concrete parent / island context to compare against.
        if not code_embedding or generation == 0 or not parent_program:
            return False

        # Novelty is island-scoped in the current design. Until islands are
        # initialized, Shinka intentionally avoids filtering proposals by
        # similarity so the archive can bootstrap.
        islands = None
        if hasattr(database, "are_all_islands_initialized"):
            islands = database
        elif hasattr(database, "islands"):
            islands = database.islands

        if (
            parent_program.island_idx is not None
            and islands is not None
            and hasattr(islands, "are_all_islands_initialized")
            and islands.are_all_islands_initialized()
        ):
            return True

        return False

    def assess_novelty_with_rejection_sampling(
        self,
        exec_fname: str,
        code_embedding: List[float],
        parent_program: Program,
        database=None,
        similarity_provider=None,
    ) -> Tuple[bool, dict]:
        """
        Perform novelty assessment with rejection sampling.

        Args:
            exec_fname: Path to the executable file containing the code
            code_embedding: Embedding vector of the proposed code
            parent_program: Parent program for island-based similarity
            database: Database instance for similarity computation

        Returns:
            Tuple of (should_accept, novelty_metadata)
        """
        similarity_provider = (
            similarity_provider if similarity_provider is not None else database
        )
        if similarity_provider is None:
            raise ValueError("Novelty assessment requires a similarity provider.")

        novelty_metadata = {
            "novelty_checks_performed": 0,
            "novelty_total_cost": 0.0,
            "novelty_explanation": "",
            "max_similarity": 0.0,
            "similarity_scores": [],
        }

        for attempt in range(self.max_novelty_attempts):
            # The actual "novelty test" begins with cosine similarity over code
            # embeddings. This is the first-pass duplicate detector, not the LLM.
            similarity_scores = similarity_provider.compute_similarity(
                code_embedding, parent_program.island_idx
            )

            if not similarity_scores:
                logger.info(
                    f"NOVELTY CHECK {attempt + 1}/{self.max_novelty_attempts}: "
                    "Accepting program due to no similarity scores."
                )
                novelty_metadata["similarity_scores"] = []
                return True, novelty_metadata

            max_similarity = max(similarity_scores)
            sorted_similarity_scores = sorted(similarity_scores, reverse=True)
            formatted_similarities = [f"{s:.2f}" for s in sorted_similarity_scores]

            logger.info(f"Top-5 similarity scores: {formatted_similarities[:5]}")

            novelty_metadata["max_similarity"] = max_similarity
            novelty_metadata["similarity_scores"] = similarity_scores

            # If the nearest neighbor is below the threshold, the proposal is
            # considered novel enough and no semantic judge is needed.
            if max_similarity <= self.similarity_threshold:
                logger.info(
                    f"NOVELTY CHECK {attempt + 1}/{self.max_novelty_attempts}: "
                    f"Accepting program due to low similarity "
                    f"({max_similarity:.3f} <= {self.similarity_threshold})"
                )
                return True, novelty_metadata

            # If the embedding gate says "too similar", we optionally ask an
            # LLM whether the proposal is still meaningfully different. This is
            # effectively a semantic tie-breaker for near-duplicates.
            should_reject = True
            novelty_cost = 0.0

            if self.novelty_llm_client is not None:
                # Today the LLM only sees the single closest neighbor. If you
                # want a stronger novelty judge, this is an obvious extension
                # point: compare against top-k nearest programs or build a
                # summary of the local neighborhood instead.
                most_similar_program = similarity_provider.get_most_similar_program(
                    code_embedding, parent_program.island_idx
                )

                if most_similar_program:
                    try:
                        # Read the current proposed code
                        proposed_code = Path(exec_fname).read_text(encoding="utf-8")
                        is_novel, explanation, cost = self.check_llm_novelty(
                            proposed_code, most_similar_program
                        )
                        should_reject = not is_novel
                        novelty_cost = cost
                        novelty_metadata["novelty_checks_performed"] += 1
                        novelty_metadata["novelty_total_cost"] += cost
                        novelty_metadata["novelty_explanation"] = explanation
                    except Exception as e:
                        logger.warning(f"Error reading code for novelty check: {e}")
                        should_reject = True  # Default to rejection on error

            if should_reject:
                logger.info(
                    f"NOVELTY CHECK {attempt + 1}/{self.max_novelty_attempts}: "
                    f"Rejecting program due to high similarity "
                    f"({max_similarity:.3f} > {self.similarity_threshold})"
                    + (
                        f" and LLM novelty check (cost: {novelty_cost:.4f})"
                        if novelty_cost > 0
                        else ""
                    )
                    + ". Retrying with different parent/inspirations."
                )
                # Rejection sampling here means:
                # "discard this candidate and let the runner try another prompt
                # context / patch attempt, up to max_novelty_attempts."
                continue
            else:
                logger.info(
                    f"NOVELTY CHECK {attempt + 1}/{self.max_novelty_attempts}: "
                    f"Accepting program despite high similarity "
                    f"({max_similarity:.3f} > {self.similarity_threshold}) "
                    f"due to LLM novelty check (cost: {novelty_cost:.4f})."
                )
                return True, novelty_metadata

        # All attempts exhausted, reject the program
        logger.info(
            f"NOVELTY CHECK: Exhausted all {self.max_novelty_attempts} attempts, "
            "rejecting program."
        )
        return False, novelty_metadata

    def check_llm_novelty(
        self, proposed_code: str, most_similar_program: Program
    ) -> Tuple[bool, str, float]:
        """
        Use an LLM to judge whether a near-duplicate is still substantively new.

        Current contract:
        - prompt contains existing code + proposed code
        - parser treats a response starting with `NOVEL` as acceptance
        - everything else counts as rejection

        To extend this without changing the runner, subclass NoveltyJudge and
        override this method. Common variants:
        - richer structured outputs
        - multi-grade novelty scoring instead of binary labels
        - domain-specific novelty criteria
        - stricter parsing / schema validation
        
        Args:
            proposed_code: The newly generated code
            most_similar_program: The most similar existing program

        Returns:
            Tuple of (is_novel, explanation, api_cost)
        """
        if not self.novelty_llm_client:
            logger.debug("Novelty LLM not configured, skipping novelty check")
            return True, "No novelty LLM configured", 0.0

        user_msg = NOVELTY_USER_MSG.format(
            language=self.language,
            existing_code=most_similar_program.code,
            proposed_code=proposed_code,
        )

        try:
            response = self.novelty_llm_client.query(
                msg=user_msg,
                system_msg=NOVELTY_SYSTEM_MSG,
                llm_kwargs=self.novelty_llm_client.get_kwargs(),
            )

            if response is None or response.content is None:
                logger.warning("Novelty LLM returned empty response")
                return True, "LLM response was empty", 0.0

            content = response.content.strip()
            api_cost = response.cost or 0.0

            # Parse the response
            is_novel = content.upper().startswith(
                "NOVEL"
            ) or content.upper().startswith("**NOVEL**")
            explanation = content
            return is_novel, explanation, api_cost

        except Exception as e:
            logger.error(f"Error in novelty LLM check: {e}")
            return True, f"Error in novelty check: {e}", 0.0

    def log_novelty_skip_message(self, reason: str) -> None:
        """Log a message about skipping novelty check."""
        logger.info(f"NOVELTY CHECK: Skipping rejection sampling - {reason}")
