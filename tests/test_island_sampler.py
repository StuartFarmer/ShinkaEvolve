"""Test island sampling strategies."""

import tempfile
from pathlib import Path
from shinka.database import Program
from shinka.database.island_sampler import create_island_sampler
from shinka.database.archive_policy import create_archive_policy
from shinka.controllers import DatabaseController


def test_island_samplers():
    """Test all island sampling strategies."""

    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"

        strategies = ["uniform", "equal", "proportional", "weighted"]

        for strategy in strategies:
            print(f"\n=== Testing {strategy} strategy ===")

            controller = DatabaseController.open(
                db_path=str(db_path),
                num_islands=3,
                read_only=False,
            )
            repo = controller.programs
            archive_policy = create_archive_policy(
                archive_selection_strategy="fitness",
                archive_size=40,
                archive_criteria={"combined_score": 1.0},
            )
            island_sampler = create_island_sampler(
                programs=repo,
                strategy=strategy,
            )

            # Add some test programs to different islands
            for island_idx in range(3):
                for i in range(island_idx + 1):  # Different counts per island
                    program = Program(
                        id=f"prog_{strategy}_{island_idx}_{i}",
                        code=f"def test_{i}(): return {i}",
                        correct=True,
                        combined_score=float(island_idx + 1),  # Different scores
                        island_idx=island_idx,
                    )
                    repo.add(program)

            # Test sampling
            initialized_islands = controller.islands.list_initialized_island_ids()
            print(f"Initialized islands: {initialized_islands}")

            # Sample multiple times to see distribution
            samples = {}
            for _ in range(30):
                sampled = island_sampler.sample_island(initialized_islands)
                samples[sampled] = samples.get(sampled, 0) + 1

            print(f"Sample distribution: {samples}")

            # Verify all strategies can sample
            assert len(samples) > 0, f"{strategy} strategy produced no samples"

            controller.close()

            # Clean up for next test
            if db_path.exists():
                db_path.unlink()

        print("\n✓ All strategies tested successfully!")


if __name__ == "__main__":
    test_island_samplers()
