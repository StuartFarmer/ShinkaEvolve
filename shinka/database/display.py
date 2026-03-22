import logging
import time
from typing import Any, Optional

import numpy as np
import rich  # type: ignore
import rich.box  # type: ignore
from rich.columns import Columns as RichColumns  # type: ignore
from rich.console import Console as RichConsole  # type: ignore
from rich.table import Table as RichTable  # type: ignore

from . import island_ops, program_reads
from .connection import Database

logger = logging.getLogger(__name__)


class DatabaseDisplay:
    """Rich console display backed by controllers/services, not raw SQL."""

    def __init__(
        self,
        *,
        db: Database,
        archive_size: int,
        num_islands: int,
        archive_policy,
        migration_interval: int = 0,
        migration_rate: float = 0.0,
        island_elitism: bool = False,
        default_console: Optional[RichConsole] = None,
    ):
        self.db = db
        self.archive_size = archive_size
        self.num_islands = num_islands
        self.archive_policy = archive_policy
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.island_elitism = island_elitism
        self.default_console = default_console
        self.last_iteration = 0

    def set_default_console(self, console: Optional[RichConsole]) -> None:
        self.default_console = console

    def set_last_iteration(self, last_iteration: int) -> None:
        self.last_iteration = last_iteration

    def _console(self, console: Optional[RichConsole]) -> RichConsole:
        return console or self.default_console or RichConsole()

    def _all_programs(self):
        with self.db.session() as session:
            return program_reads.list_all(session)

    def _best_program(self):
        with self.db.session() as session:
            return program_reads.get_best(session)

    def _top_programs(self, *, n: int, metric: Optional[str], correct_only: bool):
        with self.db.session() as session:
            return program_reads.list_top(
                session,
                n=n,
                metric=metric,
                correct_only=correct_only,
            )

    def _format_populations(self) -> str:
        with self.db.session() as session:
            return island_ops.format_populations(
                session,
                num_islands=self.num_islands,
            )

    def _cost_totals(self):
        total_api_cost = 0.0
        total_embed_cost = 0.0
        total_novelty_cost = 0.0
        total_meta_cost = 0.0
        total_compute_time = 0.0
        for program in self._all_programs():
            metadata = program.metadata or {}
            total_api_cost += float(metadata.get("api_costs", 0.0) or 0.0)
            total_embed_cost += float(metadata.get("embed_cost", 0.0) or 0.0)
            total_novelty_cost += float(metadata.get("novelty_cost", 0.0) or 0.0)
            total_meta_cost += float(metadata.get("meta_cost", 0.0) or 0.0)
            total_compute_time += float(metadata.get("compute_time", 0.0) or 0.0)
        return (
            total_api_cost,
            total_embed_cost,
            total_novelty_cost,
            total_meta_cost,
            total_compute_time,
        )

    def _format_program_row(self, prog, role_name):
        if prog.combined_score is not None:
            score = prog.combined_score
            if score > 0.8:
                score_display = f"[bold green]{score:.3f}[/bold green]"
            elif score > 0.5:
                score_display = f"[green]{score:.3f}[/green]"
            else:
                score_display = f"[yellow]{score:.3f}[/yellow]"
        else:
            score_display = "[dim]N/A[/dim]"

        island = f"I-{prog.island_idx}" if prog.island_idx is not None else "N/A"
        correct = (
            "[bold green]✓[/bold green]" if prog.correct else "[bold red]✗[/bold red]"
        )

        metadata = prog.metadata or {}
        total_cost = (
            float(metadata.get("api_costs", 0.0) or 0.0)
            + float(metadata.get("embed_cost", 0.0) or 0.0)
            + float(metadata.get("novelty_cost", 0.0) or 0.0)
            + float(metadata.get("meta_cost", 0.0) or 0.0)
        )
        cost_display = f"${total_cost:.3f}" if total_cost > 0 else "[dim]N/A[/dim]"

        time_display = "[dim]N/A[/dim]"
        if "compute_time" in metadata:
            time_val = metadata["compute_time"]
            time_display = f"{time_val / 60:.1f}m" if time_val > 60 else f"{time_val:.1f}s"

        patch_name = metadata.get("patch_name", "[dim]N/A[/dim]")[:30]
        patch_type = metadata.get("patch_type", "[dim]N/A[/dim]")

        return [
            role_name,
            str(prog.generation),
            island,
            correct,
            score_display,
            patch_name,
            patch_type,
            f"{prog.complexity:.1f}",
            cost_display,
            time_display,
        ]

    def print_program_summary(self, program, console: Optional[RichConsole] = None):
        _console = self._console(console)
        best_program = self._best_program()
        best_score_str = "[dim]N/A[/dim]"
        if best_program and best_program.combined_score is not None:
            best_score_str = f"[bold yellow]{best_program.combined_score:.3f}[/bold yellow]"

        total_cost = sum(self._cost_totals()[:4])
        table = RichTable(
            title=(
                f"[bold green]Program Evaluation Summary - "
                f"Gen {program.generation} | Total Cost: ${total_cost:.2f}[/bold green]"
            ),
            border_style="green",
            box=rich.box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
            padding=(0, 1),
            width=120,
        )
        table.add_column("GenID: " + str(program.generation), style="cyan", justify="center", width=12)
        table.add_column("Island", style="magenta", justify="center", width=8)
        table.add_column("Status", style="white", justify="center", width=14)
        table.add_column("Score", style="white", justify="right", width=8)
        table.add_column("Patch Name", style="yellow", justify="left", width=32, overflow="ellipsis")
        table.add_column("Type", style="yellow", justify="left", width=6, overflow="ellipsis")
        table.add_column("Complex", style="yellow", justify="right", width=7)
        table.add_column("Cost", style="green", justify="right", width=7)
        table.add_column("Time", style="blue", justify="right", width=5)

        status_display = "[bold green]✓ Correct[/bold green]" if program.correct else "[bold red]✗ Incorrect[/bold red]"
        score_display = "[dim]N/A[/dim]"
        if program.combined_score is not None:
            color = "bold green" if program.combined_score > 0.8 else "green" if program.combined_score > 0.5 else "yellow"
            score_display = f"[{color}]{program.combined_score:.3f}[/{color}]"

        metadata = program.metadata or {}
        total_program_cost = (
            float(metadata.get("api_costs", 0.0) or 0.0)
            + float(metadata.get("embed_cost", 0.0) or 0.0)
            + float(metadata.get("novelty_cost", 0.0) or 0.0)
            + float(metadata.get("meta_cost", 0.0) or 0.0)
        )
        cost_display = f"${total_program_cost:.3f}" if total_program_cost > 0 else "[dim]N/A[/dim]"
        time_display = "[dim]N/A[/dim]"
        if "compute_time" in metadata:
            time_val = metadata["compute_time"]
            time_display = f"{time_val / 60:.1f}m" if time_val > 60 else f"{time_val:.1f}s"

        island_display = f"I-{program.island_idx}" if program.island_idx is not None else "N/A"
        table.add_row(
            f"Best: {best_score_str}",
            island_display,
            status_display,
            score_display,
            metadata.get("patch_name", "[dim]N/A[/dim]")[:30],
            metadata.get("patch_type", "[dim]N/A[/dim]"),
            f"{program.complexity:.1f}",
            cost_display,
            time_display,
        )
        _console.print(table)

    def print_summary(self, console: Optional[RichConsole] = None) -> None:
        _console = self._console(console)
        all_programs = self._all_programs()
        correct_programs = [program for program in all_programs if program.correct]
        archive_programs = self.archive_policy(correct_programs)
        best_program = self._best_program()
        (
            total_api_cost,
            total_embed_cost,
            total_novelty_cost,
            total_meta_cost,
            total_compute_time,
        ) = self._cost_totals()

        all_scores = [float(program.combined_score) for program in all_programs if program.combined_score is not None]
        best_score = max((float(program.combined_score or 0.0) for program in correct_programs), default=0.0)
        median_score = float(np.median(all_scores)) if all_scores else 0.0
        _ = median_score

        summary_table = RichTable(
            title="[bold cyan]Program Database Summary[/bold cyan]",
            border_style="cyan",
            box=rich.box.ROUNDED,
            width=40,
        )
        summary_table.add_column("Metric", style="cyan bold", no_wrap=True)
        summary_table.add_column("Value", style="magenta")
        summary_table.add_row(
            "Overall Best Score",
            f"[bold cyan]{best_score:.2f}[/bold cyan]" if all_scores else "[dim]N/A[/dim]",
        )
        total_programs = len(all_programs)
        summary_table.add_row("Total Programs", f"[bold]{total_programs}[/bold]")
        correct_percentage = (len(correct_programs) / total_programs * 100) if total_programs > 0 else 0
        summary_table.add_row(
            "Correct Programs",
            f"[bold]{len(correct_programs)}[/bold] / {total_programs} ({correct_percentage:.0f}%)",
        )
        archive_percentage = (
            (len(archive_programs) / self.archive_size * 100)
            if self.archive_size > 0
            else 0
        )
        summary_table.add_row(
            "Archived Programs",
            f"[bold]{len(archive_programs)}[/bold] / {self.archive_size} ({archive_percentage:.0f}%)",
        )
        if self.num_islands > 0:
            summary_table.add_row("Island Populations", self._format_populations())
            migration_info = None
            if self.migration_interval > 0:
                migration_info = (
                    f"{self.migration_interval}G, "
                    f"{self.migration_rate * 100:.0f}%"
                )
                if self.island_elitism:
                    migration_info += "(E)"
            if migration_info:
                summary_table.add_row("Migration Policy", migration_info)

        best_program_table_renderable = None
        if best_program:
            best_program_table = RichTable(
                title="[bold yellow]Best Program[/bold yellow]",
                border_style="yellow",
                box=rich.box.ROUNDED,
                width=40,
            )
            best_program_table.add_column("Attribute", style="green bold")
            best_program_table.add_column("Value", style="yellow")
            short_id = best_program.id[:8] + "..." if len(best_program.id) > 8 else best_program.id
            best_program_table.add_row("ID", f"[dim]{short_id}[/dim]")
            best_program_table.add_row("Generation", str(best_program.generation))
            best_program_table.add_row("Complexity", f"{best_program.complexity:.2f}")
            best_program_table.add_row(
                "Embedding[0]",
                f"{best_program.embedding[0]:.2f}" if best_program.embedding else "N/A",
            )
            if best_program.combined_score is not None:
                best_program_table.add_row(
                    "Metric: Score",
                    f"[bold green]{best_program.combined_score:.2f}[/bold green]",
                )
            else:
                best_program_table.add_row("Metrics", "[dim]N/A[/dim]")
            best_program_table.add_row(
                "Timestamp",
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(best_program.timestamp)),
            )
            best_program_table_renderable = best_program_table

        cost_table = RichTable(
            title="[bold magenta]Cost & Stats Summary[/bold magenta]",
            border_style="magenta",
            box=rich.box.ROUNDED,
            width=40,
        )
        cost_table.add_column("Metric", style="magenta bold")
        cost_table.add_column("Value", style="green")
        total_cost = total_api_cost + total_embed_cost + total_novelty_cost + total_meta_cost
        if total_cost > 0:
            cost_table.add_row("Total API Cost", f"[bold]${total_api_cost:.2f}[/bold]")
            cost_table.add_row("Total Embedding Cost", f"[bold]${total_embed_cost:.2f}[/bold]")
            cost_table.add_row("Total Novelty Cost", f"[bold]${total_novelty_cost:.2f}[/bold]")
            cost_table.add_row("Total Meta Cost", f"[bold]${total_meta_cost:.2f}[/bold]")
            cost_table.add_row("Total Combined Cost", f"[bold]${total_cost:.2f}[/bold]")
        if total_compute_time > 0:
            hours = int(total_compute_time // 3600)
            minutes = int((total_compute_time % 3600) // 60)
            seconds = int(total_compute_time % 60)
            cost_table.add_row("Total Compute", f"{hours}h {minutes}m {seconds}s")

        tables_to_display = [summary_table]
        if best_program_table_renderable:
            tables_to_display.append(best_program_table_renderable)
        tables_to_display.append(cost_table)
        _console.print(RichColumns(tables_to_display))

        top_programs = self._top_programs(
            n=10,
            metric="combined_score",
            correct_only=True,
        )
        highlight_table = RichTable(
            title="[bold green]Top 10 Best Performing Programs[/bold green]",
            border_style="green",
            box=rich.box.ROUNDED,
            show_lines=True,
            width=120,
        )
        highlight_table.add_column("Rank", style="magenta bold", justify="center", width=6)
        highlight_table.add_column("Gen", style="cyan bold", justify="center", width=6)
        highlight_table.add_column("✓/✗", style="red bold", justify="center", width=4)
        highlight_table.add_column("Score", style="green bold", justify="right", width=8)
        highlight_table.add_column("Complexity", style="yellow", justify="right", width=10)
        highlight_table.add_column("Patch Name", style="blue", justify="left", width=32, overflow="ellipsis")
        highlight_table.add_column("Type", style="cyan", justify="left", width=8)
        highlight_table.add_column("Island", style="magenta", justify="center", width=8)
        highlight_table.add_column("Children", style="blue", justify="right", width=8)
        highlight_table.add_column("Timestamp", style="dim", width=19)

        if not top_programs:
            _console.print("[yellow]No programs with scores in the database to display.[/yellow]")
            return

        for rank, prog in enumerate(top_programs, 1):
            combined_score_str = f"{prog.combined_score:.3f}" if prog.combined_score is not None else "N/A"
            correct_str = "[bold green]✓[/bold green]" if prog.correct else "[bold red]✗[/bold red]"
            island_display = f"I{prog.island_idx}" if prog.island_idx is not None else "N/A"
            ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(prog.timestamp))
            if rank == 1:
                rank_str = "[bold gold1]#1[/bold gold1]"
                score_str = f"[bold gold1]{combined_score_str}[/bold gold1]"
            elif rank == 2:
                rank_str = "[bold bright_white]#2[/bold bright_white]"
                score_str = f"[bold bright_white]{combined_score_str}[/bold bright_white]"
            elif rank == 3:
                rank_str = "[bold orange1]#3[/bold orange1]"
                score_str = f"[bold orange1]{combined_score_str}[/bold orange1]"
            else:
                rank_str = f"#{rank}"
                score_str = combined_score_str

            highlight_table.add_row(
                rank_str,
                str(prog.generation),
                correct_str,
                score_str,
                f"{prog.complexity:.1f}",
                (prog.metadata or {}).get("patch_name", "N/A")[:30],
                (prog.metadata or {}).get("patch_type", "N/A")[:6],
                island_display,
                str(prog.children_count or 0),
                ts_str,
            )
        _console.print(highlight_table)

    def print_sampling_summary(
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
        console: Optional[RichConsole] = None,
    ):
        _console = self._console(console)
        gen_display = target_generation if target_generation is not None else parent.generation + 1
        total_cost = sum(self._cost_totals()[:4])
        title_parts = [
            f"[bold red]Parent & Context Sampling Summary - Gen {gen_display} | Total Cost: ${total_cost:.2f}"
        ]
        if (
            novelty_attempt is not None
            and max_novelty_attempts is not None
            and resample_attempt is not None
            and max_resample_attempts is not None
        ):
            title_parts.append(
                f" (Novelty: {novelty_attempt}/{max_novelty_attempts}, Resample: {resample_attempt}/{max_resample_attempts})"
            )
        title_parts.append("[/bold red]")

        table = RichTable(
            title="".join(title_parts),
            border_style="red",
            show_header=True,
            header_style="bold cyan",
            width=120,
        )
        table.add_column("Role", style="cyan bold", width=12)
        table.add_column("Gen", style="magenta", justify="center", width=5)
        table.add_column("Island", style="red", justify="center", width=8)
        table.add_column("✓/✗", style="white", justify="center", width=6)
        table.add_column("Score", style="green", justify="right", width=8)
        table.add_column("Patch Name", style="yellow", justify="left", width=32, overflow="ellipsis")
        table.add_column("Type", style="yellow", justify="left", width=6, overflow="ellipsis")
        table.add_column("Complex", style="blue", justify="right", width=7)
        table.add_column("Cost", style="green", justify="right", width=7)
        table.add_column("Time", style="cyan", justify="right", width=5)

        parent_label = "[bold red]FIX TARGET[/bold red]" if is_fix_mode else "[bold]PARENT[/bold]"
        table.add_row(*self._format_program_row(parent, parent_label))
        if ancestor_inspirations:
            for i, prog in enumerate(ancestor_inspirations):
                table.add_row(*self._format_program_row(prog, f"ANCESTOR-{i + 1}"))
        for i, prog in enumerate(archive_inspirations):
            table.add_row(*self._format_program_row(prog, f"Archive-{i + 1}"))
        for i, prog in enumerate(top_k_inspirations):
            table.add_row(*self._format_program_row(prog, f"TopK-{i + 1}"))
        _console.print(table)
