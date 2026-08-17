from __future__ import annotations

import builtins
import logging
import re
import sys
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from multiprocessing import cpu_count
from typing import Any

from rich.box import ROUNDED
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    FileSizeColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from rich.text import Text


_console: Console | None = None
_rich_handler: RichHandler | None = None
_rich_print_installed = False
_builtin_print = builtins.print
DEFAULT_CPU_THREADS = max(1, min(4, cpu_count()))


class ProgressHandle:
    def __init__(self, progress: Progress, task_id: int):
        self.progress = progress
        self.task_id = task_id

    def update(self, amount: int | float = 1) -> None:
        self.progress.advance(self.task_id, amount)

    def advance(self, amount: int | float = 1) -> None:
        self.update(amount)


def progress_handle(progress: Progress, task_id: int) -> ProgressHandle:
    return ProgressHandle(progress, task_id)


def get_console() -> Console:
    global _console
    if _console is None:
        _console = Console(
            file=sys.stderr,
            highlight=False,
            log_path=False,
            markup=False,
            soft_wrap=True,
        )
    return _console


def rich_print(
    *objects: Any,
    sep: str = " ",
    end: str = "\n",
    file: Any = None,
    flush: bool = False,
) -> None:
    if file not in (None, sys.stdout, sys.stderr):
        _builtin_print(
            *objects,
            sep=sep,
            end=end,
            file=file,
            flush=flush,
        )
        return

    console = get_console()
    styled_objects = []
    for obj in objects:
        if not isinstance(obj, str):
            styled_objects.append(obj)
            continue

        match = re.match(r"^(\s*)(\[[^\]\r\n]{1,32}\])(\s*)(.*)$", obj, re.S)
        if match is None:
            styled_objects.append(obj)
            continue

        styled = Text(match.group(1))
        styled.append(match.group(2), style="bold cyan")
        styled.append(match.group(3))
        styled.append(match.group(4))
        styled_objects.append(styled)

    console.print(
        *styled_objects,
        sep=sep,
        end=end,
        markup=False,
        highlight=False,
    )
    if flush:
        console.file.flush()


def install_rich_print() -> None:
    global _rich_print_installed
    if not _rich_print_installed:
        builtins.print = rich_print
        _rich_print_installed = True


def configure_logging(level: int = logging.INFO) -> None:
    global _rich_handler
    root_logger = logging.getLogger()
    if _rich_handler is None:
        _rich_handler = RichHandler(
            console=get_console(),
            show_path=False,
            markup=False,
            rich_tracebacks=True,
        )
    root_logger.handlers = [_rich_handler]
    root_logger.setLevel(level)


class _MetricColumn(ProgressColumn):
    """Render compact task metrics without adding a column when they are empty."""

    def render(self, task):
        metrics = task.fields.get("metrics", "")
        return Text(str(metrics), style="yellow", no_wrap=True)


class _IterationsPerSecondColumn(ProgressColumn):
    """Render the current training throughput."""

    def render(self, task):
        speed = task.speed
        if speed is None or speed <= 0:
            return Text("-- it/s", style="yellow", no_wrap=True)
        return Text(f"{speed:.2f} it/s", style="yellow", no_wrap=True)


def _progress_columns(*, download: bool = False, training: bool = False):
    columns = [
        TextColumn(
            "{task.description}",
            style="bold cyan",
            markup=False,
        ),
        BarColumn(
            complete_style="cyan",
            finished_style="bold green",
            pulse_style="bright_blue",
            bar_width=None,
        ),
        TaskProgressColumn(style="white"),
    ]

    if training:
        columns.extend(
            [
                MofNCompleteColumn(),
                TextColumn("·", style="dim", justify="center"),
                TimeElapsedColumn(),
                TextColumn("·", style="dim", justify="center"),
                TimeRemainingColumn(),
                TextColumn("·", style="dim", justify="center"),
                _IterationsPerSecondColumn(),
                TextColumn("·", style="dim", justify="center"),
                _MetricColumn(),
            ]
        )
    else:
        columns.extend(
            [
                TextColumn("·", style="dim", justify="center"),
                TimeElapsedColumn(),
                TextColumn("·", style="dim", justify="center"),
                TimeRemainingColumn(),
            ]
        )

    if download:
        columns = [
            TextColumn(
                "{task.description}",
                style="bold cyan",
                markup=False,
            ),
            BarColumn(
                complete_style="cyan",
                finished_style="bold green",
                pulse_style="bright_blue",
                bar_width=None,
            ),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TextColumn("·", style="dim", justify="center"),
            FileSizeColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TextColumn("·", style="dim", justify="center"),
            TimeRemainingColumn(),
        ]

    return columns


def create_progress(
    *,
    download: bool = False,
    leave: bool = False,
    training: bool = False,
    disable: bool = False,
) -> Progress:
    return Progress(
        *_progress_columns(download=download, training=training),
        console=get_console(),
        refresh_per_second=10,
        transient=not leave,
        disable=disable,
        redirect_stdout=False,
        redirect_stderr=False,
    )


@contextmanager
def progress_task(
    total: int | float | None,
    description: str,
    *,
    initial: int | float = 0,
    download: bool = False,
    leave: bool = False,
    training: bool = False,
    disable: bool = False,
):
    progress = create_progress(
        download=download,
        leave=leave,
        training=training,
        disable=disable,
    )
    with progress:
        task_id = progress.add_task(
            description,
            total=total,
            completed=initial,
            metrics="",
        )
        yield progress, task_id


def track(
    sequence: Iterable[Any],
    *,
    total: int | float | None = None,
    description: str = "",
    leave: bool = False,
) -> Iterator[Any]:
    if total is None:
        try:
            total = len(sequence)  # type: ignore[arg-type]
        except TypeError:
            pass
    with progress_task(total, description, leave=leave) as (progress, task_id):
        for item in sequence:
            yield item
            progress.advance(task_id)


def _parameter_counts(module: Any) -> tuple[int, int]:
    module = getattr(module, "module", module)
    seen: set[int] = set()
    total = 0
    trainable = 0
    for parameter in module.parameters():
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
    return total, trainable


def _format_count(value: int) -> str:
    return f"{value:,}"


def print_model_summary(
    models: Iterable[tuple[str, Any]],
    *,
    title: str = "Model Summary",
) -> dict[str, tuple[int, int]]:
    table = Table(
        title=title,
        box=ROUNDED,
        header_style="bold cyan",
        border_style="cyan",
        title_style="bold cyan",
        show_lines=False,
        padding=(0, 1),
    )
    table.add_column("Component", style="cyan")
    table.add_column("Parameters", justify="right")
    table.add_column("Trainable", justify="right", style="green")

    summary: dict[str, tuple[int, int]] = {}
    total_parameters = 0
    total_trainable = 0
    for name, module in models:
        if module is None:
            continue
        total, trainable = _parameter_counts(module)
        summary[name] = (total, trainable)
        total_parameters += total
        total_trainable += trainable
        table.add_row(name, _format_count(total), _format_count(trainable))

        base_module = getattr(module, "module", module)
        for child_name, child in base_module.named_children():
            child_total, child_trainable = _parameter_counts(child)
            table.add_row(
                f"  {child_name}",
                _format_count(child_total),
                _format_count(child_trainable),
            )

    table.add_section()
    table.add_row(
        "Total",
        _format_count(total_parameters),
        _format_count(total_trainable),
        style="bold",
    )
    get_console().print(table)
    return summary


def print_settings_panel(
    rows: Iterable[tuple[str, Any]],
    *,
    title: str | None = None,
) -> None:
    table = Table.grid(padding=(0, 1), expand=False)
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(style="white", overflow="fold")
    for label, value in rows:
        table.add_row(str(label), str(value))

    get_console().print(
        Panel(
            table,
            title=title,
            title_align="left",
            border_style="cyan",
            box=ROUNDED,
            expand=False,
            padding=(0, 1),
        )
    )
