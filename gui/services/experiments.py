"""Rebuilding an extracted experiment's config for another vocoder.

A thin seam over ``rvc.configs.experiments``, the same code the Gradio tab
runs.  It is JSON and file copies, so it runs here rather than in the worker,
and is imported on first use to keep the window's startup free of it.
"""

from __future__ import annotations

from . import paths


def list_experiments() -> list[str]:
    from rvc.configs import experiments

    return experiments.list_experiments(paths.LOGS_DIR)


def vocoder_choices() -> list[tuple[str, str]]:
    """``(label, id)`` for every enabled vocoder."""
    from rvc.configs.vocoders import get_vocoder_choices

    return list(get_vocoder_choices())


def default_vocoder() -> str:
    from rvc.configs import experiments

    return experiments.default_target_vocoder()


def describe(experiment: str) -> tuple[str, str | None]:
    """Markdown status, and the vocoder the current config is for (or None)."""
    from rvc.configs import experiments

    return experiments.describe(paths.LOGS_DIR / experiment)


def rebuild(experiment: str, vocoder: str, keep_backup: bool, move_checkpoints: bool) -> str:
    """Rewrite the config; returns the Markdown report."""
    from rvc.configs import experiments

    return experiments.rebuild(
        paths.LOGS_DIR / experiment, vocoder, keep_backup, move_checkpoints
    )
