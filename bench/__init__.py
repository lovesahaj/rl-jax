"""Shared benchmarking utilities for control and RL methods on CartPole.

Public API:
    rollout, evaluate, aggregate, compute_episode_metrics,
    time_callable, format_markdown_table, save_results, load_results.
"""

from .metrics import (
    BENCH_Q,
    BENCH_R,
    EpisodeStats,
    aggregate,
    compute_episode_metrics,
    evaluate,
    format_markdown_table,
    load_results,
    rollout,
    save_results,
    time_callable,
    with_canonical_reward,
)

__all__ = [
    "BENCH_Q",
    "BENCH_R",
    "EpisodeStats",
    "aggregate",
    "compute_episode_metrics",
    "evaluate",
    "format_markdown_table",
    "load_results",
    "rollout",
    "save_results",
    "time_callable",
    "with_canonical_reward",
]
