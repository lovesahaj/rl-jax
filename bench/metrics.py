"""Reusable metrics and rollout helpers for benchmarking control / RL agents.

A "policy" is any callable ``policy(env) -> action`` or
``policy(env) -> (action, force)``. If ``force`` is omitted it is inferred
from the discrete action via ``env.force_mag``.

Typical usage from a script::

    from bench import evaluate, format_markdown_table

    summary = evaluate(
        name="LQR",
        env_factory=lambda seed: ProjectEnv(seed=seed),
        policy_factory=lambda env: (lambda e: e.choose_action()),
        num_episodes=100,
        max_steps=500,
    )
    print(format_markdown_table([summary]))
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

# Canonical cost weights used for cross-method ``trajectory_cost`` comparison.
# Solvers may optimise their own internal Q/R, but the benchmark always
# evaluates the resulting trajectory against these weights so the column is
# apples-to-apples across LQR / iLQR / MPC / RL.
BENCH_Q = np.diag([1.0, 1.0, 100.0, 10.0])
BENCH_R = np.array([[1.0]])

# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------


@dataclass
class EpisodeStats:
    """Per-episode trajectory captured during a rollout."""

    states: np.ndarray  # shape (T, state_dim)
    actions: np.ndarray  # shape (T,)
    forces: np.ndarray  # shape (T,)
    rewards: np.ndarray  # shape (T,)
    terminated: bool
    truncated: bool
    steps: int
    seed: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Canonical-reward wrapper (cross-method comparability)
# -----------------------------------------------------------------------------


def with_canonical_reward(env):
    """Patch ``env.step`` so reward = 1.0 per non-terminal step, 0.0 on
    terminate. Lets us compare ``total_reward`` across methods that train on
    different shaped costs (LQR, Q-learning, MC, DeepQ all use different
    reward functions internally)."""
    original_step = env.step

    def canonical_step(action):
        state, _, terminated, truncated, info = original_step(action)
        reward = 0.0 if terminated else 1.0
        return state, reward, terminated, truncated, info

    env.step = canonical_step  # type: ignore[method-assign]
    return env


# -----------------------------------------------------------------------------
# Rollout
# -----------------------------------------------------------------------------


def _coerce_action_force(out: Any, env) -> tuple[Any, float]:
    """Normalise the policy output into ``(action, force)``.

    Discrete envs: ``policy(env)`` returns an int → we infer ``force`` from
    ``env.force_mag``. Continuous envs: ``policy(env)`` may return a force
    array (we extract its scalar) or a ``(action, force)`` tuple to be
    explicit. ``action`` is forwarded to ``env.step`` unchanged.
    """
    if isinstance(out, tuple) and len(out) == 2:
        action, force = out
        return action, float(np.asarray(force).reshape(-1)[0])

    action = out
    arr = np.asarray(action).reshape(-1)
    # discrete action: scalar int-like
    if arr.size == 1:
        try:
            a_int = int(arr[0])
        except (TypeError, ValueError):
            a_int = None
        if a_int is not None and float(arr[0]) == a_int:
            mag = float(getattr(env, "force_mag", 1.0))
            force = mag * (1.0 if a_int == 1 else -1.0)
            return a_int, force
        return action, float(arr[0])
    # continuous vector action: take first element as the applied force
    return action, float(arr[0])


def rollout(
    env,
    policy: Callable[[Any], Any],
    max_steps: int = 500,
    seed: int | None = None,
    record_render: bool = False,
) -> EpisodeStats:
    """Run a single episode and return its trajectory.

    ``policy(env)`` may return either ``action`` or ``(action, force)``.
    """
    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()

    states, actions, forces, rewards = [], [], [], []
    frames: list[np.ndarray] = []

    terminated = truncated = False
    steps = 0
    for _ in range(max_steps):
        states.append(np.asarray(env.state, dtype=np.float32))
        action, force = _coerce_action_force(policy(env), env)
        actions.append(action)
        forces.append(force)

        _, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        steps += 1

        if record_render:
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))

        if terminated or truncated:
            break

    extras: dict[str, Any] = {}
    if record_render and frames:
        extras["frames"] = np.stack(frames)

    actions_arr: np.ndarray
    try:
        actions_arr = np.asarray(actions, dtype=np.float32)
    except (TypeError, ValueError):
        actions_arr = np.asarray(actions, dtype=object)

    return EpisodeStats(
        states=np.asarray(states, dtype=np.float32),
        actions=actions_arr,
        forces=np.asarray(forces, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminated=bool(terminated),
        truncated=bool(truncated),
        steps=steps,
        seed=seed,
        extras=extras,
    )


# -----------------------------------------------------------------------------
# Per-episode metrics
# -----------------------------------------------------------------------------


def compute_episode_metrics(
    stats: EpisodeStats,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    balanced_threshold: int = 500,
) -> dict[str, float]:
    """Compute scalar metrics from a single rollout.

    - ``episode_length``: number of environment steps survived.
    - ``total_reward``: undiscounted sum of rewards.
    - ``trajectory_cost``: ``sum(x' Q x + u' R u)`` if ``Q`` and ``R`` provided.
    - ``final_state_cost``: ``x_T' Q x_T`` if ``Q`` provided.
    - ``force_mean_abs`` / ``force_peak``: actuator magnitude stats.
    - ``force_smoothness``: mean ``|Δu|`` between consecutive timesteps
      (lower is smoother).
    - ``balanced``: 1.0 if the agent survived ``>= balanced_threshold`` steps
      *without* termination, else 0.0.
    """
    T = stats.steps
    metrics: dict[str, float] = {
        "episode_length": float(T),
        "total_reward": float(stats.rewards.sum()) if T else 0.0,
        "balanced": float((not stats.terminated) and T >= balanced_threshold),
    }

    if T == 0:
        return metrics

    forces = stats.forces.astype(np.float64)
    metrics["force_mean_abs"] = float(np.mean(np.abs(forces)))
    metrics["force_peak"] = float(np.max(np.abs(forces)))
    metrics["force_smoothness"] = (
        float(np.mean(np.abs(np.diff(forces)))) if T > 1 else 0.0
    )

    if Q is not None:
        Q = np.asarray(Q, dtype=np.float64)
        states = stats.states.astype(np.float64)
        state_cost = np.einsum("ti,ij,tj->t", states, Q, states)
        metrics["final_state_cost"] = float(states[-1] @ Q @ states[-1])
        if R is not None:
            R_arr = np.asarray(R, dtype=np.float64).reshape(1, 1)
            ctrl_cost = (forces**2) * float(R_arr[0, 0])
            metrics["trajectory_cost"] = float(np.sum(state_cost + ctrl_cost))
        else:
            metrics["trajectory_cost"] = float(np.sum(state_cost))

    return metrics


# -----------------------------------------------------------------------------
# Aggregation across episodes
# -----------------------------------------------------------------------------


def aggregate(
    per_episode: Sequence[dict[str, float]],
    name: str = "method",
) -> dict[str, Any]:
    """Average per-episode metric dicts into mean / std summary."""
    if not per_episode:
        return {"name": name, "num_episodes": 0}

    keys = set().union(*(d.keys() for d in per_episode))
    summary: dict[str, Any] = {"name": name, "num_episodes": len(per_episode)}
    for k in keys:
        vals = np.array([d[k] for d in per_episode if k in d], dtype=np.float64)
        if vals.size == 0:
            continue
        summary[f"{k}_mean"] = float(vals.mean())
        summary[f"{k}_std"] = float(vals.std())
    if "balanced_mean" in summary:
        summary["stability_rate"] = summary["balanced_mean"]
    return summary


# -----------------------------------------------------------------------------
# High-level evaluation loop
# -----------------------------------------------------------------------------


def evaluate(
    name: str,
    env_factory: Callable[[int], Any],
    policy_factory: Callable[[Any], Callable[[Any], Any]],
    num_episodes: int = 100,
    max_steps: int = 500,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    balanced_threshold: int = 500,
    seeds: Iterable[int] | None = None,
    on_episode: Callable[[int, EpisodeStats, dict[str, float]], None] | None = None,
) -> dict[str, Any]:
    """Run ``num_episodes`` rollouts and return an aggregated summary.

    ``env_factory(seed)`` builds a fresh env. ``policy_factory(env)`` returns
    a policy callable bound to that env. Pass ``seeds`` for reproducible runs;
    by default ``range(num_episodes)`` is used.
    """
    if seeds is None:
        seeds = list(range(num_episodes))
    seeds = list(seeds)

    per_ep: list[dict[str, float]] = []
    for i, seed in enumerate(seeds):
        env = env_factory(seed)
        policy = policy_factory(env)
        stats = rollout(env, policy, max_steps=max_steps, seed=seed)
        m = compute_episode_metrics(
            stats, Q=Q, R=R, balanced_threshold=balanced_threshold
        )
        per_ep.append(m)
        if on_episode is not None:
            on_episode(i, stats, m)
        # be polite to pygame / matplotlib backends
        close = getattr(env, "close", None)
        if callable(close):
            close()

    summary = aggregate(per_ep, name=name)
    summary["per_episode"] = per_ep
    return summary


# -----------------------------------------------------------------------------
# Timing
# -----------------------------------------------------------------------------


def time_callable(
    fn: Callable[[], Any],
    n_runs: int = 5,
    warmup: int = 1,
) -> dict[str, float]:
    """Wall-clock-time ``fn`` across ``n_runs`` calls after ``warmup`` calls.

    Useful for measuring JIT vs no-JIT speedups: pass a closure that performs
    one full rollout / one solver call.
    """
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    arr = np.array(times)
    return {
        "mean_s": float(arr.mean()),
        "std_s": float(arr.std()),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
        "n_runs": int(n_runs),
    }


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


_DEFAULT_COLUMNS: tuple[tuple[str, str, str], ...] = (
    # (summary_key, header, format)
    ("name", "Method", "{}"),
    ("episode_length_mean", "Balanced Steps", "{:.1f}"),
    ("trajectory_cost_mean", "Final Cost", "{:.2f}"),
    ("total_reward_mean", "Mean Test Reward", "{:.2f}"),
    ("stability_rate", "Stability %", "{:.0%}"),
    ("force_smoothness_mean", "Force Smoothness", "{:.2f}"),
)


def format_markdown_table(
    summaries: Sequence[dict[str, Any]],
    columns: Sequence[tuple[str, str, str]] | None = None,
    notes: dict[str, str] | None = None,
) -> str:
    """Render a list of ``evaluate`` summaries as a markdown table."""
    cols = list(columns) if columns is not None else list(_DEFAULT_COLUMNS)
    notes_map: dict[str, str] = dict(notes) if notes is not None else {}
    if notes is not None:
        cols = cols + [("__note__", "Notes", "{}")]

    header = "| " + " | ".join(h for _, h, _ in cols) + " |"
    align = (
        "| "
        + " | ".join(
            ":---" if i == 0 or h == "Notes" else "---:"
            for i, (_, h, _) in enumerate(cols)
        )
        + " |"
    )

    lines = [header, align]
    for s in summaries:
        row = []
        for key, _, fmt in cols:
            if key == "__note__":
                row.append(notes_map.get(str(s.get("name", "")), ""))
                continue
            v = s.get(key)
            row.append(fmt.format(v) if v is not None else "—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------


def save_results(path: str, summaries: Sequence[dict[str, Any]]) -> None:
    """Save summaries (without per-episode trajectories) as ``.npz``."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    flat: dict[str, np.ndarray] = {}
    for s in summaries:
        name = s.get("name", "method")
        for k, v in s.items():
            if k in {"per_episode", "name"}:
                continue
            flat[f"{name}__{k}"] = np.asarray(v)
    np.savez(path, **flat)


def load_results(path: str) -> dict[str, dict[str, Any]]:
    """Inverse of :func:`save_results`."""
    data = np.load(path, allow_pickle=True)
    out: dict[str, dict[str, Any]] = {}
    for full_key in data.files:
        name, _, k = full_key.partition("__")
        out.setdefault(name, {"name": name})[k] = (
            data[full_key].item() if data[full_key].shape == () else data[full_key]
        )
    return out
