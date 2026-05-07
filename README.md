# CartPole Control with JAX

Implementations of classical control and reinforcement-learning algorithms for CartPole, written in JAX. Covers LQR, iLQR, MPC, tabular Q-learning / SARSA / Monte-Carlo control, and a Deep Q-Network — all benchmarked on a shared evaluation harness (`bench/`) that scores every method on the same canonical reward and quadratic cost so the numbers are directly comparable.

## Benchmark results

Every method runs through `bench.evaluate` on perturbed initial states (`[0, 0, 0.1, 1] + 𝒰(-0.05, 0.05)⁴`) and is scored against shared `Q = diag(1, 1, 100, 10), R = [[1]]`. RL methods are wrapped with `bench.with_canonical_reward` so their `total_reward` is steps survived, regardless of training-time reward shaping.

| Method     | Balanced Steps | Final Cost | Mean Test Reward | Stability % | Force Smoothness | Notes |
| :--------- | -------------: | ---------: | ---------------: | ----------: | ---------------: | :---- |
| LQR        |          500.0 |     50,625 |           500.00 |        100% |            15.00 | Bang-bang ±10 N — robust but high control cost |
| iLQR       |           35.0 |        958 |            34.00 |          0% |             0.40 | Open-loop plan; collapses on perturbed starts |
| MPC        |          909.2 |  **1,294** |       **909.10** |         90% |         **0.07** | Best closed-loop controller |
| Q-learning |          193.1 |     19,986 |           192.15 |          3% |            14.11 | Tabular, off-policy |
| SARSA      |          329.4 |     33,931 |           328.37 |          1% |            15.22 | Tabular, on-policy |
| MC-control |          186.4 |     19,490 |           185.41 |          3% |            11.25 | First-visit Monte-Carlo |
| DeepQ      |          500.0 |     50,257 |           500.00 |        100% |            13.01 | Neural Q-network, learned bang-bang |

`Final Cost` = mean Σ (xᵀQx + uᵀRu) over the rollout. `Stability %` = fraction of episodes balanced for ≥500 steps without termination.

### Headlines

- **MPC dominates closed-loop**: 90% stability, lowest cost (1,294 vs 50,625 for LQR — a **39× reduction**), smoothest control (0.07 vs 15.0 N — a **>200× reduction in actuator chatter**), all at ~0.3 ms per receding-horizon solve and ~1 iteration per step after warm-start.
- **iLQR ≠ MPC.** Once a real convergence tolerance was wired in, iLQR settled in 28.8 ± 22 iterations — but its open-loop replay survives 0/20 perturbed rollouts. The lack of feedback, not the optimizer, is what kills it.
- **LQR & DeepQ both achieve 100% stability** but in a high-cost bang-bang regime (`force_mean_abs = 10` exactly). Stability without smoothness — useful baselines, not desirable controllers.
- **Tabular RL (Q / SARSA / MC) balances partially but rarely fully**: SARSA has the longest mean survival (329 steps) but 1% full-stability rate. The (10-bin)⁴ × 2 discretisation is the ceiling.
- **DeepQ closes the gap** to LQR-level stability but converges to the same bang-bang policy — the network learned a robust switch, not a smooth controller.

## Algorithms

| File                     | Algorithm                                | Action space          |
| ------------------------ | ---------------------------------------- | --------------------- |
| `control/lqr.py`         | Linear Quadratic Regulator (LQR)         | Discrete (left/right) |
| `control/ilqr.py`        | Iterative LQR (iLQR), open-loop          | Continuous force      |
| `control/mpc.py`         | Model Predictive Control via iLQR        | Continuous force      |
| `rl/q_learning.py`       | Tabular Q-learning, ε-greedy             | Discrete              |
| `rl/sarsa.py`            | Tabular SARSA, on-policy                 | Discrete              |
| `rl/mc_control.py`       | First-visit Monte-Carlo control          | Discrete              |
| `rl/deep_q_learning.py`  | Deep Q-Network (Flax NNX)                | Discrete              |
| `bench/`                 | Shared rollout / metrics / table helpers | —                     |

### LQR (`control/lqr.py`)

Linearises the CartPole dynamics around the upright equilibrium and solves the continuous algebraic Riccati equation (CARE) via `scipy.linalg.solve_continuous_are` to get the optimal feedback gain **K**. The continuous force is thresholded to a discrete left/right action — the source of LQR's bang-bang behaviour in the table above.

### iLQR (`control/ilqr.py`)

Full nonlinear trajectory optimisation over a horizon of T=200 steps. Exact first- and second-order derivatives of the dynamics and cost are computed automatically with `jax.jacfwd` and `jax.hessian`. Backward pass (Riccati recursion) and forward pass (line search) run inside `jax.lax.scan` so they compile under `@jax.jit`. Convergence is detected by relative loss tolerance (`1e-4`).

### MPC (`control/mpc.py`)

Receding-horizon control: at each environment step, iLQR is re-run from the current state over a T=150 horizon, only the first action is applied, and the control sequence is warm-started by shifting. The terminal cost uses the discrete-time infinite-horizon LQR Riccati solution (`solve_discrete_are`) so the finite horizon inherits the long-run optimal value function. After warm-start, each step typically converges in 1 iteration (`mean_iters/step ≈ 1.01`).

### Tabular RL (`rl/q_learning.py`, `rl/sarsa.py`, `rl/mc_control.py`)

Q-learning, SARSA, and first-visit Monte-Carlo control over a 10-bin × 10-bin × 10-bin × 10-bin discretisation of the state space. Each uses a different cost shaping during training (`-cost`, `-100/-500` terminal penalties) but the benchmark wraps the env with `bench.with_canonical_reward` so test-time reward is steps-survived.

### Deep Q-Network (`rl/deep_q_learning.py`)

3-layer MLP Q-function (Flax NNX, 128 hidden units), Adam optimiser, ε-greedy exploration with decay. Reward shaping = `1 - cost`. Logs to Weights & Biases during training.

## State space

The CartPole state is `[x, ẋ, θ, θ̇]` — cart position, cart velocity, pole angle, pole angular velocity. An episode terminates when `|x| > 2.4 m` or `|θ| > 12°`.

## Setup

Requires Python ≥ 3.13. Dependencies are managed with [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

Or with pip:

```bash
pip install jax gymnasium flax optax matplotlib pygame
```

## Running

Run from the project root using module syntax so package imports resolve:

```bash
# Controllers (instant; no training)
uv run python -m control.lqr
uv run python -m control.ilqr
uv run python -m control.mpc

# Reinforcement learning (~minutes; trains then benchmarks)
uv run python -m rl.q_learning
uv run python -m rl.sarsa
uv run python -m rl.mc_control
uv run python -m rl.deep_q_learning
```

Each script writes `output/<method>_metrics.npz`. Combine them into the comparison table with:

```python
from bench import load_results, format_markdown_table
import glob
summaries = {}
for f in sorted(glob.glob("output/*_metrics.npz")):
    summaries.update(load_results(f))
order = ["LQR", "iLQR", "MPC", "Q-learning", "SARSA", "MC-control", "DeepQ"]
print(format_markdown_table([summaries[n] for n in order if n in summaries]))
```

## Benchmark module (`bench/`) - Written by Claude

A small library that every algorithm script imports:

- `rollout(env, policy, max_steps, seed)` — runs one episode, returns trajectory.
- `compute_episode_metrics(stats, Q, R, balanced_threshold)` — episode length, total reward, `Σ xᵀQx + uᵀRu`, force mean / peak / smoothness, balanced flag.
- `evaluate(name, env_factory, policy_factory, num_episodes, max_steps, Q, R)` — high-level loop returning a summary dict.
- `time_callable(fn, n_runs, warmup)` — wall-clock helper for JIT-vs-no-JIT timing.
- `with_canonical_reward(env)` — patches `env.step` so reward = 1 / step (cross-method comparability).
- `BENCH_Q`, `BENCH_R` — canonical cost weights every method is scored against.
- `save_results` / `load_results` — `.npz` round-trip for summaries.
- `format_markdown_table(summaries)` — README-ready table.

Policies are `policy(env) -> action` or `policy(env) -> (action, force)`. Both discrete and continuous action spaces are handled.

## Key JAX patterns

- `@jax.jit` on `dynamics`, `rollout`, `backward_pass`, `forward_pass` for compiled execution.
- `jax.lax.scan` replaces Python for-loops so the entire rollout and Riccati recursion are JIT-compatible.
- `jax.vmap` vectorises derivative computations over the time horizon.
- `jax.jacfwd` / `jax.hessian` / `jax.grad` provide exact linearisations — no finite-differences needed.
