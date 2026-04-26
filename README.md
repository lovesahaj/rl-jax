# CartPole Control with JAX

Implementations of classical and model-based control algorithms for the CartPole environment, written in JAX. Covers LQR, iLQR, and MPC with a custom Gymnasium environment and optional PPO scaffolding.

## Algorithms

| File          | Algorithm                            | Action space          |
| ------------- | ------------------------------------ | --------------------- |
| `lqr.py`      | Linear Quadratic Regulator (LQR)     | Discrete (left/right) |
| `ilqr.py`     | Iterative LQR (iLQR)                 | Continuous force      |
| `mpc.py`      | Model Predictive Control via iLQR    | Continuous force      |
| `cartpole.py` | PPO actor-critic scaffold (Flax NNX) | Discrete              |

### LQR (`lqr.py`)

Linearises the CartPole dynamics around the upright equilibrium and solves the continuous algebraic Riccati equation (CARE) via `scipy.linalg.solve_continuous_are` to get the optimal feedback gain **K**. The continuous force is thresholded to a discrete left/right action. Produces matplotlib plots of cart position, velocity, pole angle, angular velocity, and applied force.

### iLQR (`ilqr.py`)

Full nonlinear trajectory optimisation over a horizon of T=200 steps. Exact first- and second-order derivatives of the dynamics and cost are computed automatically with `jax.jacfwd` and `jax.hessian`. The backward pass (Riccati recursion) and forward pass (line search) are implemented with `jax.lax.scan` so they compile under `@jax.jit`.

### MPC (`mpc.py`)

Receding-horizon control: at each environment step, iLQR is re-run from the current state over a T=100 horizon, only the first action is applied, and the control sequence is warm-started by shifting. This gives closed-loop feedback without a terminal controller.

### PPO scaffold (`cartpole.py`)

Actor-critic skeleton using **Flax NNX** (`nnx.Linear`, `nnx.relu`) and **Optax**. Rollout buffer and value estimation utilities are defined; full PPO update loop is in progress.

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

```bash
# LQR with pygame visualisation + matplotlib plots
uv run python lqr.py

# iLQR open-loop trajectory optimisation
uv run python ilqr.py

# MPC (receding horizon iLQR)
uv run python mpc.py
```

All three scripts open a pygame window for live rendering and print iteration/loss information to stdout.

## Key JAX patterns

- `@jax.jit` on `dynamics`, `rollout`, `backward_pass`, `forward_pass` for compiled execution.
- `jax.lax.scan` replaces Python for-loops so the entire rollout and Riccati recursion are JIT-compatible.
- `jax.vmap` vectorises derivative computations over the time horizon.
- `jax.jacfwd` / `jax.hessian` / `jax.grad` provide exact linearisations — no finite-differences needed.
