import os
import time

import jaxtyping as jtype
import numpy as np
from gymnasium import spaces

import jax
import jax.numpy as jnp
import jax.random as jran

from bench import (
    BENCH_Q,
    BENCH_R,
    aggregate,
    compute_episode_metrics,
    save_results,
)
from bench import rollout as bench_rollout
from env.cartpole import NUM_STATES, CartPoleEnv, dynamics

OUTPUT_DIR = "output"

MAX_ITER = 500
TOLERANCE = 1e-4
LAMBDA = 1e-6
ALPHAS = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]

STATE_DIM = (4,)
ACTION_DIM = (1,)

Q = jnp.diag(jnp.array([1, 0.1, 10, 0.1], dtype=jnp.float32))
Q_F = Q.copy() * 10
R = jnp.diag(jnp.array([1.0], dtype=jnp.float32))
X_GOAL = jnp.zeros(STATE_DIM, dtype=jnp.float32)

T = 200
DT = 0.02

A_fn = jax.jacfwd(dynamics, argnums=0)
B_fn = jax.jacfwd(dynamics, argnums=1)


@jax.jit
def l(x: jtype.Array, u: jtype.Array) -> jtype.Float:
    diff = x - X_GOAL
    return diff.T @ Q @ diff + u.T @ R @ u


lx_fn = jax.grad(l, argnums=0)
lu_fn = jax.grad(l, argnums=1)
lxx_fn = jax.hessian(l, argnums=0)
luu_fn = jax.hessian(l, argnums=1)
lux_fn = jax.jacfwd(jax.grad(l, argnums=1), argnums=0)


@jax.jit
def lf(x):
    diff = x - X_GOAL
    return diff.T @ Q_F @ diff


Vx_fn = jax.grad(lf)
Vxx_fn = jax.hessian(lf)


class ProjectEnv(CartPoleEnv):
    def __init__(self, render_mode=None, seed=40):
        super().__init__(spaces.Box(-10, 10), render_mode, seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.key, subkey = jran.split(self.key)
        perturb = jran.uniform(
            subkey, shape=(NUM_STATES,), minval=-0.05, maxval=0.05
        )
        self.state = jnp.array([0.0, 0.0, 0.1, 1.0], dtype=jnp.float32) + perturb
        return self.state, {}

    def _get_next_state(self, force):
        return dynamics(self.state, force)


@jax.jit
def rollout(x0: jtype.Array, U: jtype.Array):
    def step(x, u):
        x_next = dynamics(x, u)
        cost = l(x, u)
        return x_next, (x_next, cost)

    x_last, (X_tail, costs) = jax.lax.scan(f=step, init=x0, xs=U)
    X = jnp.concatenate((x0[None, :], X_tail), axis=0)
    loss = jnp.sum(costs) + lf(x_last)

    return X, U, loss


@jax.jit
def derivatives(X, U):
    A = jax.vmap(A_fn)(X[:-1], U)
    B = jax.vmap(B_fn)(X[:-1], U)
    lx = jax.vmap(lx_fn)(X[:-1], U)
    lu = jax.vmap(lu_fn)(X[:-1], U)
    lxx = jax.vmap(lxx_fn)(X[:-1], U)
    luu = jax.vmap(luu_fn)(X[:-1], U)
    lux = jax.vmap(lux_fn)(X[:-1], U)
    Vx = Vx_fn(X[-1])
    Vxx = Vxx_fn(X[-1])
    return A, B, lx, lu, lxx, luu, lux, Vx, Vxx


@jax.jit
def backward_pass(A, B, lx, lu, lxx, luu, lux, Vx, Vxx):
    def step(carry, xs):
        Vx, Vxx = carry
        A, B, lx, lu, lxx, luu, lux = xs

        Qx = lx + A.T @ Vx
        Qu = lu + B.T @ Vx
        Qxx = lxx + A.T @ Vxx @ A
        Quu = luu + B.T @ Vxx @ B
        Qux = lux + B.T @ Vxx @ A

        Quu = Quu + LAMBDA * jnp.eye(1)

        kt = -jnp.linalg.solve(Quu, Qu)
        Kt = -jnp.linalg.solve(Quu, Qux)

        Vx = Qx + Kt.T @ Quu @ kt + Kt.T @ Qu + Qux.T @ kt
        Vxx = Qxx + Kt.T @ Quu @ Kt + Kt.T @ Qux + Qux.T @ Kt

        return (Vx, Vxx), (kt, Kt)

    _, (k, K) = jax.lax.scan(
        f=step,
        init=(Vx, Vxx),
        xs=(A, B, lx, lu, lxx, luu, lux),
        reverse=True,
    )

    return k, K


@jax.jit
def forward_pass(x0, U, X, k, K, alpha):
    def step(x_new, xs):
        x_old, u_old, kt, Kt = xs
        dx = x_new - x_old
        u_new = u_old + alpha * kt + Kt @ dx
        u_new = jnp.clip(u_new, -10.0, 10.0)
        x_next = dynamics(x_new, u_new)
        return x_next, (x_next, u_new)

    _, (X_tail, U_new) = jax.lax.scan(
        f=step,
        init=x0,
        xs=(X[:-1], U, k, K),
    )

    X_new = jnp.concatenate((x0[None, :], X_tail), axis=0)
    running_cost = jax.vmap(l)(X_new[:-1], U_new)
    loss_new = jnp.sum(running_cost) + lf(X_new[-1])

    return U_new, X_new, loss_new


def _solve(x0):
    """Run iLQR once, returning the optimised U and the iteration count.

    Stops on either: line-search rejection (no α improves loss) or relative
    loss change below ``TOLERANCE``.
    """
    U = jnp.zeros((T, *(ACTION_DIM)))
    X, U, loss = rollout(x0, U)
    iters = 0
    for i in range(MAX_ITER):
        prev_loss = float(loss)
        A, B, lx, lu, lxx, luu, lux, Vx, Vxx = derivatives(X, U)
        k, K = backward_pass(A, B, lx, lu, lxx, luu, lux, Vx, Vxx)
        accepted = False
        for alpha in ALPHAS:
            U_new, X_new, loss_new = forward_pass(x0, U, X, k, K, alpha)
            if loss_new < loss:
                X, U, loss = X_new, U_new, loss_new
                accepted = True
                break
        iters = i + 1
        if not accepted:
            break
        if abs(prev_loss - float(loss)) / max(1.0, abs(prev_loss)) < TOLERANCE:
            break
    return U, iters, float(loss)


def benchmark():
    Q_np = BENCH_Q
    R_np = BENCH_R
    per_episode = []
    iters_log = []
    solve_times = []

    for seed in range(20):
        env = ProjectEnv(seed=seed)
        x0, _ = env.reset()

        t0 = time.perf_counter()
        U_opt, iters, _ = _solve(x0)
        solve_times.append(time.perf_counter() - t0)
        iters_log.append(iters)

        U_np = np.asarray(U_opt)
        counter = {"i": 0}

        def policy(env, U=U_np, c=counter):
            i = min(c["i"], len(U) - 1)
            u = U[i]
            c["i"] = i + 1
            return jnp.asarray(u, dtype=jnp.float32), float(u[0])

        stats = bench_rollout(env, policy, max_steps=T)
        per_episode.append(compute_episode_metrics(stats, Q=Q_np, R=R_np, balanced_threshold=T))
        env.close()

    summary = aggregate(per_episode, name="iLQR")
    summary["convergence_iters_mean"] = float(np.mean(iters_log))
    summary["convergence_iters_std"] = float(np.std(iters_log))
    summary["solve_mean_s"] = float(np.mean(solve_times))
    summary["solve_std_s"] = float(np.std(solve_times))

    save_results(os.path.join(OUTPUT_DIR, "ilqr_metrics.npz"), [summary])
    print(f"iLQR benchmark saved to {OUTPUT_DIR}/ilqr_metrics.npz")
    print(f"  stability_rate={summary.get('stability_rate', 0):.2%}  "
          f"mean_iters={summary['convergence_iters_mean']:.1f}  "
          f"mean_solve={summary['solve_mean_s']*1e3:.1f}ms")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    benchmark()
