import os

import jaxtyping as jtype
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces

import jax
import jax.numpy as jnp

from env.cartpole import CartPoleEnv, dynamics

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
        self.state = jnp.array([0, 0.0, 0.1, 1], dtype=jnp.float32)
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


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    env = ProjectEnv(render_mode="console")
    x0, info = env.reset()
    U = jnp.zeros((T, *(ACTION_DIM)))
    X, U, loss = rollout(env.state, U)

    loss_history = []

    for i in range(MAX_ITER):
        A, B, lx, lu, lxx, luu, lux, Vx, Vxx = derivatives(X, U)
        k, K = backward_pass(A, B, lx, lu, lxx, luu, lux, Vx, Vxx)

        accepted = False

        for alpha in ALPHAS:
            U_new, X_new, loss_new = forward_pass(x0, U, X, k, K, alpha)

            if loss_new < loss:
                X = X_new
                U = U_new
                loss = loss_new
                accepted = True
                break

        loss_history.append(float(loss))
        print(i, float(loss), accepted)

        if not accepted:
            break

    state_history = []
    action_history = []

    for t in range(T):
        state_history.append(np.array(env.state))
        action_history.append(np.array(U[t]))
        obs, reward, terminated, truncated, info = env.step(U[t])
        env.render()

        if terminated or truncated:
            break

    env.close()

    state_history = np.array(state_history)
    action_history = np.array(action_history)
    loss_history = np.array(loss_history)

    np.savez(
        os.path.join(OUTPUT_DIR, "ilqr_results.npz"),
        state_history=state_history,
        action_history=action_history,
        loss_history=loss_history,
    )

    time = np.arange(len(state_history)) * DT

    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("iLQR Optimization Loss")
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, "ilqr_loss.png"), dpi=150, bbox_inches="tight")
    plt.close()

    labels = [
        "Cart position [m]",
        "Cart velocity [m/s]",
        "Pole angle [rad]",
        "Angular velocity [rad/s]",
    ]
    fnames = [
        "ilqr_cart_position.png",
        "ilqr_cart_velocity.png",
        "ilqr_pole_angle.png",
        "ilqr_angular_velocity.png",
    ]

    for i, (label, fname) in enumerate(zip(labels, fnames)):
        plt.figure()
        plt.plot(time, state_history[:, i])
        plt.xlabel("Time [s]")
        plt.ylabel(label)
        plt.title(label)
        plt.grid()
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches="tight")
        plt.close()

    plt.figure()
    plt.plot(time, action_history[:, 0])
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.title("iLQR Control Force")
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, "ilqr_force.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
