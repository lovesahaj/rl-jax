import math
from typing import Any

import gymnasium as gym
import jaxtyping as jtype
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces

import jax
import jax.numpy as jnp
import jax.random as jran


def dprint(x: Any, name: str):
    print(f"{name} = {x}")


MAX_ITER = 100
MAX_MPC_STEPS = 500
TOLERANCE = 1e-4
LAMBDA = 1e-2
ALPHAS = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]

STATE_DIM = (4,)
ACTION_DIM = (1,)

TERMINAL_ANGLE = jnp.deg2rad(12)
TERMINAL_POS = 2.4

# CartPole physical constants
GRAVITY = 9.8
MASSCART = 1.0
MASSPOLE = 0.1
TOTAL_MASS = MASSCART + MASSPOLE
LENGTH = 0.5
POLEMASS_LENGTH = MASSPOLE * LENGTH

# iLQR parameters
Q = jnp.diag(jnp.array([1, 0.1, 10, 0.1], dtype=jnp.float32))
Q_F = Q.copy() * 10
R = jnp.diag(jnp.array([1.0], dtype=jnp.float32))
X_GOAL = jnp.zeros(STATE_DIM, dtype=jnp.float32)

# Horizon
T = 100
DT = 0.02


def _check_if_terminated(state: jtype.Array) -> bool:
    x, _, theta, _ = state

    return bool(
        x < -TERMINAL_POS
        or x > TERMINAL_POS
        or theta < -TERMINAL_ANGLE
        or theta > TERMINAL_ANGLE
    )


@jax.jit
def dynamics(x: jtype.Array, u: jtype.Array):
    x, x_dot, theta, theta_dot = x
    force = u[0]

    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    temp = (force + POLEMASS_LENGTH * theta_dot**2 * sin_theta) / TOTAL_MASS

    theta_acc = (GRAVITY * sin_theta - cos_theta * temp) / (
        LENGTH * (4.0 / 3.0 - MASSPOLE * cos_theta**2 / TOTAL_MASS)
    )

    x_acc = temp - POLEMASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS

    x = x + DT * x_dot
    x_dot = x_dot + DT * x_acc
    theta = theta + DT * theta_dot
    theta_dot = theta_dot + DT * theta_acc

    return jnp.array([x, x_dot, theta, theta_dot], dtype=jnp.float32)


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


class ProjectEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "console", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode=None, seed=40):
        self.action_space = spaces.Box(-10, 10)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=STATE_DIM,
            dtype=np.float32,
        )

        self.render_mode = render_mode
        self.state = jnp.zeros(STATE_DIM, dtype=jnp.float32)

        self.key = jran.key(seed)

        # Rendering
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

        # iLQR

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.key = jran.key(seed)

        self.state = jnp.array([0, 0.0, 0.1, 1], dtype=jnp.float32)

        info = {}
        return self.state, info

    def step(self, action):
        # assert self.action_space.contains(action), f"Invalid action: {action}"

        self.state = self._get_next_state(action)

        terminated = _check_if_terminated(self.state)
        truncated = False

        reward = 0.0 if terminated else 1.0

        info = {"current_state": "stopped" if terminated or truncated else "moving"}

        return self.state, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "console":
            print(f"Current state: {self.state}")
            return None

        if self.render_mode in {"human", "rgb_array"}:
            return self._render_pygame()

    def _get_next_state(self, force):
        return dynamics(self.state, force)

    def _render_pygame(self):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise ImportError(
                "pygame is required for human/rgb_array rendering. "
                "Install it with: pip install pygame"
            ) from e

        if self.screen is None:
            pygame.init()

            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))

        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = TERMINAL_POS * 2
        scale = self.screen_width / world_width

        cart_y = 300
        pole_width = 10
        pole_len = scale * (2 * LENGTH)

        cart_width = 50
        cart_height = 30

        x = float(self.state[0])
        theta = float(self.state[2])

        screen = self.screen
        screen.fill((255, 255, 255))

        cart_x = x * scale + self.screen_width / 2.0

        # Draw track
        pygame.draw.line(
            screen,
            (0, 0, 0),
            (0, cart_y),
            (self.screen_width, cart_y),
            2,
        )

        # Cart polygon
        left = cart_x - cart_width / 2
        right = cart_x + cart_width / 2
        top = cart_y - cart_height / 2
        bottom = cart_y + cart_height / 2

        cart_coords = [
            (left, bottom),
            (left, top),
            (right, top),
            (right, bottom),
        ]

        gfxdraw.aapolygon(screen, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(screen, cart_coords, (0, 0, 0))

        # Pole polygon
        axle_x = cart_x
        axle_y = top

        pole_angle = -theta + math.pi / 2.0

        pole_dx = pole_len * math.cos(pole_angle)
        pole_dy = -pole_len * math.sin(pole_angle)

        pole_end_x = axle_x + pole_dx
        pole_end_y = axle_y + pole_dy

        pygame.draw.line(
            screen,
            (180, 80, 40),
            (axle_x, axle_y),
            (pole_end_x, pole_end_y),
            pole_width,
        )

        # Axle
        gfxdraw.aacircle(screen, int(axle_x), int(axle_y), 6, (80, 80, 80))
        gfxdraw.filled_circle(screen, int(axle_x), int(axle_y), 6, (80, 80, 80))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(screen)),
                axes=(1, 0, 2),
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()

            self.screen = None
            self.clock = None
            self.isopen = False


# step 1 - this wouldn't work if i used @jax.jit
# def rollout(
#     x0: jtype.Array,
#     U: jtype.Array,
# ):
#     X = jnp.zeros((T, *(STATE_DIM)))
#     X[0] = x0
#     for t in range(T):
#         X[t + 1] = dynamics(X[t], U[t])
#
#     loss = 0
#
#     for t in range(T):
#         loss += l(X[t], U[t])
#
#     loss += lf(X[-1])
#
#     return loss


# step 1 - Jax compatible
@jax.jit
def rollout(
    x0: jtype.Array,
    U: jtype.Array,
):
    def step(x, u):
        x_next = dynamics(x, u)
        cost = l(x, u)
        return x_next, (x_next, cost)
        # x_next -> would be used as the carry
        # for next step and (x_next, cost)
        # is what we want as the output

    x_last, (X_tail, costs) = jax.lax.scan(f=step, init=x0, xs=U)
    X = jnp.concatenate((x0[None, :], X_tail), axis=0)
    loss = jnp.sum(costs) + lf(x_last)

    return X, U, loss


# step 2 - getting the derivatives
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


# def backward_pass(A, B, lx, lu, lxx, luu, lux, Vx, Vxx):
#     k_list = []
#     K_list = []
#
#     for t in reversed(range(T)):
#         At = A[t]
#         Bt = B[t]
#
#         Qx = lx[t] + At.T @ Vx
#         Qu = lu[t] + Bt.T @ Vx
#
#         Qxx = lxx[t] + At.T @ Vxx @ At
#         Quu = luu[t] + Bt.T @ Vxx @ Bt
#         Qux = lux[t] + Bt.T @ Vxx @ At
#
#         Quu = Quu + LAMBDA * jnp.eye(1)
#
#         kt = -jnp.linalg.solve(Quu, Qu)
#         Kt = -jnp.linalg.solve(Quu, Qux)
#
#         Vx = Qx + Kt.T @ Quu @ kt + Kt.T @ Qu + Qux.T @ kt
#         Vxx = Qxx + Kt.T @ Quu @ Kt + Kt.T @ Qux + Qux.T @ Kt
#
#         k_list.append(kt)
#         K_list.append(Kt)
#
#     k = jnp.stack(k_list[::-1])
#     K = jnp.stack(K_list[::-1])
#
#     return k, K


@jax.jit
def backward_pass(A, B, lx, lu, lxx, luu, lux, Vx, Vxx):
    def step(carry, xs):
        Vx, Vxx = carry
        A, B, lx, lu, lxx, luu, lux = xs

        At = A
        Bt = B

        Qx = lx + At.T @ Vx
        Qu = lu + Bt.T @ Vx

        Qxx = lxx + At.T @ Vxx @ At
        Quu = luu + Bt.T @ Vxx @ Bt
        Qux = lux + Bt.T @ Vxx @ At

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
    env = ProjectEnv(render_mode="human")
    x0, info = env.reset()
    U = jnp.zeros((T, *(ACTION_DIM)))

    for mpc_step in range(MAX_MPC_STEPS):
        for i in range(MAX_ITER):
            X, U, loss = rollout(x0, U)

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

            print(i, float(loss), accepted)

            if not accepted:
                break

        x0, reward, terminated, truncated, info = env.step(U[0])
        env.render()

        U = jnp.concatenate((U[1:], U[-1:]), axis=0)

        if terminated or truncated:
            break


if __name__ == "__main__":
    main()
