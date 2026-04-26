import math

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from scipy.linalg import solve_continuous_are

import jax.numpy as jnp
import jax.random as jran

NUM_STATES = 4
NUM_ACTIONS = 2


class ProjectEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "console", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode=None, seed=40):
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(NUM_STATES,),
            dtype=np.float32,
        )

        self.render_mode = render_mode
        self.state = jnp.zeros(NUM_STATES, dtype=jnp.float32)

        self.key = jran.key(seed)

        self.terminal_angle = jnp.deg2rad(12)
        self.terminal_pos = 2.4

        # CartPole physical constants
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02

        # Rendering
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

        # LQR
        self.A = jnp.array(
            [
                [0, 1, 0, 0],
                [0, 0, -(self.masspole * self.gravity) / self.masscart, 0],
                [0, 0, 0, 1],
                [
                    0,
                    0,
                    ((self.masscart + self.masspole) * self.gravity)
                    / (self.length * self.masscart),
                    0,
                ],
            ],
        )

        B = jnp.array([0, 1 / self.masscart, 0, -1 / (self.length * self.masscart)])
        self.B = B[:, None]

        self.Q = jnp.diag(jnp.array([1, 1, 100, 10], dtype=jnp.float32))
        self.R = jnp.array([[1]], dtype=jnp.float32)

        self.P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = jnp.linalg.solve(self.R, self.B.T @ self.P)

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

        terminated = self._check_if_terminated(self.state)
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

    def _get_next_state(self, action):
        x, x_dot, theta, theta_dot = self.state

        force = self.force_mag if action == 1 else -self.force_mag

        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)

        temp = (
            force + self.polemass_length * theta_dot**2 * sin_theta
        ) / self.total_mass

        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * cos_theta**2 / self.total_mass)
        )

        x_acc = temp - self.polemass_length * theta_acc * cos_theta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc

        return jnp.array([x, x_dot, theta, theta_dot], dtype=jnp.float32)

    def _check_if_terminated(self, state):
        x, _, theta, _ = state

        return bool(
            x < -self.terminal_pos
            or x > self.terminal_pos
            or theta < -self.terminal_angle
            or theta > self.terminal_angle
        )

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

        world_width = self.terminal_pos * 2
        scale = self.screen_width / world_width

        cart_y = 300
        pole_width = 10
        pole_len = scale * (2 * self.length)

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

    def choose_action(self):
        state = self.state
        F = -self.K @ state
        F_scalar = float(F.item())

        return 1 if F_scalar > 0 else 0


def main():
    env = ProjectEnv(render_mode="human")
    obs, info = env.reset()

    state_history = []
    force_history = []
    action_history = []
    reward_history = []

    for _ in range(500):
        state_history.append(env.state)
        F_scalar = env.choose_action()
        action = 1 if F_scalar > 0 else 0
        force_history.append(F_scalar)
        action_history.append(action)

        obs, reward, terminated, truncated, info = env.step(action)
        reward_history.append(reward)

        env.render()

        if terminated or truncated:
            break

    env.close()

    state_history = np.array(state_history)
    force_history = np.array(force_history)
    action_history = np.array(action_history)
    reward_history = np.array(reward_history)

    time = np.arange(len(state_history)) * env.tau

    plt.figure()
    plt.plot(time, state_history[:, 0])
    plt.xlabel("Time [s]")
    plt.ylabel("Cart position x [m]")
    plt.title("Cart Position")
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(time, state_history[:, 1])
    plt.xlabel("Time [s]")
    plt.ylabel("Cart velocity x_dot [m/s]")
    plt.title("Cart Velocity")
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(time, state_history[:, 2])
    plt.axhline(float(env.terminal_angle), linestyle="--")
    plt.axhline(float(-env.terminal_angle), linestyle="--")
    plt.xlabel("Time [s]")
    plt.ylabel("Pole angle theta [rad]")
    plt.title("Pole Angle")
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(time, state_history[:, 3])
    plt.xlabel("Time [s]")
    plt.ylabel("Angular velocity theta_dot [rad/s]")
    plt.title("Pole Angular Velocity")
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(time, force_history)
    plt.xlabel("Time [s]")
    plt.ylabel("LQR force F")
    plt.title("Raw LQR Force")
    plt.grid()
    plt.show()

    plt.figure()
    plt.step(time, action_history, where="post")
    plt.xlabel("Time [s]")
    plt.ylabel("Discrete action")
    plt.title("Discrete Action Chosen from LQR Force")
    plt.yticks([0, 1], ["left", "right"])
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
