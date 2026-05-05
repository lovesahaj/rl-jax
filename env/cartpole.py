import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import jax
import jax.numpy as jnp
import jax.random as jran

GRAVITY = 9.8
MASSCART = 1.0
MASSPOLE = 0.1
TOTAL_MASS = MASSCART + MASSPOLE
LENGTH = 0.5
POLEMASS_LENGTH = MASSPOLE * LENGTH
FORCE_MAG = 10.0
TAU = 0.02

TERMINAL_ANGLE = jnp.deg2rad(12)
TERMINAL_POS = 2.4

NUM_STATES = 4
NUM_ACTIONS = 2


@jax.jit
def dynamics(x: jax.Array, u: jax.Array) -> jax.Array:
    """Continuous cartpole dynamics. u is shape (1,) containing the applied force."""
    x_pos, x_dot, theta, theta_dot = x
    force = u[0]

    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    temp = (force + POLEMASS_LENGTH * theta_dot**2 * sin_theta) / TOTAL_MASS
    theta_acc = (GRAVITY * sin_theta - cos_theta * temp) / (
        LENGTH * (4.0 / 3.0 - MASSPOLE * cos_theta**2 / TOTAL_MASS)
    )
    x_acc = temp - POLEMASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS

    x_pos = x_pos + TAU * x_dot
    x_dot = x_dot + TAU * x_acc
    theta = theta + TAU * theta_dot
    theta_dot = theta_dot + TAU * theta_acc

    return jnp.array([x_pos, x_dot, theta, theta_dot], dtype=jnp.float32)


class CartPoleEnv(gym.Env):
    """Base CartPole environment. Subclass to customise action space, reset state, and reward."""

    metadata = {
        "render_modes": ["human", "console", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, action_space, render_mode=None, seed=40):
        self.action_space = action_space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(NUM_STATES,), dtype=np.float32
        )
        self.render_mode = render_mode
        self.state = jnp.zeros(NUM_STATES, dtype=jnp.float32)
        self.key = jran.key(seed)

        self.gravity = GRAVITY
        self.masscart = MASSCART
        self.masspole = MASSPOLE
        self.total_mass = TOTAL_MASS
        self.length = LENGTH
        self.polemass_length = POLEMASS_LENGTH
        self.force_mag = FORCE_MAG
        self.tau = TAU
        self.terminal_angle = TERMINAL_ANGLE
        self.terminal_pos = TERMINAL_POS

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.key = jran.key(seed)
        self.state = jnp.zeros(NUM_STATES, dtype=jnp.float32)
        return self.state, {}

    def step(self, action):
        self.state = self._get_next_state(action)
        terminated = self._check_if_terminated(self.state)
        truncated = False
        reward = 0.0 if terminated else 1.0
        info = {"current_state": "stopped" if terminated or truncated else "moving"}
        return self.state, reward, terminated, truncated, info

    def _get_next_state(self, action):
        """Discrete action (0 → left, 1 → right) to next state."""
        force = self.force_mag if action == 1 else -self.force_mag
        return dynamics(self.state, jnp.array([force], dtype=jnp.float32))

    def _check_if_terminated(self, state):
        x, _, theta, _ = state
        return bool(
            x < -self.terminal_pos
            or x > self.terminal_pos
            or theta < -self.terminal_angle
            or theta > self.terminal_angle
        )

    def render(self):
        if self.render_mode == "console":
            print(f"Current state: {self.state}")
            return None
        if self.render_mode in {"human", "rgb_array"}:
            return self._render_pygame()

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

        pygame.draw.line(screen, (0, 0, 0), (0, cart_y), (self.screen_width, cart_y), 2)

        left = cart_x - cart_width / 2
        right = cart_x + cart_width / 2
        top = cart_y - cart_height / 2
        bottom = cart_y + cart_height / 2
        cart_coords = [(left, bottom), (left, top), (right, top), (right, bottom)]
        gfxdraw.aapolygon(screen, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(screen, cart_coords, (0, 0, 0))

        axle_x = cart_x
        axle_y = top
        pole_angle = -theta + math.pi / 2.0
        pole_dx = pole_len * math.cos(pole_angle)
        pole_dy = -pole_len * math.sin(pole_angle)
        pole_end_x = axle_x + pole_dx
        pole_end_y = axle_y + pole_dy

        pygame.draw.line(
            screen, (180, 80, 40), (axle_x, axle_y), (pole_end_x, pole_end_y), pole_width
        )
        gfxdraw.aacircle(screen, int(axle_x), int(axle_y), 6, (80, 80, 80))
        gfxdraw.filled_circle(screen, int(axle_x), int(axle_y), 6, (80, 80, 80))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None

        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None
            self.isopen = False
