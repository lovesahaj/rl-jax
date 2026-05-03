import math

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from scipy.linalg import solve_continuous_are
from tqdm import tqdm

import jax.numpy as jnp
import jax.random as jran

NUM_STATES = 4
NUM_ACTIONS = 2
NUM_BINS = 10

ALPHA = 0.1
GAMMA = 0.99

# Define bin edges for each state variable
# Using linspace: the *internal* edges only (no need for outer bounds)
STATE_BINS = [
    jnp.linspace(-2.4, 2.4, NUM_BINS - 1),  # x (cart position)
    jnp.linspace(-3.0, 3.0, NUM_BINS - 1),  # x_dot (cart velocity)
    jnp.linspace(-0.21, 0.21, NUM_BINS - 1),  # theta (pole angle)
    jnp.linspace(-3.0, 3.0, NUM_BINS - 1),  # theta_dot (angular velocity)
]


def make_key(state) -> tuple:
    # Convert JAX array to numpy so np.digitize works
    state = np.array(state)

    key = tuple(
        int(jnp.clip(jnp.digitize(s, bins), 0, NUM_BINS - 1))
        for s, bins in zip(state, STATE_BINS)
    )
    return key


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

        # Q Table
        self.q_table = jnp.zeros(
            [NUM_BINS] * NUM_STATES + [NUM_ACTIONS],
            dtype=jnp.float32,
        )
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.key = jran.key(seed)

        self.key, subkey = jran.split(self.key)
        self.state = jran.uniform(
            subkey, shape=(NUM_STATES,), minval=-0.05, maxval=0.05
        )
        return self.state, {}

    def step(self, action):
        # assert self.action_space.contains(action), f"Invalid action: {action}"

        self.state = self._get_next_state(action)

        terminated = self._check_if_terminated(self.state)
        truncated = False

        cost = -100.0 if terminated else -self.cost()

        info = {"current_state": "stopped" if terminated or truncated else "moving"}

        return self.state, cost, terminated, truncated, info

    def cost(self):
        x, x_dot, theta, theta_dot = self.state

        return 0.01 * abs(x) + 0.1 * (x**2) + 1 * (theta**2) + 0.1 * abs(theta_dot)

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
        """
        need to implement the epsilon greedy

        A = [
            1 - epsilon + epsilon / |action_space|, if action == max_action
            epsilon / |action_space| otherwise
        ]
        """

        probs = jnp.ones(NUM_ACTIONS, dtype=jnp.float32)
        actions = self.q_table[*make_key(self.state), :]

        max_action_idx = int(jnp.argmax(actions))
        probs = probs * (self.epsilon / NUM_ACTIONS)
        probs = probs.at[max_action_idx].set(
            1 - self.epsilon + (self.epsilon / NUM_ACTIONS)
        )
        self.key, subkey = jran.split(self.key)

        return jran.choice(
            subkey,
            jnp.arange(0, NUM_ACTIONS),
            p=probs,
        )

    def q_update(self, cur_state, next_state, action, reward, terminated):
        """
        Q(s, a) <- Q(s, a) + ALPHA * (r + GAMMA * max_a' Q(s', a') - Q(s, a))
        """
        cur_key = make_key(cur_state)
        next_key = make_key(next_state)

        current_q = self.q_table[*cur_key, action]
        if terminated:
            target_q = reward
        else:
            target_q = reward + GAMMA * jnp.max(self.q_table[*next_key])

        self.q_table = self.q_table.at[*cur_key, action].set(
            current_q + ALPHA * (target_q - current_q)
        )


def train(num_episodes=2000):
    env = ProjectEnv()
    episode_rewards = []

    loop = tqdm(range(num_episodes))
    for ep in loop:
        env.reset()
        total_reward = 0

        for _ in range(500):
            action = int(env.choose_action())
            cur_state = env.state
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = env.state
            env.q_update(cur_state, next_state, action, reward, terminated)
            total_reward += reward

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        loop.set_postfix({"Reward": total_reward})
        env.update_epsilon()

    return episode_rewards, env


def test(env):
    env.render_mode = "human"
    episode_rewards = []

    for ep in range(50):
        env.reset()
        total_reward = 0

        for _ in range(500):
            action = int(env.choose_action())
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            env.render()

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        env.update_epsilon()

    return episode_rewards


if __name__ == "__main__":
    rewards, env = train()
    test(env)
