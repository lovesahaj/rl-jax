import math

import gymnasium as gym
import optax
from flax import nnx
from gymnasium import spaces

import jax
import jax.numpy as jnp
import jax.random as jran

ROLLOUT_BUFFER = 512
BATCH_SIZE = 64
EPOCHS = 10
DISCOUNT_FACTOR = 0.99
CLIPPING_FACTOR = 0.2
LEARNING_RATE = 3e-4

NUM_STATES = 4
NUM_ACTIONS = 2


class ProjectEnv(gym.Env):
    metadata = {"render_modes": ["human", "console"]}

    def __init__(self, render_mode=None, seed=40):
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.observation_space = spaces.Box(
            low=-jnp.inf, high=jnp.inf, shape=(NUM_STATES,), dtype=jnp.float32
        )

        self.render_mode = render_mode
        self.state = jnp.zeros(NUM_STATES, dtype=jnp.float32)

        self.key = jran.key(seed=seed)

        self.terminal_angle = jnp.deg2rad(12)
        self.terminal_pos = 2.4

        # CartPole Physical Constants
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates (kinematic integration step)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = jnp.zeros(4, dtype=jnp.float32)
        info = {}

        return self.state, info

    def step(self, action):
        self.state = self._get_next_state(action)

        reward = self._calculate_reward(self.state, action)

        terminated = self._check_if_terminated(self.state)
        truncated = False

        info = {"current_state": "stopped" if terminated or truncated else "moving"}

        return self.state, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "console":
            print(f"Current state: {self.state}")

    def _get_next_state(self, action):
        x, x_dot, theta, theta_dot = self.state

        force = self.force_mag if action == 1 else -self.force_mag

        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        temp = (
            force + self.polemass_length * theta_dot**2 * sin_theta
        ) / self.total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * cos_theta**2 / self.total_mass)
        )
        # Linear acceleration of the cart
        x_acc = temp - self.polemass_length * theta_acc * cos_theta / self.total_mass

        # Euler Integration to get the new state
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc

        # Return as a float32 numpy array
        return jnp.array([x, x_dot, theta, theta_dot], dtype=jnp.float32)

    def _calculate_reward(self, state, action):
        # In CartPole, you get +1 reward for every step you survive
        # If the state is terminated, you usually don't get a reward for that final fatal step.
        terminated = self._check_if_terminated(state)
        if not terminated:
            return 1.0
        else:
            return 0.0

    def _check_if_terminated(self, state):
        x, _, theta, _ = state

        return bool(
            x < -self.terminal_pos
            or x > self.terminal_pos
            or theta < -self.terminal_angle
            or theta > self.terminal_angle
        )


class NeuralNetwork(nnx.Module):
    def __init__(self, n_features, n_hiddens, n_targets, *, rngs: nnx.Rngs):
        self.n_features = n_features
        self.layer1 = nnx.Linear(n_features, n_hiddens, rngs=rngs)
        self.layer2 = nnx.Linear(n_hiddens, n_hiddens, rngs=rngs)
        self.layer3 = nnx.Linear(n_hiddens, n_targets, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.layer1(x))
        x = nnx.relu(self.layer2(x))
        x = self.layer3(x)
        return x


@jax.jit
def action_selection(log_probs: jax.Array) -> jax.Array:
    return jnp.expand_dims(
        jnp.argmax(log_probs, axis=2), axis=2
    )  # this would give me (ROLLOUT_BUFFER, 1)


@jax.jit
def value_estimation(critic: nnx.Module, state: jax.Array) -> jax.Array:
    return critic(state)


def rollout(cartpole: ProjectEnv):
    state, _ = cartpole.reset()


def main():
    cartpole = ProjectEnv(render_mode="console")
    cartpole.reset()

    actor = NeuralNetwork(
        n_features=4,
        n_hiddens=64,
        n_targets=2,
        rngs=nnx.Rngs(0),
    )
    critic = NeuralNetwork(
        n_features=4,
        n_hiddens=64,
        n_targets=1,
        rngs=nnx.Rngs(0),
    )

    states = jnp.zeros((BATCH_SIZE, ROLLOUT_BUFFER, NUM_STATES), dtype=jnp.float32)
    actions = jnp.zeros((BATCH_SIZE, ROLLOUT_BUFFER, NUM_ACTIONS), dtype=jnp.float32)
    log_probs = jnp.zeros((BATCH_SIZE, ROLLOUT_BUFFER, 1), dtype=jnp.float32)
    rewards = jnp.zeros((BATCH_SIZE, ROLLOUT_BUFFER, 1), dtype=jnp.float32)
    values = jnp.zeros((BATCH_SIZE, ROLLOUT_BUFFER, 1), dtype=jnp.float32)
    dones = jnp.zeros((BATCH_SIZE, ROLLOUT_BUFFER, 1), dtype=jnp.bool)

    print(states.shape)
    print(action_selection(actor, states).shape)
    print(value_estimation(critic, states).shape)


if __name__ == "__main__":
    main()
