import os

import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from tqdm import tqdm

import jax.numpy as jnp
import jax.random as jran

from env.cartpole import CartPoleEnv, NUM_STATES, NUM_ACTIONS

OUTPUT_DIR = "output"

NUM_BINS = 10
ALPHA = 0.1
GAMMA = 0.99

STATE_BINS = [
    jnp.linspace(-2.4, 2.4, NUM_BINS - 1),   # x
    jnp.linspace(-3.0, 3.0, NUM_BINS - 1),   # x_dot
    jnp.linspace(-0.21, 0.21, NUM_BINS - 1), # theta
    jnp.linspace(-3.0, 3.0, NUM_BINS - 1),   # theta_dot
]


def make_key(state) -> tuple:
    state = np.array(state)
    return tuple(
        int(jnp.clip(jnp.digitize(s, bins), 0, NUM_BINS - 1))
        for s, bins in zip(state, STATE_BINS)
    )


class ProjectEnv(CartPoleEnv):
    def __init__(self, render_mode=None, seed=40):
        super().__init__(spaces.Discrete(NUM_ACTIONS), render_mode, seed)

        self.q_table = jnp.zeros(
            [NUM_BINS] * NUM_STATES + [NUM_ACTIONS], dtype=jnp.float32
        )
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.key, subkey = jran.split(self.key)
        self.state = jran.uniform(
            subkey, shape=(NUM_STATES,), minval=-0.05, maxval=0.05
        )
        return self.state, {}

    def step(self, action):
        self.state = self._get_next_state(action)
        terminated = self._check_if_terminated(self.state)
        truncated = False
        cost = -100.0 if terminated else -self.cost()
        info = {"current_state": "stopped" if terminated or truncated else "moving"}
        return self.state, cost, terminated, truncated, info

    def cost(self):
        x, x_dot, theta, theta_dot = self.state
        return 0.01 * abs(x) + 0.1 * (x**2) + 1 * (theta**2) + 0.1 * abs(theta_dot)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def choose_action(self):
        probs = jnp.ones(NUM_ACTIONS, dtype=jnp.float32)
        actions = self.q_table[*make_key(self.state), :]
        max_action_idx = int(jnp.argmax(actions))
        probs = probs * (self.epsilon / NUM_ACTIONS)
        probs = probs.at[max_action_idx].set(
            1 - self.epsilon + (self.epsilon / NUM_ACTIONS)
        )
        self.key, subkey = jran.split(self.key)
        return jran.choice(subkey, jnp.arange(0, NUM_ACTIONS), p=probs)

    def q_update(self, cur_state, next_state, action, reward, terminated):
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


def moving_average(x, window=50):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_training(rewards, window=50, save_dir=OUTPUT_DIR):
    os.makedirs(save_dir, exist_ok=True)
    rewards = np.asarray(rewards, dtype=np.float32)
    costs = -rewards

    reward_ma = moving_average(rewards, window)
    cost_ma = moving_average(costs, window)

    plt.figure(figsize=(10, 5))
    plt.plot(costs, alpha=0.35, label="Episode cost")
    plt.plot(
        np.arange(window - 1, window - 1 + len(cost_ma)),
        cost_ma,
        linewidth=2,
        label=f"{window}-episode moving average",
    )
    plt.xlabel("Episode")
    plt.ylabel("Cost / Loss")
    plt.title("Q-learning Training Cost")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "q_learning_cost.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.35, label="Episode reward")
    plt.plot(
        np.arange(window - 1, window - 1 + len(reward_ma)),
        reward_ma,
        linewidth=2,
        label=f"{window}-episode moving average",
    )
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Q-learning Training Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "q_learning_reward.png"), dpi=150, bbox_inches="tight")
    plt.close()


def train(num_episodes=2000):
    env = ProjectEnv()
    episode_rewards = []

    loop = tqdm(range(num_episodes))

    for ep in loop:
        env.reset()
        total_reward = 0.0

        for _ in range(500):
            action = int(env.choose_action())
            cur_state = env.state
            obs, reward, terminated, truncated, info = env.step(action)
            next_state = env.state

            env.q_update(cur_state, next_state, action, reward, terminated)
            total_reward += float(reward)

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)
        loop.set_postfix(
            {
                "Reward": round(total_reward, 3),
                "Cost": round(-total_reward, 3),
                "Epsilon": round(env.epsilon, 3),
            }
        )
        env.update_epsilon()

    return episode_rewards, env


def test(env, num_episodes=50):
    env.render_mode = "console"
    episode_rewards = []
    old_epsilon = env.epsilon
    env.epsilon = 0.0

    for ep in range(num_episodes):
        env.reset()
        total_reward = 0.0

        for _ in range(500):
            action = int(env.choose_action())
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            env.render()

            if terminated or truncated:
                break

        episode_rewards.append(total_reward)

    env.epsilon = old_epsilon
    return episode_rewards


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rewards, env = train(num_episodes=2000)

    plot_training(rewards, window=50)

    test_rewards = test(env, num_episodes=50)

    np.savez(
        os.path.join(OUTPUT_DIR, "q_learning_results.npz"),
        train_rewards=np.array(rewards),
        test_rewards=np.array(test_rewards),
    )
    print(f"Results saved to {OUTPUT_DIR}/")
