import os

import numpy as np
from gymnasium import spaces
from tqdm import tqdm

import jax.numpy as jnp
import jax.random as jran

from bench import BENCH_Q, BENCH_R, evaluate, save_results, with_canonical_reward
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

    def choose_action(self, state):
        probs = jnp.ones(NUM_ACTIONS, dtype=jnp.float32)
        actions = self.q_table[*make_key(state), :]
        max_action_idx = int(jnp.argmax(actions))
        probs = probs * (self.epsilon / NUM_ACTIONS)
        probs = probs.at[max_action_idx].set(
            1 - self.epsilon + (self.epsilon / NUM_ACTIONS)
        )
        self.key, subkey = jran.split(self.key)
        return jran.choice(subkey, jnp.arange(0, NUM_ACTIONS), p=probs)

    def q_update(self, cur_state, next_state, action, next_action, reward, terminated):
        cur_key = make_key(cur_state)
        next_key = make_key(next_state)
        current_q = self.q_table[*cur_key, action]
        if terminated:
            target_q = reward
        else:
            target_q = reward + GAMMA * self.q_table[*next_key, next_action]
        self.q_table = self.q_table.at[*cur_key, action].set(
            current_q + ALPHA * (target_q - current_q)
        )


def train(num_episodes=2000):
    env = ProjectEnv()
    episode_rewards = []

    loop = tqdm(range(num_episodes))

    for ep in loop:
        state, _ = env.reset()
        total_reward = 0.0
        action = int(env.choose_action(state))

        for _ in range(500):
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = int(env.choose_action(next_state))

            env.q_update(state, next_state, action, next_action, reward, terminated)
            total_reward += float(reward)

            if terminated or truncated:
                break

            state = next_state
            action = next_action

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


def benchmark(trained_env, train_rewards=None, num_episodes=100, max_steps=500):
    """Evaluate the greedy policy of a trained SARSA agent."""
    q_table = trained_env.q_table

    def env_factory(seed):
        e = ProjectEnv(seed=seed)
        e.q_table = q_table
        e.epsilon = 0.0
        return with_canonical_reward(e)

    summary = evaluate(
        name="SARSA",
        env_factory=env_factory,
        policy_factory=lambda env: (lambda e: int(e.choose_action(e.state))),
        num_episodes=num_episodes,
        max_steps=max_steps,
        Q=BENCH_Q,
        R=BENCH_R,
        balanced_threshold=max_steps,
    )

    if train_rewards is not None and train_rewards.size:
        ma = np.convolve(train_rewards, np.ones(50) / 50, mode="valid") if train_rewards.size >= 50 else train_rewards
        threshold = 0.9 * np.max(ma)
        idx = int(np.argmax(ma >= threshold))
        summary["convergence_episode"] = float(idx + 50 if train_rewards.size >= 50 else idx)
        summary["final_train_reward"] = float(np.mean(train_rewards[-50:]))

    save_results(os.path.join(OUTPUT_DIR, "sarsa_metrics.npz"), [summary])
    print(f"SARSA benchmark saved to {OUTPUT_DIR}/sarsa_metrics.npz")
    print(f"  stability_rate={summary.get('stability_rate', 0):.2%}  "
          f"mean_test_reward={summary.get('total_reward_mean', 0):.2f}  "
          f"mean_steps={summary.get('episode_length_mean', 0):.1f}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rewards, env = train(num_episodes=2000)
    benchmark(env, train_rewards=np.asarray(rewards, dtype=np.float64))
