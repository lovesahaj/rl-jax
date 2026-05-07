import os

import numpy as np
import optax
from flax import nnx
from gymnasium import spaces
from tqdm import tqdm

import jax
import jax.numpy as jnp
import jax.random as jran
import wandb

from bench import BENCH_Q, BENCH_R, evaluate, save_results, with_canonical_reward
from env.cartpole import CartPoleEnv, NUM_STATES, NUM_ACTIONS

OUTPUT_DIR = "output"

ALPHA = 0.1
LR = 1e-3
GAMMA = 0.99


@nnx.jit
def dqn_update(model, optimizer, state, next_state, action, reward, terminated):
    state = jnp.asarray(state, dtype=jnp.float32)
    next_state = jnp.asarray(next_state, dtype=jnp.float32)
    action = jnp.asarray(action, dtype=jnp.int32)
    reward = jnp.asarray(reward, dtype=jnp.float32)
    done = jnp.asarray(terminated, dtype=jnp.float32)

    next_q = model(next_state)
    target = reward + GAMMA * (1.0 - done) * jnp.max(next_q)
    target = jax.lax.stop_gradient(target)

    def loss_fn(model):
        q_values = model(state)
        q_sa = q_values[action]
        return (q_sa - target) ** 2

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)

    return loss


class NeuralNetwork(nnx.Module):
    def __init__(self, n_features, n_hiddens, n_targets, *, rngs: nnx.Rngs):
        self.layer1 = nnx.Linear(n_features, n_hiddens, rngs=rngs)
        self.layer2 = nnx.Linear(n_hiddens, n_hiddens, rngs=rngs)
        self.layer3 = nnx.Linear(n_hiddens, n_targets, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.layer1(x))
        x = nnx.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class ProjectEnv(CartPoleEnv):
    def __init__(self, render_mode=None, seed=40, max_steps: int = 500):
        super().__init__(spaces.Discrete(NUM_ACTIONS), render_mode, seed)

        self.max_steps = max_steps
        self.step_count = 0

        self.q_function = NeuralNetwork(NUM_STATES, 128, NUM_ACTIONS, rngs=nnx.Rngs(0))
        self.optim = nnx.Optimizer(self.q_function, optax.adam(LR), wrt=nnx.Param)

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.key, self.action_key = jran.split(self.key)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.key, subkey = jran.split(self.key)
        self.state = jran.uniform(
            subkey, shape=(NUM_STATES,), minval=-0.05, maxval=0.05
        )
        self.step_count = 0
        return self.state, {}

    def step(self, action):
        self.state = self._get_next_state(action)
        self.step_count += 1
        terminated = self._check_if_terminated(self.state)
        truncated = self.step_count >= self.max_steps and not terminated
        reward = 1 - self.cost()
        info = {
            "current_state": "stopped" if terminated or truncated else "moving",
            "step_count": self.step_count,
        }
        return self.state, reward, terminated, truncated, info

    def cost(self):
        x, x_dot, theta, theta_dot = self.state
        return 1 * (theta**2) + 0.1 * (x**2) + 0.01 * (x_dot**2 + theta_dot**2)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def choose_action(self, state):
        self.action_key, sub1, sub2 = jran.split(self.action_key, 3)
        if float(jran.uniform(sub1)) < self.epsilon:
            return int(jran.randint(sub2, (), 0, NUM_ACTIONS))
        state = jnp.asarray(state, dtype=jnp.float32)
        q_values = self.q_function(state)
        return int(jnp.argmax(q_values))

    def q_update(self, state, next_state, action, reward, terminated):
        loss = dqn_update(
            self.q_function, self.optim, state, next_state, action, reward, terminated
        )
        return float(loss)


def train(num_episodes=1000):
    env = ProjectEnv(max_steps=500)
    episode_rewards = []

    wandb.init(
        entity="lovesahaj1225-the-university-of-edinburgh",
        project="RL Training",
        config={
            "num_episodes": num_episodes,
            "alpha": ALPHA,
            "gamma": GAMMA,
            "epsilon_start": env.epsilon,
            "epsilon_min": env.epsilon_min,
            "epsilon_decay": env.epsilon_decay,
            "num_states": NUM_STATES,
            "num_actions": NUM_ACTIONS,
            "max_steps": env.max_steps,
        },
    )

    loop = tqdm(range(num_episodes))

    for ep in loop:
        state, _ = env.reset()
        total_reward = 0.0
        total_loss = 0.0
        episode_length = 0

        for _ in range(env.max_steps):
            action = env.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            total_loss += env.q_update(state, next_state, action, reward, terminated)
            total_reward += float(reward)
            episode_length += 1

            if terminated or truncated:
                break

            state = next_state

        episode_rewards.append(total_reward)
        wandb.log(
            {
                "reward": total_reward,
                "cost": -total_reward,
                "loss": total_loss,
                "epsilon": env.epsilon,
                "episode_length": episode_length,
                "terminated": int(terminated),
                "truncated": int(truncated),
            },
            step=ep,
        )
        loop.set_postfix(
            {
                "Reward": round(total_reward, 3),
                "Cost": round(-total_reward, 3),
                "Length": episode_length,
                "Terminated": terminated,
                "Truncated": truncated,
                "Epsilon": round(env.epsilon, 3),
                "Loss": round(total_loss, 3),
            }
        )
        env.update_epsilon()

    return episode_rewards, env


def benchmark(trained_env, train_rewards=None, num_episodes=100, max_steps=500):
    """Evaluate the greedy policy of a trained Deep Q-learning agent."""
    q_function = trained_env.q_function

    def env_factory(seed):
        e = ProjectEnv(seed=seed, max_steps=max_steps)
        e.q_function = q_function
        e.epsilon = 0.0
        return with_canonical_reward(e)

    summary = evaluate(
        name="DeepQ",
        env_factory=env_factory,
        policy_factory=lambda env: (lambda e: int(e.choose_action(e.state))),
        num_episodes=num_episodes,
        max_steps=max_steps,
        Q=BENCH_Q,
        R=BENCH_R,
        balanced_threshold=max_steps,
    )

    if train_rewards is not None and train_rewards.size:
        ma = (
            np.convolve(train_rewards, np.ones(50) / 50, mode="valid")
            if train_rewards.size >= 50
            else train_rewards
        )
        threshold = 0.9 * np.max(ma)
        idx = int(np.argmax(ma >= threshold))
        summary["convergence_episode"] = float(idx + 50 if train_rewards.size >= 50 else idx)
        summary["final_train_reward"] = float(np.mean(train_rewards[-50:]))

    save_results(os.path.join(OUTPUT_DIR, "deep_q_learning_metrics.npz"), [summary])
    print(f"DeepQ benchmark saved to {OUTPUT_DIR}/deep_q_learning_metrics.npz")
    print(
        f"  stability_rate={summary.get('stability_rate', 0):.2%}  "
        f"mean_test_reward={summary.get('total_reward_mean', 0):.2f}  "
        f"mean_steps={summary.get('episode_length_mean', 0):.1f}"
    )


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rewards, env = train(num_episodes=2000)
    benchmark(env, train_rewards=np.asarray(rewards, dtype=np.float64))
