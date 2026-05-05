import os

import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx
from gymnasium import spaces
from tqdm import tqdm

import jax
import jax.numpy as jnp
import jax.random as jran
import wandb

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
    plt.savefig(
        os.path.join(save_dir, "deep_q_learning_cost.png"), dpi=150, bbox_inches="tight"
    )
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
    plt.savefig(
        os.path.join(save_dir, "deep_q_learning_reward.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()


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


def test(env, num_episodes=50):
    env.render_mode = "human"
    episode_rewards = []
    old_epsilon = env.epsilon
    env.epsilon = 0.0

    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0.0

        for _ in range(env.max_steps):
            action = env.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            env.render()

            if terminated or truncated:
                break

            state = next_state

        episode_rewards.append(total_reward)

    env.epsilon = old_epsilon
    return episode_rewards


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rewards, env = train(num_episodes=2000)

    plot_training(rewards, window=50)

    test_rewards = test(env, num_episodes=50)

    np.savez(
        os.path.join(OUTPUT_DIR, "deep_q_learning_results.npz"),
        train_rewards=np.array(rewards),
        test_rewards=np.array(test_rewards),
    )
    print(f"Results saved to {OUTPUT_DIR}/")
