import os

import numpy as np
from gymnasium import spaces
from scipy.linalg import solve_continuous_are

import jax.numpy as jnp
import jax.random as jran

from bench import BENCH_Q, BENCH_R, evaluate, save_results, time_callable
from env.cartpole import CartPoleEnv, NUM_STATES

OUTPUT_DIR = "output"
NUM_EVAL_EPISODES = 100
MAX_EVAL_STEPS = 500


class ProjectEnv(CartPoleEnv):
    def __init__(self, render_mode=None, seed=40):
        super().__init__(spaces.Discrete(2), render_mode, seed)

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
        self.key, subkey = jran.split(self.key)
        perturb = jran.uniform(
            subkey, shape=(NUM_STATES,), minval=-0.05, maxval=0.05
        )
        self.state = jnp.array([0.0, 0.0, 0.1, 1.0], dtype=jnp.float32) + perturb
        return self.state, {}

    def choose_action(self):
        F = -self.K @ self.state
        F_scalar = float(F.item())
        return 1 if F_scalar > 0 else 0


def benchmark():
    """Evaluate the LQR controller across many seeds and save metrics."""
    summary = evaluate(
        name="LQR",
        env_factory=lambda seed: ProjectEnv(seed=seed),
        policy_factory=lambda env: (lambda e: e.choose_action()),
        num_episodes=NUM_EVAL_EPISODES,
        max_steps=MAX_EVAL_STEPS,
        Q=BENCH_Q,
        R=BENCH_R,
    )

    timing_env = ProjectEnv()
    timing_env.reset()
    timing = time_callable(lambda: timing_env.choose_action(), n_runs=200, warmup=5)
    timing_env.close()
    summary["solve_mean_s"] = timing["mean_s"]
    summary["solve_std_s"] = timing["std_s"]

    save_results(os.path.join(OUTPUT_DIR, "lqr_metrics.npz"), [summary])
    print(f"LQR benchmark saved to {OUTPUT_DIR}/lqr_metrics.npz")
    print(f"  stability_rate={summary.get('stability_rate', 0):.2%}  "
          f"mean_steps={summary.get('episode_length_mean', 0):.1f}  "
          f"mean_cost={summary.get('trajectory_cost_mean', 0):.2f}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    benchmark()
