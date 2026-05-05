import os

import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from scipy.linalg import solve_continuous_are

import jax.numpy as jnp

from env.cartpole import CartPoleEnv

OUTPUT_DIR = "output"


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
        self.state = jnp.array([0, 0.0, 0.1, 1], dtype=jnp.float32)
        return self.state, {}

    def choose_action(self):
        F = -self.K @ self.state
        F_scalar = float(F.item())
        return 1 if F_scalar > 0 else 0


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    env = ProjectEnv(render_mode="console")
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

    np.savez(
        os.path.join(OUTPUT_DIR, "lqr_results.npz"),
        state_history=state_history,
        force_history=force_history,
        action_history=action_history,
        reward_history=reward_history,
    )

    time = np.arange(len(state_history)) * env.tau

    plt.figure()
    plt.plot(time, state_history[:, 0])
    plt.xlabel("Time [s]")
    plt.ylabel("Cart position x [m]")
    plt.title("Cart Position")
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, "lqr_cart_position.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(time, state_history[:, 1])
    plt.xlabel("Time [s]")
    plt.ylabel("Cart velocity x_dot [m/s]")
    plt.title("Cart Velocity")
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, "lqr_cart_velocity.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(time, state_history[:, 2])
    plt.axhline(float(env.terminal_angle), linestyle="--")
    plt.axhline(float(-env.terminal_angle), linestyle="--")
    plt.xlabel("Time [s]")
    plt.ylabel("Pole angle theta [rad]")
    plt.title("Pole Angle")
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, "lqr_pole_angle.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(time, state_history[:, 3])
    plt.xlabel("Time [s]")
    plt.ylabel("Angular velocity theta_dot [rad/s]")
    plt.title("Pole Angular Velocity")
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, "lqr_angular_velocity.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(time, force_history)
    plt.xlabel("Time [s]")
    plt.ylabel("LQR force F")
    plt.title("Raw LQR Force")
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, "lqr_force.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.step(time, action_history, where="post")
    plt.xlabel("Time [s]")
    plt.ylabel("Discrete action")
    plt.title("Discrete Action Chosen from LQR Force")
    plt.yticks([0, 1], ["left", "right"])
    plt.grid()
    plt.savefig(os.path.join(OUTPUT_DIR, "lqr_actions.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
