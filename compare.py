import os

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "output"


def load(name):
    path = os.path.join(OUTPUT_DIR, name)
    if not os.path.exists(path):
        return None
    return np.load(path)


def moving_average(x, window=50):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def compare_control():
    """Compare LQR, iLQR, MPC state trajectories and control force."""
    methods = {
        "LQR": load("lqr_results.npz"),
        "iLQR": load("ilqr_results.npz"),
        "MPC": load("mpc_results.npz"),
    }
    available = {k: v for k, v in methods.items() if v is not None}

    if not available:
        print("No control method results found. Run lqr.py / ilqr.py / mpc.py first.")
        return

    DT = 0.02
    state_labels = [
        ("Cart Position", "x [m]", 0),
        ("Cart Velocity", "x_dot [m/s]", 1),
        ("Pole Angle", "theta [rad]", 2),
        ("Angular Velocity", "theta_dot [rad/s]", 3),
    ]

    for title, ylabel, idx in state_labels:
        plt.figure(figsize=(10, 5))
        for name, data in available.items():
            s = data["state_history"]
            t = np.arange(len(s)) * DT
            plt.plot(t, s[:, idx], label=name)
        plt.xlabel("Time [s]")
        plt.ylabel(ylabel)
        plt.title(f"Comparison — {title}")
        plt.legend()
        plt.grid(True)
        fname = f"compare_control_{ylabel.split()[0].lower()}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {fname}")

    plt.figure(figsize=(10, 5))
    for name, data in available.items():
        a = data["action_history"]
        t = np.arange(len(a)) * DT
        force = a[:, 0] if a.ndim == 2 else a
        plt.plot(t, force, label=name)
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.title("Comparison — Control Force")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "compare_control_force.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved compare_control_force.png")


def compare_rl():
    """Compare Q-learning and SARSA training curves."""
    methods = {
        "Q-learning": load("q_learning_results.npz"),
        "SARSA": load("sarsa_results.npz"),
    }
    available = {k: v for k, v in methods.items() if v is not None}

    if not available:
        print("No RL results found. Run q_learning.py / sarsa.py first.")
        return

    window = 50

    for metric, sign, ylabel, title_suffix in [
        ("train_rewards", 1, "Reward", "Training Reward"),
        ("train_rewards", -1, "Cost", "Training Cost"),
    ]:
        plt.figure(figsize=(10, 5))
        for name, data in available.items():
            raw = np.asarray(data[metric], dtype=np.float32) * sign
            ma = moving_average(raw, window)
            plt.plot(raw, alpha=0.25)
            plt.plot(
                np.arange(window - 1, window - 1 + len(ma)),
                ma,
                linewidth=2,
                label=f"{name} ({window}-ep avg)",
            )
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.title(f"Comparison — {title_suffix}")
        plt.legend()
        plt.grid(True)
        fname = f"compare_rl_{ylabel.lower()}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {fname}")

    # Test performance bar chart
    test_data = {k: v for k, v in available.items() if "test_rewards" in v}
    if test_data:
        names = list(test_data.keys())
        means = [np.mean(test_data[n]["test_rewards"]) for n in names]
        stds = [np.std(test_data[n]["test_rewards"]) for n in names]

        plt.figure(figsize=(7, 5))
        x = np.arange(len(names))
        plt.bar(x, means, yerr=stds, capsize=6, width=0.4)
        plt.xticks(x, names)
        plt.ylabel("Mean Test Reward")
        plt.title("Comparison — Test Performance")
        plt.grid(True, axis="y")
        plt.savefig(os.path.join(OUTPUT_DIR, "compare_rl_test.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved compare_rl_test.png")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    compare_control()
    compare_rl()
    print("Done.")
