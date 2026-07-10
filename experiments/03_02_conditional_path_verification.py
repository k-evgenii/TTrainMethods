"""
Flow Matching for Generative Modeling -- small-scale replication.

Reproduces the central mechanism of the paper (Lipman et al., 2022) on a 2D
checkerboard toy distribution using Conditional Flow Matching with the
Optimal Transport (OT) conditional path (paper Section 4.1, Example II).

This file is built up stage by stage:
  Stage 1: data verification          <- this stage
  Stage 2: conditional-path verification
  Stage 3: model + one training step
  Stage 4: short training run
  Stage 5: Euler generation
  Stage 6: full training run
  Stage 7: midpoint / NFE experiment
"""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np

SEED = 42
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def sample_checkerboard(batch_size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample points uniformly from the active squares of a checkerboard.

    "Active" squares are those where floor(x) + floor(y) is even, over the
    region [-4, 4] x [-4, 4]. Uses rejection sampling: draw uniformly over
    the bounding square, keep only points that land on an active cell.

    Returns: float32 array of shape (batch_size, 2).
    """
    accepted = np.empty((0, 2), dtype=np.float32)
    while accepted.shape[0] < batch_size:
        # Oversample by 2x per round since roughly half of cells are active.
        candidates = rng.uniform(-4.0, 4.0, size=(batch_size * 2, 2)).astype(np.float32)
        is_active = (np.floor(candidates[:, 0]) + np.floor(candidates[:, 1])) % 2 == 0
        accepted = np.concatenate([accepted, candidates[is_active]], axis=0)
    return accepted[:batch_size]


def plot_checkerboard_target(samples: np.ndarray, path: Path) -> None:
    plt.figure(figsize=(5, 5))
    plt.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.6)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.gca().set_aspect("equal")
    plt.title(f"Checkerboard target samples (n={samples.shape[0]})")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def conditional_path(
    x0: np.ndarray, x1: np.ndarray, t: float, sigma_min: float = 0.01
) -> np.ndarray:
    """Evaluate the OT conditional path x_t = sigma_t * x0 + t * x1.

    sigma_t = 1 - (1 - sigma_min) * t   (paper eq. 20, with linear sigma).

    At t=0 this returns x0 exactly. At t=1 it returns sigma_min * x0 + x1,
    i.e. essentially x1 with a small amount of residual noise. This is pure
    algebra -- no model, no training, just the formula evaluated directly.
    """
    sigma_t = 1.0 - (1.0 - sigma_min) * t
    return sigma_t * x0 + t * x1


def plot_conditional_path_snapshots(
    x0: np.ndarray, x1: np.ndarray, times: list[float], path: Path
) -> None:
    fig, axes = plt.subplots(1, len(times), figsize=(4 * len(times), 4))
    for ax, t in zip(axes, times):
        x_t = conditional_path(x0, x1, t)
        ax.scatter(x_t[:, 0], x_t[:, 1], s=4, alpha=0.6)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")
        ax.set_title(f"t = {t}")
    fig.suptitle("Conditional OT path: analytic x_t, no model involved")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    rng = np.random.default_rng(SEED)

    x1 = sample_checkerboard(5000, rng)

    # Sanity checks before we trust this as a training target.
    print(f"x1 shape:  {x1.shape}, dtype: {x1.dtype}")
    print(f"x1 mean:   {x1.mean(axis=0)}")
    print(f"x1 std:    {x1.std(axis=0)}")
    print(f"x1 min/max: {x1.min(axis=0)} / {x1.max(axis=0)}")

    # Every sample must actually be on an active cell -- catches off-by-one
    # errors in the floor/parity logic before they contaminate everything
    # downstream.
    cell_parity = (np.floor(x1[:, 0]) + np.floor(x1[:, 1])) % 2
    assert np.all(cell_parity == 0), "Found samples outside active cells!"
    print("All samples confirmed on active checkerboard cells.")

    plot_checkerboard_target(x1, OUTPUT_DIR / "checkerboard_target.png")
    print(f"\nSaved: {OUTPUT_DIR / 'checkerboard_target.png'}")

    # --- Stage 2: conditional path verification (still no model/training) ---
    n_path_samples = 1000
    x0_path = rng.standard_normal(size=(n_path_samples, 2)).astype(np.float32)
    x1_path = sample_checkerboard(n_path_samples, rng)

    # At t=0, x_t must equal x0 exactly -- a direct algebraic check.
    x_t0 = conditional_path(x0_path, x1_path, t=0.0)
    assert np.allclose(x_t0, x0_path), "x_t at t=0 should equal x0 exactly!"
    print("\nConditional path at t=0 matches x0 exactly, as expected.")

    # At t=1, x_t should be very close to x1 (off by sigma_min * x0).
    x_t1 = conditional_path(x0_path, x1_path, t=1.0)
    max_deviation = np.max(np.abs(x_t1 - x1_path))
    print(f"Conditional path at t=1: max deviation from x1 = {max_deviation:.4f}"
          f" (should be small, bounded by sigma_min * |x0|)")

    plot_conditional_path_snapshots(
        x0_path, x1_path, [0.0, 0.25, 0.5, 0.75, 1.0],
        OUTPUT_DIR / "conditional_path_snapshots.png",
    )
    print(f"Saved: {OUTPUT_DIR / 'conditional_path_snapshots.png'}")


if __name__ == "__main__":
    main()