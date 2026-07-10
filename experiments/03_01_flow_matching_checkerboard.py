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


def main() -> None:
    rng = np.random.default_rng(SEED)

    x1 = sample_checkerboard(20000, rng)

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


if __name__ == "__main__":
    main()