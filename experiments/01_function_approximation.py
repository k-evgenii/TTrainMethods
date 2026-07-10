import os
import random
import time
from pathlib import Path

# Suppress TensorFlow informational messages.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


SEED = 42
OUTPUT_DIR = Path("outputs")

# Seed python/numpy/TF directly rather than via tf.keras.utils.set_random_seed.
# That convenience wrapper also sets an internal tf_keras seed generator that,
# in the legacy Keras 2 backend NVIDIA's container uses, has a real bug: it
# calls random.randint(1, 1e9) with a float upper bound, which Python 3.12
# rejects outright. Seeding the three sources directly gives the identical
# reproducibility guarantee without touching that broken code path.
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

OUTPUT_DIR.mkdir(exist_ok=True)


def target_function(x: np.ndarray) -> np.ndarray:
    """Function that the neural network will approximate."""
    return np.sin(2.0 * x) + 0.2 * x**2


def build_model() -> tf.keras.Model:
    """Construct a small multilayer perceptron."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(32, activation="tanh"),
            tf.keras.layers.Dense(32, activation="tanh"),
            tf.keras.layers.Dense(1),
        ]
    )


def main() -> None:
    # Find out what TensorFlow can actually see before doing anything else.
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPUs visible to TensorFlow: {gpus}")

    # By default TF grabs ALL GPU memory up front, even for a tiny model.
    # On a laptop GPU that's also driving your display, that's wasteful -
    # this makes it allocate memory incrementally as it's actually needed.
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    device = "/GPU:0" if gpus else "/CPU:0"
    print(f"Training on: {device}\n")

    # Shape: (256, 1)
    x_train = np.linspace(-3.0, 3.0, 256, dtype=np.float32).reshape(-1, 1)
    y_train = target_function(x_train).astype(np.float32)

    # Everything below - building the model, training, and predicting -
    # is pinned to whichever device we found above. Wrapping it explicitly
    # (rather than just letting TF decide) means there's no ambiguity about
    # what actually ran where.
    with tf.device(device):
        model = build_model()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="mean_squared_error",
        )

        model.summary()

        start_time = time.time()
        history = model.fit(
            x_train,
            y_train,
            epochs=300,
            batch_size=32,
            validation_split=0.2,
            verbose=0,
        )
        elapsed = time.time() - start_time

        x_test = np.linspace(-3.0, 3.0, 500, dtype=np.float32).reshape(-1, 1)
        y_true = target_function(x_test)
        y_pred = model.predict(x_test, verbose=0)

    final_train_loss = history.history["loss"][-1]
    final_validation_loss = history.history["val_loss"][-1]

    print(f"\nTraining time:   {elapsed:.2f}s  (on {device})")
    print(f"Training loss:   {final_train_loss:.8f}")
    print(f"Validation loss: {final_validation_loss:.8f}")
    print(f"Input shape:     {x_train.shape}")
    print(f"Output shape:    {y_pred.shape}")

    # Plot the function approximation.
    plt.figure(figsize=(8, 5))
    plt.plot(x_test, y_true, label="True function")
    plt.plot(x_test, y_pred, linestyle="--", label="Neural network")
    plt.scatter(x_train[::8], y_train[::8], s=12, label="Training samples")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("TensorFlow neural-network function approximation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "function_approximation.png",
        dpi=150,
    )
    plt.close()

    # Plot the optimisation process.
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error")
    plt.title("Training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "training_history.png",
        dpi=150,
    )
    plt.close()

    model.save(OUTPUT_DIR / "function_approximator.keras")

    print("\nCreated:")
    print("  outputs/function_approximation.png")
    print("  outputs/training_history.png")
    print("  outputs/function_approximator.keras")


if __name__ == "__main__":
    main()