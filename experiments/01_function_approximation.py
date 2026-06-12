import os
from pathlib import Path

# Suppress TensorFlow informational messages.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


SEED = 42
OUTPUT_DIR = Path("outputs")

tf.keras.utils.set_random_seed(SEED)
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
    # Shape: (256, 1)
    x_train = np.linspace(-3.0, 3.0, 256, dtype=np.float32).reshape(-1, 1)
    y_train = target_function(x_train).astype(np.float32)

    model = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mean_squared_error",
    )

    model.summary()

    history = model.fit(
        x_train,
        y_train,
        epochs=300,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
    )

    x_test = np.linspace(-3.0, 3.0, 500, dtype=np.float32).reshape(-1, 1)
    y_true = target_function(x_test)
    y_pred = model.predict(x_test, verbose=0)

    final_train_loss = history.history["loss"][-1]
    final_validation_loss = history.history["val_loss"][-1]

    print(f"\nTraining loss:   {final_train_loss:.8f}")
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
