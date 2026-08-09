import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qreal_vqc import QRealVQC


def prepare_data(n_qubits=3, max_train=120, seed=0):
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=seed, stratify=data.target
    )
    pipe = make_pipeline(StandardScaler(), PCA(n_components=n_qubits, random_state=seed))
    x_train = pipe.fit_transform(x_train)
    x_test = pipe.transform(x_test)
    scale = np.maximum(np.std(x_train, axis=0, keepdims=True), 1e-9)
    x_train = np.clip(x_train / scale, -np.pi, np.pi)
    x_test = np.clip(x_test / scale, -np.pi, np.pi)
    return x_train[:max_train], x_test, y_train[:max_train], y_test


def main():
    x_train, x_test, y_train, y_test = prepare_data()
    base = QRealVQC(3, 1, "complex")
    init = base.initial_parameters(seed=17)

    fitted = {}
    for backend in ("complex", "structured_real", "naive_real"):
        model = QRealVQC(3, 1, backend)
        theta, history = model.fit(
            x_train, y_train, parameters=init, epochs=6, learning_rate=0.08
        )
        p = model.probabilities(x_test, theta)
        acc = accuracy_score(y_test, p >= 0.5)
        fitted[backend] = (model, theta, p)
        print(f"{backend:16s} loss={history[-1]:.6f} test_accuracy={acc:.4f}")

    err = np.max(np.abs(fitted["complex"][2] - fitted["structured_real"][2]))
    print(f"max complex/structured_real prediction error: {err:.3e}")


if __name__ == "__main__":
    main()
