"""GP multiclase de sklearn: tres ajustes uno contra el resto, con Laplace."""

import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier

from ..configuration import CLASSES
from .kernels import build_kernel


def build_model(config, covariance):
    """Cada ajuste fija los parámetros; la grilla compara configuraciones completas."""
    return GaussianProcessClassifier(
        kernel=build_kernel(config, covariance),
        optimizer=None,
        random_state=config["seed"],
        multi_class="one_vs_rest",
        n_jobs=1,
    )


def predict(model, X):
    """Probabilidades en orden CLASSES y argmax; un empate elige la primera clase.

    Sin dominante es una observación propia, no un sinónimo de incertidumbre.
    """
    if list(model.classes_) != CLASSES:
        raise ValueError("El entrenamiento debe contener las tres clases")
    if not len(X):
        return np.empty((0, len(CLASSES))), np.array([], dtype=str)
    probabilities = model.predict_proba(X)
    return probabilities, np.asarray(CLASSES)[probabilities.argmax(axis=1)]
