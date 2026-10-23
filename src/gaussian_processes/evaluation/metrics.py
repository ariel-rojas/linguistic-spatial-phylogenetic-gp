"""Métricas multiclase compartidas por todas las evaluaciones."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)

from ..configuration import CLASSES


def classification_metrics(observed, probabilities):
    """Brier = media de sum_c (p_c - indicador_c)^2, rango [0, 2].

    Recall de clase ausente: null. Macro-F1 usa las tres clases y cero para términos
    indefinidos; balanced_accuracy promedia únicamente recalls observables.
    """
    observed = np.asarray(observed)
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(observed), len(CLASSES)) or not len(observed):
        raise ValueError("Dimensiones de probabilidades incorrectas o evaluación vacía")
    if not set(observed) <= set(CLASSES):
        raise ValueError("Etiqueta observada desconocida")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=1), 1)
    ):
        raise ValueError("Probabilidades inválidas")
    labels = np.asarray(CLASSES)[probabilities.argmax(axis=1)]
    precision, recall, f1, support = precision_recall_fscore_support(
        observed, labels, labels=CLASSES, zero_division=0
    )
    targets = observed[:, None] == np.asarray(CLASSES)[None, :]
    return {
        "classes": CLASSES,
        "n_test": len(observed),
        "correct": int((observed == labels).sum()),
        "incorrect": int((observed != labels).sum()),
        "accuracy": accuracy_score(observed, labels),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)),
        "log_loss": log_loss(observed, probabilities, labels=CLASSES),
        "brier_multiclass": float(
            np.mean(np.sum((probabilities - targets) ** 2, axis=1))
        ),
        "confusion_matrix": confusion_matrix(observed, labels, labels=CLASSES).tolist(),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]) if support[i] else None,
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(CLASSES)
        },
    }
