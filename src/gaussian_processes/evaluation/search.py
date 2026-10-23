"""GridSearchCV con test externo intacto y validación interna por identidad."""

import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterGrid

from ..configuration import CLASSES, load_configuration
from ..data.loading import load_data
from ..data.phylogeny import load_phylogeny
from ..models.features import build_features
from ..models.gp import build_model
from ..models.spatial import parameter_paths
from .experiment import evaluate, save_run
from .splits import inner_splits, split_data


def select_parameters(training_rows, X_train, config, out, covariance):
    """Solo recibe train: el test externo no está disponible para la búsqueda.

    Cada candidato mantiene optimizer=None. Se minimiza la media de log-loss
    interna; el mejor se reajusta automáticamente sobre todo este entrenamiento.
    """
    folds = inner_splits(training_rows, config)
    paths = parameter_paths(config)
    grid = {paths[key]: values for key, values in config["grid"].items()}
    count = len(ParameterGrid(grid))
    print(f"Grilla: {count} combinaciones × {len(folds)} particiones", flush=True)
    search = GridSearchCV(
        build_model(config, covariance),
        grid,
        scoring="neg_log_loss",
        cv=folds,
        refit=True,
        n_jobs=1,
        error_score="raise",
        return_train_score=False,
        verbose=2,
    )
    search.fit(X_train, training_rows.observado)
    results = pd.DataFrame(search.cv_results_)
    results["mean_validation_log_loss"] = -results.mean_test_score
    for key, path in paths.items():
        results[key] = results[f"param_{path}"]
    columns = [
        *paths,
        "rank_test_score",
        "mean_validation_log_loss",
        "std_test_score",
        "mean_fit_time",
    ]
    for number in range(len(folds)):
        column = f"fold_{number + 1}_log_loss"
        results[column] = -results[f"split{number}_test_score"]
        columns.append(column)
    results[columns].sort_values("rank_test_score").to_csv(
        out / "grid_results.csv", index=False
    )
    selected = dict(config)
    selected.update({key: search.best_params_[path] for key, path in paths.items()})
    summary = {
        "n_candidates": count,
        "cv_folds": len(folds),
        "selection_metric": "log_loss",
        "best_mean_validation_log_loss": -search.best_score_,
        "best_parameters": {key: selected[key] for key in paths},
        "boundary_parameters": [
            key
            for key in paths
            if selected[key] in (min(config["grid"][key]), max(config["grid"][key]))
        ],
    }
    return search.best_estimator_, selected, summary


def run_search(filename):
    """Separa test, selecciona dentro de train, evalúa test una vez e imputa."""
    config, out = load_configuration(filename)
    out.mkdir(parents=True, exist_ok=True)
    search_folder = out / "search"
    search_folder.mkdir(exist_ok=True)
    data = load_data()
    covariance, phylogeny = load_phylogeny(config, data)
    eligible = data.loc[data.elegible_modelo].copy().reset_index(drop=True)
    X = build_features(eligible)
    known = eligible.observado.isin(CLASSES).to_numpy()
    observed = eligible.loc[known].copy()
    X_known = X[known]
    train, test = split_data(observed, config)
    manifest = observed[["ID", "unidad", "observado"]].copy()
    manifest["role"] = "train"
    manifest.loc[observed.index[test], "role"] = "test"
    manifest["validation_fold"] = pd.NA
    training = observed.iloc[train]
    for number, (_, validation) in enumerate(inner_splits(training, config), start=1):
        manifest.loc[training.index[validation], "validation_fold"] = number
    manifest.to_csv(search_folder / "splits.csv", index=False)
    print(
        f"Test reservado: {len(test)}; entrenamiento para búsqueda: {len(train)}",
        flush=True,
    )
    model, selected, summary = select_parameters(
        training, X_known[train], config, search_folder, covariance
    )
    rows, metrics = evaluate(observed, X_known, selected, covariance, model=model)
    save_run(
        out,
        config,
        selected,
        rows,
        metrics,
        eligible,
        X,
        covariance,
        phylogeny,
        summary,
    )
