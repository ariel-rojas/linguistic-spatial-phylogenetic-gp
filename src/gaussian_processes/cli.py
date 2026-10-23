"""Comandos separados para datos, búsqueda, referencia fija, familias y mapa."""

import argparse

from threadpoolctl import threadpool_limits

from .configuration import DEFAULT_CONFIGURATION


def main():
    """Expone las etapas y limita hilos para evitar sobrecarga en matrices GP."""
    parser = argparse.ArgumentParser(
        description="Imputación OV / VO / Sin dominante con GP"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    bayesian = commands.add_parser(
        "bayesian", help="GP categórico: evaluar e imputar con NUTS"
    )
    bayesian.add_argument("--config", default=DEFAULT_CONFIGURATION)
    bayesian.add_argument("--sampling-config", default="config/bayesian.json")
    commands.add_parser(
        "prepare", help="Preparar WALS, auditar identidades y construir covarianza MCC"
    )
    commands.add_parser(
        "prepare-geography", help="Preparar contexto físico Natural Earth y GMBA"
    )
    commands.add_parser(
        "report",
        help="Comparar las tres ejecuciones preparadas en results/comparison.md",
    )
    for name, help_text in [
        ("search", "Seleccionar hiperparámetros, evaluar e imputar"),
        ("train", "Evaluar parámetros fijos, sin grilla"),
        ("evaluate-families", "Reservar cada familia, con parámetros fijos"),
        ("evaluate-spatial", "Validación por bloques espaciales, con parámetros fijos"),
        ("map", "Generar mapa interactivo de la búsqueda"),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument(
            "--config",
            default=DEFAULT_CONFIGURATION,
            help="Configuración JSON (por defecto: spline esférico)",
        )
        # El mapa también puede mostrar la referencia fija y su familia reservada.
        if name == "map":
            command.add_argument(
                "--fixed", action="store_true", help="Visualizar la ejecución train"
            )
            command.add_argument(
                "--bayesian", action="store_true", help="Visualizar la ejecución NUTS"
            )
        if name == "evaluate-spatial":
            command.add_argument(
                "--block-km",
                type=float,
                default=2000,
                help="Tamaño aproximado de bloque",
            )
            command.add_argument(
                "--buffer-km",
                type=float,
                default=250,
                help="Separación mínima train/test",
            )
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.command == "bayesian":
            from .bayesian.workflow import run_bayesian

            run_bayesian(args.config, args.sampling_config)
        elif args.command == "prepare":
            from .data.preparation import prepare_data

            prepare_data()
        elif args.command == "search":
            from .evaluation.search import run_search

            run_search(args.config)
        elif args.command == "prepare-geography":
            from .data.geography import prepare_geography

            prepare_geography()
        elif args.command == "report":
            from .evaluation.report import comparison_report

            comparison_report()
        elif args.command == "evaluate-spatial":
            from .evaluation.spatial import evaluate_spatial

            evaluate_spatial(args.config, args.block_km, args.buffer_km)
        elif args.command == "train":
            from .evaluation.experiment import train

            train(args.config)
        elif args.command == "evaluate-families":
            from .evaluation.all_families import evaluate_all_families

            evaluate_all_families(args.config)
        else:
            from .visualization.map import build_map

            build_map(args.config, fixed=args.fixed, bayesian=args.bayesian)
