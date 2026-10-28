"""Comandos del pipeline; el flujo bayesiano es la ruta estándar."""

import argparse

from threadpoolctl import threadpool_limits

from .configuration import DEFAULT_CONFIGURATION


def _add_classic_command(commands, name, help_text):
    """Registra una herramienta clásica y sus opciones propias."""
    command = commands.add_parser(name, help=help_text)
    command.add_argument("--config", default=DEFAULT_CONFIGURATION)
    if name == "map":
        command.add_argument("--fixed", action="store_true")
    if name == "evaluate-spatial":
        command.add_argument("--block-km", type=float, default=2000)
        command.add_argument("--buffer-km", type=float, default=250)
    return command


def main():
    """Expone el ajuste bayesiano como flujo principal y sklearn como alternativa."""
    parser = argparse.ArgumentParser(
        description="Imputación multiclase con un proceso gaussiano bayesiano"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    for name, help_text in [
        ("run", "Flujo estándar: evaluar e imputar con inferencia bayesiana"),
        ("bayesian", "Alias de compatibilidad para el flujo estándar run"),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", default=DEFAULT_CONFIGURATION)
        command.add_argument("--sampling-config", default="config/bayesian.json")

    commands.add_parser(
        "prepare", help="Preparar WALS, enlazar identidades y construir MCC"
    )
    commands.add_parser(
        "prepare-geography", help="Preparar capas físicas opcionales del mapa"
    )
    classic = commands.add_parser(
        "classic", help="Herramientas alternativas con clasificadores de scikit-learn"
    )
    classic_commands = classic.add_subparsers(
        dest="classic_command", required=True
    )
    for name, help_text in [
        ("search", "Búsqueda clásica de hiperparámetros y evaluación"),
        ("train", "Evaluación clásica con parámetros fijos"),
        ("evaluate-families", "Evaluación clásica reservando cada familia"),
        ("evaluate-spatial", "Evaluación clásica por bloques espaciales"),
        ("map", "Mapa de resultados clásicos"),
        ("report", "Comparación de ejecuciones clásicas guardadas"),
    ]:
        _add_classic_command(classic_commands, name, help_text)

    map_command = commands.add_parser(
        "map", help="Abrir el mapa bayesiano guardado (predeterminado)"
    )
    map_command.add_argument("--config", default=DEFAULT_CONFIGURATION)
    map_choice = map_command.add_mutually_exclusive_group()
    map_choice.add_argument(
        "--classic", action="store_true", help="Usar resultados de la búsqueda clásica"
    )
    map_choice.add_argument(
        "--fixed", action="store_true", help="Usar resultados clásicos de parámetros fijos"
    )
    map_choice.add_argument(
        "--bayesian", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.command in {"run", "bayesian"}:
            from .bayesian.workflow import run_bayesian

            run_bayesian(args.config, args.sampling_config)
            return
        if args.command == "prepare":
            from .data.preparation import prepare_data

            prepare_data()
            return
        if args.command == "prepare-geography":
            from .data.geography import prepare_geography

            prepare_geography()
            return
        if args.command == "map":
            from .visualization.map import build_map

            build_map(
                args.config,
                fixed=args.fixed,
                bayesian=not (args.classic or args.fixed),
            )
            return

        if args.command == "classic":
            action = args.classic_command
        else:
            action = args.command

        if action == "search":
            from .evaluation.search import run_search

            run_search(args.config)
        elif action == "train":
            from .evaluation.experiment import train

            train(args.config)
        elif action == "evaluate-families":
            from .evaluation.all_families import evaluate_all_families

            evaluate_all_families(args.config)
        elif action == "evaluate-spatial":
            from .evaluation.spatial import evaluate_spatial

            evaluate_spatial(args.config, args.block_km, args.buffer_km)
        elif action == "report":
            from .evaluation.report import comparison_report

            comparison_report()
        elif action == "map":
            from .visualization.map import build_map

            build_map(args.config, fixed=args.fixed)


if __name__ == "__main__":
    main()
