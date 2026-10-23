"""Inicio local: uv run --extra viewer python -m tree_viewer [--port 8769]."""

import argparse

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizador local de árboles")
    parser.add_argument("--port", type=int, default=8769)
    args = parser.parse_args()
    uvicorn.run("tree_viewer.app:app", host="127.0.0.1", port=args.port)
