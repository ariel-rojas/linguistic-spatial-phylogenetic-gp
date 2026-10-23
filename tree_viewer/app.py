"""API local de solo lectura y frontend; carga diferida con caché de tres árboles."""

import json
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .prepare import DATA
from .readers import load_tree

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Árboles lingüísticos", docs_url="/api/docs")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@lru_cache(maxsize=1)
def catalog():
    """Catálogo validado por prepare; reiniciar servidor si se actualizan fuentes."""
    path = DATA / "catalog.json"
    if not path.exists():
        raise HTTPException(
            503, "Ejecuta uv run --extra viewer python -m tree_viewer.prepare"
        )
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=3)
def tree(tree_id):
    """Solo acepta ID del catálogo, nunca rutas enviadas por el navegador."""
    entry = next((e for e in catalog()["entries"] if e["id"] == tree_id), None)
    if entry is None:
        raise HTTPException(404, "Árbol no publicado")
    return load_tree(entry)


@app.get("/")
def index():
    """Página única sin servicios externos ni bibliotecas remotas."""
    return FileResponse(STATIC / "index.html")


@app.get("/api/catalog")
def get_catalog():
    """Devuelve las combinaciones realmente disponibles y su procedencia."""
    return catalog()


@app.get("/api/trees/{tree_id}")
def get_tree(tree_id: str):
    """La geometría viaja sin las grandes anotaciones de todos los nodos."""
    return {k: v for k, v in tree(tree_id).items() if k != "metadata"}


@app.get("/api/trees/{tree_id}/nodes/{node_id}")
def get_node(tree_id: str, node_id: int):
    """Metadata completa de un nodo, conservando nombres y valores originales."""
    data = tree(tree_id)
    if node_id < 0 or node_id >= len(data["nodes"]):
        raise HTTPException(404, "Nodo inexistente")
    return {"node": data["nodes"][node_id], **data["metadata"][node_id]}
