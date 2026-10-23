"""Descarga verificada e indexación de los archivos originales de la release.

Los tar permanecen comprimidos: se indexan sus miembros para lectura bajo demanda.
Así evitamos miles de archivos sueltos. No se extraen rutas del archivo comprimido.
"""

import hashlib
import json
import re
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datos/tree_viewer"
RELEASE = "https://api.github.com/repos/rbouckaert/global-language-tree-pipeline/releases/tags/v2.0.0"


def download(url, target, digest=None):
    """Reutiliza descargas completas; valida SHA-256 antes de publicarlas."""
    if target.exists() and (
        not digest or hashlib.sha256(target.read_bytes()).hexdigest() == digest
    ):
        return
    temporary = target.with_suffix(target.suffix + ".part")
    print(f"Descargando {target.name}", flush=True)
    urllib.request.urlretrieve(url, temporary)
    actual = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if digest and actual != digest:
        raise ValueError(f"SHA-256 incorrecto: {target.name}")
    temporary.replace(target)


def prepare():
    """Descubre todos los cortes publicados y enlaza tablas por muestra/tiempo."""
    DATA.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        RELEASE, headers={"User-Agent": "language-tree-viewer"}
    )
    with urllib.request.urlopen(request) as response:
        release = json.load(response)
    entries = {}
    sources = []
    for asset in release["assets"]:
        name = asset["name"]
        match = re.fullmatch(r"(\d+)_(tre|csv)\.tgz", name)
        if name != "all_tree.tgz" and not match:
            continue
        target = DATA / name
        digest = (asset.get("digest") or "").removeprefix("sha256:") or None
        download(asset["browser_download_url"], target, digest)
        sources.append(
            {
                "file": name,
                "url": asset["browser_download_url"],
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            }
        )
        print(f"Indexando {name}", flush=True)
        with tarfile.open(target, "r:gz") as archive:
            for member in archive:
                basename = Path(member.name).name
                if not member.isfile() or basename.startswith("._"):
                    continue
                if match:
                    identity = re.search(
                        r"es4_tree_(\d+)_Global_alltime_(\d+)_2\.(tre|csv)$", basename
                    )
                    if not identity:
                        continue
                    sample, years, extension = identity.groups()
                else:
                    identity = re.search(r"tree_(\d+).*\.(?:tre|tree)$", basename)
                    if not identity:
                        # Se valida explícitamente el formato real, sin asignar números por posición.
                        print(f"Miembro no reconocido: {basename}", flush=True)
                        continue
                    sample, years, extension = identity.group(1), "0", "tre"
                key = f"posterior-{sample}-{years}"
                item = entries.setdefault(
                    key,
                    {
                        "id": key,
                        "kind": "posterior",
                        "sample": int(sample),
                        "years": int(years),
                    },
                )
                field = "tree" if extension == "tre" else "metadata"
                if field in item:
                    raise ValueError(f"Duplicado en catálogo: {key}/{field}")
                item[field] = {
                    "archive": name,
                    "member": member.name,
                    "offset": member.offset_data,
                    "size": member.size,
                }
    missing = [
        key
        for key, entry in entries.items()
        if "tree" not in entry or (entry["years"] and "metadata" not in entry)
    ]
    if missing:
        raise ValueError(f"Pares árbol/metadata incompletos: {missing[:10]}")
    catalog = {
        "release": release["html_url"],
        "sources": sources,
        "entries": [
            {"id": "mcc", "kind": "mcc", "sample": None, "years": 0},
            *sorted(entries.values(), key=lambda x: (x["years"], x["sample"])),
        ],
    }
    (DATA / "catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Catálogo: {len(catalog['entries'])} árboles", flush=True)


if __name__ == "__main__":
    prepare()
