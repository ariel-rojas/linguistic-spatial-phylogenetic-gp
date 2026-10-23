# Imputación lingüística con covarianza espacial y filogenética

Este proyecto analiza e imputa la clase de orden de constituyentes OV, VO o
«Sin dominante» en lenguas del World Atlas of Language Structures (WALS). Combina
un proceso espacial con una covarianza obtenida de un árbol lingüístico fechado.
Las imputaciones son predicciones para revisión; no reemplazan observaciones.

La categoría «Sin dominante» representa una propiedad lingüística observada. No
significa que el modelo tenga dudas entre OV y VO.

## Empezar

Se requiere Python 3.12 o posterior y [uv](https://docs.astral.sh/uv/).
Desde la carpeta del proyecto, en PowerShell:

```powershell
uv sync --locked
uv run gaussian-processes prepare
uv run gaussian-processes search
uv run gaussian-processes map
```

`prepare` descarga las fuentes que falten y construye los datos de trabajo.
`search` evalúa una grilla de configuraciones y escribe sus resultados bajo
`results/`. `map` genera un mapa HTML para la búsqueda elegida.

Para usar el modelo categórico bayesiano con NUTS:

```powershell
uv sync --locked --extra bayesian
uv run --extra bayesian gaussian-processes bayesian
```

Sus configuraciones están en `config/`. Se puede hacer una copia y cambiar el
campo `name` para que cada experimento escriba en una carpeta distinta. La guía
de comandos, datos, evaluación y salidas está en
[la documentación del flujo](docs/workflow.md).

## Qué hace el modelo

Las ubicaciones de los idiomas son puntos: el modelo no recibe territorios
poligonales. La componente espacial produce una superficie suave sobre la esfera.
La componente filogenética correlaciona lenguas según sus ramas compartidas en el
árbol. Ambas forman tendencias latentes para las tres clases.

La configuración predeterminada usa un spline esférico y una correlación del árbol
MCC normalizada con ceros entre familias. El motor clásico usa scikit-learn; el
experimental bayesiano estima amplitudes y suavizado con NUTS. Son dos análisis
diferentes y sus métricas no se deben comparar como si solo cambiara la inferencia.

## Datos y reproducción

WALS aporta el inventario de lenguas y el orden de constituyentes. Glottolog/LOW
ayuda a resolver identidades y familias. La filogenia procede de Global Language
Tree Pipeline. Las copias de trabajo, descargas, resultados y artefactos grandes
se generan localmente y están excluidos de Git por defecto. Consultá
[fuentes y preparación](docs/pipeline.md) y [filogenia](docs/phylogeny.md) para
conocer entradas, licencias y limitaciones.

## Código y publicación documental

El código Python está en `src/gaussian_processes/`. Las guías y la página estática
se encuentran en `docs/`. GitHub Pages puede publicar esa carpeta desde la rama
principal. Tras subir el repositorio, elegí **Settings → Pages → GitHub Actions**
para activar el flujo automático. La página explica el proceso y enlaza el código
fuente. Los archivos descargados y los resultados analíticos no se publican con
el sitio.

El proyecto es de investigación y sus resultados dependen de la cobertura y
calidad de las fuentes lingüísticas, de la muestra y de los supuestos del modelo.
