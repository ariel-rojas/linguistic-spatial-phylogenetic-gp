# Flujo del proyecto y diseño de componentes

> Actualización 21/10/2026: el modelo predeterminado usa correlación filogenética normalizada con ceros entre familias. Las fórmulas con C cruda y tasas por milenio describen la variante histórica; para la actual se usa R por bloques y amplitud marginal. Ver [transformación, configuración y unidades](phylogeny.md#family-covariance). El comando `bayesian` implementa NUTS; ver [flujo bayesiano](bayesian_inference.md).

```text
WALS + auditoría de identificadores low + árbol MCC fijado por hash
    ↓ prepare
lenguas.csv + covariance.npy + tips.csv + manifest.json
    ↓ features: [x, y, z, índice de hoja]
split externo estratificado y agrupado (aprox. 20 %)
    ↓ solo train
108 candidatos × 3 folds internos → menor log-loss
    ↓
ajuste ganador en train → evaluación en test
    ↓
ajuste nuevo con todos los observados → imputaciones → mapa
```

## Responsabilidades

| Módulo | Funciones/clases principales | Responsabilidad |
|---|---|---|
| `data/preparation.py` | `prepare_data`, `link_genealogy`, `group_identifiers` | WALS, metadatos low, identidad, cobertura |
| `data/phylogeny.py` | `read_tree`, `build_shared_time_covariance`, `prepare_phylogeny`, `load_phylogeny` | Árbol, matriz y validación de artefactos |
| `models/features.py` | `build_features`, `coordinates_xyz` | Coordenadas + índice; no usa etiquetas |
| `models/base.py` | `KernelComponent`, `ColumnKernel` | Contrato abstracto y selección de columnas |
| `models/spatial.py` | `SpatialMaternComponent`, `GeodesicExponentialComponent`, `SphericalSplineComponent` | Estrategias y rutas de parámetros sklearn |
| `models/spherical.py` | `great_circle_distance`, `GeodesicExponentialKernel`, `SphericalSplineKernel` | Geometría superficial y covarianzas válidas |
| `data/geography.py` | `prepare_geography`, `classify_locations` | Cartografía física descriptiva y manifiesto |
| `evaluation/spatial.py` | `spatial_splits`, `evaluate_spatial` | Bloques geográficos con buffer, valores fijos |
| `visualization/geography.py` | `add_geographic_layers`, `geographic_details` | Contexto opcional en mapa y fichas |
| `models/phylogenetic.py` | `BrownianPhylogeneticComponent`, `PhylogeneticKernel` | Peso evolutivo y consulta de C |
| `models/kernels.py` | `build_kernel` | Composición de componentes por suma |
| `models/gp.py` | `build_model`, `predict` | GPC multiclase y probabilidades |
| `evaluation/search.py` | `select_parameters`, `run_search` | GridSearchCV sin acceso al test |
| `evaluation/experiment.py` | `evaluate`, `impute`, `save_run` | Evaluación, reajuste y trazabilidad |
| `evaluation/splits.py` | `split_data`, `inner_splits` | Particiones e identidades disjuntas |
| `evaluation/all_families.py` | `evaluate_all_families` | Una familia completa fuera en cada ajuste |
| `visualization/colors.py` | `COLORS`, `LABELS` | Colores plenos y etiquetas compartidas |
| `visualization/map.py` | `build_map` | Capas y fichas con tres probabilidades |

## Por qué estas clases

El diseño usa **Strategy** (estrategia) y composición. `KernelComponent` establece un solo contrato abstracto: `build()` devuelve un kernel de sklearn. `SpatialMaternComponent` y `BrownianPhylogeneticComponent` son dos estrategias concretas. Sus objetos almacenan la configuración necesaria; `build_kernel` los construye y suma sus resultados.

Hay dos adaptadores pequeños. `ColumnKernel` hace que cada componente vea únicamente sus columnas. `PhylogeneticKernel`, que hereda de la clase abstracta `Kernel` de sklearn, consulta la matriz por índices: implementa llamada rectangular, diagonal y gradiente vacío. El peso filogenético es un `ConstantKernel` externo. No se implementa el algoritmo del GP: sklearn conserva la inferencia y la composición aditiva.

Esta separación evita una clase gigante y evita un framework de plugins. No hay registro automático ni contenedor de dependencias. La matriz se carga una vez por comando y se pasa explícitamente a los modelos; no se lee un archivo en cada evaluación del kernel.

## Cambiar un componente

1. Crear una clase que herede de `KernelComponent` y cuyo `build()` devuelva un kernel sklearn válido.
2. Incorporarla a `build_spatial_component`, que selecciona por `spatial_kernel`. Si necesita otros datos, documentar qué columnas recibe y cómo se preparan.
3. Declarar sus parámetros en `SPATIAL_PARAMETERS` de `models/spatial.py` y en el JSON. La validación y las rutas de GridSearchCV se derivan de ese único registro; no se duplican listas en search.py.
4. Probar simetría/semidefinición, diagonal, consulta train/test, `clone` y `set_params`, y finalmente un ajuste pequeño.

Para cambiar valores basta editar el JSON. Los tres kernels espaciales están implementados y se explican en [spatial_models.md](spatial_models.md). Las alternativas filogenéticas de [alternativas filogenéticas](phylogeny.md#alternatives) siguen siendo investigación.

## Reglas para seguir el proceso

Los índices del árbol son globales para el par matriz/tabla de hojas. Nunca se renumeran al crear folds. Las coordenadas y el árbol no usan etiquetas OV/VO; sus bloques pueden prepararse antes del split. Los parámetros se seleccionan solo dentro de train. El GP final para imputar sí utiliza todos los datos observados, después de cerrar la evaluación.

Las columnas auditadas de WALS conservan nombres como `observado`, `predicho`, `estado`, `unidad` y `familia_id`. Las nuevas columnas son `tree_glottocode` y `tree_tip_index`. No se sobrescribe la observación al imputar. Funciones y clases están en inglés; comentarios y documentación, en español.

## Ruta bayesiana alternativa

```text
misma tabla + correlación filogenética por familias
  → mismo split externo agrupado y estratificado
  → simulación del prior sin etiquetas
  → NUTS solo train → diagnósticos → probabilidades e intervalos test
  → NUTS nuevo con todos los observados
  → diagnósticos → imputaciones conjuntas → tabla + mapa
```

No utiliza grid search. Ver [priors, funciones y ecuaciones](bayesian_inference.md).
