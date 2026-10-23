# Resultados

La ejecución predeterminada está en **spherical_spline_family/** (correlación normalizada por familias). Su evaluación fija está en **fixed/map.html**; su búsqueda está pendiente. **spherical_spline/** conserva la búsqueda histórica global, sin normalizar.
**geodesic_exponential/** contiene la comparación superficial sencilla y **phylogenetic_mcc/** conserva la referencia Matérn sobre cuerda. Son experimentos distintos, no duplicados.
El resumen conjunto es **comparison.md** (`uv run gaussian-processes report`).

Estado al 20/10/2026: búsquedas completas en spherical_spline y phylogenetic_mcc. La búsqueda geodésica está pendiente; su mapa disponible está en geodesic_exponential/fixed/map.html. Las tres estrategias tienen evaluaciones fija y espacial. Solo phylogenetic_mcc tiene el recorrido de todas las familias (184). El informe no confunde archivos de partición con búsquedas terminadas.

```text
<nombre-del-experimento>/
  map.html                 test, train, ambos, observaciones e imputaciones
  predictions.csv          una fila por registro WALS; role identifica su uso
  run.json                 métricas, configuración, selección y trazabilidad
  search/
    grid_results.csv       108 candidatos: parámetros, log-loss por fold y media
    splits.csv             train/test y fold de validación por registro
  families/
    metrics_by_family.csv  una fila por familia reservada
    predictions.csv        predicciones fuera de muestra
    run.json               resumen, métricas por clase, exclusiones y trazabilidad
    distribution.png       distribución del rendimiento entre familias
  fixed/                   referencia opcional, sin búsqueda de hiperparámetros
    map.html
    predictions.csv
    run.json
    family_evaluation.csv  evaluación de la única familia configurada
  spatial/
    predictions.csv       cada lengua fuera de muestra una vez; bloque y fold
    run.json              métricas, parámetros fijos, buffer e IDs retirados
```

En la tabla principal, `role` vale `train`, `test`, `imputed` o `excluded`.
Train conserva la observación y no lleva predicción ni probabilidad de evaluación.
Test lleva predicciones del modelo ajustado exclusivamente con train.
Las imputaciones proceden de un ajuste posterior con todos los observados.
Los excluidos conservan `motivo_exclusion`; no desaparecen del inventario.

En `search/splits.csv`, `validation_fold=1` significa que ese registro validó en
el primer fold interno y entrenó en los restantes. El test externo nunca tiene
fold interno. `unidad` identifica registros que deben permanecer juntos.

`run.json` reúne matriz de confusión, métricas por clase, parámetros, versiones
y hashes. En `families/run.json` también se guardan los ID sin familia conocida y
las métricas de cada clase por familia. Ese protocolo usa parámetros fijos.

Cada comando reemplaza sus salidas. Para conservar un experimento, usa otro `name`
en el JSON. No se acumulan logs, capturas ni copias históricas aquí.
El reporte histórico MCC está en [../docs/phylogenetic_report.md](../docs/phylogenetic_report.md).
Las fuentes geográficas se guardan una sola vez en `datos/geography/`, fuera de results.

## Inferencia bayesiana

`spherical_spline_family/bayesian/` contiene un flujo separado: `run.json`,
`predictions.csv`, `evaluation.npz`, `imputation.npz`, `report.md`, `map.html` y `posterior.html`.
Primero revisa los diagnósticos de ambos ajustes en el reporte. Los intervalos
son de probabilidades, no de códigos de clase. Los NPZ conservan dependencia
entre imputaciones y permiten análisis posteriores sin fijar etiquetas como ciertas.
