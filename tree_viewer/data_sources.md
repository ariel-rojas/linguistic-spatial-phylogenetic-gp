# Fuentes inspeccionadas y reglas de interpretación

Inspección realizada el 7 de octubre de 2026. Se revisó la release y se leyeron muestras reales de archivos comprimidos en memoria; no se descargaron todas las colecciones al proyecto.

## Inventario

| Fuente de v2.0.0 | Contenido | Uso en el visualizador |
|---|---|---|
| `global-language-summary-tree-relabelled_cartoonised.tree.gz` | MCC NEXUS, copia anotada y duplicado sin anotaciones | Vista resumen por defecto |
| `global-language-trees-6636-taxa.tgz` | Distribución posterior NEXUS | Lectura de muestras, metadata presente en esa fuente |
| `all_tree.tgz` | 1000 árboles posteriores individuales, sin burn-in | Navegación sencilla por muestra; no asumir anotaciones completas |
| `3500_tre.tgz`, `4250_tre.tgz`, `5000_tre.tgz` | Backbones en cortes históricos | Vistas de linajes/subclados |
| `3500_csv.tgz`, `4250_csv.tgz`, `5000_csv.tgz` | Tablas asociadas a esos backbones | Metadata por punta del corte |

La colección de 4250 no es un nuevo posterior inferido exclusivamente con lenguas antiguas. Se deriva cortando muestras posteriores globales. La publicación describe esos tiempos como años antes del presente; no convertirlos a fechas de calendario ni restarlos del año actual sin una definición explícita de ese origen.

## Par real revisado

- Árbol: `4250_tre/es4_tree_712_Global_alltime_4250_2.tre`.
- Tabla: `4250_csv/es4_tree_712_Global_alltime_4250_2.csv`.
- **353 puntas y 353 filas**, con igualdad de conjuntos entre etiquetas de puntas y columna `tips`.
- Ejemplo de identificador local: `tips = 6644`; `tree = 712`.
- `Crown = 195.586579198`, `Time = 191.336579198`, `Time2 = 4250`.
- Se verifica `(Crown - Time) × 1000 = 4250` dentro de precisión numérica.
- La profundidad máxima del backbone leído fue aproximadamente `191.336579202`, consistente con `Time`.

En consecuencia, **Time es una distancia temporal desde la raíz hasta el corte**, no la antigüedad del corte antes del presente. Tampoco se debe utilizar una raíz de 129.6 milenios del MCC para fechar la muestra 712: cada árbol tiene sus propias edades.

Los archivos dentro de los tar no están ordenados igual: los primeros `.tre` y `.csv` inspeccionados tenían distintos IDs. Deben emparejarse por muestra y corte y luego por `tips`, jamás por posición de archivo o fila.

## Metadata de la tabla inspeccionada

Se encontraron 47 columnas:

```text
tree, REGION, SUBREGION, longitude, latitude, es_hm, es_lm, length,
clade_age, es_sq, DPLACE, Forage1, Forage2, area, speakers, island,
distancetocityyear, popd, temperature, friction, distancetocityyear2,
L1_Users, friction_log, distancetocityyear_log, popd_log,
distancetocityyear2_log, area_log, speakers_log, L1_Users_log, tips,
region, subregion, temperature2, dummy, clade_age2, Time, Crown,
Time2, tree_type, group, island_raw, popd_raw, friction_raw,
distancetocityyear_raw, distancetocityyear2_raw, area_raw, temperature_raw
```

Se conservan todas bajo sus nombres originales, incluso cuando no existe una unidad o traducción documentada. La interfaz permite filtrar por nombre y valor sin renombrar los datos.

El script de preparación realiza agregaciones, transformaciones, escalados e imputaciones con `mice`. Por ello, un valor de estas tablas no debe mostrarse automáticamente como una observación directa. `clade_age` se transforma en el script: no equivale sin más a la edad de un nodo en el Newick. El sufijo `_raw` tampoco basta para garantizar que una celda nunca fue imputada; el pipeline procesa varias columnas conjuntamente y la tabla puede no incluir una máscara por celda.

## Discrepancias que el lector debe tratar explícitamente

El script actual calcula `Time2` con signo negativo, mientras el CSV inspeccionado guarda `4250` positivo. Se conserva el valor original y se valida la edad por archivo y por `Crown - Time`; no se aplicará ciegamente una regla de signo tomada de master. El código actual del repositorio puede haber cambiado desde la generación de los assets; los hashes de cada fuente fijan qué se está viendo.

Las etiquetas numéricas internas del Newick son identificadores potenciales, no probabilidades de soporte por defecto. Deben conservarse como etiquetas y nombres de campo distintos. Un número local puede repetirse en otro árbol y no identifica el mismo ancestro.

## Preservación del contexto al cambiar árboles

Las hojas completas pueden enlazarse por glottocode. Los backbones usan IDs locales de subclados. El enlace a sus lenguas descendientes requerirá la muestra completa y una correspondencia verificada con la derivación del corte; no basta un join numérico con el MCC. El algoritmo fuente corta hacia la raíz y elimina ciertas puntas originales: tampoco se debe asumir que todas las lenguas modernas quedan representadas individualmente.

Se validaron los pares árbol/tabla para las muestras 1, 712 y 1000 en las tres fechas. No se implementa una correspondencia entre hojas modernas y linajes históricos: la interfaz indica «sin correspondencia comprobada» y vuelve a la raíz. Entre árboles actuales sí se conservan glottocodes o su MRCA.

## Fuentes

- [Release v2.0.0](https://github.com/rbouckaert/global-language-tree-pipeline/releases/tag/v2.0.0).
- [Generación de tablas y backbones](https://github.com/rbouckaert/global-language-tree-pipeline/blob/master/TreeSetAnalysisScripts/code/analysis/004a_time_slice_datasets.R).
- [Conteo de clados en cortes temporales](https://github.com/rbouckaert/global-language-tree-pipeline/blob/master/TreeSetAnalysisScripts/code/analysis/004b_tree_slicing_three_points.R).
- [Guía de los análisis posteriores](https://github.com/rbouckaert/global-language-tree-pipeline/blob/master/TreeSetAnalysisScripts/README.md).
