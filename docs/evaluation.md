# Evaluación: clases, métricas y protocolos

Esta guía reúne la interpretación de las tres clases, la selección de parámetros
y la evaluación por familias. Las métricas sirven para ambos motores. La grilla
y el recorrido familiar son de sklearn; el flujo NUTS está en [la guía bayesiana](bayesian_inference.md).


<a id="classes"></a>

## Clases y métricas

WALS 83A distingue OV, VO y ausencia de orden dominante. No son niveles de una magnitud: codificarlas 0, 1, 2 no autoriza una regresión numérica. El significado lingüístico de la tercera clase se describe en [WALS 83](https://wals.info/chapter/83).

### Clasificador

Se usa `GaussianProcessClassifier(multi_class="one_vs_rest")`. Hay tres problemas binarios: OV frente al resto, Sin dominante frente al resto y VO frente al resto. Cada problema usa una función latente GP, enlace logístico y aproximación de Laplace. Las probabilidades binarias se normalizan para obtener tres probabilidades que suman uno. Esto **no es** un GP categórico conjunto con softmax; esa alternativa está implementada en el comando `bayesian`. [Implementación sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html).

Las columnas tienen orden fijo: `p_OV`, `p_SD`, `p_VO`. La etiqueta es el argmax. Un empate exacto se resuelve por orden OV, Sin dominante, VO, únicamente como convención reproducible. No existe ya `umbral_vo`. Ejemplo: (0.45, 0.10, 0.45) expresa duda entre OV/VO; no implica Sin dominante.

### Métricas

Sea y_i la clase real, p_ic su probabilidad predicha y c una de las tres categorías:

\[
\operatorname{LogLoss}=-\frac1n\sum_i\log p_{i,y_i},
\qquad
\operatorname{Brier}=\frac1n\sum_i\sum_{c=1}^3(p_{ic}-1[y_i=c])^2.
\]

Ambas usan probabilidades, no solo argmax. Menor es mejor. Brier se exporta sin dividir por dos, en [0,2]; no debe compararse numéricamente con el Brier binario anterior de [0,1].

- **Accuracy:** aciertos / registros.
- **Precision por clase:** proporción correcta entre predicciones de esa clase.
- **Recall por clase:** proporción recuperada entre observaciones reales de esa clase.
- **F1 por clase:** media armónica de precision y recall.
- **Macro-F1:** media del F1 de las tres clases; da visibilidad a Sin dominante.
- **Weighted-F1:** pondera por frecuencias observadas; las mayoritarias pesan más.
- **Balanced accuracy:** media de recalls de las clases presentes en el conjunto evaluado.
- **Matriz de confusión:** filas reales y columnas predichas, en orden OV, Sin dominante, VO.

Si una familia reservada no contiene una clase, su recall es `null`, no un acierto perfecto. Para macro-F1, precision y F1 se usa cero en términos indefinidos y se mantiene el denominador de tres clases. Una familia toda OV predicha perfectamente tiene accuracy=1 y balanced_accuracy=1, pero macro-F1=1/3. Por eso hay que leer estas cifras junto con el soporte y no comparar medias familiares como si todas cubrieran las mismas clases.

La muestra contiene 82 registros Sin dominante frente a 1296 OV/VO. La estratificación preserva esa frecuencia; no rebalancea artificialmente las clases. Un modelo puede tener buena accuracy y aun así no reconocer la tercera categoría. Los reportes permiten comprobarlo directamente.


<a id="search"></a>

## Hiperparámetros y validación sklearn

### Covarianza y estrategias

La configuración predeterminada es `config/spherical_spline_family.json`, con correlación filogenética normalizada por familias. `config/spherical_spline.json` conserva la referencia global histórica.
La [guía espacial](spatial_models.md) explica sus parámetros `max_degree` y
`smoothing`, además del exponencial geodésico. Las configuraciones históricas suman `q C_MCC` cruda; la actual usa correlación por bloques y amplitud marginal. Las grillas tienen 108 combinaciones. Las particiones descritas aquí son comunes.

La fórmula y grilla siguientes corresponden a la **referencia Matérn histórica**
`config/multiclass.json`; no deben aplicarse a las otras estrategias.

\[
K_{ij}=a M_\nu(d_{ij}/\ell)+q C_{ij}.
\]

C contiene el tiempo compartido desde la raíz del MCC, en milenios, sin normalizar. Su diagonal H ronda 129.601. Por tanto la varianza marginal total es a + qH; q no es una fracción ni la varianza marginal filogenética. Los pesos no miden causalmente contacto/herencia.

a = `spatial_variance`; q = `phylogenetic_rate`; ell = `length_scale_km`; nu es la suavidad espacial. d es la cuerda entre coordenadas xyz sobre una esfera de 6371 km, no el arco geodésico.

Con r=d/ell:

\[
M_{1/2}(r)=e^{-r},\quad M_{3/2}(r)=(1+\sqrt3 r)e^{-\sqrt3 r},
\]
\[
M_{5/2}(r)=(1+\sqrt5 r+5r^2/3)e^{-\sqrt5 r}.
\]

[Matérn en sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html).

### Grilla

| Parámetro | Candidatos |
|---|---|
| length_scale_km | 500, 1500, 3000, 6000 |
| spatial_variance | 0.25, 1, 4 |
| phylogenetic_rate | 0.002, 0.008, 0.032 |
| nu | 0.5, 1.5, 2.5 |

Son 108 combinaciones y tres folds internos: 324 ajustes multiclase. Cada ajuste entrena tres GP binarios uno-contra-el-resto con aproximación de Laplace. `optimizer=None` evita una segunda optimización dentro de cada candidato. La grilla modifica parámetros sklearn mediante `set_params`, incluido nu.

Las tasas filogenéticas implican varianzas marginales aproximadas de 0.259, 1.037 y 4.147 en este árbol. Se mantiene C intacta; se selecciona el peso evolutivo. Los rangos son una elección inicial explícita, no un resultado científico.

Se minimiza la media aritmética de log-loss interna (`neg_log_loss` en sklearn); un empate exacto conserva el primer candidato según el orden de la grilla. El ganador se reajusta en todo train; posteriormente se evalúa una vez en test y se realiza un ajuste nuevo con todos los observados para imputar. [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).

### Partición externa

`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, primera partición. Aproxima un 20 % y las proporciones de clases; no divide identidades para lograr porcentajes exactos. Se agrupan transitivamente glottocodes e ISO compartidos. El enlace al árbol es exacto por glottocode, por lo que una misma hoja tampoco cruza train/test.

Familias y regiones pueden estar en ambos lados: este protocolo evalúa lenguas con vecinas/parientes potencialmente observadas. Para extrapolar a familias completas, usar el protocolo familiar.

La población actual contiene 1378 registros observados: 644 OV, 652 VO y 82 Sin dominante. Los identificadores y tamaños exactos de la ejecución están en `search/splits.csv` y el reporte. Cambiar cobertura, semilla o grupos puede cambiar el split. [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).

| Clase | Train | Test |
|---|---:|---:|
| OV | 516 | 128 |
| Sin dominante | 65 | 17 |
| VO | 521 | 131 |
| Total | 1102 | 276 |

### Particiones internas

Solo el train externo entra a una segunda `StratifiedGroupKFold`, de tres folds. Todas las combinaciones comparten esas particiones. Las verificaciones exigen identidades disjuntas y tres clases en entrenamiento y validación. La columna `validation_fold` de `search/splits.csv` permite inspeccionarlas.

El árbol y las coordenadas son metadatos, sin etiquetas tipológicas. La matriz no se reestima al ver test ni se poda por fold. Un futuro transformador aprendido con y sí tendría que ajustarse dentro de cada partición.

### Interpretación

El mejor candidato es el mejor de esta grilla bajo ese criterio, no un óptimo global. El campo `search` de `run.json` informa parámetros en extremos; no se amplía automáticamente la grilla después de mirar test. Las desviaciones entre folds no son intervalos bayesianos.

Los resultados anteriores usan otra covarianza y otra cobertura; no constituyen una comparación controlada con esta ejecución. Además, el proyecto ya ha examinado datos de test en otras iteraciones: la evaluación es exploratoria, aunque esta búsqueda mantenga aisladas las etiquetas externas.

[Inferencia bayesiana estudiada, sin implementación](bayesian_inference.md).


<a id="families"></a>

## Reservar familias completas

`uv run gaussian-processes evaluate-families` recorre todas las familias raíz elegibles, deja una fuera, entrena con las demás y predice la reservada. Usa las tres clases y los cuatro valores principales del JSON, sin búsqueda de hiperparámetros. `test_family` no limita este recorrido.

Cada registro conocido elegible con familia documentada aparece exactamente una vez como prueba. Los 76 conocidos sin familia low se excluyen únicamente de este experimento y sus ID se registran en `run.json`; sí participan en el split aleatorio y en el ajuste para imputar. Se verifican identidades disjuntas; un conflicto de identificadores entre familias detiene el proceso. El resultado queda en `results/<name>/families/`:

- `metrics_by_family.csv`: tamaño, aciertos y métricas por familia.
- `predictions.csv`: todas las probabilidades fuera de muestra.
- `run.json`: medias, distribución descriptiva, métricas por clase y familia, exclusiones, parámetros y hashes.
- `distribution.png`: histograma de exactitud y relación con el tamaño familiar.

La media por familia da el mismo peso a familias grandes y pequeñas. La exactitud agrupada pondera cada registro por igual. Las familias de una lengua solo pueden acertar o fallar; esto puede generar muchas exactitudes 0 y 1. Los percentiles no son intervalos de confianza y los ajustes comparten entrenamiento.

Una familia reservada puede no contener las tres clases: balanced_accuracy solo promedia recalls observables; macro-F1 conserva tres clases, con cero en términos indefinidos. Las métricas probabilísticas siempre reciben tres columnas. El entrenamiento sí debe contener las tres clases.

Los hiperparámetros seleccionados por el split aleatorio no se cargan automáticamente: su selección habría usado etiquetas de familias que esta evaluación pretende ocultar. Para estimar generalización de un procedimiento que también selecciona hiperparámetros, habría que repetir una búsqueda interna dentro de cada entrenamiento familiar. Si interesa generalizar a familias nuevas, los folds internos deberían separar familias también, siempre que su cobertura de clases lo permita. Ese procedimiento anidado no está implementado.

Para reservar únicamente la familia configurada ejecuta `uv run gaussian-processes train`. El resultado opcional se guarda en `fixed/family_evaluation.csv` y `fixed/run.json` (campo `held_out_family_metrics`). También usa valores fijos.

Con el árbol global, una familia reservada puede compartir ramas profundas con las familias de entrenamiento. La covarianza filogenética entre familias ya no está forzada a cero. El tiempo compartido se calcula siempre desde la raíz original del MCC, sin podar o volver a enraizar dentro de cada fold. La clasificación low se usa para definir los grupos reservados, no para calcular la covarianza.
