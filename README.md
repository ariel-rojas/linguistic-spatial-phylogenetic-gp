# Imputación lingüística con procesos gaussianos

Este proyecto estudia e imputa el orden de objeto y verbo registrado por WALS:
**OV**, **VO** o **Sin dominante**. Combina semejanza geográfica y parentesco
lingüístico para predecir las lenguas con ese dato faltante. Cada imputación es
una propuesta estadística que requiere revisión; no convierte un valor faltante
en una observación conocida.

«Sin dominante» es una categoría lingüística observada. No significa que el
modelo tenga poca certeza entre OV y VO.

## Empezar

Requisitos: Python 3.12 o posterior y [uv](https://docs.astral.sh/uv/).
El método principal es el modelo bayesiano categórico:

~~~powershell
uv sync --locked --extra bayesian
uv run gaussian-processes prepare
uv run gaussian-processes run
uv run gaussian-processes map
~~~

La primera preparación necesita internet para descargar fuentes que todavía no
estén guardadas localmente. El comando run realiza dos ajustes bayesianos: uno
predice un test reservado para medir rendimiento y otro vuelve a ajustar con
todas las etiquetas conocidas para imputar.

El grado efectivo de los armónicos esféricos también se infiere como un
parámetro continuo con prior acotado. Un corte suave permite que NUTS lo ajuste
y propaga esa incertidumbre a las probabilidades de predicción.

## Qué combina

- **WALS** aporta el inventario de lenguas, coordenadas y valores observados o
  faltantes del parámetro 83A.
- **LOW** ayuda a enlazar identidades y familias; no agrega filas de lenguas
  fuera del inventario WALS.
- El **árbol MCC** aporta tiempos de ramas compartidas para construir la
  covarianza filogenética.
- La componente **espacial** expresa qué ubicaciones pueden compartir una
  tendencia latente. La opción predeterminada es el spline esférico.

El método clásico de scikit-learn se conserva como alternativa en el grupo de
comandos classic; por ejemplo, uv run gaussian-processes classic search.
Sus métricas no se deben mezclar directamente con las bayesianas porque la
formulación categórica y la forma de estimar parámetros son distintas.

## Documentación

- [Flujo completo, fuentes, comandos, evaluación y resultados](docs/workflow.md).
- [Modelo bayesiano paso a paso, ecuaciones, logits y probabilidades](docs/bayesian_inference.md).
- [Referencia de kernels, covarianza filogenética y efectos de grupo](docs/model_reference.md).
- [Resultados generados y cómo interpretarlos](results/README.md).
- El visualizador de árboles ahora vive como proyecto local independiente en
  `../visualizador-arboles`; allí tiene su propio README y comandos.

## Mapa publicado

La [página del proyecto](https://ariel-rojas.github.io/linguistic-spatial-phylogenetic-gp/)
contiene el [mapa global bayesiano](https://ariel-rojas.github.io/linguistic-spatial-phylogenetic-gp/mapa-bayesiano/).
Muestra capas de observaciones, entrenamiento, test, ambos e imputaciones,
además de distribuciones posteriores por lengua. El mapa publicado es una
instantánea: volver a ejecutar el modelo localmente no cambia la página.
La guía de [flujo y comandos](docs/workflow.md#publicar-el-mapa) explica cómo
actualizarla.

Los datos descargados y los archivos de resultados grandes se guardan localmente
y no se publican por defecto. Consultá la documentación de cada fuente y sus
licencias antes de redistribuirla. El rendimiento depende de la cobertura de
WALS, de los enlaces de identidad y de los supuestos del modelo.
