# Guía del proyecto: de las fuentes a la imputación

Este documento recorre el proceso completo para modelar el orden de objeto y
verbo en WALS: qué datos entran, cómo se conectan, cómo se evalúa el modelo y
dónde revisar sus predicciones. La formulación matemática del modelo principal
está en [Inferencia bayesiana](bayesian_inference.md); los supuestos de las
covarianzas se explican en [Referencia del modelo](model_reference.md).

## Qué se predice

El parámetro **83A de WALS** registra una de tres respuestas:

| Respuesta | Significado |
|---|---|
| **OV** | El objeto precede al verbo |
| **Sin dominante** | WALS registra que no predomina uno de esos órdenes |
| **VO** | El verbo precede al objeto |

Un dato faltante no es una cuarta respuesta: es una observación sin etiqueta
que el modelo puede intentar imputar. «Sin dominante» es una categoría
lingüística real, no una medida de duda del modelo.

## De dónde vienen las filas y las relaciones

| Fuente | Aporte al proyecto |
|---|---|
| **WALS** | Define el inventario de lenguas, el valor 83A, las coordenadas y los faltantes. |
| **LOW** (`languages-of-the-world`) | Ayuda a enlazar identificadores, glottocodes y clasificación familiar. |
| **Global Language Tree Pipeline, v2.0.0** | Aporta el árbol MCC y las longitudes temporales de sus ramas. |
| **Natural Earth y GMBA** (opcionales) | Añaden contexto de tierra, océano y relieve al mapa; no entran en el modelo. |

WALS define las filas: LOW y el árbol no agregan lenguas al inventario que se
modela. El enlace prioriza un glottocode exacto; los códigos ISO se usan solo
cuando WALS no trae glottocode y la correspondencia identifica una lengua
individual. Los nombres parecidos no bastan para fusionar registros.

La preparación valida coordenadas, enlaces y duplicados. Registros que comparten
identificadores se agrupan en una misma **unidad de identidad**, para que una
misma lengua no se filtre de entrenamiento a test. Los casos ambiguos o sin
coordenadas se mantienen en la auditoría con un motivo, aunque no se puedan
modelar.

## Cómo se construyen las dependencias

### Componente filogenética

Para dos hojas del árbol, la covarianza Browniana es el tiempo compartido desde
la raíz hasta su ancestro común:

$$
C_{ij}=\sum_{e\in P_i\cap P_j}t_e,
$$

donde $P_i$ y $P_j$ son los caminos de cada lengua y $t_e$ es la longitud de
una rama. El proyecto normaliza $C$ a una matriz de correlaciones con diagonal
uno y, en la configuración estándar, pone en cero las asociaciones entre
familias diferentes. Una lengua sin familia asignable recibe un bloque propio.
El árbol MCC y las asignaciones familiares se mantienen fijos durante cada
ajuste.

### Componente espacial

El ajuste bayesiano estándar representa ubicaciones con armónicos esféricos y
reduce progresivamente los patrones de escala fina mediante un corte logístico
suave. NUTS infiere el grado efectivo continuo $L$ y el suavizado. La base
llega hasta el límite superior de $L$; no se vuelve a crear para cada muestra.
El kernel esférico previo es isotrópico: pares a igual distancia angular tienen
la misma covarianza antes de observar las etiquetas. La posterior puede ser
espacialmente asimétrica porque las observaciones son desiguales. La
[referencia del modelo](model_reference.md) distingue esta formulación de los
kernels clásicos y explica sus límites.

Las covarianzas representan semejanza esperada entre efectos latentes. No son
una medida directa de contacto, préstamo o causalidad. El término filogenético
puede contener señal geográfica heredada del árbol; el término espacial se
conserva por separado como parte del diseño del análisis.

## Cómo se separan entrenamiento y test

Solo las filas elegibles con una respuesta observada participan en la
evaluación. Con `test_fraction = 0.2`, `split_data` usa
`StratifiedGroupKFold` con cinco particiones y reserva la primera como test.
La semilla de esta ejecución es `20261026`: quedaron **1103 lenguas en
entrenamiento** y **275 en test**.

- **Estratificada** busca conservar las proporciones de las tres clases; no
  puede garantizarlas exactamente porque las unidades de identidad no se
  dividen.
- **Agrupada** mantiene juntas las filas conectadas por identificadores
  compartidos.
- Es un test aleatorio de identidades, no una prueba de transferencia a una
  familia o región enteramente nueva. Las evaluaciones clásicas por familia y
  por bloques espaciales responden preguntas distintas y usan parámetros fijos.

Las etiquetas test no entran en el ajuste de evaluación. Sus coordenadas y su
posición en el árbol sí: esos datos explicativos estarían disponibles aunque
faltara la respuesta lingüística.

## Modelo principal: dos ajustes bayesianos

El comando `run` ejecuta dos ajustes independientes del mismo modelo:

1. **Evaluación.** NUTS ajusta con las 1103 etiquetas de entrenamiento y
   predice las 275 lenguas test. Estas predicciones se guardan antes de que las
   etiquetas test se usen en otro ajuste.
2. **Imputación.** NUTS vuelve a ajustar desde cero con las **1378 respuestas
   observadas** y predice **940 valores faltantes elegibles**.

No se reutilizan literalmente las muestras posteriores del ajuste de
evaluación: para imputar se aprovechan también las etiquetas que estaban en
test. Las dos ejecuciones usan el mismo modelo, priors, configuración de
muestreo y regla de datos; cambia el conjunto de respuestas disponibles.

El modelo suma un intercepto global y dos campos latentes, uno espacial y otro
filogenético. Dos contrastes latentes se convierten en tres logits y luego en
probabilidades para OV, Sin dominante y VO. NUTS estima conjuntamente los
campos, sus amplitudes, el suavizado y $L$. La guía bayesiana detalla las
matrices, los priors, la verosimilitud, las iteraciones y la incertidumbre.

### Configuración del muestreo bayesiano

Los valores de esta ejecución están en `config/bayesian.json`:

| Parámetro | Valor | Interpretación |
|---|---:|---|
| Cadenas | 4 | Recorridos independientes de NUTS |
| Calentamiento | 1000 por cadena | Iteraciones adaptativas que se descartan |
| Muestras guardadas | 2000 por cadena | 8000 muestras posteriores por ajuste |
| Máximo de draws predictivos | 4000 | Estados usados para formar probabilidades predictivas |
| Prior de $L$ | Uniforme continuo entre 4 y 16 | Rango permitido para el corte espectral |
| Ancho del corte | 0.25 grados | Suavidad fija de la transición logística |
| Semilla | 20261026 | Reproduce esta configuración y partición |

Cambiar el prior de $L$, las iteraciones o los demás controles modifica
`config/bayesian.json`. Un rango posterior concentrado cerca del límite debe
motivar un análisis de sensibilidad con un límite más amplio; no se debe
interpretar el límite configurado como una escala física descubierta por los
datos.

## Resultado de la ejecución guardada

La salida de `run.json` y `report.md` corresponde a la semilla `20261026`:

| Medida en el test (275 lenguas) | Resultado |
|---|---:|
| Aciertos / errores | **239 / 36** |
| Accuracy | 0.869 |
| Balanced accuracy | 0.652 |
| Macro-F1 | 0.659 |
| Log-loss | 0.399 |
| Brier multiclase | 0.198 |

La matriz de confusión está en orden OV, Sin dominante, VO; las filas son los
valores reales y las columnas las predicciones:

| Real \\ Predicha | OV | Sin dominante | VO |
|---|---:|---:|---:|
| OV | 122 | 3 | 4 |
| Sin dominante | 7 | 2 | 7 |
| VO | 14 | 1 | 115 |

El test tenía 129 casos OV, 16 Sin dominante y 130 VO. El recall fue 0.946,
0.125 y 0.885, respectivamente. Por tanto, el rendimiento global oculta una
limitación clara: el modelo recupera solo **2 de los 16** casos Sin dominante.
Accuracy mide aciertos globales; macro-F1 da igual peso a cada clase; log-loss
y Brier evalúan las probabilidades, no solo la clase ganadora. Ninguna de estas
métricas demuestra que el modelo sea el mejor disponible ni que generalice a
familias nuevas.

Los diagnósticos de muestreo pasaron los límites configurados en ambos ajustes:

| Ajuste | R-hat máximo global | ESS bulk mínimo | ESS tail mínimo | Divergencias | Profundidad saturada |
|---|---:|---:|---:|---:|---:|
| Train para test | 1.0011 | 2052 | 2179 | 0 | 0 |
| Todas las etiquetas observadas | 1.0019 | 2392 | 2824 | 0 | 0 |

Esto respalda la exploración de la posterior en esta ejecución; no evalúa por
sí solo la calidad predictiva ni la adecuación de los supuestos.

El grado efectivo $L$ tuvo media 14.81 e intervalo creíble del 95 %
12.57–15.96 en evaluación; en el ajuste de imputación, media 15.04 e intervalo
13.23–15.96. Ambas distribuciones se acercan al límite superior 16. El
resultado describe la inferencia **dentro del rango elegido** y sugiere probar
si el rango superior debe ampliarse antes de hacer una interpretación fuerte.

## Cómo leer una imputación

La clase mostrada es la de mayor probabilidad media. Para cada lengua conviene
revisar también las tres probabilidades, sus intervalos creíbles del 95 % y la
entropía predictiva. Los intervalos resumen incertidumbre posterior de cada
probabilidad; no garantizan una probabilidad del 95 % de que la etiqueta sea
correcta. «Sin dominante» sigue siendo una clase observada; la incertidumbre se
expresa por la distribución de probabilidades.

Las medidas de influencia espacial y filogenética comparan la predicción
completa con un cálculo contrafactual que omite un componente. Son medidas de
sensibilidad del modelo, no porcentajes causales ni pesos que deban sumar uno.
El mapa distingue observaciones, train, test e imputaciones. Las imputaciones
son propuestas que requieren revisión lingüística.

## Comandos de uso

Ejecutar en PowerShell desde la raíz del repositorio:

```powershell
# Instalar el proyecto con las dependencias bayesianas
uv sync --locked --extra bayesian

# Preparar datos, enlaces de identidades y covarianza del árbol
uv run gaussian-processes prepare

# Opcional: añadir contexto de tierra, océano y montañas al mapa
uv run gaussian-processes prepare-geography

# Evaluar e imputar con NUTS; genera también los mapas locales
uv run gaussian-processes run

# Abrir el mapa generado localmente
uv run gaussian-processes map
```

Si las fuentes y matrices ya están preparadas, no hace falta repetir `prepare`
antes de cada ejecución. Para modificar hiperparámetros bayesianos, edita
`config/bayesian.json`; para cambiar el kernel, los datos de configuración o el
nombre de carpeta de la salida, usa otro JSON de `config/` y pasa `--config`.

El método clásico de scikit-learn queda disponible bajo `classic`:

```powershell
# Búsqueda clásica de hiperparámetros y evaluación
uv run gaussian-processes classic search

# Evaluaciones clásicas con familia reservada o bloques espaciales
uv run gaussian-processes classic evaluate-families
uv run gaussian-processes classic evaluate-spatial --block-km 2000 --buffer-km 250

# Pruebas del código
uv run python -m unittest discover -s tests -v
```

La búsqueda clásica tiene su propio conjunto de métricas y selección. No es el
flujo bayesiano estándar ni sus valores deben compararse sin considerar las
diferencias entre las formulaciones.

## Archivos de resultado y estructura del código

La carpeta `results/spherical_spline_family/bayesian/` contiene `run.json`,
`report.md`, `predictions.csv`, `evaluation.npz`, `imputation.npz`, `map.html`
y `posterior.html`. El archivo `run.json` conserva configuración, hashes,
métricas y diagnósticos; los `.npz` guardan muestras y predicciones detalladas.
Los datos fuente y los resultados grandes permanecen locales y están excluidos
del repositorio. Consulta [results/README.md](../results/README.md) antes de
abrir o reutilizar un archivo de muestras.

| Código | Responsabilidad |
|---|---|
| `data/preparation.py`: `prepare_data`, `read_wals`, `link_genealogy` | Descargar/leer fuentes, enlazar identidades y auditar filas. |
| `data/phylogeny.py`: `prepare_phylogeny`, `build_shared_time_covariance` | Preparar el MCC y su matriz de tiempo compartido. |
| `evaluation/splits.py`: `split_data` | Crear particiones estratificadas por clase y agrupadas por identidad. |
| `bayesian/design.py`: `spherical_basis`, `basis_scales`, `phylogenetic_design` | Construir bases y covarianzas para los dos campos. |
| `bayesian/model.py`: `categorical_gp` | Declarar priors, logits y verosimilitud categórica. |
| `bayesian/inference.py`: `sample_posterior`, `probability_draws`, `diagnose` | Muestrear con NUTS, predecir y revisar diagnósticos. |
| `bayesian/workflow.py`: `run_bayesian`, `fit_stage` | Ejecutar evaluación e imputación y guardar resultados. |
| `visualization/map.py`, `visualization/posterior.py` | Crear el mapa y el explorador de distribuciones posteriores. |

## Publicar la página de GitHub Pages

La página pública usa archivos estáticos en `docs/mapa-bayesiano/`. Una corrida
local no cambia automáticamente el sitio. Después de revisar el mapa y el
explorador nuevos, se copian y se envían junto a la documentación:

```powershell
Copy-Item results\spherical_spline_family\bayesian\map.html docs\mapa-bayesiano\map.html -Force
Copy-Item results\spherical_spline_family\bayesian\posterior.html docs\mapa-bayesiano\posterior.html -Force
git add README.md docs
git commit -m "Actualizar documentación y mapa bayesiano"
git push
```

El visualizador de árboles es un proyecto local independiente en
`../visualizador-arboles`; su instalación y sus comandos están en su propio
README. No forma parte de la página de imputación.
