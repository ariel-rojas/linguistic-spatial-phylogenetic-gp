# GP bayesiano: flujo completo implementado

El comando `bayesian` utiliza NumPyro/NUTS y una verosimilitud categórica conjunta.
`train` y `search` conservan sklearn con Laplace y uno-contra-el-resto. Una diferencia
predictiva entre ellos no puede atribuirse exclusivamente a integrar hiperparámetros.

## 1. Datos y separación

WALS aporta lenguas, coordenadas y OV/VO/Sin dominante. LOW aporta familias y
metadatos de identidad. El árbol MCC aporta tiempos compartidos. `load_data`
carga la tabla y `load_phylogeny` verifica hashes, alinea hojas, normaliza C y
pone cero entre familias. Cada hoja sin familia tiene bloque individual.
El archivo original no se modifica. La fuente tiene 2185 hojas enlazadas.

Los elegibles incluyen 1378 observados y 940 faltantes. `split_data` usa
StratifiedGroupKFold, aproximadamente 20%, semilla 42: 1102 train y 276 test.
Conserva identidades compartidas en la misma partición y aproxima las proporciones
de clases. La primera inferencia usa SOLO etiquetas train. Se permiten coordenadas
y árbol conocidos de test, pero sus etiquetas no entran a la verosimilitud.

Después de evaluar se inicia un segundo muestreo con las 1378 observaciones para
imputar los 940 faltantes. Las predicciones de test no se reemplazan por las de
ese segundo ajuste. No hay grid search dentro de este flujo. La primera ruta bayesiana implementada
es este split aleatorio agrupado; los comandos de evaluación por familias y
bloques espaciales siguen usando sklearn, no NUTS.

## 2. Qué son F y F^(s)

F es una tabla de tendencias ocultas, con una fila por lengua y dos columnas de
contrastes. No contiene etiquetas ni probabilidades. Dos contrastes permiten
construir tres logits de suma cero, sin elegir una clase de referencia.

F^(s) significa una realización plausible de esa tabla en la muestra posterior s.
El superíndice NO es una potencia. La expresión «amplitudes + suavizado + interceptos + F» era una enumeración del contenido de la muestra, NO una suma matemática. Cada muestra incluye también sus amplitudes,
suavizado e interceptos. No representa una iteración de mejora ni otra base de datos.

Ejemplo inventado: logits (1.2, -0.8, 0) en orden OV, SD, VO producen probabilidades
(0.696, 0.094, 0.210). Si la observación es OV, la contribución a la verosimilitud
es 0.696. Otra muestra de parámetros y tendencias produce otras probabilidades.

## 3. Covarianza y representación

R_filo[i,j] = C[i,j] / sqrt(C[i,i] C[j,j]), con cero entre familias diferentes.
K = sigma_spatial² R_spatial(lambda) + sigma_phylogenetic² R_filo.

Para evitar factorizar K en cada paso usamos una representación equivalente:

```
g = sigma_spatial * Phi * diag(sqrt(w / sum(w))) * z_spatial
h = sigma_phylogenetic * L * z_phylogenetic
F = g + h
logits = (alpha + F) * A.T
```

Phi contiene armónicos reales multiplicados por sqrt(4*pi). Para cada armónico de
grado l, w = 1/(1 + lambda*[l(l+1)]²). Se incluyen todos los órdenes y el grado
cero: reproduce exactamente el spline de Legendre existente. Grado 16 equivale
a 289 coeficientes por contraste. El modo constante y el intercepto tienen priors
propios: no interpretamos sus contribuciones constantes por separado.

L es Cholesky de la correlación de hojas ÚNICAS de train. Las filas WALS que
comparten hoja comparten el efecto. NUTS recibe L en formato disperso BCOO para
aprovechar los ceros entre familias, sin aproximar sus valores. No se agrega
nugget a los alias. Los z son
normales estándar independientes con dos columnas.

A es una matriz 3x2: filas de norma uno, suma cero y productos entre filas -1/2.
Por ello los tres logits tienen prior simétrico y cada clase tiene covarianza K
para su función. La restricción evita la dirección común redundante de softmax.
La normalidad corresponde a las tendencias, no a los códigos de clase.

Geografía y filogenia pueden explicar patrones similares: sus amplitudes pueden
compensarse en la posterior. No interpretamos su reparto como atribución causal.
El bloqueo familiar tampoco elimina geografía de las relaciones internas del árbol.

## 4. Priors y verosimilitud

En `posterior.html`, abrir **«Entender el modelo: priors, rangos, datos y evaluación»**.
La guía usa `run.json` de esa ejecución, no el archivo de configuración actual.
Incluye las fórmulas de θ, dimensiones reales de Z, priors analíticos, controles de
NUTS, comprobación predictiva previa y métricas por clase. `visualization/bayesian_context.py`
calcula cuantiles y masas con SciPy; `posterior_assets/workflow.js` organiza la explicación.

Un **soporte** es el conjunto de valores permitidos. Un **intervalo central del 95 %**
del prior no es una restricción: queda un 5 % de probabilidad fuera. El intervalo
posterior es distinto y depende de las etiquetas. Los gráficos de priors muestran
el 99,8 % central sin renormalizar sus masas; λ usa eje log10 para mostrar su escala.
Los coeficientes Z son normales estándar independientes en el prior; sus posteriores
pueden depender entre sí. Su transformación por Φ y L produce efectos correlacionados.

Priors de config/bayesian.json:

- sigma_spatial y sigma_phylogenetic ~ HalfNormal(1).
- log(lambda) ~ Normal(log(0.001), 1.5²).
- alpha: dos contrastes independientes Normal(0, 1.5²).
- z_spatial y z_phylogenetic: coeficientes independientes Normal(0,1).

Las amplitudes son desviaciones estándar latentes, no tasas por milenio.
Los valores puntuales spatial_variance, phylogenetic_rate, smoothing y la grilla
del JSON sklearn NO se emplean como estimaciones en NUTS. Se reutilizan grado,
datos, estructura filogenética y protocolo de separación.

```
pi[i,c] = exp(logit[i,c]) / sum_r exp(logit[i,r])
y[i] ~ Categorical(pi[i])
verosimilitud = producto_i pi[i,clase_observada_i]
posterior ∝ verosimilitud × prior(z) × prior(alpha) × prior(amplitudes,lambda)
```

Antes del ajuste `prior_predictive` simula 300 escenarios sin usar etiquetas.
Resume las frecuencias esperadas por clase y cuántas probabilidades son extremas.
Queda en run.json; no certifica automáticamente que los priors sean adecuados.

## 5. Cómo itera NUTS

`categorical_gp` declara el modelo y `sample_posterior` ejecuta NUTS. JAX calcula
gradientes en doble precisión. Una posición de la cadena incluye amplitudes,
lambda, alpha y todos los z. NUTS incorpora un impulso auxiliar, construye
trayectorias guiadas por gradientes de la log posterior y evita prolongarlas
cuando comienzan a regresar sobre sí mismas. No busca quedarse con un máximo.

Durante warmup adapta el muestreador; esas iteraciones se descartan. Las cadenas
posteriores permiten aproximar integrales por promedios. Las cadenas se ejecutan
secuencialmente en CPU; no se requiere GPU.

Se revisan R-hat de rango y ESS bulk/tail de TODOS los parámetros, incluidos z,
divergencias y saturación de profundidad. Criterio: Rhat<1.01, ESS bulk/tail>=400,
cero divergencias y saturaciones. Si falla queda diagnostics_warning visible en
reporte y mapa. No se cambian umbrales para ocultar problemas.

`posterior_predictive_training` compara frecuencias reales con réplicas de etiquetas
simuladas de la posterior. No es una métrica fuera de muestra. NUTS no muestrea
las etiquetas discretas faltantes: se generan durante la predicción.

## 6. Imputación e incertidumbre

El explorador también muestra una sensibilidad contrafactual para cada caso. Para cada
muestra guardada se calculan las probabilidades completas y las de dos escenarios
auxiliares, omitiendo el efecto espacial o el filogenético. La distancia de variación
total entre vectores categóricos es `TV(p,q) = 1/2 Σ_c |p_c − q_c|`, entre 0 y 1.
Se resumen estos cambios a través de las muestras para mostrar la incertidumbre.
Esto no es un peso estimado, una fracción de varianza ni una atribución causal: mide la
sensibilidad de la predicción al retirar un término y mantener el resto de esa muestra.
Los dos efectos pueden ser importantes simultáneamente o compensarse.

Por muestra posterior, la componente espacial se evalúa en Phi de las nuevas
coordenadas. Esto es exacto para nuestro kernel de rango finito.

Para filogenia, con R_train = L L.T:

```
A_target = R_target,train * inverse(L.T)
V_target = R_target,target - A_target * A_target.T
h_target = sigma_phylogenetic * (A_target*z_phylogenetic + sqrt(V_target)*epsilon)
```

epsilon es normal estándar y se genera de nuevo por muestra. Se usan sistemas
triangulares, no inversas explícitas. La raíz de V se calcula por autovalores:
se rechaza negatividad material y solo se recorta error de redondeo.

Las predicciones de todas las lenguas objetivo se generan juntas para conservar
su dependencia. Para una familia ausente de train, la media filogenética es cero
pero su varianza permanece; la componente espacial puede informarla.

Cada muestra da logits y luego softmax. Se promedian probabilidades, NO logits.
Los percentiles 2.5 y 97.5 son intervalos creíbles puntuales del 95% de cada
probabilidad, no intervalos de la etiqueta ni bandas simultáneas.

También se simulan etiquetas de la distribución categórica de cada muestra para
hacer análisis con múltiples imputaciones. La etiqueta argmax es solo un resumen.
Se guardan entropía predictiva y diferencia entre entropía de la media y media de
entropías (información epistémica). Ninguna sustituye estudiar calibración.

La incertidumbre incluye coeficientes, hiperparámetros continuos y residual nuevo.
NO incluye árboles alternativos, errores de etiquetas, coordenadas inciertas,
otros grados/kernels ni incertidumbre de los priors elegidos. La ausencia de datos
no se modela como un proceso propio. Sin dominante no significa incertidumbre.

## 7. Módulos y funciones

| Archivo en src/gaussian_processes/bayesian | Funciones | Responsabilidad |
|---|---|---|
| design.py | spherical_basis, basis_scales, phylogenetic_design | Álgebra del GP |
| model.py | categorical_gp | Priors, tendencias y verosimilitud |
| inference.py | sample_posterior, diagnose | NUTS y diagnósticos |
| inference.py | prior_predictive, probability_draws | Simulación y predicción conjunta |
| workflow.py | run_bayesian, fit_stage | Separación, dos ajustes y resultados |
| workflow.py | summarize_predictions, write_report | Intervalos y reporte |

## 8. Ejecución y archivos

```powershell
uv sync --locked --extra bayesian
uv run --extra bayesian gaussian-processes bayesian
uv run --extra bayesian gaussian-processes map --bayesian
# Cambiar controles/prior usando un JSON propio:
uv run --extra bayesian gaussian-processes bayesian --sampling-config config/bayesian.json
```

Resultados en results/spherical_spline_family/bayesian/:

- run.json: configuración, priors predictivos, diagnósticos, métricas y hashes.
- predictions.csv: roles y probabilidades medias con intervalos puntuales del 95%.
- evaluation.npz: posterior de train y predicciones conjuntas de test.
- imputation.npz: posterior de todos los observados e imputaciones conjuntas.
- report.md y map.html: lectura humana de resultados y fichas por lengua.

Los NPZ guardan muestras de parámetros por cadena y draw, probabilidades, etiquetas
simuladas y sus IDs. Las probabilidades utilizan una subselección determinista de
hasta predictive_draws estados equilibrada entre cadenas; se guardan todos los
coeficientes posteriores. No hay muestreo independiente por lengua.

```python
import numpy as np
run = np.load('results/spherical_spline_family/bayesian/imputation.npz', allow_pickle=False)
print(run['classes'])
print(run['target_ids'][:5])
print(run['probabilities'].shape)  # muestra, lengua, clase
print(run['posterior_sigma_spatial'].shape)  # cadena, draw
```

Reutilizar el mismo name reemplaza su ejecución. Para comparar configuraciones,
copia el JSON base y cambia name. Evalúa log-loss, Brier, macro-F1, recall de
Sin dominante, diagnósticos y sensibilidad a priors, no solo accuracy. El test
histórico ya se ha consultado; la comparación actual es exploratoria.

## Referencias

- [NUTS, Hoffman y Gelman](https://jmlr.org/papers/v15/hoffman14a.html).
- [Clasificación GP, Rasmussen y Williams](https://gaussianprocess.org/gpml/chapters/RW3.pdf).
- [NUTS de NumPyro](https://num.pyro.ai/en/stable/mcmc.html).
- [Muestreo elíptico](https://proceedings.mlr.press/v9/murray10a.html).
- [INLA, Martino y Riebler](https://arxiv.org/abs/1907.01248).

## Mapa publicado en GitHub Pages

La [vista mundial interactiva](mapa-bayesiano/) muestra la instantánea de la
ejecución bayesiana completada. Sus capas permiten alternar datos observados,
test, train, train junto con test e imputaciones. Al abrir una lengua se ven las
probabilidades medias y sus intervalos; el enlace de la ficha abre el explorador
de histogramas posteriores y parámetros globales.

El mapa es una salida pública del proyecto: sus puntos muestran nombres,
coordenadas WALS, clase observada, predicción y probabilidades. La página conserva
la atribución de WALS y de los mapas base. No se publican las fuentes descargadas,
los CSV de resultados ni los archivos NPZ de muestras.

La página conserva una copia de `map.html` y `posterior.html` junto a su índice.
Después de generar un ajuste y un mapa nuevos, reemplazá esas dos copias desde
PowerShell y subí el cambio:

```powershell
uv run --extra bayesian gaussian-processes bayesian
uv run --extra bayesian gaussian-processes map --bayesian
Copy-Item results\spherical_spline_family\bayesian\map.html docs\mapa-bayesiano\map.html -Force
Copy-Item results\spherical_spline_family\bayesian\posterior.html docs\mapa-bayesiano\posterior.html -Force
git add docs
git commit -m "Update published Bayesian map"
git push
```

El flujo de GitHub Actions reconstruye la página cuando cambia `docs/`. La
ejecución actual se publica como instantánea; cambiar los resultados locales por
sí solo no actualiza la web.

## Explorar las distribuciones en la web

`map --bayesian` genera también `posterior.html`, sin volver a muestrear. En las
fichas de test e imputaciones aparece «Explorar distribuciones e incertidumbre».
El explorador permite buscar por lengua, código WALS o familia y alternar test
e imputaciones. Funciona sin conexión; el mapa base sí requiere internet.

- **Predicción final:** tres barras categóricas con medias posteriores que suman 100%.
- **Probabilidad de cada clase:** histogramas de las muestras, medias e intervalos
  creíbles puntuales del 95%. La altura es porcentaje de muestras por intervalo.
- **Parámetros del ajuste:** posteriores de amplitudes, suavizado e interceptos,
  con trazas de las amplitudes/suavizado y diagnósticos. Son parámetros globales
  del ajuste correspondiente, no parámetros propios de cada lengua.

`visualization/posterior.py` valida NPZ/CSV/configuración, calcula histogramas y
genera el HTML. `posterior_assets/` separa estructura, estilo e interacción.
Los histogramas contienen todas las muestras guardadas de probabilidades; las
trazas muestran hasta 200 puntos por cadena. No se estiman densidades suavizadas
ni nuevas posteriores al mover un selector. La tabla F no se grafica como si
fuera una etiqueta o una probabilidad: sus efectos se reflejan al transformar
cada muestra en las probabilidades que sí se muestran.
