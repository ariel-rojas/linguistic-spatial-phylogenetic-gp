# Inferencia bayesiana: del dato a las probabilidades

Esta guía explica el modelo bayesiano que ejecuta **gaussian-processes run**.
Empieza por las matrices y termina en las probabilidades de OV, Sin dominante
y VO. El clasificador clásico de scikit-learn queda como alternativa y se
resume en la [guía del proyecto](workflow.md).

## 1. Qué pregunta responde

Para cada lengua con orden conocido, usamos su ubicación y su posición en el
árbol para aprender patrones compartidos. Para una lengua con orden faltante,
el resultado no es solo una etiqueta: es una distribución posterior sobre las
tres clases.

El orden de clases en el código y en los archivos es:

$$
(\mathrm{OV},\ \mathrm{Sin\ dominante},\ \mathrm{VO}).
$$

La categoría Sin dominante describe un valor lingüístico que WALS registra.
No es una manera de decir que el modelo duda entre OV y VO.

## 2. Las piezas y sus tamaños

L es ahora un parámetro continuo de la inferencia, con prior uniforme entre
**4 y 16**. La base computacional llega al límite superior configurado, con
$M_{\max}=289$ patrones (1 + 3 + 5 + … + 33). Cada muestra posterior asigna
pesos distintos a esos patrones mediante un corte suave alrededor de su propio
L. Se usan dos contrastes latentes para formar las probabilidades de tres
clases.

| Símbolo | Tamaño | Qué representa |
|---|---:|---|
| Φ | N × Mₘₐₓ | Valor de cada patrón esférico en cada lengua |
| Dλ,L | Mₘₐₓ × Mₘₐₓ | Peso de cada escala según suavizado y corte L |
| Zₛ | Mₘₐₓ × 2 | Coeficientes normales de los patrones espaciales |
| g | N × 2 | Efecto espacial en los dos contrastes |
| Q | N × r | Factor cuya covarianza reproduce la filogenia |
| Z-filo | r × 2 | Coeficientes aleatorios del efecto filogenético |
| h | N × 2 | Efecto filogenético en los dos contrastes |
| H | N × 2 | Tendencias latentes completas, incluido el intercepto |
| A | 3 × 2 | Contrastes que traducen tendencias a logits de clase |
| ℒ | N × 3 | Tres logits, uno por clase y lengua |

N cuenta lenguas; Mₘₐₓ cuenta patrones espaciales hasta el máximo configurado
de la base; r cuenta las hojas únicas necesarias para representar el efecto
del árbol en ese ajuste.

## 3. Qué es una tendencia latente

No empezamos diciendo «esta lengua es OV». Primero calculamos dos cantidades
continuas no observadas para cada lengua. Se llaman latentes porque no se miden
directamente: sirven para expresar inclinaciones relativas antes de convertirlas
en probabilidades.

Para la lengua $i$, esas dos tendencias completas son la fila $H_i$. Una
muestra posterior $s$ contiene una posible versión de los parámetros y de
los campos latentes. No existe una suma de cuatro tipos de datos llamada
«amplitud + suavizado + intercepto + F»: esos son parámetros y piezas del
cálculo, no sumandos intercambiables. La suma que sí ocurre es:

$$
H_i=\alpha+g_i+h_i.
$$

Aquí $\alpha$ es un intercepto de dos componentes, $g_i$ el efecto espacial
y $h_i$ el efecto filogenético. La fórmula se aplica por fila, es decir, por
lengua.

## 4. Cómo la suavidad espacial se convierte en covarianza

La ubicación $s_i$ se evalúa en armónicos esféricos reales:

$$
\Phi_{i,\ell m}=\sqrt{4\pi}\,Y_{\ell m}(s_i),
\qquad 0\leq \ell\leq L_{\max},\quad -\ell\leq m\leq\ell.
$$

Los grados bajos representan cambios amplios sobre el planeta; los grados altos
permiten detalle más fino. Para que los detalles más pequeños no dominen, el
modelo asigna a cada coeficiente un peso:

$$
w_\ell(\lambda,L)=\frac{\operatorname{logistic}((L+1/2-\ell)/\tau)}{1+\lambda[\ell(\ell+1)]^2},
\qquad
D_{\lambda,L}=\operatorname{diag}
\left(\sqrt{\frac{w_{\ell}(\lambda,L)}{\sum_{j=1}^{M_{\max}}w_{\ell_j}(\lambda,L)}}\right).
$$

En el denominador, $j$ recorre las columnas hasta el máximo configurado,
incluidos los órdenes $m$ repetidos por grado. La función logística reduce
gradualmente el peso alrededor de L; $\tau$ fija el ancho de esa transición y
se mantiene fijo. El coeficiente $\lambda$ controla la suavidad adicional: al
crecer, reduce más los grados altos. El modelo infiere L y $\lambda$; no usa
la grilla clásica.

Los coeficientes estándar son normales independientes:

$$
Z_s\sim\mathcal N(0,I),
\qquad
g=\sigma_s\Phi D_{\lambda,L} Z_s.
$$

Dimensiones: $(N\times M_{\max})(M_{\max}\times M_{\max})(M_{\max}\times2)=N\times2$. Al propagar la
normal por esas matrices, la covarianza de cada columna de $g$ es:

$$
\operatorname{Cov}(g_{\cdot k}\mid\sigma_s,\lambda,L)
=\sigma_s^2\Phi D_{\lambda,L}^2\Phi^\mathsf T
=\sigma_s^2R_{\mathrm{espacial}}(\lambda,L).
$$

Este es el puente exacto entre la base suave y el kernel. El código no necesita
crear y factorizar una matriz espacial densa de tamaño $N\times N$: construye
la superficie mediante la base común y reduce suavemente los patrones por
encima de L. Con una transición estrecha, se aproxima al spline esférico
regularizado de **SphericalSplineKernel**. Esta versión permite a NUTS estimar
L de manera continua; es una representación bayesiana de un término suave,
emparentada con un GAM; no llama
a **mgcv::gam** ni usa una base de nudos de spline cúbico.

La misma $\sigma_s$ y la misma $\lambda$ se comparten entre las dos columnas
de $g$; sus coeficientes normales $Z_s$ son independientes en el prior.
La base incluye el patrón constante de grado cero. Tanto este modo como
$\alpha$ tienen priors propios, pero sus desplazamientos constantes no se
interpretan por separado.

La correlación espacial previa depende solo de la distancia angular: es
isótropa. Los datos pueden llevar a una superficie posterior con picos y valles
regionales, pero el alcance de correlación no cambia automáticamente según la
región. Más detalle en [Referencia del modelo](model_reference.md).

## 5. Cómo se convierte el árbol en un efecto aleatorio

La matriz filogenética se construye a partir del tiempo de ramas compartidas,
se normaliza a diagonal uno y se ponen ceros entre familias. Se factoriza como
$R_{\mathrm{filogenética}}=QQ^\mathsf T$, con tratamiento conjunto de hojas
repetidas para que aliases de la misma punta compartan el efecto.

El campo se genera así:

$$
Z_f\sim\mathcal N(0,I),
\qquad
h=\sigma_f QZ_f.
$$

Por tanto,

$$
\operatorname{Cov}(h_{\cdot k}\mid\sigma_f)
=\sigma_f^2R_{\mathrm{filogenética}}.
$$

La misma amplitud $\sigma_f$ se comparte entre las dos columnas de $h$.

Si una lengua objetivo comparte ramas con las hojas de entrenamiento, la media
de su efecto se condiciona a ellas. Si pertenece a una familia ausente del
entrenamiento, su media filogenética es cero y conserva incertidumbre: la
componente espacial y el intercepto todavía pueden contribuir.

Los campos espacial y filogenético son independientes en el prior dados sus
parámetros. Después de observar las etiquetas, sus distribuciones posteriores
pueden asociarse o compensarse. Por eso no se deben interpretar sus amplitudes
como porcentajes causales de geografía y genealogía.

## 6. El producto matricial que crea los logits

Primero sumamos las tendencias y el intercepto de cada fila:

$$
H=\mathbf 1_N\alpha^\mathsf T+g+h,
\qquad H\in\mathbb R^{N\times2}.
$$

La matriz de contrastes que usa el código es:

$$
A=\sqrt{\frac34}
\begin{pmatrix}
1 & 1/\sqrt3\\
-1 & 1/\sqrt3\\
0 & -2/\sqrt3
\end{pmatrix},
\qquad A\in\mathbb R^{3\times2}.
$$

Sus tres filas corresponden, en este orden, a OV, Sin dominante y VO. Para
obtener un puntaje por clase multiplicamos:

$$
\underbrace{H}_{N\times2}
\underbrace{A^\mathsf T}_{2\times3}
=
\underbrace{\mathcal L}_{N\times3}.
$$

Para una sola lengua, si $H_i=(u_i,v_i)$, los tres logits son:

$$
\begin{aligned}
\ell_{i,\mathrm{OV}}&=\sqrt{3/4}\,(u_i+v_i/\sqrt3),\\
\ell_{i,\mathrm{Sin\ dominante}}&=\sqrt{3/4}\,(-u_i+v_i/\sqrt3),\\
\ell_{i,\mathrm{VO}}&=\sqrt{3/4}\,(-2v_i/\sqrt3).
\end{aligned}
$$

Por ejemplo, $u_i$ inclina la comparación hacia OV frente a Sin dominante;
$v_i$ mueve juntas esas dos clases frente a VO. Las dos direcciones son
simétricas entre las tres categorías. No se agrega un tercer parámetro latente
independiente para VO: su puntaje se obtiene de las mismas dos coordenadas.

### ¿Por qué dos contrastes producen tres logits?

Se necesitan tres puntajes porque hay tres clases. Sin embargo, los tres logits
no tienen tres grados de libertad: sumar la misma constante $c$ a todos no
cambia las probabilidades:

$$
\operatorname{softmax}(\ell_1+c,\ell_2+c,\ell_3+c)
=\operatorname{softmax}(\ell_1,\ell_2,\ell_3).
$$

Por eso un vector de tres logits tiene una dirección redundante. Esta
parametrización elimina esa dirección imponiendo que los tres logits sumen
cero. La matriz $A$ cumple $\mathbf 1_3^\mathsf T A=0$, tiene filas de
longitud uno y productos entre filas de $-1/2$. Así las tres clases reciben
el mismo tratamiento a priori. **El tercer logit es necesario como puntaje de
la tercera clase, pero no introduce un tercer efecto latente libre.**

## 7. De logits a probabilidades y respuestas observadas

Los logits son puntajes reales sin límite fijo, no probabilidades. Para cada
lengua aplicamos softmax:

$$
p_{ic}=
\frac{\exp(\ell_{ic})}
{\sum_{d\in\{\mathrm{OV},\mathrm{Sin\ dominante},\mathrm{VO}\}}
\exp(\ell_{id})}.
$$

Cada $p_{ic}$ está entre cero y uno y las tres probabilidades suman uno. Las
razones entre probabilidades se interpretan directamente a partir de la
diferencia de logits:
$p_{ic}/p_{id}=\exp(\ell_{ic}-\ell_{id})$.

Por ejemplo, si $H_i=(1,0)$, los logits son aproximadamente
$(0.866,-0.866,0)$ y softmax da probabilidades aproximadas
$(0.626,0.111,0.263)$. La primera clase recibe el mayor puntaje, pero las
otras dos conservan probabilidad.

Si la etiqueta observada de la lengua $i$ es $y_i$, su probabilidad bajo el
modelo es $p_{i,y_i}$. La verosimilitud de las etiquetas usadas en el ajuste es:

$$
p(\mathbf y\mid\text{parámetros})
=\prod_{i\in\mathrm{train}}p_{i,y_i},
\qquad
\log p(\mathbf y\mid\cdot)
=\sum_{i\in\mathrm{train}}\log p_{i,y_i}.
$$

Se multiplican solo las probabilidades de las clases que realmente se
observaron. Una etiqueta faltante no participa en esa verosimilitud.

## 8. Priors y posterior

Los priors actuales están en **config/bayesian.json**:

| Cantidad | Prior | Intuición |
|---|---|---|
| Grado máximo efectivo L | Uniforme continua entre 4 y 16 | Antes de ver etiquetas, todas las posiciones del corte dentro del rango tienen igual densidad previa |
| σₛ, σ-filo | HalfNormal(1) | Amplitudes no negativas de los dos campos |
| log(λ) | Normal(media = log(0.001), desvío = 1.5) | Suavidad positiva, con incertidumbre en escala logarítmica |
| Cada elemento de α | Normal(media = 0, desvío = 1.5) | Tendencia global antes de ver etiquetas |
| Coeficientes Zₛ y Z-filo | Normal(0, 1), independientes | Variables estándar que, transformadas por las bases, se correlacionan |

La posterior tiene la forma:

$$
p(L,\alpha,\sigma_s,\sigma_f,\lambda,Z_s,Z_f\mid\mathbf y)
\ \propto\
p(\mathbf y\mid L,\alpha,\sigma_s,\sigma_f,\lambda,Z_s,Z_f)
\ p(L)p(\alpha)p(\sigma_s)p(\sigma_f)p(\lambda)p(Z_s)p(Z_f).
$$

Los datos informan qué combinaciones de tendencias y parámetros explican las
etiquetas, mientras los priors regularizan combinaciones extremas. La
distribución posterior de L indica dónde se ubica el corte espectral más
compatible con las etiquetas. El árbol, la estructura familiar y la
definición de kernel siguen fijos: no se promedian árboles ni kernels.

Los límites del prior y el ancho del corte suave están en **config/bayesian.json**.
El campo **max_degree** de **config/spherical_spline_family.json** solo lo usa
el modelo clásico; el modelo bayesiano no fija L con ese campo. Los valores
puntuales de suavidad, amplitud espacial y peso filogenético de ese archivo
también son propios del flujo clásico. En el flujo bayesiano, L, las amplitudes
y la suavidad se infieren con sus distribuciones previas.

## 9. Cómo se infieren los parámetros y L

La posterior anterior es muy multidimensional y no se puede integrar a mano.
NUTS actualiza conjuntamente L, las amplitudes, el suavizado, los interceptos
y los coeficientes latentes. Evalúa la log-posterior y sus gradientes mediante
JAX. Esos gradientes guían trayectorias por zonas de probabilidad alta; NUTS
decide dinámicamente cuándo detener cada recorrido. Tras descartar el
calentamiento, las muestras conservadas forman una aproximación de la
distribución posterior conjunta.

El corte logístico hace que el efecto espacial cambie suavemente con L, por lo
que NUTS puede explorar valores continuos entre 4 y 16. La base tiene dimensión
fija y llega hasta el máximo configurado. Una muestra con L bajo atenúa más
los grados altos, pero no los elimina con una máscara dura. El ancho $\tau$
queda fijo en 0,25 grados; conviene incluirlo en análisis de sensibilidad antes
de interpretarlo como escala física.

Con la configuración guardada hay cuatro cadenas, 1000 iteraciones de
calentamiento y 2000 muestras por cadena. Cada ajuste produce 8000 muestras
posteriores de parámetros. Para calcular predicciones se seleccionan hasta 4000
estados de forma determinista y repartida entre cadenas.

| Control en config/bayesian.json | Valor actual | Función |
|---|---:|---|
| chains | 4 | Cadenas que exploran la posterior desde recorridos distintos |
| warmup | 1000 | Iteraciones para adaptar el muestreador; luego se descartan |
| draws | 2000 | Muestras que se guardan por cadena |
| target_accept | 0.9 | Objetivo de aceptación que regula el tamaño de paso de NUTS |
| max_tree_depth | 9 | Límite de la trayectoria que construye NUTS |
| degree_min | 4 | L mínimo permitido por el prior uniforme |
| degree_max | 16 | L máximo del prior y límite superior de la base |
| degree_taper_width | 0.25 | Ancho fijo de la transición logística alrededor de L |
| predictive_draws | 4000 | Máximo de muestras posteriores usadas para predicción |
| prior_draws | 300 | Escenarios simulados antes del ajuste para revisar los priors |

Se puede cambiar, por ejemplo, **draws** para guardar más muestras. Eso aumenta
el coste de cada ajuste y no corrige por sí solo una mala mezcla: hay que volver
a revisar ESS, R-hat, divergencias y saturaciones. El resumen posterior de L se
informa por separado para evaluación e imputación. Si se concentra cerca del
límite superior, conviene ampliar el límite de la base y volver a evaluar. La
distribución aparece en **run.json** y en el histograma de parámetros del
explorador posterior.

R-hat y ESS se calculan para L y los parámetros globales. R-hat compara la
variación dentro y entre cadenas; ESS estima cuántas muestras independientes
equivalentes hay. Los coeficientes latentes se resumen aparte porque son
muchos y cada uno puede tener mezcla lenta. Son controles de muestreo, no
medidas de acierto predictivo.

Antes de muestrear, el programa simula 300 escenarios predictivos desde los
priors, sin etiquetas, y guarda resúmenes en **run.json**. Esto permite revisar
si los priors implican probabilidades absurdamente extremas; no demuestra por sí
solo que sean adecuados.

## 10. Por qué hay dos ajustes: evaluar y luego imputar

El flujo completo separa dos objetivos:

| Etapa | Etiquetas conocidas disponibles | Salida |
|---|---|---|
| Evaluación | Solo train: 1103 lenguas | Predicciones de las 275 lenguas test |
| Imputación | Las 1378 lenguas observadas | Predicciones de las 940 lenguas elegibles con respuesta faltante |

En la primera etapa, las etiquetas test no se entregan a NUTS. Sí se pueden usar
sus coordenadas y sus hojas del árbol para formar la predicción, porque esos
atributos no son la respuesta que se está ocultando.

Después se inicia otro muestreo NUTS independiente con **todas** las respuestas
observadas, incluidas las del antiguo test. Por eso las imputaciones no usan
exactamente las muestras del modelo train-only: vuelven a estimar la posterior
con más etiquetas. Es el uso correcto de los datos una vez que la evaluación
está hecha. Las predicciones de test se conservan desde la primera etapa para
no reportar como test una predicción de un modelo que ya vio esa etiqueta.

## 11. Predicción e incertidumbre

Para cada muestra posterior se usan conjuntamente su L y su suavizado para
construir el efecto espacial; por tanto, la incertidumbre de L se propaga a la
predicción. Se calculan los tres logits y se aplica softmax. Después se
promedian **probabilidades**, no logits:

$$
\bar p_{ic}=\frac1S\sum_{s=1}^S p_{ic}^{(s)}.
$$

La clase **predicho** es la de mayor probabilidad media. Los percentiles 2.5 y
97.5 de las muestras de cada $p_{ic}$ forman un intervalo creíble puntual del
95% para esa probabilidad. Los extremos de las tres clases no tienen por qué
sumar uno; cada vector de probabilidades en cada muestra sí suma uno.

Además, el programa simula etiquetas categóricas desde cada muestra posterior.
Eso permite análisis con múltiples imputaciones y conserva dependencias entre
lenguas faltantes. Una sola etiqueta de máxima probabilidad es un resumen, no
una observación confirmada.

- **Entropía predictiva**: incertidumbre de la distribución media entre clases.
- **Información epistémica**: cuánto cambia la distribución entre muestras
  posteriores. Se calcula como entropía de la media menos la media de las
  entropías.
- **Influencia espacial o filogenética**: cambio entre la predicción completa
  y un contrafactual que omite un componente. Se mide con
  $\mathrm{TV}(p,q)=\tfrac12\sum_c|p_c-q_c|$. Es sensibilidad descriptiva,
  no una fracción causal ni un porcentaje de peso.

La incertidumbre actual incluye variación de parámetros y efectos latentes y
residuo filogenético condicional para las lenguas objetivo. No incluye árboles
alternativos, errores de etiqueta, incertidumbre de coordenadas, otras formas
de kernel, cambios de familia o incertidumbre sobre el mecanismo de datos
faltantes.

## 12. Archivos, comandos y código

### Comandos

~~~powershell
uv sync --locked --extra bayesian
uv run gaussian-processes prepare
uv run gaussian-processes run

# Cambiar el rango/prior continuo de L, otros priors o controles de muestreo:
# editar config/bayesian.json y volver a ejecutar run
uv run gaussian-processes run --config config/spherical_spline_family.json --sampling-config config/bayesian.json

# Abrir el mapa y las fichas posteriores ya guardadas:
uv run gaussian-processes map

# Ejecutar el método clásico como alternativa:
uv run gaussian-processes classic search
~~~

El comando **prepare** se necesita cuando todavía no están preparados los datos
y el árbol. **run** ejecuta las dos etapas bayesianas, escribe sus salidas y
genera mapa y explorador posterior. El antiguo comando **bayesian** se conserva
como alias de **run**.

### Dónde encontrar cada operación

| Archivo | Función | Para qué sirve |
|---|---|---|
| **bayesian/design.py** | **spherical_basis** | Construye Φ y los grados de cada columna |
| **bayesian/design.py** | **basis_scales** | Construye Dλ y normaliza la diagonal espacial |
| **bayesian/design.py** | **phylogenetic_design** | Alinea y factoriza los efectos del árbol |
| **bayesian/model.py** | **categorical_gp** | Declara priors, efectos, logits y verosimilitud |
| **bayesian/inference.py** | **sample_posterior** | Ejecuta NUTS para L y los demás parámetros; obtiene diagnósticos |
| **bayesian/inference.py** | **probability_draws** | Convierte cada muestra en probabilidades conjuntas |
| **bayesian/workflow.py** | **run_bayesian** | Separa evaluación e imputación y guarda resultados |
| **bayesian/workflow.py** | **summarize_predictions** | Medias, intervalos, clase y entropías |
| **bayesian/decomposition.py** | **predictive_components**, **total_variation** | Predicciones completas y sensibilidad contrafactual a cada término |
| **visualization/posterior.py** | **build_posterior_explorer** | Genera histogramas y fichas de probabilidades por lengua |

### Archivos de salida

Se escriben en **results/spherical_spline_family/bayesian/**:

- **run.json**: configuración, priors, hashes, métricas y diagnósticos.
- **predictions.csv**: rol de cada fila, clase, probabilidades medias e
  intervalos.
- **evaluation.npz**: muestras y predicciones conjuntas del test desde el
  ajuste train-only.
- **imputation.npz**: posterior y predicciones conjuntas para faltantes desde
  el ajuste con todos los observados.
- **report.md**, **map.html** y **posterior.html**: lecturas y exploración
  visual de esa ejecución.

Los histogramas por lengua muestran distribuciones de probabilidad entre
muestras, no un histograma de etiquetas observadas. Los parámetros σₛ, σ-filo
y λ son globales al ajuste, no valores
independientes elegidos para cada lengua.

## Referencias

- [GAM y términos suaves de mgcv](https://rweb.stat.umn.edu/R/library/mgcv/html/gam.html).
- [Suavizadores GP de rango reducido en mgcv](https://web.mit.edu/r/current/lib/R/library/mgcv/html/smooth.construct.gp.smooth.spec.html).
- [NUTS, Hoffman y Gelman](https://jmlr.org/papers/v15/hoffman14a.html).
- [NUTS en NumPyro](https://num.pyro.ai/en/stable/mcmc.html).
- [Clasificación con procesos gaussianos, Rasmussen y Williams](https://gaussianprocess.org/gpml/chapters/RW3.pdf).
