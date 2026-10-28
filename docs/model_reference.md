# Referencia del modelo: geografía, genealogía y grupos

Esta página aclara qué expresa cada componente del modelo y qué supuestos
incorpora. El flujo de datos y los comandos están en la
[guía del proyecto](workflow.md); la formulación completa de logits y
probabilidades está en la [guía bayesiana](bayesian_inference.md).

## 1. Kernel: semejanza entre efectos latentes

Un kernel es una regla para convertir dos entradas en una covarianza. Si el
kernel espacial entre dos ubicaciones es alto, antes de observar las etiquetas
esperamos que sus efectos latentes tiendan a ser parecidos. El kernel no es una
probabilidad de contacto, una tasa de préstamo ni una medida causal.

La idea de bases suaves y kernel está conectada. Si una superficie es
$f(s)=B(s)\beta$ y los coeficientes tienen prior
$\beta\sim\mathcal N(0,\Sigma_\beta)$, entonces:

$$
\mathrm{Cov}(f(s),f(t))
=B(s)\Sigma_\beta B(t)^\mathsf T.
$$

Una penalización cuadrática sobre coeficientes equivale a restringir sus
varianzas gaussianas. Esta es la conexión matemática entre muchos suavizadores
de GAM y procesos gaussianos. En este proyecto no se llama a una rutina de GAM:
se programa una base esférica finita y se calcula la covarianza que induce.

## 2. Tres opciones espaciales del proyecto

El kernel espacial del flujo bayesiano estándar es una base de armónicos
esféricos con corte espectral suave. El parámetro continuo $L$ se infiere con
un prior uniforme de **config/bayesian.json**; la base llega hasta su límite
superior. La covarianza condicionada a los parámetros es:

$$
R_{\mathrm{spline}}(s,t)
=
\frac{\sum_{\ell=0}^{L_{\max}}(2\ell+1)w_\ell
P_\ell(\cos\theta(s,t))}
{\sum_{\ell=0}^{L_{\max}}(2\ell+1)w_\ell},
\qquad
w_\ell=\frac{\mathrm{logistic}((L+1/2-\ell)/\tau)}{1+\lambda[\ell(\ell+1)]^2}.
$$

Los armónicos esféricos hacen que la base viva sobre una esfera y no sobre un
rectángulo de latitud-longitud. Los grados bajos forman patrones amplios; los
altos, patrones más finos. En el ajuste bayesiano actual, $L$ tiene un prior
uniforme continuo entre 4 y 16; el máximo 16 requiere 289 patrones por
contraste. $\tau$ fija el ancho de la transición del corte logístico y
$\lambda$ regula el suavizado espectral adicional. Las probabilidades
imputadas integran la incertidumbre posterior de ambos parámetros.

La covarianza previa es **isótropa**: dos pares con igual distancia angular
tienen la misma correlación. El campo posterior puede formar regiones
asimétricas porque los datos no están distribuidos uniformemente; lo que no
cambia por región es la regla previa de decaimiento. La truncación espectral
puede producir lóbulos pequeños positivos o negativos. Una correlación negativa
es una relación entre desviaciones latentes, no una probabilidad negativa ni
una barrera geográfica.

En el flujo clásico, el archivo de configuración deja fijo o busca $L$ junto
con los otros hiperparámetros. Las otras estrategias disponibles en ese motor
son:

| Nombre | Regla espacial | Qué permite variar |
|---|---|---|
| **spherical_spline** | Base armónica esférica con penalización espectral | Grado máximo y suavidad |
| **geodesic_exponential** | $\exp(-d_{\mathrm{superficie}}/\ell)$ | Alcance $\ell$ en kilómetros |
| **matern_chord** | Matérn de scikit-learn sobre distancia de cuerda en coordenadas xyz | Alcance y $\nu$, dentro de los valores configurados |

El exponencial geodésico equivale al caso Matérn $\nu=1/2$ sobre distancia de
gran círculo. El Matérn sobre cuerda es una opción distinta y conserva la
implementación nativa de scikit-learn. Para comparar o cambiar la estrategia
clásica se usa **--config**. El flujo bayesiano actual admite el spline
esférico, no las otras dos opciones.

El kernel espacial es global: una sola regla de suavidad para todo el planeta.
Un GAM puede producir una superficie flexible, pero eso no le da
automáticamente una longitud de correlación distinta en cada región. Para
aprender alcances locales haría falta una covarianza no estacionaria válida o
un modelo de convolución con parámetros locales, además de datos suficientes y
validación por bloques geográficos. Esa adaptación todavía no está
implementada.

## 3. Cómo representa el modelo el parentesco

El árbol usado proviene de la release 2.0.0 de Global Language Tree Pipeline.
Incluye 6636 hojas y un resumen MCC de una muestra posterior de árboles. MCC
significa árbol de máxima credibilidad de clados: selecciona un árbol resumen
según el soporte posterior de sus clados; no es el promedio de topologías ni
una garantía de que todas las ramas sean ciertas.

El árbol combina restricciones de clasificación lingüística, filogenias
léxicas publicadas, calibraciones temporales y un modelo de difusión
geográfica. Por eso la matriz derivada de él puede conservar señal geográfica
además de parentesco. En este proyecto se mantiene de todos modos un término
espacial separado, de acuerdo con la pregunta de investigación.

Para una hoja $i$, sea $P_i$ el camino desde la raíz hasta ella. La
covarianza browniana original es la longitud total de las ramas que dos
caminos comparten:

$$
C_{ij}=\sum_{e\in P_i\cap P_j}t_e.
$$

Puede imaginarse una desviación aleatoria independiente en cada rama. Cada
lengua hereda la suma de las desviaciones de su camino. Dos lenguas comparten
las desviaciones anteriores a su separación; las ramas terminales aportan su
propia varianza a la diagonal. Por eso $C$ es una matriz de covarianza válida.

La configuración predeterminada hace dos transformaciones:

1. **Normaliza** a correlación:
   $R_{ij}=C_{ij}/\sqrt{C_{ii}C_{jj}}$, con diagonal uno.
2. **Bloquea familias**: si dos hojas pertenecen a familias diferentes,
   $R_{ij}=0$. Dentro de cada familia conserva los valores relativos del
   árbol. Una asignación familiar desconocida se trata como una hoja individual.

El bloqueo impone una decisión explícita: en esta componente no hay correlación
filogenética entre familias. No dice que dos familias tengan historias
independientes en el mundo real; dice cómo se parametriza esta comparación.
La versión no bloqueada sigue disponible en la configuración, y la matriz
fuente nunca se reemplaza por la transformada.

El ajuste actual condiciona en un único MCC. No integra la incertidumbre de
los 1000 árboles posteriores, ni la incertidumbre de las fechas de las ramas.
El explorador local de árboles es una herramienta descriptiva: cambiar el
árbol que se ve no cambia el árbol utilizado por el modelo.

## 4. Qué significa aquí un «efecto de grupo»

Conviene separar tres usos de la palabra grupo:

1. **Agrupar identidades en train/test** impide que aliases de una lengua
   aparezcan a ambos lados de la evaluación. Es una protección contra fuga de
   información; no añade parámetros.
2. **Reservar una familia para evaluación** pregunta cómo predice el modelo
   una familia sin etiquetas de esa familia en train. No crea un efecto
   aleatorio familiar.
3. **Un efecto aleatorio** añade una variable compartida o correlacionada entre
   miembros de un grupo. El efecto filogenético actual sí es un efecto aleatorio
   estructurado: las semejanzas las determina $R_{\mathrm{filogenética}}$.

No hay ahora un intercepto categórico adicional por familia, macroárea, país o
fuente de documentación. Añadir otro intercepto familiar podría duplicar la
señal que ya modela el árbol y volver difícil separar sus amplitudes. No se
debe añadir solo porque las observaciones tengan una etiqueta de familia.

## 5. La suma y su interpretación

En la versión clásica:

$$
K=aR_{\mathrm{espacial}}+qR_{\mathrm{filogenética}}.
$$

$a$ y $q$ son amplitudes de covarianza; no son porcentajes que deban sumar
uno. Como las matrices tienen diagonal uno en la configuración predeterminada,
cada parámetro escala la varianza marginal de su componente. Aun así, no es
válido interpretar $a/(a+q)$ como «porcentaje de geografía», porque los
componentes pueden explicar patrones parecidos y se estiman condicionados a
los mismos datos.

En el flujo bayesiano se escriben dos campos:
$F_i=\alpha+g_i+h_i$. Sus amplitudes se muestrean y se describen mediante
predicciones completas y contrafactuales que omiten un término cada vez. La
distancia entre probabilidades en esos contrafactuales mide sensibilidad de la
predicción, no importancia causal.

## 6. Contexto geográfico opcional

**prepare-geography** añade polígonos de tierra y océano de Natural Earth y
regiones montañosas de GMBA. Se usan para clasificar o dibujar el punto WALS y
para informar al usuario si una coordenada cae en mar o en una región
montañosa. No intervienen en el kernel, no cambian la ubicación y no imponen
resistencia al movimiento lingüístico.

No pertenecer a una región GMBA no demuestra que el terreno sea llano. Tampoco
se usa un polígono como territorio completo de cada lengua: la unidad espacial
actual es el punto de coordenadas de WALS.

## 7. Posibles extensiones y límites

- **Dependencia espacial local:** permitir que suavidad o alcance cambien con
  la ubicación requiere una covarianza no estacionaria válida; no basta con
  colorear por región una correlación radial.
- **Muchas features a la vez:** el código actual ajusta una respuesta. Un
  modelo conjunto podría compartir información entre features y aprender
  correlaciones entre ellas, pero requiere una matriz lengua × feature y un
  diseño que trate explícitamente los datos faltantes.
- **Áreas lingüísticas poligonales:** un centroide es una aproximación; una
  covarianza areal debe integrar el proceso sobre polígonos o definir una red
  de vecindad. Dibujar polígonos no implementa ese modelo.
- **Árboles alternativos:** propagar incertidumbre filogenética requiere
  repetir o integrar el ajuste sobre árboles y comprobar correspondencia de
  puntas y hojas.

Estas son alternativas de investigación, no capacidades del código actual.

## Referencias breves

- [Release del Global Language Tree Pipeline, v2.0.0](https://github.com/rbouckaert/global-language-tree-pipeline/releases/tag/v2.0.0).
- [Guzmán Naranjo y Becker (2022), control estadístico de sesgo tipológico](https://doi.org/10.1515/lingty-2021-0002).
- [Guzmán Naranjo y Mertner (2023), efectos areales en tipología](https://doi.org/10.1515/lingty-2022-0037).
- [Paciorek y Schervish (2006), covarianzas no estacionarias](https://www.stat.berkeley.edu/users/paciorek/files/environmetrics/paci.sche.2006.pdf).
