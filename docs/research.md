# Procesos espaciales, múltiples features y polígonos

Esta revisión separa tres ideas que se parecen en el mapa pero responden a
preguntas distintas: controlar dependencia geográfica con puntos, estudiar muchas
features a la vez y usar territorios poligonales.

## Proceso gaussiano con localizaciones puntuales

Guzmán Naranjo y Becker (2022) estudian asociaciones tipológicas controlando
dependencia genealógica y areal. Para el componente espacial representan cada
lengua mediante una coordenada (latitud y longitud) y usan una matriz de
covarianza de proceso gaussiano: pares próximos pueden tener efectos latentes
parecidos y la asociación decae con la distancia. Su caso compara orden OV/VO y
posición de afijos, y separa el efecto espacial del control filogenético.

El flujo conceptual es parecido al nuestro: covariables o etiquetas observadas,
un campo espacial latente, un término filogenético y una probabilidad para la
respuesta. La formulación exacta es distinta: el artículo estudia otra respuesta,
otro diseño y un GP exponencial cuadrático en el espacio de coordenadas; nuestro
motor bayesiano es una likelihood categórica de tres clases, con superficie de
armónicos esféricos, covarianza del árbol MCC y priors explícitos para amplitudes
y suavizado. Por eso tomamos la intuición del GP, no afirmamos replicar el modelo.

## Qué añade el trabajo multifeature de Naranjo y Mertner

En 2023, Guzmán Naranjo y Mertner estudian inventarios de fonemas de 645 lenguas
africanas. Cada fonema se codifica como presente/ausente. El modelo combina:

- regresión filogenética para la relación genealógica;
- GP para la relación areal;
- probit multivariado para que fonemas distintos puedan correlacionarse.

La escala e intensidad del proceso espacial se aprenden con información de los
fonemas conjuntamente. Así, un fonema muy frecuente por casualidad no decide por
sí solo el alcance espacial. Esta es una estrategia adecuada para explorar si el
alcance del contacto se comparte entre features. Nuestro programa, en cambio,
ajusta una sola feature cada vez y no aprende correlación entre rasgos.

El artículo de 2024 *Spatial effects with missing data* desarrolla multivAreate 2,
con inferencia espacial aproximada, faltantes y comparación en varios conjuntos.
Sigue siendo una formulación para localizaciones/puntos y múltiples resultados;
no es un paso que reemplaza automáticamente cada punto por un polígono.

## Qué cambia al representar áreas con polígonos

Un GP espacial convencional requiere que cada observación tenga una ubicación o
que se defina una covarianza válida entre pares de observaciones. Un polígono no
tiene una única distancia a otro polígono: se podría elegir distancia entre
centroides, mínima distancia entre bordes, área de solapamiento o integrar la
covarianza a través de ambas superficies. Cada definición expresa un supuesto
distinto y debe garantizar que la matriz resultante siga siendo semidefinida
positiva para que sea una covarianza válida.

Mertner (2025), en su tesis sobre modelos espaciales con áreas de lenguas,
explica que el GP puntual no se aplica directamente a la unidad poligonal que
usa en ese capítulo. Utiliza un modelo autorregresivo condicional (CAR): primero
construye una matriz de vecindad a partir de intersección o solapamiento de
territorios; los polígonos aislados se conectan a su vecino más cercano, y los
vecinos reciben pesos por distancia. El modelo representa dependencias por esa
red, no por un campo continuo de distancia como el GP. Las decisiones sobre
vecindad y pesos pasan a ser parte esencial del supuesto.

Una segunda idea, McNew, Derungs y Moran (2018), usa teselaciones de Voronoi para
visualizar áreas asociadas a puntos y mostrar huecos de cobertura. Es una técnica
cartográfica, no una estimación de territorios reales ni un GP areal. Los límites
dependen de qué lenguas estén en la muestra, por lo que no conviene describirlos
como fronteras lingüísticas observadas.

Por lo tanto, si en el futuro queremos usar territorios de Ethnologue u otra
fuente, hay que elegir explícitamente el objetivo: (a) mostrar polígonos en el
mapa; (b) agregar una covarianza puntual sobre centroides; (c) construir un GP
areal integrado sobre polígonos; o (d) definir vecindad y probar un CAR. (a) es
visualización; (b) es una aproximación sencilla que pierde forma y solapamiento;
(c) es más fiel pero requiere una covarianza areal válida y más cálculo; (d) es
viable si la pregunta es sobre vecinos/solapamiento, pero reemplaza el decaimiento
continuo por relaciones de una red. Ninguna opción debe mezclar sus unidades con
la covarianza MCC sin explicitar cómo se suman los procesos.

## ¿Por qué cada feature podría necesitar otra mezcla espacial/filogenética?

Para una feature concreta, las dos amplitudes son globales: la geografía no pesa
más para una lengua que para otra como parámetro. El efecto latente espacial y el
filogenético sí toman valores distintos por ubicación/linaje. Al ajustar otra
feature por separado, cambian las etiquetas y puede cambiar la posterior de las
amplitudes globales. Una feature heredada principalmente dentro de familias puede
apoyar mayor amplitud filogenética; otra extendida entre familias vecinas puede
apoyar mayor amplitud espacial. Ambas señales están confundidas cuando familias
y regiones se solapan.

Para compartir información entre features, un desarrollo gradual sería:

1. establecer una versión independiente por feature como referencia;
2. unificar entrenamiento y armar una tabla lengua × feature que retenga el
   mecanismo de faltantes;
3. probar una escala de decaimiento común y campos espaciales propios por feature;
4. permitir amplitudes feature-específicas con distribución jerárquica común;
5. solo después incluir correlación explícita entre features y validarla.

Si queremos que la geografía tenga importancia distinta para cada lengua, eso
sería un modelo de coeficientes variables en el espacio, por ejemplo
\(F_i=\sigma_s(x_i)g_i+\sigma_p(x_i)h_i\). Es mucho más exigente: ahora habría
que estimar dos superficies adicionales de pesos. Sin muchas observaciones por
región y controles contra confusión, el modelo podría repartir arbitrariamente
una señal entre componentes. No es lo que estima el código actual.

## Referencias

- [Guzmán Naranjo y Becker (2022), *Statistical bias control in typology*](https://doi.org/10.1515/lingty-2021-0002).
- [Guzmán Naranjo y Mertner (2023), *Estimating areal effects in typology*](https://doi.org/10.1515/lingty-2022-0037).
- [Guzmán Naranjo, Mertner y Urban (2024), *Spatial effects with missing data*](https://doi.org/10.1515/opli-2024-0032).
- [McNew, Derungs y Moran (2018), *Towards faithfully visualizing global linguistic diversity*](https://lrec.elra.info/lrec2018-main-129).
- [Mertner (2025), tesis *Spatial models of linguistic data in Africa and beyond*](https://publikationen.uni-tuebingen.de/xmlui/handle/10900/173835).
