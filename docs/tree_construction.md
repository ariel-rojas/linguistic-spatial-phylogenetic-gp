# Construcción del árbol y fundamento de la covarianza

Guía científica. Para ejecutar el proyecto consulta [el flujo](pipeline.md).

## Cómo construyeron el árbol

### 2.1. Inventario e identificadores

Los autores parten de Glottolog 4.0 y seleccionan lenguas, no dialectos. Tras excluir determinadas categorías —entre ellas lenguas extintas, artificiales, de signos, pidgins, criollos y algunas lenguas coloniales recientes— quedan 6636 lenguas contemporáneas. El identificador es el glottocode. Estas decisiones delimitan la muestra del estudio; no implican una valoración sobre las categorías excluidas.

### 2.2. Relaciones establecidas dentro de familias

Glottolog aporta 323 unidades entre familias y lenguas no afiliadas, incluidas 124 no afiliadas. Sus agrupaciones se imponen como restricciones de monofilia: los miembros de un grupo deben descender de un ancestro común que no incluya lenguas externas al grupo. Las bifurcaciones no resueltas pueden explorarse dentro de esas restricciones.

Por tanto, el árbol combina relaciones lingüísticas documentadas con relaciones inferidas. El vínculo profundo entre dos familias no tiene necesariamente el mismo respaldo que las relaciones internas de una familia bien estudiada.

### 2.3. Filogenias léxicas anteriores

Se incorporan resultados de nueve análisis de familias o grandes subgrupos: Atlantic-Congo, Austronesian, Indo-European, Turkic, Uralic, Dravidian, Pama-Nyungan, Semitic y Sino-Tibetan. El repositorio incluye conjuntos posteriores y tablas que enlazan sus nombres con glottocodes, procedentes de D-PLACE.

No se analiza una única lista de cognados comparable entre todas las lenguas del planeta. Se trasladan restricciones de los estudios anteriores: clados compatibles con soporte posterior superior al 95 %, junto con información sobre media y desviación de sus edades. Los autores resuelven correspondencias de dialectos y códigos retirados para esa transferencia. No debemos reutilizar esas sustituciones automáticamente para emparejar registros WALS.

### 2.4. Calibración temporal

La clasificación indica quién está emparentado; las calibraciones informan cuándo se separaron los linajes. Se utilizan estudios léxicos fechados, evidencia histórica, lingüística y arqueológica. Donde falta información, se emplean intervalos amplios; por ejemplo, entre 600 y 12000 años para el ancestro común de ciertas familias.

Hay dos edades diferentes: la separación de una familia respecto de otras (`originate`) y el ancestro común más reciente de sus miembros muestreados (`tMRCA`). No deben confundirse.

Las calibraciones están expresadas en miles de años: el archivo contiene, por ejemplo, `normal(55.0, 2.55)` para Sahul y `normal(16.00, 0.77)` para Americas. Son distribuciones que representan restricciones e incertidumbre, no fechas observadas sin error. Una rama representa la diferencia temporal entre sus extremos.

### 2.5. Geografía

Se usan ubicaciones de Glottolog y fuentes suplementarias. Un modelo de difusión sobre una esfera asigna probabilidades a desplazamientos a lo largo de las ramas. Permite variación de velocidades entre ramas. Así, los datos geográficos ayudan a resolver partes inciertas del árbol y relaciones entre familias.

La geografía no actúa simplemente como una regla de agrupar vecinos. Interactúa con la estructura permitida, los tiempos y los parámetros de difusión. En nuestro proyecto se mantendrá adicionalmente el término espacial solicitado; posponemos estudiar la separación interpretativa de ambas contribuciones.

### 2.6. Supuestos sobre el pasado profundo

Se supone un origen único en África subsahariana y una cota superior de 250000 años para la raíz. También se imponen restricciones asociadas a Sahul y a la colonización de América, incluida la monofilia del conjunto americano definido por los autores y restricciones de desplazamiento oceánico.

Estas son hipótesis de construcción, no demostraciones de una protolengua mundial mediante comparación lingüística. El manuscrito reconoce el carácter especulativo de las relaciones profundas.

### 2.7. Diversificación e inferencia bayesiana

El prior de diversificación `multi-Yule` permite tasas distintas entre familias y para el tramo profundo que las conecta. Esto evita exigir que familias de edades parecidas tengan necesariamente tamaños parecidos.

BEAST explora topologías, tiempos y parámetros mediante MCMC. De forma esquemática:

\[
p(T,\theta\mid\text{evidencia})\propto
p(\text{ubicaciones}\mid T,\theta)\,
p(T\mid\theta,\text{restricciones y calibraciones})\,p(\theta).
\]

Esta ecuación resume el procedimiento; parte de la evidencia léxica previa entra como restricciones y calibraciones, no como una nueva verosimilitud global de cognados. Una propuesta incompatible con una restricción dura no se acepta. Las restantes se exploran según su probabilidad posterior, no solo buscando un único óptimo.

Para hacer viable el cálculo se combinan análisis con muestreo global y cobertura completa de diferentes macrorregiones. El resultado distribuido incluye 1000 árboles posteriores. Esto representa incertidumbre sobre relaciones y fechas, condicionada a los supuestos anteriores.

### 2.8. Árbol resumen MCC

El archivo `global-language-summary-tree-relabelled_cartoonised.tree.gz` contiene un árbol de máxima credibilidad de clados (MCC), en NEXUS. MCC resume el soporte de los clados en la muestra posterior; no es la media aritmética de los árboles ni prueba que todas sus ramas sean ciertas.

Las anotaciones de soporte, alturas e intervalos son distintas de las longitudes de ramas. Para esta primera integración usaremos las longitudes publicadas tras los dos puntos del árbol, sin recalcularlas a partir de las anotaciones. El esquema de resumen de alturas no debe suponerse a partir del nombre MCC.

El NEXUS utiliza una tabla `Translate` para convertir identificadores numéricos en nombres de hojas. Hay que resolverla antes de extraer el glottocode. La inspección anterior del archivo encontró 6636 hojas distintas y una profundidad raíz-hoja prácticamente constante de 129.601172582915 unidades, coherente con miles de años. No se debe interpretar ese número como años ni como una fecha conocida con esa precisión.

## Del árbol a la matriz

### 3.1. Covarianza utilizada

Sea `depth(v)` la suma de las longitudes de ramas desde la raíz original hasta el nodo `v`. Definimos:

\[
C_{ij}=\operatorname{depth}(\operatorname{MRCA}(i,j)),\qquad
C_{ii}=\operatorname{depth}(i).
\]

MRCA es el ancestro común más reciente. El valor fuera de la diagonal es el tiempo recorrido conjuntamente desde la raíz antes de que ambos linajes se separen. **No es la edad del MRCA medida hacia atrás desde el presente.**

En un árbol ultramétrico de altura `H`, si el MRCA vivió hace `t` unidades, `C_ij = H - t`. Para la distancia de camino entre puntas `D_ij`, la identidad general es `C_ij = (depth(i) + depth(j) - D_ij) / 2`. La matriz `D` se puede exportar para interpretar divergencias, pero no usar directamente como covarianza: tiene diagonal cero y entradas positivas, incompatible con una matriz de covarianza no trivial.

### 3.2. Ejemplo verificable

Árbol de ejemplo, con números didácticos, no del archivo real:

```text
          ┌── A: 2
raíz ──8──┤
          └── B: 2
   └───────── C: 10
```

Su Newick es `((A:2,B:2):8,C:10);`. Las tres profundidades son 10. A y B comparten 8; A y C solo comparten la raíz, cuya profundidad es cero:

\[
C=\begin{pmatrix}10&8&0\\8&10&0\\0&0&10\end{pmatrix}.
\]

Se conservan 10 y 8; no se convierten en 1 y 0.8. Una bifurcación antigua implica menos tiempo compartido y menor covarianza. Las ramas terminales contribuyen a la diagonal.

### 3.3. Por qué es una covarianza válida

Imaginemos cambios latentes independientes en cada rama, con varianza proporcional a su duración. Cada lengua hereda los cambios de todas las ramas de su camino. Dos lenguas comparten exactamente los cambios anteriores a su separación. Por eso su covarianza es la suma de las duraciones compartidas.

Formalmente, si `A_ie` indica si la rama `e` pertenece al camino de `i` y `l_e >= 0`, entonces `C = A diag(l) Aᵀ`. Esto demuestra que la matriz es semidefinida positiva. Es una justificación matemática; no obliga a usar esos indicadores como features ni a conservar el kernel anterior.

Se trata de un movimiento browniano sobre el árbol para la **propensión latente** de cada clasificador. OV, VO y Sin dominante siguen siendo categorías. No estamos suponiendo que sus códigos numéricos evolucionen como una variable continua ni implementando un modelo de transiciones discretas entre órdenes.

La raíz tiene valor latente fijo de referencia: no se añade varianza ancestral compartida anterior a ella. Por tanto, dos lenguas cuyo MRCA sea la raíz tienen covarianza filogenética cero. Añadir un intercepto ancestral aleatorio sería otra decisión de modelo y queda fuera de esta primera integración.

### 3.4. Kernel completo y unidades

\[
K_{ij}=a\,M_\nu(d^{\mathrm{espacial}}_{ij};\ell)+q\,C_{ij}.
\]

- `a`: amplitud de covarianza espacial.
- `ell`: escala espacial en km; se conserva la distancia de cuerda actual.
- `nu`: suavidad del Matérn, incluido en la grilla.
- `q`: varianza latente acumulada por unidad temporal del árbol, propuesta de nombre `phylogenetic_rate`.

Multiplicar por `q` ajusta el peso del proceso en el clasificador; no modifica ni normaliza el árbol ni la matriz almacenada. Con `q = 1` se utiliza numéricamente `C` sin cambio de amplitud. No hay escala temporal de decaimiento adicional en esta propuesta browniana.

Los valores antiguos de `genealogical_variance` no se deben copiar sin revisar: antes la diagonal genealógica era uno; ahora ronda 129.6. Como grilla inicial explícita, proponemos `phylogenetic_rate: [0.002, 0.008, 0.032]`, que produce varianzas marginales aproximadas de 0.26, 1.04 y 4.15 para esa altura. Son candidatos de partida, no estimaciones obtenidas de etiquetas. La matriz continúa en sus unidades originales. Con las 4 escalas espaciales, 3 amplitudes espaciales y 3 valores de nu actuales se conservan **108 combinaciones**.

## Fuentes revisadas

- [Manuscrito de Bouckaert et al., versión 2](https://doi.org/10.31235/osf.io/f8tr6_v2), especialmente métodos, páginas 19–26; [PDF](https://osf.io/download/hctka/).
- [Release v2.0.0: árbol MCC y muestra posterior](https://github.com/rbouckaert/global-language-tree-pipeline/releases/tag/v2.0.0).
- [Preparación de restricciones y datos](https://github.com/rbouckaert/global-language-tree-pipeline/blob/master/monos/data/README.md).
- [Incorporación de filogenias léxicas](https://github.com/rbouckaert/global-language-tree-pipeline/blob/master/monos/data/lexical/README.md).
- [Calibraciones temporales](https://github.com/rbouckaert/global-language-tree-pipeline/blob/master/monos/data/calibrations.nex) y [plantilla BEAST](https://github.com/rbouckaert/global-language-tree-pipeline/blob/master/monos/data/template.xml). Los enlaces a master son contexto metodológico; el artefacto de producción quedará fijado por release y hash.
- [BEAST TreeAnnotator y resumen MCC](https://www.beast2.org/treeannotator/).
- [ape: covarianzas filogenéticas, `vcv.phylo`](https://stat.ethz.ch/CRAN/web/packages/ape/refman/ape.html).
- [Bio.Phylo: lectura y manipulación de árboles](https://biopython.org/docs/latest/Tutorial/chapter_phylo.html).
- [Interfaz Kernel de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Kernel.html) y [procesos gaussianos](https://scikit-learn.org/stable/modules/gaussian_process.html).
