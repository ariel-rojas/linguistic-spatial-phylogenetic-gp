# Cómo leer el árbol que utiliza el modelo

> Actualización 21/10/2026: el modelo predeterminado usa correlación filogenética normalizada con ceros entre familias. Las fórmulas con C cruda y tasas por milenio describen la variante histórica; para la actual se usa R por bloques y amplitud marginal. Ver [transformación, configuración y unidades](phylogeny.md#family-covariance). El comando `bayesian` implementa NUTS; ver [flujo bayesiano](bayesian_inference.md).

## Fuente concreta

Usamos el árbol resumen MCC distribuido en la [release v2.0.0](https://github.com/rbouckaert/global-language-tree-pipeline/releases/tag/v2.0.0), con 6636 hojas. No calculamos un MCC nuevo en este proyecto ni elegimos un árbol mirando OV/VO.

El archivo NEXUS comprimido contiene dos declaraciones: `STATE_1` con anotaciones y `STATE_0` sin ellas. Verificamos que al quitar comentarios sus Newick son exactamente iguales, con las mismas longitudes y topología. Se selecciona `STATE_1`; la segunda declaración no es una muestra posterior adicional usada en el GP.

Su SHA-256 es `35578af18d987e58172c88793b33e8ba60626404bdad246610cd94e73146cb46`, coincidente con el digest publicado por GitHub. La raíz y todas las ramas se conservan. La unidad es el milenio y la profundidad raíz-hoja ronda 129.6011726. Las cifras decimales son precisión del archivo, no certeza histórica de la fecha.

## Qué son los 1000 árboles

El estudio distribuye una muestra posterior de 1000 árboles globales. Cada árbol es una hipótesis completa sobre cómo se relacionan las lenguas y cuándo se separan. Comparten el inventario de hojas, pero pueden diferir en agrupaciones y fechas allí donde hay incertidumbre. Son muestras de inferencia MCMC, no 1000 reconstrucciones independientes ni 1000 observaciones lingüísticas; puede existir autocorrelación.

El modelo original combina clasificación Glottolog, filogenias léxicas previas, calibraciones y geografía, con supuestos de diversificación e historia profunda. El árbol no se obtuvo entrenando sobre nuestra etiqueta WALS 83A. El [fundamento metodológico](tree_construction.md) explica su construcción paso a paso y enlaza el manuscrito.

## Qué significa MCC exactamente

MCC significa **Maximum Clade Credibility**, máxima credibilidad de clados. Un clado es un ancestro y todos sus descendientes. Para cada clado, se calcula la proporción de árboles posteriores en que aparece. Si aparece en 850 de 1000 árboles, su soporte estimado es 0.85, condicionado al modelo y la evidencia del estudio.

En la implementación MCC de BEAST revisada, se puntúa cada topología candidata mediante la suma de los logaritmos de los soportes de sus clados internos, equivalente a maximizar su producto:

\[
S(T)=\sum_{c\in\mathrm{clados}(T)}\log\hat p(c),\qquad
T_{MCC}=\operatorname*{argmax}_{T\;\mathrm{muestreado}} S(T).
\]

Ejemplo didáctico con dos clados informativos: soportes (0.95, 0.80) dan producto 0.76; (0.90, 0.70) dan 0.63. Se prefiere el primero bajo ese criterio. No interpretamos ese producto como una probabilidad conjunta exacta del árbol: los clados son dependientes.

El MCC es una topología de la muestra elegida por ese criterio; no es necesariamente el árbol de máxima probabilidad posterior conjunta, ni el consenso de mayoría, ni una media aritmética de topologías. TreeAnnotator puede mantener alturas originales o resumirlas mediante opciones como media/mediana. No reconstruimos esa elección retrospectivamente: usamos las longitudes del archivo publicado.

La página divulgativa de TreeAnnotator describe el criterio de forma simplificada como suma de probabilidades; el [código MCC de BEAST](https://github.com/CompEvol/BeastFX/blob/master/src/beastfx/app/treeannotator/services/MCCTopologyService.java) llama explícitamente a la puntuación logarítmica. [Opciones de TreeAnnotator](https://www.beast2.org/treeannotator/).

## Estructura de un árbol

- **Hojas**: las lenguas identificadas por glottocode.
- **Nodos internos**: ancestros inferidos; no necesitan representar una lengua histórica atestiguada.
- **Raíz**: ancestro de referencia de todo el árbol.
- **Ramas**: conexiones entre un nodo y su descendiente; sus longitudes representan duraciones.
- **Topología**: patrón de agrupación y bifurcación, independientemente de las duraciones.
- **Anotaciones**: soporte posterior, alturas, intervalos y otras cantidades del estudio. No son todas longitudes de rama.

Un ejemplo Newick:

```text
((A:2,B:2):8,C:10);
```

Se lee desde los paréntesis interiores: A y B forman un grupo, con dos unidades desde su ancestro hasta cada hoja. Ese ancestro está conectado a la raíz por ocho unidades. C se conecta directamente a la raíz por diez unidades. El punto y coma termina el árbol. Así, A y B se separaron hace dos milenios en el ejemplo y comparten ocho milenios desde la raíz.

NEXUS envuelve esta estructura con bloques de metadatos. En nuestro archivo, `Translate` relaciona números de hojas con etiquetas como `aari1239_Aari_South_Omotic`. Un fragmento conceptual sería:

```text
Begin trees;
Translate 1 aari1239_Aari_South_Omotic, 2 otra1234_Otra_lengua;
tree ejemplo = (1:2,2:2);
End;
```

Este fragmento es ilustrativo, no una relación real entre esas lenguas. El lector resuelve Translate explícitamente antes de guardar los glottocodes. Las anotaciones `[&posterior=...,height=...]` se preservan en el original y se omiten al calcular las ramas.

## Matriz usada por el GP

Definimos `depth(v)` como la suma de ramas desde la raíz original hasta v. Entonces:

\[
C_{ij}=\mathrm{depth}(\mathrm{MRCA}(i,j)),\qquad C_{ii}=\mathrm{depth}(i).
\]

Para el ejemplo anterior:

\[
C=\begin{pmatrix}10&8&0\\8&10&0\\0&0&10\end{pmatrix}.
\]

Es una covarianza browniana con raíz fija: cambios latentes independientes se acumulan por rama con varianza proporcional a su duración. Las lenguas comparten los cambios anteriores a su separación. No hay varianza adicional anterior a la raíz. C no se normaliza y no se pondera por soportes posteriores de nodos.

La definición equivale a C = A diag(longitudes) Aᵀ para una matriz indicadora de caminos A; ello garantiza semidefinición positiva. El código no usa esa representación como features: suma cada rama al bloque de sus hojas descendientes. Bio.Phylo permite contrastar pares mediante MRCA y profundidad, como hacen las pruebas.

Se almacenan 2185 hojas distintas enlazadas a WALS, con su orden explícito. Esta submatriz conserva el árbol completo y su raíz; no se vuelve a enraizar al quitar hojas. El tamaño de la matriz es aproximadamente 38.2 MB (36.4 MiB). Varias filas WALS pueden compartir hoja; permanecen juntas en los splits.

## Qué se conserva y qué queda fuera

El kernel es `a * Matern + q * C`. La topología y las ramas están fijas; la grilla ajusta q, los parámetros espaciales y nu. El componente filogenético modela propensiones latentes, mientras el clasificador sigue tratando OV/VO/Sin dominante como categorías.

La inferencia del árbol incorpora geografía y mantenemos además el término espacial. Sus pesos no tienen una interpretación causal directa. Tampoco propagamos la incertidumbre entre los 1000 árboles: las probabilidades actuales son condicionales al MCC. [Alternativas estudiadas](phylogeny.md#alternatives).


<a id="family-covariance"></a>

## Correlación normalizada por familias

Desde el 21/10/2026 la configuración predeterminada es `config/spherical_spline_family.json`.
`train` y `search` usan sklearn con Laplace y uno-contra-el-resto. El comando
`bayesian` implementa NUTS y una categórica conjunta: ver [flujo bayesiano](bayesian_inference.md).

### Operación y significado

La matriz fuente C contiene tiempos compartidos en milenios. Se conserva sin cambios
en `datos/phylogeny/covariance.npy`. Al cargarla para el modelo se calcula:

`R[i,j] = C[i,j] / sqrt(C[i,i] * C[j,j])`

Después se pone R[i,j] = 0 si las hojas pertenecen a familias diferentes. La diagonal
queda en uno y las correlaciones dentro de cada familia se conservan. Son ceros
numéricos, no datos faltantes ni `NaN`. Cada bloque es una submatriz principal de
una correlación válida; el bloqueo completo preserva semidefinición positiva.

El kernel final es `a * R_spatial + b * R_phylogenetic`. El parámetro histórico
`phylogenetic_rate` representa ahora b, varianza marginal filogenética, cuando se
activa la normalización. Con matriz cruda sigue siendo tasa por milenio.

El árbol actual tiene altura H = 129.60117258291532 milenios. Para conservar la
amplitud anterior usamos b = q * H: 0.008 pasa a 1.0368093806633225. La grilla
también se convierte multiplicando sus tres tasas por H; conserva 108 candidatos.
Esto aísla el efecto del bloqueo, sin reducir accidentalmente la contribución
filogenética al normalizar. Para un árbol no ultramétrico esta equivalencia con
una sola altura no sería exacta.

### Familias y casos sin clasificación

`familia_id` proviene del ancestro superior de la clasificación LOW enlazada a WALS.
Los alias de una hoja comparten una asignación. Una asignación conocida completa
alias vacíos; dos asignaciones conocidas contradictorias detienen la ejecución.
Cada hoja sin familia recibe un bloque individual, incluidas lenguas aisladas:
no se inventa una familia común de desconocidas. Las filas sin hoja no participan.
Se utiliza la tabla completa de metadatos, nunca las etiquetas de la feature.

El manifiesto de cada ejecución añade `model_covariance`: estructura, normalización,
hash de matriz efectiva, hash de asignaciones familiares y número de hojas sin
familia. Los campos originales siguen describiendo el archivo fuente crudo.

### Configuración y comandos

```json
"phylogenetic_structure": "within_family",
"phylogenetic_normalize": true
```

Para comparar con correlaciones globales cambia solo `within_family` por `global`
y el `name` del experimento. Mantén amplitudes y splits para esa comparación.
Los JSON históricos sin estas opciones siguen utilizando `global` y `false`.
No cambies solo la normalización al comparar amplitudes: revisa sus unidades.

```powershell
uv run gaussian-processes train
uv run gaussian-processes map --fixed
## Búsqueda nueva, independiente de los resultados históricos:
uv run gaussian-processes search
uv run gaussian-processes map
## Reproducir la configuración histórica:
uv run gaussian-processes train --config config/spherical_spline.json
```

La nueva ejecución está en `results/spherical_spline_family/`. `fixed/` contiene
el ajuste con parámetros fijos; la nueva búsqueda de 108 candidatos está pendiente.
`report` sigue comparando los tres experimentos espaciales históricos.

### Verificación del 21/10/2026

Las 32 pruebas pasan. Se ejecutó `train` y se generó `map --fixed` con los datos
reales. Las 2185 hojas tienen diagonal uno; 94 hojas sin familia reciben bloques
individuales (no confundir hojas con filas elegibles). Comparación exploratoria
con el ajuste fijo histórico, mismas amplitudes marginales y test:

| Covarianza | Aciertos / 276 | Log-loss | Macro-F1 |
|---|---:|---:|---:|
| Global cruda histórica | 226 | 0.48653 | 0.56342 |
| Normalizada por familias | 238 | 0.46810 | 0.62733 |

La nueva variante reconoce solo 1/17 casos Sin dominante. No es una búsqueda nueva
ni una validación independiente: el test ya se ha consultado en otras iteraciones.

### Flujo del código

1. `configuration.load_configuration`: valida opciones y selecciona salida.
2. `data.phylogeny.load_phylogeny`: verifica hashes y alineación con hojas.
3. `align_tip_families`: asigna familias o bloques individuales por hoja.
4. `models.phylogenetic_covariance.PhylogeneticCovariance.transform`: normaliza y bloquea una copia.
5. `models.phylogenetic.PhylogeneticKernel`: consulta la matriz por índices.
6. `models.kernels.build_kernel`: suma espacial y filogenético ponderados.
7. Las evaluaciones y el mapa conservan su flujo y registran la transformación.

### Interpretación y límites

Entre familias solo queda covarianza espacial en el kernel. Dentro de cada familia
el árbol puede seguir reflejando geografía; el bloqueo no elimina esa influencia.
Reservar una familia completa impide transferirle efectos filogenéticos desde el
train. Esto cambia la pregunta predictiva y no equivale a imputar dentro de una
familia conocida. Los bloques conservan el tiempo compartido desde la raíz global;
no se vuelve a enraizar cada familia.


<a id="alternatives"></a>

## Alternativas de investigación

Documento de investigación. La base implementada es la covarianza browniana del MCC, con opciones de normalización y bloqueo por familias. Las restantes alternativas de esta sección son investigación y no activan kernels adicionales.

Sea C la matriz de tiempos compartidos, D la distancia de camino entre hojas, t_i su profundidad y s_ij la profundidad de su ancestro común. Todas las alternativas conservarían el término espacial por separado, pero cambian los supuestos de la componente filogenética.

### 1. Browniano de tiempo compartido: referencia actual

K_phy = q C. Los cambios latentes se acumulan por rama; no existe retorno hacia una media. q determina la tasa de acumulación de varianza. Se conserva la raíz fija.

Ventajas: un único peso, interpretación directa, duraciones originales y semidefinición positiva. Limitaciones: historia profunda con influencia persistente y una sola tasa para todo el árbol. La difusión es de la propensión latente, no una transición literal OV→VO.

La construcción estándar está documentada en [ape: vcv.phylo](https://github.com/emmanuelparadis/ape/blob/master/R/vcv.phylo.R).

### 2. Exponencial de distancia / OU estacionario

\[
K_{ij}=v\exp(-\alpha D_{ij}),\qquad v>0,\quad\alpha>0.
\]

D es la suma de ramas al recorrer el camino entre hojas. La dependencia decae con la separación; `1/alpha` tiene unidades temporales. Puede interpretarse como un proceso Ornstein–Uhlenbeck homogéneo con raíz en su distribución estacionaria. v es su varianza estacionaria, no la tasa de difusión browniana. Introduce una escala temporal de pérdida de memoria.

Es una covarianza válida sobre un árbol con ramas no negativas. No se debe sustituirla por un Matérn arbitrario sobre D y asumir automáticamente la misma garantía: la validez depende de la distancia y del kernel elegidos. La formulación exponencial aparece como `corMartins` en [ape](https://stat.ethz.ch/CRAN/web/packages/ape/refman/ape.html).

La elección de raíz importa. Con raíz fija, el OU tiene otra covarianza:

\[
K_{ij}=\frac{\sigma^2}{2\alpha}
\exp[-\alpha(t_i+t_j-2s_{ij})]
(1-\exp[-2\alpha s_{ij}]).
\]

Al tender alpha a cero, se recupera sigma² s_ij, la covarianza browniana actual. El OU estacionario simple no tiene ese mismo límite manteniendo v fijo. Es una derivación del proceso sobre ramas; especificar la condición inicial evita mezclar modelos distintos.

Coste científico: un parámetro temporal adicional y posible confusión con la escala espacial. Sería la primera alternativa razonable si queremos estudiar cuánto persiste la dependencia filogenética con el tiempo.

### 3. Señal filogenética de Pagel lambda

\[
K_{phy}=q[\lambda C+(1-\lambda)\operatorname{diag}(C)],\qquad 0\leq\lambda\leq1.
\]

Con lambda=1 recuperamos la referencia; con lambda=0 solo queda variación independiente entre hojas. La diagonal se conserva. Esta combinación es semidefinida positiva y permite estudiar cuánto acoplamiento entre linajes favorecen las observaciones.

El término independiente se refiere a hojas del árbol; dos registros asociados a una misma hoja no se convierten en dos linajes nuevos. No es ruido de medición de la etiqueta. Lambda no estima qué porcentaje del rasgo se debe causalmente a la herencia. [Referencia corPagel](https://stat.ethz.ch/CRAN/web/packages/ape/refman/ape.html).

### 4. Tasas distintas por grupos de ramas

Asignar una tasa no negativa q_e a cada rama:

\[
K_{phy}=A\operatorname{diag}(q_e l_e)A^T.
\]

Las longitudes l_e se conservan; cambia la cantidad de varianza acumulada por unidad temporal en cada región del árbol. Pueden distinguirse, por ejemplo, ramas profundas e internas a familias. La fórmula garantiza semidefinición positiva si las tasas no son negativas.

Permite heterogeneidad evolutiva, pero una tasa por rama sería excesiva para nuestros datos. Serían necesarios pocos grupos con interpretación y regularización. No forma parte de la implementación actual.

### 5. Usar varios árboles

Hay dos operaciones diferentes:

**Promediar covarianzas:** preparar C_1,...,C_S sobre exactamente las mismas hojas y unidades, y usar C_promedio = sum(w_s C_s), con pesos no negativos que suman uno. Sigue siendo una covarianza válida. Produce un único GP cuya covarianza resume varios árboles.

**Promediar predicciones:** ajustar un modelo condicionado a cada árbol y promediar sus probabilidades. Esto conserva diferencias entre las predicciones de los árboles, pero multiplica el coste de ajuste. No equivale a un GP con la covarianza promedio: la clasificación es no lineal y una mezcla de normales no es, en general, una sola normal.

Usar pesos iguales de la muestra filogenética externa es un análisis modular que mantiene la distribución del árbol del estudio original. Una inferencia bayesiana conjunta permitiría que las etiquetas WALS actualizaran los pesos de los árboles mediante p(y|T), integrando también parámetros. No basta llamar bayesiano a cualquier promedio.

Si se hace selección de hiperparámetros, todo ajuste y selección debe permanecer dentro de train; ningún árbol se elige por rendimiento en test. Una exploración futura de 20–50 árboles sería una prueba de sensibilidad manejable, antes de los 1000. [Distribución publicada](https://github.com/rbouckaert/global-language-tree-pipeline/releases/tag/v2.0.0).

### 6. Cambiar la representación computacional

Un browniano sobre el árbol también puede representarse mediante incrementos independientes por rama o mediante una matriz de precisión de los nodos. Al marginalizar correctamente los nodos internos y fijar la raíz, se recupera la misma covarianza entre hojas. Esto puede ayudar a escalar, pero no es una hipótesis evolutiva nueva.

Un Laplaciano o una inversa elegidos sin especificar raíz, condiciones de contorno y marginalización no garantizan equivalencia. Para el tamaño actual, la matriz explícita es más fácil de inspeccionar.

### Fuera del kernel GP

Otra línea sería un proceso de Markov de estados discretos OV, VO y Sin dominante sobre ramas. Una matriz de tasas de transición definiría probabilidades mediante exp(Qt). Modelaría cambios de estado directamente, pero exige otra verosimilitud y tratamiento de ancestros: no es un reemplazo pequeño del kernel de nuestro GPC. Se deja fuera para mantener sencilla esta etapa.

### Orden sugerido de exploración

Primero analizar la referencia MCC y sus errores por clase/familia. Después probar sensibilidad a otros árboles con idénticos splits. Si la pregunta es la memoria temporal, comparar OU; si es la intensidad del acoplamiento, estudiar lambda. Mantener un protocolo de evaluación común y no añadir simultáneamente todas estas dimensiones a la grilla.
