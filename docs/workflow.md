# Cómo pasa el modelo de las lenguas a las probabilidades

Esta guía acompaña al programa y a la página interactiva. Explica qué matriz se
construye, por qué multiplicamos por factores de matrices y qué puede significar
comparar la parte espacial con la genealógica.

## 1. De una covarianza al producto con variables normales

Supongamos primero un vector de efectos latentes \(f\), uno por lengua. Queremos
que dos lenguas relacionadas tengan efectos parecidos. Especificamos una matriz de
covarianza \(K\), donde \(K_{ij}\) mide la semejanza esperada entre los efectos de
las lenguas \(i\) y \(j\):

\[
\mathbf f\sim\mathcal N(\mathbf 0, K).
\]

Una forma práctica de generar exactamente esa normal multivariada es comenzar con
ruido independiente \(\mathbf z\sim\mathcal N(\mathbf 0,I)\) y transformarlo:

\[
\mathbf f=L\mathbf z,\qquad LL^\top=K.
\]

La covarianza resultante es

\[
\operatorname{Cov}(L\mathbf z)
=L\operatorname{Cov}(\mathbf z)L^\top
=LIL^\top=LL^\top=K.
\]

Por eso aparece el producto: multiplicar ruido blanco por una raíz matricial
impone la estructura de semejanza deseada. No es una aproximación intuitiva; es
una manera algebraica de construir una variable normal con covarianza \(K\).

## 2. Por qué hay dos efectos y varias matrices

En este modelo separamos las dos fuentes de semejanza:

\[
K=\sigma_s^2R_s(\lambda)+\sigma_p^2R_p.
\]

\(R_s\) es la correlación espacial, \(R_p\) la correlación filogenética,
\(\sigma_s,\sigma_p\) son amplitudes no negativas y \(\lambda\) controla el
suavizado espacial. Los dos procesos llevan coeficientes independientes. Como
suma de variables normales independientes, su covarianza total es la suma de
las dos covarianzas.

### Superficie espacial

La base \(\Phi\) contiene armónicos esféricos evaluados en cada ubicación. Cada
armónico representa un patrón espacial de cierta escala. Su grado \(\ell_j\)
indica aproximadamente cuán fino es: grados mayores permiten cambios más
locales. Aplicamos pesos dependientes del suavizado:

\[
w_j(\lambda)=\frac{1}{1+\lambda[\ell_j(\ell_j+1)]^2},
\qquad
d_j=\sqrt{\frac{w_j}{\sum_r w_r}},
\qquad D=\operatorname{diag}(d_1,\ldots,d_m).
\]

Después generamos

\[
\mathbf g=\sigma_s\Phi D\mathbf z_s,
\qquad \mathbf z_s\sim\mathcal N(0,I).
\]

Su covarianza es

\[
\operatorname{Cov}(\mathbf g)
=\sigma_s^2\Phi D I D^\top\Phi^\top
=\sigma_s^2\Phi D^2\Phi^\top
=\sigma_s^2R_s(\lambda).
\]

El producto ordena las operaciones: ruido independiente → ajustar cada escala
con \(D\) → evaluar y sumar patrones sobre el mapa con \(\Phi\) → escalar la
magnitud total con \(\sigma_s\). El resultado tiene un valor por lengua.
Al ser una base finita, este componente tiene rango finito y se evalúa en
ubicaciones nuevas usando sus coordenadas.

### Componente filogenética

El árbol produce una matriz \(R_p\) según ramas compartidas, normalizada y con
ceros entre familias en la configuración actual. Si \(R_p=LL^\top\), entonces

\[
\mathbf h=\sigma_pL\mathbf z_p,
\qquad \mathbf z_p\sim\mathcal N(0,I),
\]

y por el mismo cálculo \(\operatorname{Cov}(\mathbf h)=\sigma_p^2R_p\).
Si dos lenguas comparten mucha historia evolutiva, sus filas de \(L\) producen
efectos parecidos. Un cero entre familias significa que esta componente no las
correlaciona entre sí; la componente espacial todavía puede hacerlo.

Finalmente:

\[
\mathbf F=\mathbf g+\mathbf h,
\qquad \mathbf u=\boldsymbol\alpha+\mathbf F.
\]

El código utiliza dos columnas en cada matriz (los dos contrastes entre tres
clases), de modo que \(Z_s,Z_p\in\mathbb R^{m\times2}\). Las amplitudes y el
suavizado son escalares globales por ajuste. No se ajusta un parámetro espacial
o filogenético distinto para cada lengua.

## 3. De los efectos latentes a OV, Sin dominante y VO

Los dos contrastes \(u_1,u_2\) se convierten en tres puntuaciones:

\[
\eta_{OV}=\tfrac{\sqrt3}{2}u_1+\tfrac12u_2,\quad
\eta_{SD}=-\tfrac{\sqrt3}{2}u_1+\tfrac12u_2,\quad
\eta_{VO}=-u_2.
\]

La *softmax* las convierte a probabilidades positivas que suman uno:

\[
\pi_{ic}=\frac{\exp(\eta_{ic})}{\sum_{r\in\{OV,SD,VO\}}\exp(\eta_{ir})},
\qquad y_i\sim\operatorname{Categorical}(\boldsymbol\pi_i).
\]

La verosimilitud multiplica la probabilidad que el modelo asigna a cada etiqueta
observada. NUTS explora la posterior conjunta de amplitudes, suavizado,
interceptos y coeficientes latentes. Cada muestra se transforma en probabilidades;
su promedio es la predicción final, y su variación muestra incertidumbre.

## 4. ¿El efecto espacial y el filogenético tienen «pesos» por lengua?

Actualmente no estimamos un porcentaje de importancia local. \(\sigma_s\) y
\(\sigma_p\) son globales al ajuste. Sin embargo, el cambio que los efectos
producen puede variar por lengua porque cambian \(\Phi\), las relaciones del
árbol, y la distribución posterior de los campos latentes. Además, espacio y
parentesco pueden explicar patrones similares; las amplitudes pueden compensarse.

La página calcula una sensibilidad descriptiva para cada caso. Para cada muestra
posterior construye la probabilidad completa \(p\), otra omitiendo \(g\), y otra
omitiendo \(h\), y calcula:

\[
\operatorname{TV}(p,q)=\tfrac12\sum_c|p_c-q_c|.
\]

TV vale entre 0 y 1; en la página se expresa como porcentaje del cambio total de
probabilidad entre clases. Un cambio grande al omitir espacio indica que esa
predicción es sensible al término espacial dentro del modelo. No es una
fracción de varianza, un coeficiente estimado, ni un efecto causal. Ambos
cambios pueden ser grandes a la vez, por lo que no los forzamos a sumar 100 %.
También pueden cambiar ante otros supuestos, datos o parámetros.

## 5. ¿Qué pasa al analizar muchas features?

El análisis actual de una feature ajusta sus propias etiquetas y su posterior.
Al repetirlo independientemente para otras features, cada ajuste puede aprender
amplitudes espaciales y filogenéticas distintas a partir de sus datos. Eso no
significa que cada lengua tenga un «peso» distinto; el reparto sigue siendo
global dentro de cada ajuste. Con pocos datos por feature, la estimación puede
ser inestable.

Una extensión multifeature tiene varias opciones:

1. **Compartir únicamente el rango espacial:** todas las features informan
   sobre la escala geográfica, y cada una mantiene sus propios campos latentes.
2. **Amplitudes parcialmente agrupadas:** cada feature tiene su amplitud, pero
   se contrae hacia un promedio común aprendido con todas.
3. **Modelo multivariado:** correlaciona features que tienden a aparecer juntas,
   además de los efectos espaciales y filogenéticos.

La segunda o tercera opción permitiría aprendizaje compartido sin obligar a que
todas las features respondan igual. Una variación local de los pesos espaciales
sería aún más flexible, pero requeriría covariables y datos suficientes para
identificar superficies de peso; no la asumiría como primer paso.

## 6. Qué explorar para mejorar «Sin dominante»

La categoría minoritaria tiene menos ejemplos y el split aleatorio conserva su
proporción aproximada. Conviene revisar primero los desaciertos y su definición
de origen. Luego se pueden comparar, dentro de validación y reportando todas las
clases: priors de intercepto o prevalencia más informativos; ponderación de
clases o verosimilitud con pesos (cambia la interpretación de las probabilidades);
una estructura ordinal/jerárquica si la lingüística justifica que Sin dominante
está relacionada con variación OV/VO; umbrales de decisión para priorizar recall;
y validación repetida estratificada por identidad y familias.

Duplicar casos o usar SMOTE en los efectos latentes no sería la primera opción:
puede crear evidencia artificial y romper la dependencia del árbol. Una mayor
recuperación de la clase se debe valorar junto con falsos positivos, recall,
macro-F1 y calibración.

## 7. Referencias que guían las extensiones

- Guzmán Naranjo y Becker (2022), [Statistical bias control in typology](https://doi.org/10.1515/lingty-2021-0002).
- Guzmán Naranjo y Mertner (2023), [Estimating areal effects in typology](https://doi.org/10.1515/lingty-2022-0037).
- Guzmán Naranjo, Mertner y Urban (2024), [Spatial effects with missing data](https://doi.org/10.1515/opli-2024-0032).
- McNew, Derungs y Moran (2018), [Towards faithfully visualizing global linguistic diversity](https://lrec.elra.info/lrec2018-main-129).
- Mertner (2025), [Spatial models of linguistic data in Africa and beyond](https://publikationen.uni-tuebingen.de/xmlui/handle/10900/173835).

Estos trabajos usan modelos y objetivos distintos al clasificador categórico de
este repositorio; sus métodos inspiran alternativas, no son idénticos a nuestro
código.
