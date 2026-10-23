# Suavizado espacial, distancias y cartografía

> Actualización 21/10/2026: el modelo predeterminado usa correlación filogenética normalizada con ceros entre familias. Las fórmulas con C cruda y tasas por milenio describen la variante histórica; para la actual se usa R por bloques y amplitud marginal. Ver [transformación, configuración y unidades](phylogeny.md#family-covariance). El comando `bayesian` implementa NUTS; ver [flujo bayesiano](bayesian_inference.md).

## Modelo compartido

Los tres experimentos utilizan `K = a R_espacial + q C_MCC`. Se conserva exactamente
la matriz filogenética, su orden de hojas y los tiempos originales en milenios.
Solo cambia R. Todas las R tienen diagonal uno; por tanto a es varianza espacial
y q sigue siendo varianza filogenética por milenio. El clasificador es el mismo
`GaussianProcessClassifier` de sklearn, con Laplace y tres modelos uno-contra-resto.

## Suavizado esférico tipo GAM (predeterminado)

Representamos la superficie latente como una suma de armónicos esféricos reales:
`f(s) = sum(beta_lm * Y_lm(s))`, para grados l entre 0 y L y todos sus órdenes m.
Los grados bajos permiten patrones amplios; los altos agregan detalle. Hay
`(L+1)^2` funciones. Se penalizan tanto la norma como la rugosidad:

```text
penalización = sum_lm [1 + smoothing * (l(l+1))²] * beta_lm²
varianza previa de beta_lm ∝ 1 / [1 + smoothing * (l(l+1))²]
```

La norma adicional hace propio el prior, incluso para la función constante l=0.
No se invierte una penalización singular ni se deja un intercepto impropio.
Este es un suavizado esférico regularizado inspirado en GAM, no una reproducción
exacta del spline predeterminado de mgcv ni un GAM multinomial independiente.

El teorema de adición permite evaluar la covarianza sin almacenar la base:

```text
w_l = (2l+1) / [1 + smoothing * (l(l+1))²]
R(s,t) = sum_l w_l P_l(cos(theta(s,t))) / sum_l w_l
```

P_l es el polinomio de Legendre. Todos los pesos espectrales son positivos,
por lo que la matriz es semidefinida positiva. La truncación es explícita y puede
limitar el detalle: L=8,16,32 ofrece 81,289,1089 funciones respectivamente.
`smoothing` modifica pesos relativos; `spatial_variance` escala toda la covarianza.
No son dos nombres para la misma amplitud. Más suavizado favorece grados bajos,
pero no equivale a un alcance expresable directamente en kilómetros.

Se incluyen todos los órdenes de cada grado para conservar invariancia ante
rotaciones. La constante es aleatoria y penalizada; no es un intercepto libre.
La correlación puede ser negativa a ciertas distancias y no representa una
probabilidad de contacto. El GP denso de sklearn no aprovecha automáticamente el
rango reducido para acelerar la inferencia; no se promete una mejora de velocidad.

## Exponencial geodésico

`R(s,t) = exp(-d_superficie / length_scale_km)`. Calculamos arcos sobre una esfera
de radio 6371 km a partir de xyz normalizados, con una expresión atan2 estable
en coincidencias y antípodas. No hay discontinuidad especial en ±180°.

Es Matérn nu=1/2 sobre distancia de círculo máximo. No admite nu=1.5 o 2.5:
sustituir directamente distancia euclídea por geodésica en esos Matérn no define
un kernel válido globalmente en la esfera. Tampoco representa distancia de
viaje, recorrido sobre el relieve ni una geodésica elipsoidal.

## Referencia histórica

`config/multiclass.json` conserva el Matérn nativo de sklearn sobre distancia de
cuerda xyz, con nu ajustable. Omitir `spatial_kernel` significa `matern_chord`,
para reproducir configuraciones y resultados históricos sin reinterpretarlos.

## Configurar y comparar

| Configuración | Parámetros espaciales | Candidatos |
|---|---|---:|
| spherical_spline.json | max_degree, smoothing, spatial_variance | 108 |
| geodesic_exponential.json | length_scale_km, spatial_variance | 108 |
| multiclass.json | length_scale_km, nu, spatial_variance | 108 |

Las tres grillas incluyen `phylogenetic_rate`. El entrenamiento fijo utiliza a=1
y q=0.008 para comparar estrategias con amplitudes constantes. La búsqueda permite
reajustar esos pesos en train. Las bases y la geometría no se aprenden con etiquetas.

```powershell
uv run gaussian-processes train
uv run gaussian-processes search
uv run gaussian-processes map
uv run gaussian-processes search --config config/geodesic_exponential.json
uv run gaussian-processes map --config config/geodesic_exponential.json
uv run gaussian-processes evaluate-spatial --block-km 2000 --buffer-km 250
uv run gaussian-processes evaluate-families
```

Los comandos de modelado y mapa aceptan `--config`. Para estudiar
una sola familia se utiliza `train` y el campo `test_family`. `evaluate-families`
recorre todas. Ambos usan valores principales fijos, no el ganador aleatorio.

## Evaluación espacial

`evaluate-spatial` construye celdas cúbicas en xyz e intersecta ese cubrimiento con
la esfera. El tamaño es aproximado: no son regiones de igual área. Las celdas que
comparten una identidad lingüística se unen transitivamente. GroupKFold reparte
bloques completos en tres folds; cada lengua se evalúa una vez. Un fold puede
contener varios bloques separados, no es necesariamente una única región continua.

Después se retiran de train las identidades con algún punto a menos de 250 km
de cualquier punto test. El margen usa arcos superficiales. No se fuerza
estratificación por clases, porque la separación geográfica tiene prioridad;
train debe conservar las tres clases. Se guardan IDs retirados y bloques de test.

Se utilizan parámetros fijos, no parámetros seleccionados con las etiquetas
reservadas. Para seleccionarlos bajo este protocolo haría falta CV espacial
anidada. El experimento es exploratorio: ya se ha consultado el test aleatorio
en iteraciones anteriores. No se amplían grillas a partir de esos resultados.

Para reconstruir la tabla comparativa, ejecuta `uv run gaussian-processes report`
después de completar `search`, `train` y `evaluate-spatial` para las tres
configuraciones. El reporte verifica datos, árbol y test compartidos. No hace falta
repetir el `search` histórico si sus artefactos siguen siendo válidos. El mapa se
genera aparte con `map`; el informe queda en `results/comparison.md`.

## Cartografía física

`uv run gaussian-processes prepare-geography` descarga y reutiliza:

- Natural Earth 5.1.2: tierra y océano, escala 1:10 millones (no diez metros).
- GMBA v2 Standard Basic: regiones montañosas y nombres.

Los polígonos completos clasifican las coordenadas WALS. La copia del mapa se
simplifica a 0.03 grados solo para dibujo. El manifiesto registra fuentes,
licencias, hashes y el hash de lenguas.csv. Los atributos no cambian elegibilidad,
etiquetas, coordenadas ni covarianzas. Una coordenada marítima se marca para
revisión; puede reflejar redondeo, costas generalizadas o islas pequeñas.

No pertenecer a una región GMBA no demuestra terreno llano. El contexto es moderno
y corresponde al punto WALS, no a toda la distribución histórica de la lengua.
No se ha incorporado DEM, rugosidad, rutas ni penalizaciones por barreras.

`datos/geography/sources/` conserva los tres originales. `languages.csv` reúne
atributos por ID, `context.geojson` las geometrías ligeras y `manifest.json` la
procedencia. No se generan copias por candidato. El mapa funciona también sin
estas capas si todavía no se han preparado.

## Fuentes

- [Relación suavizado / efectos gaussianos en mgcv](https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/gam.vcomp.html).
- [Splines sobre la esfera](https://stat.ethz.ch/R-manual/R-devel/library/mgcv/html/smooth.construct.sos.smooth.spec.html).
- [Gneiting: covarianzas esféricas, expansión espectral y restricciones Matérn](https://arxiv.org/pdf/1111.7077).
- [Natural Earth](https://www.naturalearthdata.com/downloads/10m-physical-vectors/), dominio público.
- [Snethlage et al. 2022](https://doi.org/10.1038/s41597-022-01256-y) y [datos GMBA v2](https://doi.org/10.48601/earthenv-t9k2-1407), CC BY 4.0.
