# Repaso del GP, efectos de grupo y lectura del manuscrito

> Actualización 21/10/2026: el modelo predeterminado usa correlación filogenética normalizada con ceros entre familias. Las fórmulas con C cruda y tasas por milenio describen la variante histórica; para la actual se usa R por bloques y amplitud marginal. Ver [transformación, configuración y unidades](phylogeny.md#family-covariance). El comando `bayesian` implementa NUTS; ver [flujo bayesiano](bayesian_inference.md).

Revisión del código y artefactos: 20 de octubre de 2026. Esta nota explica lo implementado y separa alternativas sin implementar. El proyecto de multilingüismo es independiente.

## 1. Datos y resultados reales

WALS 83A aporta 2694 registros, de los cuales 1378 observados son elegibles: 644 OV, 652 VO y 82 Sin dominante. Se imputan 940 faltantes elegibles. Hay 376 registros excluidos del modelo (incluyen observados y faltantes); los 32 sin coordenadas permanecen en tablas, pero no pueden dibujarse. LOW audita identidades y familias. La matriz filogenética corresponde al MCC v2.0.0: 6636 hojas en el árbol completo y 2185 hojas distintas enlazadas a WALS en la matriz. Ni LOW ni el árbol crean por sí solos nuevas filas de observación.

El test externo contiene 276 registros, el train 1102. Las identidades se mantienen juntas y las clases se estratifican aproximadamente. Cada búsqueda completa compara 108 candidatos y tres folds dentro de train; selecciona por log-loss, reajusta sobre train y evalúa test. Luego ajusta otro modelo con los 1378 observados para imputar los 940 faltantes.

| Búsqueda guardada | Aciertos test | Macro-F1 | Log-loss | Recall Sin dominante |
|---|---:|---:|---:|---:|
| Spline esférico | 236/276 | 0,6217 | 0,4583 | 1/17 |
| Matérn sobre cuerda | 233/276 | 0,6403 | 0,4472 | 2/17 |
| Exponencial geodésico | Pendiente | — | — | — |

Las tres estrategias tienen evaluación fija y espacial. El spline fijo obtiene 1044/1378 aciertos fuera de bloques espaciales (75,76%) y reconoce 1/82 Sin dominante; no es la misma población de test ni los mismos parámetros que su búsqueda. La evaluación por las 184 familias está ejecutada solo para Matérn fijo: 978/1302 aciertos (75,12%); se excluyen 76 registros sin familia. No confundir rendimiento por lengua con media por familia.

El test ya se consultó en etapas anteriores; estas comparaciones son exploratorias. El principal límite actual es el bajo recall de la clase minoritaria, no la falta de un decimal adicional de accuracy. Los pesos ganadores a=4 y q=0,032 están en el borde superior de sus grillas: no prueban un óptimo global.

## 2. El proceso gaussiano paso a paso

Para cada clase c se ajusta un GP binario uno-contra-el-resto. Sea z_ic=1 si la observación i tiene esa clase. Introducimos una propensión latente f_c por lengua:

f_c ~ Normal(0,K),
K_ij = a R(s_i,s_j) + q C_ij.

R es correlación espacial de diagonal uno; C es duración compartida desde la raíz hasta el ancestro común, en milenios. El índice de hoja solo consulta C: no es una magnitud ordinal ni una distancia. a es varianza espacial, q varianza filogenética por milenio. K_ii=a+q C_ii; por tanto no comparar a y q directamente como porcentajes.

La observación es Bernoulli con enlace logístico: P(z_ic=1|f_ic)=1/(1+exp(-f_ic)). La posterior es proporcional al producto de esas verosimilitudes y el prior gaussiano. No es gaussiana: sklearn aproxima alrededor del modo mediante Laplace. Se integra aproximadamente la sigmoide bajo la distribución predictiva latente; no basta aplicar sigmoide a su media. Las tres probabilidades binarias se normalizan para sumar uno y se elige argmax. No hay una softmax conjunta ni regresión sobre números arbitrarios 0,1,2.

En los comandos sklearn `train` y `search`, los hiperparámetros están fijos en cada ajuste (`optimizer=None`); `search` los elige por CV. En esos comandos no se muestrean sus posteriores. El comando `bayesian` sí integra hiperparámetros continuos con NUTS; ninguno integra incertidumbre sobre árboles. La inferencia de funciones sí tiene prior y posterior aproximada; decir que no hay nada bayesiano sería incorrecto.

### Espacio

Predeterminado: spline esférico regularizado, f(s)=sum_lm beta_lm Y_lm(s). Se usan todos los órdenes de cada grado. La varianza de los coeficientes es proporcional a 1/[1+lambda(l(l+1))²]; se penaliza norma y rugosidad. El teorema de adición permite evaluarlo mediante Legendre:

R(s,t)=sum_l w_l P_l(cos theta)/sum_l w_l,
w_l=(2l+1)/[1+lambda(l(l+1))²].

El ganador usa grado 16 (289 funciones), smoothing=0,0001, a=4 y q=0,032. Es un suavizado tipo GAM expresado como kernel, no un GAM multinomial separado. La constante de grado cero es aleatoria y regularizada; no hay intercepto fijo libre.

Alternativas: exp(-distancia superficial/escala), equivalente a Matérn nu=1/2 sobre la esfera; o Matérn sklearn sobre distancia de cuerda, donde sí se compara nu=0,5/1,5/2,5. El ganador Matérn usa nu=1,5 y escala 1500 km. No atribuir ese nu al spline.

### Filogenia

Para cada rama e de longitud t_e se puede imaginar un incremento independiente delta_ec ~ Normal(0,q*t_e). El efecto de la lengua es la suma de incrementos en su camino raíz-hoja. Dos lenguas comparten los incrementos de sus ramas comunes; por eso Cov(h_ic,h_jc)=q*C_ij. Se fija el estado de raíz a cero en esta representación; no se añade una varianza de raíz independiente.

Esta equivalencia explica por qué el kernel filogenético ya es un modelo de efectos aleatorios estructurados. Condicionar en datos relaciona las componentes a posteriori, aunque sus priors se hayan definido independientes. El código ajusta la suma, no guarda una descomposición posterior de efectos por rama o por lengua.

## 3. Tres significados diferentes de grupo

1. **Identidad en validación:** `unidad` impide que filas de la misma lengua crucen train/test. No añade parámetros al modelo.
2. **Familia como evaluación:** reservar una familia cambia qué datos se usan para entrenar. Tampoco añade un efecto de familia.
3. **Efecto aleatorio:** varias observaciones comparten una variable latente o tienen variables latentes correlacionadas. Esto sí cambia el modelo: actualmente ocurre por C y R.

No tenemos un término categórico adicional por familia, macroárea, país o fuente de documentación. No modelamos pendientes específicas por familia ni dispersión residual por familia. Natural Earth y GMBA son contexto cartográfico, no covariables aprendidas ni barreras.

## 4. Formas concretas de modelar efectos de grupo

### Intercepto compartido por familia

u_gc ~ Normal(0,tau_c²), f_ic = f_espacial,ic + u_familia(i),c.

Ejemplo hipotético: las lenguas de una familia comparten una desviación hacia OV. No significa que todas deban ser OV. Sus efectos se regularizan hacia la distribución global: con pocos datos, hay mayor dependencia de esa distribución. En forma de kernel: K_familia(i,j)=tau²*1[familia_i=familia_j]. Es una forma especial de GP, con matriz Z G Z^T.

Para una familia nueva no se reutiliza el efecto de otra: el prior tiene media cero y varianza tau². La incertidumbre permanece aunque no haya desplazamiento aprendido. En contraste, el árbol puede conectar la familia reservada con ramas y clados más amplios.

### Efecto filogenético correlacionado

h_c ~ Normal(0,q C). Ya implementado. Dos lenguas de la misma familia no reciben obligatoriamente la misma semejanza: depende de sus ramas compartidas. Una familia categórica supone igual covarianza para todos sus miembros, lo cual es una simplificación mayor.

### Efectos cruzados

f_ic = alpha_c + u_familia(i),c + v_area(i),c.

Familia y área no están necesariamente anidadas: una familia puede ocupar varias regiones y una región puede contener varias familias. Nuestro K usa una versión continua del espacio y una versión estructurada de la genealogía. Añadir además familia puede duplicar señal y hacer poco identificables las varianzas; no se justifica solo por llamarlo control.

### Pendiente variable

f_ic = alpha_c + u_gc + (beta_c+b_gc)*x_i.

Permite que una asociación con un predictor x cambie de grupo. Requiere un predictor medido y variación suficiente dentro de los grupos; no es sinónimo de dar a cada familia un peso arbitrario. En nuestro dataset actual no hay un predictor sociolingüístico de esta clase incorporado.

### Efectos en dispersión

En una respuesta continua puede modelarse mu_i=alpha+u_g y log sigma_i=gamma+v_g. El manuscrito adjunto utiliza un esquema de este tipo. En una Bernoulli la varianza condicional p(1-p) está ligada a la media: no se puede agregar sin más el sigma libre de una normal. Se puede investigar ruido de etiquetas o heterogeneidad latente, pero es otro modelo.

## 5. Antecedentes concretos con Blasi

- Blasi et al. (2019), *Human sound systems are shaped by post-Neolithic changes in bite configuration*: regresión bayesiana Poisson de cantidad de labiodentales, con interceptos y pendientes por familia y área. La respuesta metodológica de 2020 explica controles y heterogeneidad. Es un antecedente de efectos categóricos cruzados, no una estimación de OV/VO. [Artículo](https://doi.org/10.1126/science.aav3218), [respuesta de los autores](https://doi.org/10.1101/2020.02.27.965400).
- Shcherbakova et al., con Blasi (2023, versión corregida), *Societies of strangers do not speak less complex languages*: efectos espaciales y filogenéticos correlacionados, más alternativas de áreas categóricas. El reanálisis usa R-INLA y respuesta gaussiana; estandariza matrices, a diferencia de nuestra C en tiempos originales. Es el antecedente más próximo a la arquitectura aditiva actual. [Artículo](https://pmc.ncbi.nlm.nih.gov/articles/PMC10431698/), [código](https://github.com/OlenaShcherbakova/Sociodemographic_factors_complexity/blob/main/models_boundness_reanalysis_phylogenetic_spatial.R). La [corrección](https://pmc.ncbi.nlm.nih.gov/articles/PMC12697922/) documenta problemas de faltantes codificados como cero; se debe consultar la versión corregida.

No son las únicas maneras utilizadas por Blasi ni una revisión exhaustiva de su obra. El término group effect no identifica un algoritmo único: importa qué comparten las observaciones y qué distribución se supone.

## 6. Manuscrito: The bandwidth of social transmission through language

Lectura del borrador local de Blasi, Cristia y colaboradores fechado el 16/10/2026. Se conserva fuera del repositorio, no se publica ni modifica. Tiene 20 páginas, comentarios editoriales y secciones incompletas.

### Resumen breve

Busca estimar cuánta exposición al habla rodea a los niños durante el desarrollo, usando grabaciones naturalistas de EL1000 y varios algoritmos. La muestra se describe como 19 comunidades/experimentos y al menos 10 lenguas. Ajusta minutos de habla por hora y extrapola hasta seis años usando horas de vigilia; discute comparaciones con adquisición y entrenamiento de modelos de lenguaje. Las cifras de palabras son conversiones aproximadas, no transcripciones contadas ni prueba causal de suficiencia del input.

### Lo explícito en el PDF

- Se agrega cada niño dentro de cada experimento, promediando sesiones con pesos proporcionales a su duración (p. 10).
- La respuesta está entre 0 y 60 minutos/hora y se modela con una normal doblemente truncada (p. 4).
- `experiment` tiene efecto sobre localización y dispersión. Las tablas 3–5 incluyen su correlación (pp. 11–12).
- Se informa HMC: cuatro cadenas, 1000 iteraciones de calentamiento y 5000 posteriores por cadena. El texto afirma convergencia, pero esas tablas no muestran R-hat, tamaño efectivo ni divergencias.
- Para extrapolar se simulan nuevos niveles de grupo y se mantienen fijos de 0 a 6 años (p. 5).

### Reconstrucción matemática compatible, no código verificado

Para niño i del experimento g:

y_ig sigue una normal de parámetros mu_g y sigma_g² truncada al intervalo de 0 a 60,
mu_g=alpha+u_g,
log(sigma_g)=gamma+v_g,
(u_g,v_g) ~ Normal_2(0,Sigma),
Sigma=[[tau_mu²,rho*tau_mu*tau_sigma],[rho*tau_mu*tau_sigma,tau_sigma²]].

El enlace log para sigma es compatible con los nombres de las tablas y la parametrización habitual de brms, pero el PDF no lo declara explícitamente. Tampoco identifica inequívocamente el software o los priors. No afirmar que se verificó el script. La correlación sí aparece en las tablas: los efectos de localización y escala no deben simularse independientemente si se quiere preservar ese modelo.

La densidad de observación incluye normalización dependiente de los parámetros:

p(y|mu,sigma) = phi((y-mu)/sigma) / {sigma*[Phi((60-mu)/sigma)-Phi(-mu/sigma)]}, para 0<=y<=60.

No es una normal recortada después de generarla: es condicionarla a caer dentro del intervalo. mu y sigma son los parámetros de la normal subyacente, no exactamente la media y desviación de la distribución truncada.

La posterior multiplica esas densidades por el prior conjunto de efectos y los priors de alpha, gamma, tau y rho. Los priors concretos no están informados; no es posible reproducir exactamente el ajuste desde el PDF solamente. Half-normal para desviaciones y LKJ para correlaciones serían opciones, no hechos sobre este manuscrito.

Tabla 3 LENA: alpha=6,52; gamma=0,63; tau_mu=1,29; tau_sigma=0,19; rho=0,63, con intervalo [0,02;0,97]. Bajo enlace log, sigma del grupo con v=0 sería exp(0,63)≈1,88 minutos/hora. Es una interpretación condicional, no la media poblacional de sigma. La fila sd_experiment refleja variación entre experimentos, distinta de sigma, que refleja dispersión entre niños dentro de un experimento.

Ejemplo hipotético: u_g=+2 y v_g=+0,2 desplazan la localización LENA a 8,52 y multiplican su escala por exp(0,2)≈1,22. No son efectos estimados de una comunidad concreta.

### Predicción de nuevas comunidades

Por cada muestra posterior de hiperparámetros se genera un nuevo par (u_nuevo,v_nuevo) de su normal bivariada. Si se desea un niño, se genera después una observación truncada; si se desea exposición media comunitaria, se usa la esperanza truncada, no una realización individual. El borrador debe precisar cuál utiliza al construir trayectorias.

Una exposición e minutos/hora se acumula como H(A)=integral_0^A [e(a)/60]*w(a) da, si a está en días y w(a) son horas despierto por día. Las palabras requieren además multiplicar por una tasa de palabras por hora de habla. El borrador deja incompleta esa tasa y advierte que una equivalencia basada en inglés no es universal.

El manuscrito informa provisionalmente medianas 7,0 y 8,6 minutos/hora y un rango de 20–75 millones de palabras a los seis años. No se interpretan como estimaciones mundiales ya validadas. La equivalencia de etiquetas LENA/BTC/BTC+/ACLEW debe aclararse. El texto geográfico tampoco coincide plenamente con la tabla de experimentos, que incluye Europa.

### Límites para interpretar los efectos

`experiment` representa proyecto/comunidad y puede absorber diferencias de edad, composición socioeconómica, desarrollo y procedimiento. No identifica un efecto cultural causal puro. Agregar sesiones evita pseudorreplicación por sesión, pero no modela explícitamente las trayectorias individuales ni su precisión desigual. El texto describe niño como unidad de observación, no muestra un intercepto aleatorio adicional por niño.

Para reproducibilidad faltan fórmula exacta, priors, tamaño final de muestra por algoritmo y grupo, diagnósticos, correspondencia de algoritmos, tratamiento conjunto al simular los efectos correlacionados y detalle de la extrapolación. La ausencia de asterisco en una correlación no equivale a independencia; un intervalo de desviación estándar que excluye cero tampoco es por sí solo una prueba contra ausencia de heterogeneidad.

## 7. Funciones y comandos

- `data/preparation.py:prepare_data`: prepara y audita; `data/phylogeny.py:build_shared_time_covariance`: construye C.
- `models/features.py:build_features`: xyz e índice; `models/kernels.py:build_kernel`: suma R y C.
- `models/gp.py:build_model`: GPC sklearn; `predict`: probabilidades y clase.
- `evaluation/splits.py:split_data/inner_splits`: separaciones; `evaluation/search.py:select_parameters`: grilla.
- `evaluation/experiment.py:evaluate/impute/save_run`: evaluación, ajuste final y archivos.
- `evaluation/report.py:comparison_report`: compara lo disponible y declara pendientes.
- `visualization/map.py:build_map`: mapa; `tree_viewer/`: aplicación independiente para explorar árboles.

```powershell
uv sync --locked
uv run gaussian-processes map
uv run gaussian-processes report
uv run python -m unittest discover -s tests -v
# Solo si quieres volver a ajustar:
uv run gaussian-processes search
# Completar la búsqueda geodésica pendiente:
uv run gaussian-processes search --config config/geodesic_exponential.json
uv run gaussian-processes map --config config/geodesic_exponential.json
# Evaluaciones nuevas con parámetros fijos:
uv run gaussian-processes evaluate-families
uv run gaussian-processes evaluate-spatial
# Visualizador opcional:
uv run --extra viewer python -m tree_viewer
```

Revisión actual: se corrigió el reporte para tolerar búsquedas pendientes sin perder comprobaciones de comparabilidad, se generaron mapas faltantes desde predicciones existentes y se corrigió documentación antigua que presentaba Matérn como predeterminado. No se cambiaron kernels, etiquetas, splits ni hiperparámetros, ni se volvió a ajustar una grilla global. Se conservan fuentes y resultados distintos para reproducibilidad. La limpieza de cachés, tmp/pdfs/model.png y results/geodesic_exponential/search/splits.csv quedó pendiente: el control automático rechazó tanto la eliminación recursiva como una versión limitada a archivos individuales con el mensaje «blocked by policy». No se eliminó ningún archivo.

Referencias técnicas: [GPC sklearn](https://scikit-learn.org/stable/modules/gaussian_process.html), [modelos distribucionales brms](https://paulbuerkner.com/brms/articles/brms_distreg.html), [fórmulas y efectos correlacionados brms](https://paulbuerkner.com/brms/reference/brmsformula.html).
