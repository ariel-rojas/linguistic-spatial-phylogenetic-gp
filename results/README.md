# Resultados del modelo

El flujo estándar guarda sus resultados en
**results/spherical_spline_family/bayesian/**. Los archivos grandes de muestras
posteriores son locales y están excluidos de Git.

~~~text
bayesian/
  run.json           configuración, fuentes, métricas, diagnósticos y posterior de L
  report.md          resumen legible de la ejecución
  predictions.csv    probabilidades e intervalos por lengua
  evaluation.npz     posterior del ajuste train-only y predicciones del test
  imputation.npz     posterior reajustada con todas las etiquetas observadas
  map.html           mapa interactivo de train, test e imputaciones
  posterior.html     probabilidades por lengua y distribuciones posteriores
~~~

El grado efectivo L tiene un prior uniforme continuo entre los límites de
config/bayesian.json. Su posterior aparece en run.json, report.md y como
histograma en posterior.html. Evaluación e imputación tienen posteriores
separadas porque el segundo ajuste incorpora también las etiquetas del test.

En predictions.csv, la columna role indica entrenamiento, test, imputación o
exclusión. p_OV, p_SD y p_VO son las probabilidades posteriores medias; las
columnas _lower y _upper dan intervalos creíbles puntuales del 95 %. Una
predicción imputada sigue siendo una propuesta, no un dato verificado.

Los comandos alternativos de scikit-learn guardan archivos solo cuando se
ejecutan, dentro de la carpeta del experimento correspondiente. La estructura
y los comandos están en la [guía del proyecto](../docs/workflow.md); el modelo
y la interpretación de sus probabilidades se explican en la
[guía bayesiana](../docs/bayesian_inference.md).

Al volver a ejecutar con el mismo nombre se reemplaza la salida de ese
experimento. Para conservar resultados de varias configuraciones, copiá los
JSON, asigná un nombre distinto y ejecutalos por separado.
