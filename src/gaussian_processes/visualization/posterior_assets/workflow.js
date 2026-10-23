/* Guía de la ejecución: usa exclusivamente el contexto archivado en el HTML.
   Los priors son masas analíticas; no confundirlos con muestras posteriores. */
function renderWorkflow() {
  const w = data.workflow,
    s = w.sampling,
    c = w.configuration,
    n = w.counts;
  const table = (headers, rows) =>
    `<div class="table-scroll"><table><thead><tr>${headers.map((h) => `<th>${escapeText(h)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((v) => `<td>${escapeText(v)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
  const priorCards = Object.entries(w.priors)
    .map(
      ([name, p]) =>
        `<div class="chart"><h4>${escapeText(name)}</h4><p><code>${escapeText(p.formula)}</code></p><p><strong>Valores posibles:</strong> ${escapeText(p.support)}</p>${histogram(p.chart, "#a85516")}<p><strong>95 % central del prior:</strong> ${num(p.interval[0])} a ${num(p.interval[1])}<br>Mediana: ${num(p.median)}</p><p>${p.logarithmic ? "Eje horizontal: log10(λ); −3 significa λ = 0,001. Línea: mediana en escala logarítmica." : "Eje horizontal: valor del parámetro. Línea: media."} Altura: masa del prior por intervalo. Sombreado: 95 % central. Se muestra el 99,8 % central; no es un límite del parámetro.</p></div>`,
    )
    .join("");
  const q = w.prior_predictive.expected_class_fraction_quantiles_025_50_975;
  $("workflow").innerHTML = `
    <h3>1 · Datos → separación → dos ajustes</h3>
    <p>WALS aporta las etiquetas OV / Sin dominante / VO y coordenadas; LOW aporta identidad y familias; el árbol MCC aporta tiempos compartidos. Solo los registros elegibles entran al ajuste.</p>
    <p><strong>${n.n_train} train → posterior de evaluación → ${n.n_test} test.</strong><br><strong>${n.n_final_fit} observados → nueva posterior final → ${n.n_imputed} imputaciones.</strong> Las etiquetas de test no intervienen en el primer ajuste; sí en el ajuste final. Las imputaciones no se usan para medir aciertos.</p>
    <p>Separación estratificada por identidad: una de ${Math.round(1 / c.test_fraction)} particiones, semilla ${c.seed}. Intenta conservar las proporciones de clases y mantiene juntas las identidades compartidas. No reserva familias ni regiones completas.</p>
    <h3>2 · θ: qué se estima y dónde entra</h3>
    <p><code>θ = (σ espacial, σ filogenética, λ, α, Z espacial, Z filogenético)</code>. Se muestrean conjuntamente. F se deriva de θ; no es otro parámetro independiente.</p>
    <div class="equation">wⱼ(λ) = 1 / {1 + λ [ℓⱼ(ℓⱼ+1)]²}<br>dⱼ = √(wⱼ / Σᵣ wᵣ)<br>g = σ espacial · Φ · diag(d) · Z espacial<br>h = σ filogenética · L · Z filogenético<br>F = g + h<br>u = α + F</div>
    <p>Φ evalúa los armónicos esféricos en las coordenadas; ℓⱼ es el grado de cada armónico. λ atenúa los detalles finos. L es un factor de la correlación filogenética (R = LLᵀ), alineado con las lenguas: transforma coeficientes independientes en efectos correlacionados. Los Z tienen dos columnas, una por contraste. α se suma a todas las filas.</p>
    <p>Las amplitudes multiplican efectos; sus cuadrados multiplican covarianzas: <code>K = σ espacial² R espacial(λ) + σ filogenética² R filogenética</code>. No son porcentajes de influencia ni atribuciones causales.</p>
    ${table(
      ["Ajuste", "Z espacial", "Z filogenético"],
      Object.entries(data.stages).map(([name, stage]) => [
        name === "evaluation" ? "Evaluación" : "Imputación",
        stage.latent_shapes.z_spatial?.join(" × ") ?? "No disponible",
        stage.latent_shapes.z_phylogenetic?.join(" × ") ?? "No disponible",
      ]),
    )}
    <h3>3 · Priors y rangos: antes de observar etiquetas</h3>
    <p><strong>No hay una grilla ni límites mínimos/máximos de búsqueda.</strong> El soporte indica qué valores permite el prior; el intervalo central del 95 % indica dónde concentra su masa, pero permite valores fuera. El intervalo posterior, en «Parámetros del ajuste», se calcula después de usar los datos. El mínimo/máximo visible de un gráfico tampoco es un límite del modelo.</p>
    <div class="grid">${priorCards}</div>
    <h3>4 · De dos contrastes a tres clases y a la posterior</h3>
    <div class="equation">η OV = (√3/2)u₁ + u₂/2<br>η Sin dominante = −(√3/2)u₁ + u₂/2<br>η VO = −u₂<br>πᵢ꜀ = exp(ηᵢ꜀) / Σᵣ exp(ηᵢᵣ)<br>yᵢ ~ Categorical(πᵢ)<br>p(θ | datos) ∝ [Πᵢ πᵢ,clase observada] · p(θ)</div>
    <p>u₁ compara OV con Sin dominante; u₂ compara ambas frente a VO. Las puntuaciones suman cero, evitando la dirección redundante de softmax. Ningún contraste es una probabilidad. La verosimilitud usa solo las etiquetas del conjunto de ajuste.</p>
    <h3>5 · NUTS y predicción</h3>
    ${table(
      ["Control", "Valor por ajuste"],
      [
        ["Cadenas", s.chains],
        ["Adaptación descartada por cadena", s.warmup],
        ["Muestras guardadas por cadena", s.draws],
        ["Total posterior", s.chains * s.draws],
        [
          "Muestras utilizadas para predecir",
          Math.min(s.predictive_draws, s.chains * s.draws),
        ],
        ["Aceptación objetivo", s.target_accept],
        ["Profundidad máxima", s.max_tree_depth],
        ["Semilla base del muestreo", s.seed],
      ],
    )}
    <p>NUTS adapta sus pasos durante warmup y luego explora la posterior conjunta. Cada iteración puede contener muchos pasos internos. Las muestras están correlacionadas: el ESS mide información efectiva. Para lenguas nuevas se evalúa la superficie espacial y se simula el residual filogenético condicional. Se aplica softmax a cada muestra y después se promedian las probabilidades; no al revés.</p>
    <h3>6 · Qué predecía el prior y cómo rindió el modelo</h3>
    <p>Antes del ajuste se simularon ${s.prior_draws} escenarios. Esta tabla muestra cuantiles de la <strong>fracción esperada de registros por clase</strong>, no hablantes ni probabilidades de una lengua concreta. No usa etiquetas y no mide performance.</p>
    ${table(
      ["Clase", "Percentil 2,5", "Mediana", "Percentil 97,5"],
      data.classes.map((name, i) => [
        name,
        pct(q[0][i]),
        pct(q[1][i]),
        pct(q[2][i]),
      ]),
    )}
    <p>En esas simulaciones, ${pct(w.prior_predictive.fraction_probability_above_095)} de las predicciones tuvieron probabilidad máxima superior al 95 %. Sirve para revisar cuánta confianza presupone el prior.</p>
    ${table(
      ["Métrica en test", "Valor"],
      [
        ["Aciertos", `${w.metrics.correct} / ${w.metrics.n_test}`],
        ["Accuracy", pct(w.metrics.accuracy)],
        ["Balanced accuracy", pct(w.metrics.balanced_accuracy)],
        ["Macro-F1", num(w.metrics.macro_f1)],
        ["Log-loss (menor es mejor)", num(w.metrics.log_loss)],
        ["Brier multiclase (menor es mejor)", num(w.metrics.brier_multiclass)],
      ],
    )}
    ${table(
      ["Clase", "Casos test", "Recall", "F1"],
      data.classes.map((name) => {
        const m = w.metrics.per_class[name];
        return [name, m.support, pct(m.recall), num(m.f1)];
      }),
    )}
    <p>La accuracy mide aciertos globales; balanced accuracy promedia recalls; macro-F1 da el mismo peso a cada clase. Log-loss y Brier evalúan probabilidades. Los diagnósticos de cadenas están en «Parámetros del ajuste»: una buena accuracy no prueba convergencia.</p>
    <h3>7 · Qué queda fijo y dónde revisar los archivos</h3>
    <p>Grado espacial máximo: ${c.max_degree}. Estructura filogenética: ${escapeText(c.phylogenetic_structure)}. Normalización: ${c.phylogenetic_normalize ? "sí" : "no"}. Con within_family, Rᵢⱼ = 0 entre familias; si se normaliza, Rᵢⱼ = Cᵢⱼ / √(Cᵢᵢ Cⱼⱼ) dentro de cada bloque.</p>
    <p>No se integran árboles alternativos, errores de etiquetas ni otras estructuras de kernel. Los valores puntuales y la grilla de sklearn no son priors de NUTS. El modo espacial constante y el intercepto pueden explicar una tendencia común: no interpretamos sus contribuciones constantes por separado.</p>
    <p><a href="report.md">Reporte resumido</a> · <a href="run.json">Registro completo de esta ejecución</a> · <a href="predictions.csv">Tabla de predicciones</a>. Los NPZ conservan las muestras. Cambiar un JSON de configuración no actualiza estos resultados: requiere ejecutar de nuevo el modelo.</p>`;
}
renderWorkflow();
