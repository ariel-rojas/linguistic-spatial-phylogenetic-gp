/* Vista autónoma: histogramas de muestras y predictiva categórica se mantienen separados. */
const data = JSON.parse(document.getElementById("data").textContent);
const $ = (id) => document.getElementById(id);
const colors = ["#2563eb", "#9333ea", "#ea580c"];
const names = [
  "OV · objeto antes del verbo",
  "Sin orden dominante",
  "VO · verbo antes del objeto",
];
const pct = (x) => (100 * x).toFixed(1) + " %";
const num = (x) => (Math.abs(x) < 0.01 ? x.toExponential(2) : x.toFixed(3));
const escapeText = (text) =>
  String(text).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
// Cada barra representa la fracción de muestras dentro de un intervalo.
function histogram(d, color, probability = false) {
  const W = 300,
    H = 170,
    left = 35,
    right = 10,
    top = 12,
    bottom = 32;
  const min = d.edges[0],
    max = d.edges.at(-1),
    peak = Math.max(...d.mass, 0.01);
  const x = (v) => left + ((v - min) / (max - min || 1)) * (W - left - right);
  const y = (v) => H - bottom - (v / peak) * (H - top - bottom);
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Histograma de muestras posteriores"><rect x="${x(d.interval[0])}" y="${top}" width="${Math.max(0, x(d.interval[1]) - x(d.interval[0]))}" height="${H - top - bottom}" fill="${color}" opacity="0.10"/>`;
  d.mass.forEach(
    (m, i) =>
      (svg += `<rect x="${x(d.edges[i]) + 0.5}" y="${y(m)}" width="${Math.max(0, x(d.edges[i + 1]) - x(d.edges[i]) - 1)}" height="${H - bottom - y(m)}" fill="${color}"><title>${probability ? pct(d.edges[i]) : num(d.edges[i])} a ${probability ? pct(d.edges[i + 1]) : num(d.edges[i + 1])}: ${pct(m)} de las muestras</title></rect>`),
  );
  svg += `<line x1="${x(d.mean)}" x2="${x(d.mean)}" y1="${top}" y2="${H - bottom}" stroke="#132e42" stroke-width="2"/><line x1="${left}" x2="${W - right}" y1="${H - bottom}" y2="${H - bottom}" stroke="#71869a"/>`;
  for (const v of [min, (min + max) / 2, max])
    svg += `<text x="${x(v)}" y="${H - 10}" text-anchor="${v === min ? "start" : v === max ? "end" : "middle"}">${probability ? (v * 100).toFixed(0) + "%" : num(v)}</text>`;
  svg += `<text x="${left - 4}" y="${top + 4}" text-anchor="end">${(peak * 100).toFixed(0)}%</text><text x="${left - 4}" y="${H - bottom}" text-anchor="end">0</text></svg>`;
  return svg;
}
// Vista resumida de cadenas; no sustituye los diagnósticos con todas las muestras.
function traceChart(d) {
  const values = d.traces.flat(),
    lo = Math.min(...values),
    hi = Math.max(...values),
    palette = ["#2563eb", "#db2777", "#059669", "#b45309"];
  let svg =
    '<svg viewBox="0 0 300 100" role="img" aria-label="Trazas por cadena">';
  d.traces.forEach((chain, c) => {
    const points = chain
      .map(
        (v, i) =>
          `${10 + (280 * i) / Math.max(1, chain.length - 1)},${80 - (70 * (v - lo)) / (hi - lo || 1)}`,
      )
      .join(" ");
    svg += `<polyline points="${points}" fill="none" stroke="${palette[c % 4]}" stroke-width="0.8" opacity="0.75"/>`;
  });
  return (
    svg +
    '<text x="10" y="97">Inicio</text><text x="290" y="97" text-anchor="end">Fin · muestras sin warmup</text></svg>'
  );
}
function selectTab(global) {
  $("local").hidden = global;
  $("global").hidden = !global;
  $("localTab").setAttribute("aria-pressed", !global);
  $("globalTab").setAttribute("aria-pressed", global);
}
// La lengua seleccionada determina tanto su predictiva como el ajuste asociado.
function render() {
  const item = data.cases.find(
    (x) => x.id === $("cases").value && x.stage === $("scope").value,
  );
  if (!item) {
    $("language").textContent = "No hay lenguas que coincidan";
    $("metadata").textContent = "Probá otro nombre, código o familia.";
    $("local").hidden = true;
    $("global").hidden = true;
    return;
  }
  selectTab($("globalTab").getAttribute("aria-pressed") === "true");
  $("language").textContent = item.name;
  $("metadata").textContent =
    `WALS: ${item.id} · ${item.family} · ${item.stage === "imputation" ? "Imputación sin verificar" : "Test reservado · observado: " + item.observed}`;
  $("prediction").innerHTML = item.distributions
    .map(
      (d, c) =>
        `<div class="bar"><div class="bar-label"><span>${names[c]}</span><strong>${pct(d.mean)}</strong></div><div class="track"><div class="fill" style="width:${100 * d.mean}%;background:${colors[c]}"></div></div></div>`,
    )
    .join("");
  $("probabilities").innerHTML = item.distributions
    .map(
      (d, c) =>
        `<div class="chart"><h4>${names[c]}</h4>${histogram(d, colors[c], true)}<p>Media <strong>${pct(d.mean)}</strong><br>Intervalo creíble del 95 %: <strong>${pct(d.interval[0])}–${pct(d.interval[1])}</strong></p></div>`,
    )
    .join("");
  $("sampleCount").textContent =
    `${item.distributions[0].n} muestras predictivas conjuntas. Los histogramas utilizan 20 intervalos iguales y su altura es masa de probabilidad por intervalo, no densidad.`;
  const sensitivity = item.component_sensitivity;
  $("componentInfluence").innerHTML = [
    ["spatial", "Efecto espacial", "#2563eb"],
    ["phylogenetic", "Efecto filogenético", "#7c3aed"],
  ]
    .map(([key, label, color]) => {
      const d = sensitivity[key];
      return `<div class="chart"><h4>${label}</h4>${histogram(d, color, true)}<p>Cambio medio: <strong>${pct(d.mean)}</strong><br>Intervalo creíble del 95 %: ${pct(d.interval[0])}–${pct(d.interval[1])}</p></div>`;
    })
    .join("");
  const stage = data.stages[item.stage],
    diagnostics = stage.diagnostics;
  $("stageNote").textContent =
    `Ajuste utilizado: ${item.stage === "imputation" ? "todos los observados para imputar" : "solo train para evaluar test"} (${stage.n_fit} observaciones). Esta selección cambia automáticamente según el caso elegido.`;
  const definitions = {
    max_degree: [
      "Grado máximo efectivo L",
      "Centro del corte logístico suave: controla qué grados de armónicos aportan a la superficie espacial.",
    ],
    sigma_spatial: [
      "Amplitud espacial σ",
      "Cuánto pueden variar las tendencias espaciales.",
    ],
    sigma_phylogenetic: [
      "Amplitud filogenética σ",
      "Cuánto pueden variar las tendencias asociadas al árbol.",
    ],
    smoothing: [
      "Suavizado λ",
      "Valores mayores reducen el peso de los detalles espaciales finos.",
    ],
    intercept_1: [
      "Intercepto · contraste 1",
      "Tendencia global en el primer eje de contraste; no es una probabilidad.",
    ],
    intercept_2: [
      "Intercepto · contraste 2",
      "Tendencia global en el segundo eje de contraste; no pertenece a una lengua.",
    ],
  };
  $("parameters").innerHTML = Object.entries(stage.parameters)
    .map(
      ([key, d]) =>
        `<div class="chart"><h4>${definitions[key][0]}</h4><p>${definitions[key][1]}</p>${histogram(d, "#15768a")}<p>Media <strong>${num(d.mean)}</strong><br>Intervalo creíble del 95 %: ${num(d.interval[0])}–${num(d.interval[1])}<br>${d.n} muestras posteriores, combinando cadenas.</p>${d.traces ? traceChart(d) : ""}</div>`,
    )
    .join("");
  $("diagnostics").innerHTML =
    `<p class="${diagnostics.passed ? "ok" : "warning"}">${diagnostics.passed ? "Cumple el criterio de diagnóstico del proyecto." : "Resultados exploratorios: no se cumple todo el criterio de diagnóstico."}</p><div class="stats">${[
      ["R-hat máximo", diagnostics.max_rhat.toFixed(4)],
      ["ESS bulk mínimo", diagnostics.min_ess_bulk.toFixed(0)],
      ["ESS tail mínimo", diagnostics.min_ess_tail.toFixed(0)],
      ["Divergencias", diagnostics.divergences],
      ["Saturaciones de profundidad", diagnostics.tree_depth_hits],
    ]
      .map(
        ([label, v]) =>
          `<div class="stat">${escapeText(label)}<strong>${v}</strong></div>`,
      )
      .join(
        "",
      )}</div><p>El criterio se aplica a L, amplitudes, suavizado e interceptos globales: R-hat &lt; 1,01; ESS bulk/tail ≥ 400; cero divergencias y cero saturaciones de profundidad. Para los ${diagnostics.latent_diagnostics.n_coefficients} coeficientes latentes: R-hat máximo ${diagnostics.latent_diagnostics.max_rhat.toFixed(3)}, ESS bulk mínimo ${diagnostics.latent_diagnostics.min_ess_bulk.toFixed(0)}. Se resumen aparte porque son muchos y algunos pueden mezclar lentamente. Ninguno de estos controles certifica que los supuestos lingüísticos sean correctos.</p>`;
  const url = new URL(location.href);
  url.searchParams.set("language", item.id);
  url.searchParams.set("stage", item.stage);
  try {
    history.replaceState(null, "", url);
  } catch (e) {
    /* Algunos navegadores restringen history bajo file://. */
  }
}
// Conserva la selección al buscar si todavía está entre las coincidencias.
function filter(preferred) {
  const q = $("search").value.toLocaleLowerCase();
  const cases = data.cases
    .filter(
      (x) =>
        x.stage === $("scope").value &&
        `${x.name} ${x.id} ${x.family}`.toLocaleLowerCase().includes(q),
    )
    .sort((a, b) => a.name.localeCompare(b.name));
  $("cases").replaceChildren(
    ...cases.map((x) => new Option(`${x.name} · ${x.id}`, x.id)),
  );
  if (cases.some((x) => x.id === preferred)) $("cases").value = preferred;
  $("count").textContent = `${cases.length} casos disponibles`;
  render();
}
$("warning").className = data.status === "complete" ? "ok" : "warning";
$("warning").textContent =
  data.status === "complete"
    ? "Muestreo completado y criterio de diagnóstico satisfecho."
    : "Muestreo terminado con advertencia de diagnóstico. Las distribuciones son exploratorias; consultá la pestaña de parámetros.";
$("localTab").onclick = () => selectTab(false);
$("globalTab").onclick = () => selectTab(true);
$("scope").onchange = () => filter();
$("search").oninput = () => filter($("cases").value);
$("cases").onchange = render;
const query = new URLSearchParams(location.search);
if (["evaluation", "imputation"].includes(query.get("stage")))
  $("scope").value = query.get("stage");
filter(query.get("language"));
