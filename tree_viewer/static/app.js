/** Selectores, búsqueda y metadata. Los textos de fuentes se insertan con textContent. */
import { TreeView } from "./tree.js";
const $ = (id) => document.getElementById(id);
let catalog,
  data,
  selected = 0,
  properties,
  requestNumber = 0,
  nodeRequest = 0,
  availableEntries = [];
const view = new TreeView($("tree"), selectNode);
const number = (value) => Math.round(value).toLocaleString("es");
function status(message, error = false) {
  $("status").textContent = message;
  $("status").classList.toggle("error", error);
}
async function json(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error((await response.text()).slice(0, 300));
  return response.json();
}
function option(value, label) {
  const item = document.createElement("option");
  item.value = value;
  item.textContent = label;
  return item;
}
function updateChoices() {
  const previousYear = $("years").value,
    previousSample = $("sample").value;
  const entries = catalog.entries.filter((e) => e.kind === $("kind").value);
  const years = [...new Set(entries.map((e) => e.years))].sort((a, b) => a - b);
  $("years").replaceChildren(
    ...years.map((y) =>
      option(y, y ? `${number(y)} años atrás` : "Actual · árbol completo"),
    ),
  );
  if (previousYear !== "" && years.includes(Number(previousYear)))
    $("years").value = previousYear;
  updateSamples(previousSample);
}
function selectedEntry() {
  return (
    availableEntries.find((e) => e.sample === Number($("sample").value)) ||
    (availableEntries.length === 1 ? availableEntries[0] : null)
  );
}
function updateNavigation() {
  const index = availableEntries.indexOf(selectedEntry());
  $("previous").disabled = index <= 0;
  $("next").disabled = index < 0 || index >= availableEntries.length - 1;
}
function updateSamples(previous = $("sample").value) {
  availableEntries = catalog.entries.filter(
    (e) => e.kind === $("kind").value && e.years === Number($("years").value),
  );
  const sample = String(previous).startsWith("posterior-")
    ? Number(previous.split("-")[1])
    : Number(previous);
  const match =
    availableEntries.find((e) => e.sample === sample) || availableEntries[0];
  $("sample").value = match.sample ?? "";
  $("sample").disabled = availableEntries.length === 1;
  $("sample").min = availableEntries[0].sample ?? 1;
  $("sample").max = availableEntries.at(-1).sample ?? 1;
  updateNavigation();
}
function selectedDescendants() {
  if (!data || data.entry.years) return [];
  const leaves = [],
    stack = [selected];
  while (stack.length) {
    const n = data.nodes[stack.pop()];
    if (n.children.length) stack.push(...n.children);
    else if (n.glottocode) leaves.push(n.glottocode);
  }
  return leaves;
}
function correspondingNode(leaves) {
  if (!leaves.length || data.entry.years) return null;
  const byCode = new Map(
    data.nodes.filter((n) => n.glottocode).map((n) => [n.glottocode, n.id]),
  );
  if (leaves.some((code) => !byCode.has(code))) return null;
  let candidate = byCode.get(leaves[0]);
  const ancestors = (id) => {
    const set = new Set();
    while (id !== null) {
      set.add(id);
      id = data.nodes[id].parent;
    }
    return set;
  };
  for (const code of leaves.slice(1)) {
    const path = ancestors(byCode.get(code));
    while (!path.has(candidate)) candidate = data.nodes[candidate].parent;
  }
  return candidate;
}
async function load() {
  const ticket = ++requestNumber;
  nodeRequest++;
  const oldLeaves = selectedDescendants(),
    id = selectedEntry()?.id;
  if (!id) {
    status("Elige un número de muestra disponible.", true);
    return;
  }
  status("Cargando árbol y verificando su metadata…");
  $("matches").replaceChildren();
  $("search").value = "";
  try {
    const result = await json(`/api/trees/${encodeURIComponent(id)}`);
    if (ticket !== requestNumber) return;
    data = result;
    view.load(data);
    const context = correspondingNode(oldLeaves);
    selected = context ?? 0;
    if (context !== null && context !== 0) view.reveal(context);
    $("tree-title").textContent =
      data.entry.kind === "mcc"
        ? "MCC · árbol resumen"
        : `Posterior ${data.entry.sample} · ${data.entry.years ? number(data.entry.years) + " años atrás" : "actual"}`;
    $("tree-count").textContent =
      `${number(data.nodes[0].leaves)} puntas · raíz: ${number(data.root_age_years)} años · ${data.entry.years ? "puntas = linajes históricos" : "puntas = lenguas actuales"}`;
    status(
      oldLeaves.length && context === null
        ? "Árbol cargado. Selección anterior sin correspondencia comprobada; se muestra la raíz."
        : context !== null && data.nodes[context].leaves !== oldLeaves.length
          ? "Se conservó el ancestro común; el clado contiene descendientes adicionales en este árbol."
          : "Árbol cargado. Selecciona un nodo para revisar todas sus propiedades.",
    );
    const url = new URL(location.href);
    url.searchParams.set("tree", id);
    history.replaceState(null, "", url);
    await selectNode(selected);
    updateNavigation();
  } catch (error) {
    if (ticket === requestNumber)
      status(`No se pudo cargar: ${error.message}`, true);
  }
}
async function selectNode(id) {
  selected = id;
  view.selected = id;
  const node = data.nodes[id],
    ticket = ++nodeRequest;
  $("node-name").textContent = node.label;
  $("node-summary").textContent =
    `${node.glottocode || node.source_label || "Nodo interno " + id}\nEdad por ramas: ${number(node.age_years)} años\nRama entrante: ${node.length.toPrecision(6)} milenios\n${number(node.leaves)} puntas descendientes`;
  $("expand").disabled = !node.children.length;
  $("focus").disabled = false;
  $("expand").textContent = view.open.has(id) ? "Cerrar clado" : "Abrir clado";
  $("metadata").textContent = "Cargando propiedades…";
  try {
    const result = await json(
      `/api/trees/${encodeURIComponent(data.entry.id)}/nodes/${id}`,
    );
    if (ticket !== nodeRequest) return;
    properties = result;
    renderMetadata();
  } catch (error) {
    if (ticket === nodeRequest) $("metadata").textContent = error.message;
  }
}
function renderMetadata() {
  if (!properties) return;
  const filter = $("metadata-filter").value.toLocaleLowerCase();
  $("metadata").replaceChildren();
  const groups = [
    ["Geometría y etiquetas", properties.node],
    ["Anotaciones originales del árbol", properties.annotations],
    ["Anotaciones de la rama", properties.edge_annotations],
    ["Tabla histórica · valores originales", properties.csv],
    ["Lengua actual · fuente y WALS", properties.language],
    ["Comentarios adicionales", properties.comments],
    ["Procedencia", data.entry],
  ];
  for (const [title, values] of groups) {
    const fields = Object.entries(values || {}).filter(([key, value]) =>
      `${key} ${JSON.stringify(value)}`.toLocaleLowerCase().includes(filter),
    );
    if (!fields.length) continue;
    const group = document.createElement("details");
    group.open =
      !title.startsWith("Comentarios") &&
      !title.startsWith("Procedencia") &&
      !title.startsWith("Geometría");
    const summary = document.createElement("summary");
    summary.textContent = title;
    group.append(summary);
    for (const [key, value] of fields) {
      const field = document.createElement("div");
      field.className = "field";
      const label = document.createElement("b");
      label.textContent = key;
      field.append(label);
      const content = document.createElement(
        typeof value === "object" || String(value).length > 150
          ? "pre"
          : "span",
      );
      content.textContent =
        typeof value === "object"
          ? JSON.stringify(value, null, 2)
          : String(value);
      field.append(content);
      group.append(field);
    }
    $("metadata").append(group);
  }
  if (!$("metadata").children.length)
    $("metadata").textContent = "Sin propiedades para este filtro.";
}
$("kind").onchange = () => {
  updateChoices();
  load();
};
$("years").onchange = () => {
  updateSamples();
  load();
};
$("sample").onchange = load;
$("load-sample").onclick = load;
$("sample").onkeydown = (event) => {
  if (event.key === "Enter") load();
};
for (const [id, step] of [
  ["previous", -1],
  ["next", 1],
])
  $(id).onclick = () => {
    const entry =
      availableEntries[availableEntries.indexOf(selectedEntry()) + step];
    if (entry) {
      $("sample").value = entry.sample;
      load();
    }
  };
$("fit").onclick = () => view.fit();
$("root").onclick = () => {
  view.load(data);
  selectNode(0);
};
$("zoom-in").onclick = () => view.scale(1.4);
$("zoom-out").onclick = () => view.scale(1 / 1.4);
$("expand").onclick = () => {
  view.toggle(selected);
  selectNode(selected);
};
$("focus").onclick = () => view.reveal(selected);
$("metadata-filter").oninput = renderMetadata;
$("search").oninput = () => {
  $("matches").replaceChildren();
  const query = $("search").value.trim().toLocaleLowerCase();
  if (!query || !data) return;
  const matches = data.nodes
    .filter((n) =>
      `${n.label} ${n.source_label} ${n.glottocode}`
        .toLocaleLowerCase()
        .includes(query),
    )
    .slice(0, 30);
  for (const node of matches) {
    const button = document.createElement("button");
    button.textContent = `${node.label} · ${node.glottocode || node.source_label || "clado " + node.id}`;
    button.onclick = () => {
      view.reveal(node.id);
      selectNode(node.id);
      $("matches").replaceChildren();
    };
    $("matches").append(button);
  }
  if (!matches.length) {
    const message = document.createElement("div");
    message.className = "empty";
    message.textContent = "Sin coincidencias en este árbol.";
    $("matches").append(message);
  }
};
try {
  catalog = await json("/api/catalog");
  const requested = new URL(location.href).searchParams.get("tree");
  const entry = catalog.entries.find((e) => e.id === requested);
  if (entry) $("kind").value = entry.kind;
  updateChoices();
  if (entry) {
    $("years").value = entry.years;
    updateSamples(entry.id);
  }
  await load();
} catch (error) {
  status(`No se pudo iniciar: ${error.message}`, true);
}
