/** Geometría temporal, colapsado y navegación SVG; no modifica datos del árbol. */
const NS = "http://www.w3.org/2000/svg";
const element = (tag, attributes = {}, text = "") => {
  const item = document.createElementNS(NS, tag);
  Object.entries(attributes).forEach(([key, value]) =>
    item.setAttribute(key, value),
  );
  item.textContent = text;
  return item;
};

export class TreeView {
  constructor(svg, onSelect) {
    this.svg = svg;
    this.onSelect = onSelect;
    this.open = new Set();
    this.zoom = 1;
    this.tx = 30;
    this.ty = 55;
    svg.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        const rect = svg.getBoundingClientRect();
        this.scale(
          Math.exp(-event.deltaY * 0.0015),
          event.clientX - rect.left,
          event.clientY - rect.top,
        );
      },
      { passive: false },
    );
    svg.addEventListener("pointerdown", (event) => {
      if (event.target.closest(".node")) return;
      this.drag = {
        x: event.clientX,
        y: event.clientY,
        tx: this.tx,
        ty: this.ty,
      };
      svg.setPointerCapture(event.pointerId);
    });
    svg.addEventListener("pointermove", (event) => {
      if (!this.drag) return;
      this.tx = this.drag.tx + event.clientX - this.drag.x;
      this.ty = this.drag.ty + event.clientY - this.drag.y;
      this.transform();
    });
    svg.addEventListener("pointerup", () => {
      this.drag = null;
    });
    svg.addEventListener("pointercancel", () => {
      this.drag = null;
    });
  }
  load(data) {
    this.data = data;
    this.nodes = data.nodes;
    this.selected = 0;
    this.open.clear();
    this.openLevels(0, 4);
    this.render();
    this.fit();
  }
  openLevels(id, levels) {
    if (levels <= 0) return;
    this.open.add(id);
    this.nodes[id].children.forEach((child) =>
      this.openLevels(child, levels - 1),
    );
  }
  reveal(id) {
    let parent = this.nodes[id].parent;
    while (parent !== null) {
      this.open.add(parent);
      parent = this.nodes[parent].parent;
    }
    this.selected = id;
    this.render();
    this.center(id);
  }
  toggle(id) {
    if (!this.nodes[id].children.length) return;
    this.open.has(id) ? this.open.delete(id) : this.open.add(id);
    this.render();
    this.center(id);
  }
  render() {
    if (!this.data) return;
    const visible = [],
      positions = new Map();
    let row = 0;
    const layout = (id) => {
      const node = this.nodes[id];
      visible.push(node);
      const children = this.open.has(id) ? node.children : [];
      children.forEach(layout);
      const y = children.length
        ? children.reduce((sum, child) => sum + positions.get(child).y, 0) /
          children.length
        : row++ * 27;
      positions.set(id, {
        x: ((node.depth * 1000) / this.data.root_age_years) * 950,
        y,
      });
    };
    layout(0);
    this.positions = positions;
    this.height = Math.max(100, row * 27);
    this.svg.replaceChildren();
    this.group = element("g");
    this.svg.append(this.group);
    const axis = element("g", { class: "axis" });
    const last = this.data.entry.years,
      first = this.data.root_age_years;
    for (let i = 0; i <= 5; i++) {
      const age = first + ((last - first) * i) / 5;
      const x = ((first - age) / first) * 950;
      axis.append(
        element("line", { x1: x, x2: x, y1: -25, y2: this.height + 10 }),
        element(
          "text",
          { x, y: -33, "text-anchor": "middle" },
          Math.round(age).toLocaleString("es") + " a",
        ),
      );
    }
    this.group.append(axis);
    visible.forEach((node) => {
      const p = positions.get(node.id);
      if (node.parent !== null) {
        const parent = positions.get(node.parent);
        this.group.append(
          element("path", {
            class: "branch",
            d: `M${parent.x},${parent.y}V${p.y}H${p.x}`,
          }),
        );
      }
      const group = element("g", {
        class: `node ${node.children.length ? "" : "leaf"} ${node.id === this.selected ? "selected" : ""}`,
        transform: `translate(${p.x},${p.y})`,
        tabindex: "0",
        role: "button",
        "aria-label": `${node.label}, ${node.leaves} puntas`,
      });
      group.append(element("circle", { r: node.children.length ? 4.5 : 3.5 }));
      const collapsed = node.children.length && !this.open.has(node.id);
      if (collapsed || !node.children.length || node.id === this.selected) {
        group.append(
          element(
            "text",
            { x: 9, y: 4 },
            collapsed ? `${node.label} · ${node.leaves} puntas +` : node.label,
          ),
        );
      }
      group.append(
        element(
          "title",
          {},
          `${node.source_label || node.label}\n${Math.round(node.age_years).toLocaleString("es")} años · ${node.leaves} puntas`,
        ),
      );
      group.addEventListener("click", () => {
        this.selected = node.id;
        this.render();
        this.onSelect(node.id);
      });
      group.addEventListener("dblclick", (event) => {
        event.preventDefault();
        this.toggle(node.id);
      });
      group.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          this.selected = node.id;
          this.render();
          this.onSelect(node.id);
        }
      });
      this.group.append(group);
    });
    this.transform();
  }
  transform() {
    this.group?.setAttribute(
      "transform",
      `translate(${this.tx},${this.ty}) scale(${this.zoom})`,
    );
    // Mantiene letras legibles al encuadrar; las ramas conservan su escala relativa.
    this.group?.querySelectorAll("text").forEach((text) => {
      text.style.fontSize = `${11 / Math.min(1, this.zoom)}px`;
    });
  }
  fit() {
    this.zoom = Math.min(
      (this.svg.clientWidth - 65) / 1200,
      (this.svg.clientHeight - 85) / this.height,
    );
    this.tx = 35;
    this.ty = 55;
    this.transform();
  }
  scale(factor, x = this.svg.clientWidth / 2, y = this.svg.clientHeight / 2) {
    const zoom = Math.max(0.02, Math.min(20, this.zoom * factor));
    const ratio = zoom / this.zoom;
    this.tx = x - (x - this.tx) * ratio;
    this.ty = y - (y - this.ty) * ratio;
    this.zoom = zoom;
    this.transform();
  }
  center(id) {
    const p = this.positions.get(id);
    if (!p) return;
    this.zoom = Math.max(this.zoom, 0.7);
    this.tx = this.svg.clientWidth * 0.38 - p.x * this.zoom;
    this.ty = this.svg.clientHeight / 2 - p.y * this.zoom;
    this.transform();
  }
}
