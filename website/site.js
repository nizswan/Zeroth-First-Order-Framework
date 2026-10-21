"use strict";

const $ = (id) => document.getElementById(id);


function updateCode() {
  const valid = $("configuration").checkValidity();
  $("copy-code").disabled = !valid;
  if (!valid) {
    $("config-code").textContent = "Choose finite, positive epsilon and beta, and a nonnegative learning rate.";
    return;
  }
  const base = $("base").value;
  const method = $("method").value;
  const inside = {auto: '"auto"', true: "True", false: "False"}[$("interior").value];
  const muon = base === "Muon";
  const taco = base === "TACO";
  const matrixOnly = muon || taco;
  const parameters = matrixOnly ? "params" : "model.parameters()";
  const extra = matrixOnly ? `# ${base} requires matrix-only trainable parameters.\nparams = [p for p in model.parameters() if p.requires_grad]\nif any(p.ndim != 2 for p in params):\n    raise ValueError("This ${base} example needs 2D parameters")\n\n` : "";
  const optimizerCode = taco
    ? `${extra}base = TACO(params, lr=${Number($("lr").value)})\n`
    : `${extra}base = torch.optim.${base}(${parameters},\n                        lr=${Number($("lr").value)}, weight_decay=0)\n`;
  $("config-code").textContent = `import torch\nfrom ZFO import ZFO${taco ? ", TACO" : ""}\n\n# Assumes model, loader, and loss_fn already exist.\n${optimizerCode}zfo = ZFO(\n    base, model=model,\n    epsilon=${Number($("epsilon").value)}, method="${method}",\n    interior_search=${inside}, beta=${Number($("beta").value)},\n)\nscheduler = torch.optim.lr_scheduler.CosineAnnealingLR(\n    base, T_max=len(loader)\n)\n\nfor inputs, targets in loader:\n    loss = zfo.step(\n        lambda: loss_fn(model(inputs), targets)\n    )\n    scheduler.step()`;

}

$("configuration").addEventListener("input", updateCode);
$("configuration").addEventListener("submit", (event) => event.preventDefault());
$("copy-code").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("config-code").textContent);
    $("copy-code").textContent = "Copied ✓";
    $("copy-status").textContent = "Configuration copied to clipboard.";
    setTimeout(() => { $("copy-code").textContent = "Copy code ⧉"; }, 1800);
  } catch {
    $("copy-status").textContent = "Clipboard unavailable. Select and copy the code directly.";
    $("copy-code").textContent = "Select code to copy";
  }
});
updateCode();

// The function playground runs entirely in the browser; expressions are parsed as math.
let functionResult = null;
let highlightedMethod = null;
const numberText = value => value === null || !Number.isFinite(value) ? "Undefined" : value === 0 ? "0" : Math.abs(value) < .0001 || Math.abs(value) >= 10000 ? value.toExponential(3) : Number(value.toPrecision(5)).toString();
const svgNS = "http://www.w3.org/2000/svg";
function svgElement(name, attrs, text) {
  const element = document.createElementNS(svgNS, name);
  for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
  if (text !== undefined) element.textContent = text;
  return element;
}
function drawFunctionChart(result) {
  const width = 900, height = 360, left = 66, right = 24, top = 30, bottom = 55;
  const limit = result.radius || result.epsilon;
  const samples = Array.from({ length: 321 }, (_, index) => limit * index / 320);
  function safeValue(fn, s) { try { const value = fn(s); return Number.isFinite(value) ? value : null; } catch { return null; } }
  const truth = samples.map(s => safeValue(s => result.f.at(result.x + result.z * s), s));
  const values = truth.filter(v => v !== null).concat(result.f0);
  let low = Math.min(...values), high = Math.max(...values);
  const span = high - low || Math.max(1, Math.abs(low) * .1);
  low -= span * .18; high += span * .18;
  const X = s => left + s / limit * (width - left - right);
  const Y = value => top + (high - value) / (high - low) * (height - top - bottom);
  const svg = svgElement("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-labelledby": "fit-plot-title fit-plot-desc" });
  svg.append(svgElement("title", { id: "fit-plot-title" }, "Your function and four local ZFO fits"), svgElement("desc", { id: "fit-plot-desc" }, `Step distance s ranges from 0 to ${numberText(result.radius)}. ${result.goal === "min" ? "Lower" : "Higher"} values are better. Exact selected points and values are listed in the method cards below.`));
  const defs = svgElement("defs", {}), clip = svgElement("clipPath", { id: "fit-plot-clip" });
  clip.append(svgElement("rect", { x: left, y: top, width: width - left - right, height: height - top - bottom })); defs.append(clip); svg.append(defs);
  for (let j = 0; j <= 4; j++) {
    const value = low + (high - low) * j / 4, y = Y(value);
    svg.append(svgElement("line", { x1: left, y1: y, x2: width - right, y2: y, stroke: "#e4e7de" }));
    svg.append(svgElement("text", { x: left - 12, y: y + 4, "text-anchor": "end" }, numberText(value)));
    const s = limit * j / 4;
    svg.append(svgElement("text", { x: X(s), y: height - bottom + 24, "text-anchor": "middle" }, numberText(s)));
  }
  svg.append(svgElement("text", { x: left, y: 15, class: "chart-axis-label" }, "Function value"));
  svg.append(svgElement("text", { x: width / 2, y: height - 8, "text-anchor": "middle", class: "chart-axis-label" }, result.radius ? "Step distance s" : "No step: the starting slope is zero"));
  const lines = svgElement("g", { "clip-path": "url(#fit-plot-clip)" });
  let clipped = false;
  function path(values) {
    let data = "", drawing = false;
    values.forEach((value, index) => {
      // Break non-finite or extreme segments instead of drawing through an asymptote.
      if (value === null || Math.abs((value - low) / (high - low)) > 20) { drawing = false; return; }
      data += `${drawing ? "L" : "M"}${X(samples[index]).toFixed(2)},${Y(value).toFixed(2)} `;
      drawing = true;
    });
    return data;
  }
  for (const model of result.models) {
    const modelValues = samples.map(s => safeValue(model.fn, s));
    clipped ||= modelValues.some(v => v === null || v < low || v > high);
    lines.append(svgElement("path", { d: path(modelValues), fill: "none", stroke: model.color, "stroke-width": highlightedMethod === model.name ? 3.5 : 2, "stroke-dasharray": "7 5", opacity: !highlightedMethod || highlightedMethod === model.name ? .9 : .2 }));
  }
  lines.append(svgElement("path", { d: path(truth), fill: "none", stroke: "#173e37", "stroke-width": 3 }));
  for (const model of result.models) {
    if (model.actual === null) continue;
    const marker = svgElement("circle", { cx: X(model.step), cy: Y(model.actual), r: highlightedMethod === model.name ? 7 : 5, fill: model.color, stroke: "#fffef9", "stroke-width": 2, opacity: !highlightedMethod || highlightedMethod === model.name ? 1 : .2 });
    marker.append(svgElement("title", {}, `${model.name}: x = ${numberText(model.x)}, f(x) = ${numberText(model.actual)}`));
    lines.append(marker);
  }
  svg.append(lines);
  $("function-chart").replaceChildren(svg);
  const messages = [];
  if (!result.z) messages.push("The starting slope is zero, so there is no gradient direction and all methods stay at x. Try another starting point to see the fits.");
  if (truth.some(v => v === null)) messages.push("The function is undefined at some sampled points; gaps are left in the curve.");
  if (clipped) messages.push("Some fits extend beyond the chart’s vertical range; their selected values are still shown below.");
  if (result.models.some(m => m.fallback)) messages.push("A Padé fit needed a Taylor2 fallback. Its card explains why.");
  if (result.models.some(m => m.actual === null)) messages.push("A selected point is outside the function’s domain; its actual value is marked Undefined.");
  if (result.models.some(m => m.actual !== null && (result.goal === "min" ? m.actual > result.f0 : m.actual < result.f0))) messages.push("Some selected steps make the actual function worse. The model prediction is only a local estimate.");
  $("function-notes").textContent = messages.join(" ");
  $("function-notes").hidden = messages.length === 0;
}
function renderFunctionResult(result) {
  const summary = $("function-summary"); summary.replaceChildren();
  for (const [label, value] of [["Starting value · f(x)", result.f0], ["Starting slope · f′(x)", result.gradient], ["Search limit · R", result.radius]]) {
    const cell = document.createElement("div"), title = document.createElement("span"), number = document.createElement("strong");
    title.textContent = label; number.textContent = numberText(value); cell.append(title, number); summary.append(cell);
  }
  $("function-chart-caption").textContent = result.z ? `Starting at x = ${numberText(result.x)}, move ${result.z > 0 ? "right" : "left"}: φ(s) = f(${numberText(result.x)} ${result.z > 0 ? "+" : "−"} s). Search from s = 0 to ${numberText(result.radius)}.` : "The starting point is stationary. Try a different x to explore a direction.";
  $("function-legend").replaceChildren();
  for (const model of [{ name: "Your function", color: "#173e37" }, ...result.models]) {
    const item = document.createElement("span"), swatch = document.createElement("i"); swatch.style.setProperty("--fit-color", model.color); item.append(swatch, document.createTextNode(model.name)); $("function-legend").append(item);
  }
  $("function-methods").replaceChildren();
  for (const model of result.models) {
    const card = document.createElement("button"); card.type = "button"; card.className = "method-card";
    card.style.setProperty("--fit-color", model.color); card.setAttribute("aria-pressed", "false"); card.setAttribute("aria-label", `Highlight ${model.name} fit`);
    const title = document.createElement("h4"), note = document.createElement("p"), list = document.createElement("dl");
    title.textContent = model.name; note.textContent = model.note;
    for (const [label, value] of [["Step s", model.step], ["Next x", model.x], ["Model predicts", model.predicted], ["Actual f(x)", model.actual]]) {
      const row = document.createElement("div"), term = document.createElement("dt"), definition = document.createElement("dd");
      term.textContent = label; definition.textContent = numberText(value);
      if (label === "Actual f(x)") row.className = "actual-result";
      row.append(term, definition); list.append(row);
    }
    card.append(title, note, list);
    card.addEventListener("click", () => {
      highlightedMethod = highlightedMethod === model.name ? null : model.name;
      for (const button of $("function-methods").children) button.setAttribute("aria-pressed", String(button.getAttribute("aria-label") === `Highlight ${highlightedMethod} fit`));
      drawFunctionChart(result);
    });
    $("function-methods").append(card);
  }
  drawFunctionChart(result);
}
function compareFunction() {
  if (!$("function-form").reportValidity()) return;
  try {
    functionResult = ZFOPlayground.compare({ expression: $("function-expression").value, x: Number($("function-x").value), lr: Number($("function-lr").value), beta: Number($("function-beta").value), epsilon: Number($("function-epsilon").value), goal: $("function-goal").value });
    highlightedMethod = null;
    renderFunctionResult(functionResult);
    $("function-error").hidden = true; $("function-results").hidden = false;
  } catch (error) {
    $("function-error").textContent = error.message; $("function-error").hidden = false; $("function-results").hidden = true;
  }
}
$("function-form").addEventListener("submit", event => { event.preventDefault(); compareFunction(); });
$("function-form").addEventListener("input", () => {
  $("function-results").hidden = true;
  $("function-error").hidden = true;
});
for (const button of document.querySelectorAll("[data-example]")) button.addEventListener("click", () => {
  const examples = { wave: ["x^2 + 0.4*sin(3*x)", 1.2], quadratic: ["(x - 1)^2", 3], exponential: ["exp(x) - 2*x", 1.5] };
  const [expression, x] = examples[button.dataset.example];
  $("function-expression").value = expression; $("function-x").value = x;
  $("function-lr").value = .2; $("function-beta").value = 2; $("function-epsilon").value = .1; $("function-goal").value = "min";
  compareFunction();
});
compareFunction();
