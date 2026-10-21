/* Self-contained expression parser and the four models in the paper's methodology. */
(function (root) {
  "use strict";
  function parseExpression(source) {
    source = source.trim().replace(/^f\s*\(\s*x\s*\)\s*=\s*/i, "").replaceAll("−", "-").replaceAll("**", "^");
    if (!source || source.length > 240) throw new Error("Enter a function of x using at most 240 characters.");
    const tokens = [];
    let pos = 0;
    while (pos < source.length) {
      const part = source.slice(pos);
      const match = /^(?:\s+|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[a-zA-Z]+|[+\-*/^()])/.exec(part);
      if (!match) throw new Error(`Unexpected character “${source[pos]}”. Use numbers, x, and the listed math operations.`);
      pos += match[0].length;
      if (!/^\s+$/.test(match[0])) tokens.push(match[0].toLowerCase());
    }
    if (tokens.length > 120) throw new Error("Please use a shorter expression.");
    let i = 0, depth = 0;
    const functions = new Set(["sin", "cos", "tan", "exp", "log", "sqrt"]);
    function primary() {
      if (++depth > 32) throw new Error("Please use fewer nested parentheses.");
      const token = tokens[i++];
      let node;
      if (token === "(") {
        node = sum();
        if (tokens[i++] !== ")") throw new Error("Close each opening parenthesis.");
      } else if (token === "x") node = { op: "x" };
      else if (token === "pi" || token === "e") node = { op: "number", value: token === "pi" ? Math.PI : Math.E };
      else if (token && /^(?:\d|\.)/.test(token)) node = { op: "number", value: Number(token) };
      else if (functions.has(token)) {
        if (tokens[i++] !== "(") throw new Error(`Write ${token} with parentheses, for example ${token}(x).`);
        node = { op: token, left: sum() };
        if (tokens[i++] !== ")") throw new Error(`Close the parentheses for ${token}.`);
      } else throw new Error(`Expected a number, x, or a supported function${token ? `; found “${token}”` : ""}.`);
      depth--;
      return node;
    }
    function power() {
      const left = primary();
      return tokens[i] === "^" ? (i++, { op: "^", left, right: unary() }) : left;
    }
    function unary() {
      if (tokens[i] === "+" || tokens[i] === "-") {
        const op = tokens[i++];
        return { op: "sign", value: op === "-" ? -1 : 1, left: unary() };
      }
      return power();
    }
    function product() {
      let left = unary();
      while (tokens[i] === "*" || tokens[i] === "/") left = { op: tokens[i++], left, right: unary() };
      return left;
    }
    function sum() {
      let left = product();
      while (tokens[i] === "+" || tokens[i] === "-") left = { op: tokens[i++], left, right: product() };
      return left;
    }
    const ast = sum();
    if (i !== tokens.length) throw new Error("Check the expression. Use * for multiplication, for example 2*x.");
    function evaluate(node, x, derivative) {
      if (node.op === "x") return [x, derivative ? 1 : 0];
      if (node.op === "number") return [node.value, 0];
      const [u, du] = evaluate(node.left, x, derivative);
      const [v, dv] = node.right ? evaluate(node.right, x, derivative) : [0, 0];
      let value, slope;
      switch (node.op) {
        case "+": value = u + v; slope = du + dv; break;
        case "-": value = u - v; slope = du - dv; break;
        case "*": value = u * v; slope = du * v + u * dv; break;
        case "/": value = u / v; slope = (du * v - u * dv) / (v * v); break;
        case "^":
          value = Math.pow(u, v);
          if (!derivative || (du === 0 && dv === 0)) slope = 0;
          else if (dv === 0) slope = v === 0 ? 0 : v * Math.pow(u, v - 1) * du;
          else slope = u > 0 ? value * (dv * Math.log(u) + v * du / u) : NaN;
          break;
        case "sign": value = node.value * u; slope = node.value * du; break;
        case "sin": value = Math.sin(u); slope = Math.cos(u) * du; break;
        case "cos": value = Math.cos(u); slope = -Math.sin(u) * du; break;
        case "tan":
          if (Math.abs(Math.cos(u)) < 1e-10) throw new Error("tan is undefined at this point. Choose another starting point or probe distance.");
          value = Math.tan(u); slope = du / Math.cos(u) ** 2; break;
        case "exp": value = Math.exp(u); slope = value * du; break;
        case "log": value = u > 0 ? Math.log(u) : NaN; slope = du / u; break;
        case "sqrt": value = Math.sqrt(u); slope = du === 0 ? 0 : du / (2 * value); break;
      }
      if (!Number.isFinite(value) || (derivative && !Number.isFinite(slope))) throw new Error("The function or its slope is not finite here. Check the function’s domain and try another x or smaller settings.");
      return [value, derivative ? slope : 0];
    }
    return { at: x => evaluate(ast, x, false)[0], slope: x => evaluate(ast, x, true)[1] };
  }
  function roots(a, b, c) {
    const scale = Math.max(Math.abs(a), Math.abs(b), Math.abs(c));
    if (!scale || !Number.isFinite(scale)) return [];
    a /= scale; b /= scale; c /= scale;
    if (Math.abs(a) < 1e-14) return Math.abs(b) < 1e-14 ? [] : [-c / b];
    const disc = b * b - 4 * a * c;
    if (disc < 0) return [];
    const q = -.5 * (b + (b >= 0 ? 1 : -1) * Math.sqrt(disc));
    return q === 0 ? [-b / (2 * a)] : [q / a, c / q];
  }
  function fitModels(f0, a, b, c, radius, goal) {
    const t2 = s => f0 + a * s + .5 * b * s * s;
    const models = [
      { name: "Taylor2", color: "#326b97", fn: t2, candidates: b ? [-a / b] : [], note: "Quadratic fit" },
      { name: "Taylor3", color: "#4c7d48", fn: s => t2(s) + c * s ** 3 / 6, candidates: roots(c / 2, b, a), note: "Cubic fit" }
    ];
    function rational(name, color, q, rho, invalid) {
      const pole = q !== 0 && -1 / q >= 0 && -1 / q <= radius;
      const fallback = invalid || !Number.isFinite(q) || !Number.isFinite(rho) || pole || 1 + q * radius < 1e-9;
      models.push({ name, color, fallback, note: fallback ? `${pole ? "Pole in interval" : "Undefined or unstable fit"} · uses Taylor2` : name === "Padé2" ? "Rational fit · [1/1]" : "Rational fit · [2/1]",
        fn: fallback ? t2 : s => f0 + (a * s + rho * s * s) / (1 + q * s),
        candidates: fallback ? (b ? [-a / b] : []) : roots(rho * q, 2 * rho, a) });
    }
    rational("Padé2", "#b16b28", -b / (2 * a), 0, Math.abs(a) < 1e-12);
    const q3 = -c / (3 * b);
    rational("Padé3", "#885a93", q3, b / 2 + a * q3, Math.abs(b) < 1e-12);
    for (const model of models) {
      model.step = 0; model.predicted = model.fn(0);
      for (const s of [radius, ...model.candidates]) {
        if (!Number.isFinite(s) || s < 0 || s > radius) continue;
        const value = model.fn(s);
        if (!Number.isFinite(value)) throw new Error("A fit exceeded the numeric range. Try a smaller learning rate or search multiplier.");
        if ((goal === "min" && value < model.predicted) || (goal === "max" && value > model.predicted)) {
          model.step = s; model.predicted = value;
        }
      }
    }
    return models;
  }
  function compare({ expression, x, lr, beta, epsilon, goal }) {
    if (![x, lr, beta, epsilon].every(Number.isFinite) || Math.abs(x) > 1e6 || lr < 1e-6 || lr > 100 || beta < 1e-6 || beta > 100 || epsilon < 1e-5 || epsilon > 10 || !["min", "max"].includes(goal)) throw new Error("Use finite values within the ranges shown by the input fields.");
    const f = parseExpression(expression), f0 = f.at(x), gradient = f.slope(x);
    if (![f0, gradient].every(Number.isFinite)) throw new Error("The function and slope must be finite at x.");
    const z = (goal === "min" ? -1 : 1) * Math.sign(gradient);
    const usual = lr * Math.abs(gradient), radius = beta * usual;
    if (!Number.isFinite(radius) || radius > 1e6) throw new Error("The search interval is too large to display. Reduce the learning rate or search multiplier.");
    let a = gradient * z, b = 0, c = 0;
    if (z) {
      const plus = f.at(x + epsilon * z), minus = f.at(x - epsilon * z);
      b = (plus - 2 * f0 + minus) / epsilon ** 2;
      c = 3 * (plus - minus - 2 * epsilon * a) / epsilon ** 3;
      if (![a, b, c].every(Number.isFinite)) throw new Error("The probes produced unstable estimates. Try a larger probe distance or another starting point.");
    }
    const models = fitModels(f0, a, b, c, radius, goal);
    for (const model of models) {
      model.x = x + z * model.step;
      try { model.actual = f.at(model.x); } catch { model.actual = null; }
    }
    return { f, f0, gradient, x, z, usual, radius, epsilon, models, goal, a, b, c };
  }
  const api = { parseExpression, roots, fitModels, compare };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ZFOPlayground = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
