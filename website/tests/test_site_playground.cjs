const { test } = require('node:test');
const assert = require('node:assert/strict');
const { parseExpression, compare, fitModels } = require('../playground-core.js');
const close = (a, b, tolerance = 1e-8) => assert.ok(Math.abs(a - b) <= tolerance * Math.max(1, Math.abs(b)), `${a} != ${b}`);
const settings = { expression: '(x - 1)^2', x: 3, lr: .5, beta: 2, epsilon: .1, goal: 'min' };

test('math parser has conventional precedence and supports the requested functions', () => {
  close(parseExpression('-2^2 + 2^3^2').at(0), 508);
  close(parseExpression('2^-2').at(0), .25);
  close(parseExpression('f(x) = sin(x)+cos(x)+tan(x)+exp(x)').at(.3), Math.sin(.3)+Math.cos(.3)+Math.tan(.3)+Math.exp(.3));
  close(parseExpression('1e-3*x + pi - e').at(2), .002 + Math.PI - Math.E);
});
test('automatic derivatives handle composite functions, powers and zero', () => {
  close(parseExpression('sin(x^2) + exp(2*x) / (1+x)').slope(.4), .8*Math.cos(.16) + Math.exp(.8)*(2/1.4 - 1/1.4**2));
  close(parseExpression('x^2').slope(0), 0);
  close(parseExpression('x^0').slope(0), 0);
  close(parseExpression('x^x').slope(2), 4*(Math.log(2)+1));
});
test('only math syntax is accepted and invalid domains are rejected', () => {
  for (const expression of ['alert(1)', 'window.location', 'x;1', 'x.constructor', '<img>', '2x', 'sin(', 'sin x', 'x'.repeat(250)]) assert.throws(() => parseExpression(expression));
  assert.throws(() => parseExpression('tan(x)').at(Math.PI/2));
  assert.throws(() => parseExpression('sqrt(x)').slope(0));
  assert.throws(() => compare({ ...settings, expression: 'log(x)', x: .01, epsilon: .1 }));
});
test('quadratic example chooses the analytic minimum and reports actual values', () => {
  const r = compare(settings);
  close(r.f0, 4); close(r.gradient, 4); close(r.z, -1); close(r.radius, 4);
  for (const model of r.models.filter(m => m.name !== 'Padé2')) { close(model.x, 1, 1e-6); close(model.actual, 0, 1e-6); }
});
test('cubic model exactly recovers a cubic from a slope and two probes', () => {
  const r = compare({ ...settings, expression: 'x^3 - 3*x', x: .5, lr: .2, beta: 2 });
  const model = r.models.find(m => m.name === 'Taylor3');
  for (const s of [0, .05, .2, .6]) close(model.fn(s), r.f.at(r.x + r.z*s));
  close(model.x, 1);
});
test('rational fits match known rational functions', () => {
  const p2 = fitModels(2, 3, -3, 4.5, 2, 'max')[2];
  for (const s of [0, .1, 1, 2]) close(p2.fn(s), 2+3*s/(1+.5*s));
  const p3 = fitModels(2, 3, 8, -12, 2, 'max')[3];
  for (const s of [0, .1, 1, 2]) close(p3.fn(s), 2+3*s+4*s*s/(1+.5*s));
});
test('Padé poles fall back to the quadratic model', () => {
  const models = fitModels(1, 1, 2, 6, 2, 'min');
  for (const model of models.slice(2)) { assert.equal(model.fallback, true); close(model.step, models[0].step); close(model.fn(.3), models[0].fn(.3)); }
});
test('stationary points and endpoints agree with dense reference searches', () => {
  for (const goal of ['min', 'max']) for (const [a,b,c] of [[-2,3,1],[1,-2,.7],[-1,4,-3],[3,0,0]]) {
    for (const model of fitModels(2, a, b, c, 2, goal)) {
      const values = Array.from({length:10001}, (_,i) => model.fn(i/5000));
      const reference = goal === 'min' ? Math.min(...values) : Math.max(...values);
      close(model.predicted, reference, 1e-6);
      assert.ok(model.step >= 0 && model.step <= 2);
    }
  }
});
test('zero slopes stay put, maximization reverses direction, invalid settings fail', () => {
  const stationary = compare({...settings, expression: 'x^2', x: 0});
  for (const m of stationary.models) { close(m.step, 0); close(m.x, 0); close(m.actual, 0); }
  const max = compare({...settings, expression: '-(x-1)^2', goal: 'max'});
  close(max.z, -1); close(max.models[0].x, 1);
  for (const overrides of [{lr: 0}, {epsilon: 0}, {x: NaN}, {beta: -1}, {goal: 'oops'}]) assert.throws(() => compare({...settings, ...overrides}));
});
