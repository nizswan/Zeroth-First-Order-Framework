"""Build the website's vector illustration, inspired by the paper's Figure 1.

The surface and 2D slice share a synthetic loss. The cubic model uses the
value/slope at zero and two symmetric probes, as in the paper illustration.
Run from any directory; no third-party dependencies are required.
"""
from pathlib import Path
from math import sqrt
import xml.etree.ElementTree as ET

OUT = Path(__file__).resolve().parents[1] / 'docs/assets/hero-direction-step.svg'
# Reuse the paper illustration's physical ray coordinate s internally.
# Website labels use the displacement multiplier a = s / BASE, so the
# base optimizer is exactly a=1 and the endpoint is beta=BOUND/BASE.
EPS, BASE, BOUND = .14, .42, 2.05

def loss(s):
    return 2.7 - 2*s + .67*s*s + .14*s**3 + .08*s**4

def surface(s, v):
    return loss(s) + .95*v*v + .32*v*s + .25*v

# Central probes recover the quadratic/cubic coefficients of the local model.
c2 = (loss(EPS) + loss(-EPS) - 2*loss(0)) / (2*EPS**2)
c3 = (loss(EPS) - loss(-EPS) + 4*EPS) / (2*EPS**3)

def model(s):
    return loss(0) - 2*s + c2*s*s + c3*s**3

SELECTED = (-2*c2 + sqrt((2*c2)**2 + 24*c3)) / (6*c3)
assert 0 < SELECTED < BOUND
assert all(abs(model(s)-loss(s)) < 1e-10 for s in (-EPS, 0, EPS))
assert abs(-2 + 2*c2*SELECTED + 3*c3*SELECTED**2) < 1e-10

def project(s, v, z):
    return (95 + 102*s + 64*v, 248 + 25*s - 39*v - 34*z)

def plot(s, z):
    return (438 + 91*s, 308 - 53*z)

def seq(a, b, n=80):
    return [a+(b-a)*i/n for i in range(n+1)]

def points(ps):
    return ' '.join(f'{x:.2f},{y:.2f}' for x,y in ps)

def path(ps, **attrs):
    return '<path d="M' + ' L'.join(f'{x:.2f},{y:.2f}' for x,y in ps) + '" ' + ' '.join(f'{k.replace("_", "-")}="{v}"' for k,v in attrs.items()) + '/>'

def circle(p, r, fill, **attrs):
    return f'<circle cx="{p[0]:.2f}" cy="{p[1]:.2f}" r="{r}" fill="{fill}" ' + ' '.join(f'{k.replace("_", "-")}="{v}"' for k,v in attrs.items()) + '/>'

def diamond(p):
    x,y=p
    return f'<path d="M{x},{y-5}l5 5-5 5-5-5Z" fill="#fafbf5" stroke="#7e8983" stroke-width="1.8"/>'

parts = ['''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 680 390" role="img" aria-labelledby="title desc">
<title id="title">From an objective landscape to one step along a direction</title>
<desc id="desc">A translucent three-dimensional loss surface is cut along the optimizer direction. Its green cross-section becomes the same green loss curve in the two-dimensional plot. Two rust-colored probes near the current point fit a dashed local model. A gray diamond marks the base optimizer step; a purple point marks the ZFO model-selected step. Heights on the surface show actual loss, which can differ from the model. Synthetic illustration.</desc>
<defs>
  <linearGradient id="surface" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#87b9af" stop-opacity=".65"/><stop offset="1" stop-color="#d4e5d9" stop-opacity=".3"/></linearGradient>
  <linearGradient id="slice" x1="0" y1="0" x2="0" y2="1"><stop stop-color="#387f6e" stop-opacity=".15"/><stop offset="1" stop-color="#387f6e" stop-opacity=".02"/></linearGradient>
  <marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 7 3.5 0 7" fill="none" stroke="#648379" stroke-width="1.2"/></marker>
</defs>
<style>text{font-family:Arial,sans-serif;fill:#29483e;font-size:15px}.heading{font-size:17px;font-weight:600}.muted{fill:#64776c;font-size:13px}.math{font-family:Georgia,serif;font-style:italic}.purple{fill:#7954a1}.rust{fill:#ad542f}</style>
<g id="landscape">
<text class="heading" x="28" y="31">01 / Trust the direction</text>
<text class="muted" x="28" y="53">The full objective landscape</text>
''']
# A projected parameter plane and nested contours establish the 3D geometry.
floor = [project(s,v,0) for s,v in [(-.28,-.92),(2.1,-.92),(2.1,.92),(-.28,.92)]]
parts.append(f'<polygon points="{points(floor)}" fill="#e1e9db" fill-opacity=".6" stroke="#c3d3c5"/>')
# Exact level sets of surface(s,v), projected onto parameter space.
for level in (1.24, 1.38, 1.6, 1.9, 2.3):
    branches = [[], []]
    for s in seq(-.28, 2.1, 500):
        linear = .32*s + .25
        disc = linear*linear - 4*.95*(loss(s)-level)
        if disc >= 0:
            for branch, sign in zip(branches, (-1, 1)):
                v = (-linear + sign*sqrt(disc))/(2*.95)
                branch.append(project(s,v,0) if -.92 <= v <= .92 else None)
    # Split at the floor boundary rather than drawing a false closing edge.
    for branch in branches:
        run = []
        for point in branch + [None]:
            if point is not None:
                run.append(point)
            elif run:
                parts.append(path(run,fill='none',stroke='#afc7bb',stroke_width='.8'))
                run = []
# Painter order from the far edge to the near edge, using smooth filled strips.
for va,vb in zip(seq(.92,-.92,27),seq(.92,-.92,27)[1:]):
    poly = [project(s,va,surface(s,va)) for s in seq(-.28,2.1,65)]
    poly += [project(s,vb,surface(s,vb)) for s in seq(2.1,-.28,65)]
    parts.append(f'<polygon points="{points(poly)}" fill="url(#surface)"/>')
for v in seq(-.92,.92,13):
    parts.append(path([project(s,v,surface(s,v)) for s in seq(-.28,2.1)],fill='none',stroke='#689c90',stroke_opacity='.36',stroke_width='.75'))
for s in seq(-.28,2.1,19):
    parts.append(path([project(s,v,surface(s,v)) for v in seq(-.92,.92)],fill='none',stroke='#689c90',stroke_opacity='.3',stroke_width='.75'))
plane=[project(s,0,z) for s,z in [(-.28,0),(2.1,0),(2.1,4.4),(-.28,4.4)]]
parts.append(f'<polygon points="{points(plane)}" fill="url(#slice)" stroke="#438873" stroke-opacity=".45"/>')
parts.append(path([project(s,0,loss(s)) for s in seq(-.28,2.1)],fill='none',stroke='#176453',stroke_width='3'))
parts.append(path([project(-.28,0,0),project(2.22,0,0)],fill='none',stroke='#648379',stroke_width='1.6',marker_end='url(#arrow)'))
for s,color in [(0,'#203f36'),(BASE,'#7e8983'),(SELECTED,'#7954a1')]:
    parts.append(path([project(s,0,0),project(s,0,loss(s))],fill='none',stroke=color,stroke_dasharray='3 4',stroke_opacity='.65'))
    for p in [project(s,0,0),project(s,0,loss(s))]:
        parts.append(diamond(p) if s==BASE else circle(p,4.5,color,stroke='#f4f3ed',stroke_width='1.3'))
for s in (-EPS,EPS):
    parts.append(circle(project(s,0,loss(s)),3.5,'#b65f37',stroke='#f4f3ed',stroke_width='1'))
parts += ['''<text x="73" y="272" class="math">θ</text>
<text x="119" y="292" class="muted">Base</text>
<text x="203" y="310" class="purple">ZFO</text>
<text x="320" y="314" class="math">d</text>
</g><g id="connector">
<path d="M349 173 C372 173 374 173 398 173" fill="none" stroke="#648379" stroke-width="1.8" marker-end="url(#arrow)"/>
<text x="374" y="154" text-anchor="middle" class="muted">slice</text>
</g>
''']
# Unfold exactly the same cross-section; the model has its own selected height.
parts.append('''<g id="line-plot">
<text class="heading" x="405" y="31">02 / Search the step</text>
<text class="muted" x="405" y="53">One slice. One local model.</text>''')
parts.append('<rect x="438" y="111" width="186.55" height="197" rx="3" fill="#7954a1" fill-opacity=".045"/>')
parts.append('<text x="531" y="104" text-anchor="middle" class="muted">Search interval [0, β]</text>')
parts.append(path([(415,121),(415,308),(643,308)],fill='none',stroke='#9db0a3',stroke_width='1'))
parts.append('<text x="408" y="112" class="muted">Loss</text>')
for s in [0,BASE,SELECTED,BOUND]:
    x,y=plot(s,model(s) if s==SELECTED else loss(s))
    parts.append(path([(x,y),(x,308)],fill='none',stroke='#7954a1' if s==SELECTED else '#a4b4a7',stroke_dasharray='3 4',stroke_width='1'))
parts.append(path([plot(s,loss(s)) for s in seq(-.22,2.1)],fill='none',stroke='#176453',stroke_width='3'))
parts.append(path([plot(s,model(s)) for s in seq(-.22,2.1)],fill='none',stroke='#7954a1',stroke_width='2.2',stroke_dasharray='7 5'))
parts.append(circle(plot(0,loss(0)),4,'#203f36'))
parts.append(diamond(plot(BASE,loss(BASE))))
for s in (-EPS,EPS):
    parts.append(circle(plot(s,loss(s)),3.5,'#b65f37',stroke='#f4f3ed',stroke_width='1'))
x,y=plot(SELECTED,model(SELECTED))
parts.append(circle((x,y),10,'#7954a1',fill_opacity='.12'))
parts.append(circle((x,y),4.5,'#7954a1',stroke='#fffef9',stroke_width='1.5'))
parts += [f'<text x="{x}" y="{y+29}" text-anchor="middle" class="purple">ZFO</text>',
'''<text x="471" y="190" class="muted">Base</text>
<path d="M427 140 425 155 M447 140 453 176" fill="none" stroke="#b65f37" stroke-width="1"/>
<text x="439" y="132" text-anchor="middle" class="rust">±ε</text>
<text x="438" y="329" text-anchor="middle" class="math">0</text>
<text x="625" y="329" text-anchor="middle" class="math">β</text>
<text x="531" y="351" text-anchor="middle">Step multiplier a</text>
</g><g id="legend">
<path d="M28 364H49" stroke="#176453" stroke-width="3"/>
<text x="57" y="369" class="muted">R<tspan baseline-shift="super" font-size="9">n</tspan> Objective</text>
<path d="M168 364H191" stroke="#7954a1" stroke-width="2.2" stroke-dasharray="6 4"/>
<text x="199" y="369" class="muted">Local model</text>
<circle cx="301" cy="364" r="3.5" fill="#b65f37"/>
<text x="312" y="369" class="muted">Probes</text>
</g></svg>''']
OUT.write_text('\n'.join(parts))
print(f'Wrote {OUT.name}: {OUT.stat().st_size:,} bytes; model-selected a = {SELECTED/BASE:.6f}')

# Stack the same vector panels on phones, preserving readable label sizes.
ET.register_namespace('', 'http://www.w3.org/2000/svg')
ns = {'svg': 'http://www.w3.org/2000/svg'}
mobile = ET.fromstring(OUT.read_text())
mobile.set('viewBox', '0 0 380 740')
mobile.find("svg:g[@id='landscape']", ns).set('transform', 'translate(5 0) scale(.98)')
mobile.find("svg:g[@id='line-plot']", ns).set('transform', 'translate(-350 337)')
mobile.find("svg:g[@id='legend']", ns).set('transform', 'translate(0 352)')
connector = mobile.find("svg:g[@id='connector']", ns)
connector.clear()
connector.append(ET.fromstring('''<path xmlns="http://www.w3.org/2000/svg" d="M190 326V350" fill="none" stroke="#648379" stroke-width="1.8" marker-end="url(#arrow)"/>'''))
connector.append(ET.fromstring('''<text xmlns="http://www.w3.org/2000/svg" x="206" y="345" class="muted">slice</text>'''))
OUT.with_name('hero-direction-step-mobile.svg').write_text(ET.tostring(mobile, encoding='unicode'))
