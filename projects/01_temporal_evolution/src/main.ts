import * as THREE from 'three'
import './style.css'
import { HarmonicEngine, type TimelineFrame } from './audio'
import { LearningMachine } from './learningMachine'

type Theme = { id: string; name: string; color: string }
type Strip = { id: string; name: string; source_short: string; color: string }
type Paper = {
  pmid: string; year: number; month: number; title: string; journal: string; theme: string
  secondary_theme: string; xyz: number[]; density: number; strips: Record<string, number>; cluster?: number
  form_clusters?: Record<string, number>
}
type SemanticCluster = {
  id: number; label: string; theme: string; color: string; vocabulary: string[]; emergence_year: number
  peak_year: number; size: number; centroid: number[]; spread: number; coherence: number; growth: number[]; form?: string
}
type Corpus = {
  strips: Strip[]; themes: Theme[]; timeline: TimelineFrame[]; papers: Paper[]
  clusters?: SemanticCluster[]; form_clusters?: Record<string, SemanticCluster[]>
}
type TemplateEntry = {
  id: string; kind: string; pointCount?: number; pointScale?: number; points?: string; edges?: string; edgeCount?: number
}
type Atlas = { version: number; entries: TemplateEntry[] }
type GeometryData = { points: THREE.Vector3[]; edges: [number, number][] }

declare global {
  var __NEURO_CORPUS__: Corpus | undefined
  var __NEURO_ATLAS__: Atlas | undefined
}

const $ = <T extends HTMLElement>(selector: string): T => {
  const element = document.querySelector<T>(selector)
  if (!element) throw new Error(`Missing ${selector}`)
  return element
}

const canvas = $('#field') as HTMLCanvasElement
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' })
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2.5))
renderer.setSize(innerWidth, innerHeight, false)
renderer.outputColorSpace = THREE.SRGBColorSpace
renderer.setClearColor(0x020609, 1)

const scene = new THREE.Scene()
scene.fog = new THREE.FogExp2(0x020609, .024)
const camera = new THREE.OrthographicCamera(-8.2, 8.2, 4.6, -4.6, -8, 8)
camera.position.z = 5
const focusScene = new THREE.Scene()
const focusCamera = new THREE.PerspectiveCamera(43, innerWidth / innerHeight, .035, 80)
focusCamera.position.set(0, 0, 4.2)
const focusRoot = new THREE.Group()
focusScene.add(new THREE.Mesh(
  new THREE.PlaneGeometry(40, 24),
  new THREE.MeshBasicMaterial({ color: 0x010507, transparent: true, opacity: .90, depthWrite: false }),
))
focusScene.children[0].position.z = -4
focusScene.add(focusRoot)

const ROW_TOP = 2.82
const ROW_STEP = .78
const CHAMBER_X = 5.18
const STREAM_START = -5.72
const STREAM_END = 3.05
const CODA_SECONDS = 12
const audio = new HarmonicEngine()
const clock = new THREE.Clock()
const uniforms = {
  uTime: { value: 0 },
  uPulse: { value: 0 },
  uPixelRatio: { value: renderer.getPixelRatio() },
  uFire: { value: 0 },
}
let corpus: Corpus
let atlas: Atlas
let learningMachine: LearningMachine
let started = false
let playing = true
let progress = 0
let speed = 1
let clean = false
let cumulative: number[] = []
let lastCardKey = -1
let artworkElapsed = 0
let codaElapsed = 0
let focusedLane = -1
let focusDistance = 4.2
let focusTargetDistance = 4.2
let dragging = false
let lastFormation = -1
let lastAxisYear = -1
let focusHalos: THREE.LineSegments[] = []
let focusClusterIds = new Set<number>()
let focusPaperCloud: THREE.Points | undefined
let focusPaperMetadata: Paper[] = []
let focusBirths: number[] = []
let focusLineMaterial: THREE.LineBasicMaterial | undefined
const activePointers = new Map<number, { x: number; y: number }>()
let pinchDistance = 0
const destinations: Record<string, THREE.Vector3[]> = {}
const focusGeometries: Record<string, GeometryData> = {}
const papersByYear = new Map<number, Paper[]>()
const raycaster = new THREE.Raycaster()
const pointerNdc = new THREE.Vector2()
raycaster.params.Points = { threshold: .055 }

const LANDMARKS = [
  { year: 1976, label: 'THE TEMPORAL FIELD OPENS' },
  { year: 1981, label: 'VISUAL RECEPTIVE FIELDS ENTER THE NOBEL CANON' },
  { year: 1986, label: 'BACKPROPAGATION REVIVES TRAINABLE NEURAL NETWORKS' },
  { year: 1991, label: 'HUMAN FUNCTIONAL MRI SIGNALS EMERGE' },
  { year: 1998, label: 'ADULT HUMAN HIPPOCAMPAL NEUROGENESIS REPORTED' },
  { year: 2005, label: 'OPTOGENETIC CONTROL OF NEURAL ACTIVITY' },
  { year: 2010, label: 'THE HUMAN CONNECTOME PROJECT BEGINS' },
  { year: 2011, label: 'THE REPRODUCIBILITY CRISIS BECOMES A METHODOLOGICAL RECKONING' },
  { year: 2013, label: 'BRAIN INITIATIVE + CEREBRAL ORGANOIDS' },
  { year: 2017, label: 'ATTENTION BECOMES A GENERAL COMPUTATIONAL ARCHITECTURE' },
  { year: 2023, label: 'A COMPLETE LARVAL FRUIT-FLY BRAIN CONNECTOME' },
  { year: 2024, label: 'THE ADULT FLY BRAIN CONNECTOME ARRIVES' },
] as const

const PAPER_VERTEX = /* glsl */`
  attribute vec3 aStream;
  attribute vec3 aTarget;
  attribute vec3 aColor;
  attribute float aBirth;
  attribute float aStrength;
  attribute float aSize;
  varying vec3 vColor;
  varying float vAlpha;
  uniform float uTime;
  uniform float uPulse;
  uniform float uPixelRatio;
  float ease(float t){ return t*t*(3.0-2.0*t); }
  void main(){
    float born = smoothstep(aBirth - .002, aBirth + .004, uTime);
    float journey = ease(clamp((uTime - aBirth) / .052, 0.0, 1.0));
    vec3 pos = mix(aStream, aTarget, journey);
    float wave = sin(uPulse * 2.1 + aBirth * 87.0 + position.x * 2.0);
    pos.z += wave * .025 * (1.0 - journey);
    pos.y += sin(journey * 3.14159) * .13 * (1.0 - aStrength);
    vec4 mv = modelViewMatrix * vec4(pos, 1.0);
    gl_Position = projectionMatrix * mv;
    float newborn = exp(-abs(uTime - aBirth) * 115.0);
    gl_PointSize = (aSize + newborn * 8.5) * uPixelRatio;
    vColor = aColor * (1.0 + newborn * 1.8);
    vAlpha = born * (.20 + aStrength * .72 + newborn * .8);
  }
`

const PAPER_FRAGMENT = /* glsl */`
  precision highp float;
  varying vec3 vColor;
  varying float vAlpha;
  void main(){
    vec2 p = gl_PointCoord - .5;
    float r = length(p);
    if(r > .5) discard;
    float core = exp(-r*r*28.0);
    float halo = exp(-r*r*7.0) * .38;
    gl_FragColor = vec4(vColor, (core + halo) * vAlpha);
  }
`

const ARCHIVE_VERTEX = /* glsl */`
  attribute vec3 aColor;
  attribute float aBirth;
  attribute float aStrength;
  attribute float aSize;
  varying vec3 vColor;
  varying float vAlpha;
  uniform float uTime;
  uniform float uPulse;
  uniform float uPixelRatio;
  void main(){
    float born = smoothstep(aBirth - .003, aBirth + .003, uTime);
    float recent = exp(-abs(uTime - aBirth) * 70.0);
    vec3 pos = position;
    pos.y += sin(position.x * 2.7 + uPulse * .18 + aBirth * 91.0) * .008;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
    gl_PointSize = (.78 + aSize * .46 + recent * 2.4) * uPixelRatio;
    vColor = aColor;
    vAlpha = born * (.065 + aStrength * .19 + recent * .32);
  }
`

const ARCHIVE_FRAGMENT = /* glsl */`
  precision highp float;
  varying vec3 vColor;
  varying float vAlpha;
  void main(){
    float r = length(gl_PointCoord - .5);
    if(r > .5) discard;
    gl_FragColor = vec4(vColor, exp(-r*r*13.0) * vAlpha);
  }
`

const FOCUS_VERTEX = /* glsl */`
  attribute vec3 aOrigin;
  attribute vec3 aColor;
  attribute float aBirth;
  attribute float aStrength;
  varying vec3 vColor;
  varying float vAlpha;
  uniform float uTime;
  uniform float uPulse;
  uniform float uPixelRatio;
  uniform float uFire;
  float ease(float t){ return t*t*(3.0-2.0*t); }
  void main(){
    float born = smoothstep(aBirth - .003, aBirth + .004, uTime);
    float journey = ease(clamp((uTime - aBirth) / .036, 0.0, 1.0));
    float arrival = exp(-abs(uTime - (aBirth + .036)) * 155.0);
    vec3 axis = normalize(aOrigin + vec3(.001, .002, .003));
    vec3 tangent = normalize(cross(axis, vec3(.13, .91, .37)));
    vec3 pos = mix(aOrigin, position, journey);
    pos += tangent * sin(journey * 15.707 + aBirth * 173.0) * .28 * (1.0 - journey);
    pos *= 1.0 + sin(uPulse * 1.8 + aBirth * 113.0) * .008 * (journey + uFire);
    vec4 mv = modelViewMatrix * vec4(pos, 1.0);
    gl_Position = projectionMatrix * mv;
    float surge = (1.0 - journey) * 4.5 + arrival * 12.0 + uFire * 1.8;
    gl_PointSize = (3.0 + aStrength * 4.8 + surge) * uPixelRatio * (3.2 / max(.72, -mv.z));
    vColor = aColor * (1.0 + arrival * 2.8 + uFire * .55);
    vAlpha = born * (.32 + aStrength * .62 + arrival * .9);
  }
`

const FOCUS_TEMPLATE_VERTEX = /* glsl */`
  attribute float aGraphPhase;
  attribute float aDegree;
  attribute float aNodeSeed;
  varying float vAlpha;
  uniform float uPulse;
  uniform float uPixelRatio;
  uniform float uFire;
  float random(float n){ return fract(sin(n) * 43758.5453123); }
  void main(){
    float epoch = floor(uPulse * 1.35);
    float recruitment = step(.72 - aDegree * .18, random(epoch * 13.7 + aNodeSeed * 97.1));
    float propagated = pow(max(0.0, sin(uPulse * (3.6 + aNodeSeed * 1.9) - aGraphPhase * 6.28318)), 18.0);
    float firing = uFire * recruitment * propagated * (.55 + aDegree * 1.25);
    vec3 pos = position;
    vec4 mv = modelViewMatrix * vec4(pos, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = (1.45 + firing * 1.05) * uPixelRatio * (3.0 / max(.8, -mv.z));
    vAlpha = .15 + firing * .11;
  }
`

const FOCUS_TEMPLATE_FRAGMENT = /* glsl */`
  precision highp float;
  varying float vAlpha;
  uniform vec3 uColor;
  void main(){
    float r = length(gl_PointCoord - .5);
    if(r > .5) discard;
    float core = exp(-r*r*30.0);
    float halo = exp(-r*r*6.0) * .34;
    gl_FragColor = vec4(uColor * (1.08 + vAlpha * .55), (core + halo) * vAlpha);
  }
`

function decodePoints(entry: TemplateEntry): THREE.Vector3[] {
  if (!entry.points || !entry.pointScale) return []
  const binary = atob(entry.points)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  const values = new Int16Array(bytes.buffer)
  const points: THREE.Vector3[] = []
  for (let i = 0; i < values.length; i += 3) points.push(new THREE.Vector3(values[i] * entry.pointScale, values[i + 1] * entry.pointScale, values[i + 2] * entry.pointScale))
  return points
}

function decodeEdges(entry: TemplateEntry): [number, number][] {
  if (!entry.edges) return []
  const binary = atob(entry.edges)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  const values = new Uint16Array(bytes.buffer)
  const edges: [number, number][] = []
  for (let i = 0; i < values.length; i += 2) edges.push([values[i], values[i + 1]])
  return edges
}

function gradientGeometry(): GeometryData {
  const points: THREE.Vector3[] = [], edges: [number, number][] = []
  const cols = 110, rows = 9
  for (let y = 0; y < rows; y++) for (let x = 0; x < cols; x++) {
    const phase = x / (cols - 1)
    points.push(new THREE.Vector3(phase * 2 - 1, (y / (rows - 1) - .5) * .7 + Math.sin(phase * Math.PI * 5 + y * .7) * .045, Math.cos(phase * 9 + y) * .12))
    if (x) edges.push([y * cols + x - 1, y * cols + x])
  }
  return { points, edges }
}

function fungalGeometry(): GeometryData {
  type Tip = { index: number; direction: THREE.Vector3; generation: number }
  const points: THREE.Vector3[] = [], edges: [number, number][] = []
  let seed = 19071976
  const random = () => { seed = (seed * 48271) % 2147483647; return seed / 2147483647 }
  const roots = 7
  let tips: Tip[] = []
  for (let root = 0; root < roots; root++) {
    const angle = (root / roots) * Math.PI * 2
    const point = new THREE.Vector3(-1.18 + random() * .08, Math.sin(angle) * .09, Math.cos(angle) * .09)
    points.push(point)
    tips.push({
      index: points.length - 1,
      direction: new THREE.Vector3(.48 + random() * .16, Math.sin(angle) * .82, Math.cos(angle) * .68).normalize(),
      generation: 0,
    })
  }

  // Persistent apical growth creates long hyphae; lateral branching and later
  // anastomosis create a reticulate mycelium instead of a diagrammatic tree.
  for (let cycle = 0; cycle < 92 && points.length < 1_050; cycle++) {
    const next: Tip[] = []
    tips.forEach(tip => {
      if (points.length >= 1_050) return
      const parent = points[tip.index]
      const drift = new THREE.Vector3(
        .045 + random() * .055,
        (random() - .5) * .145,
        (random() - .5) * .125,
      )
      const direction = tip.direction.clone().multiplyScalar(.82).add(drift).normalize()
      const step = .041 + random() * .03
      const point = parent.clone().addScaledVector(direction, step)
      point.y += Math.sin(point.x * 7.3 + tip.generation) * .002
      const index = points.push(point) - 1
      edges.push([tip.index, index])
      next.push({ index, direction, generation: tip.generation })

      const branchProbability = cycle < 14 ? .14 : .045 + Math.min(.07, tip.generation * .008)
      if (random() < branchProbability && next.length < 74 && points.length < 1_048) {
        const axis = new THREE.Vector3(random() - .5, random() - .5, random() - .5).normalize()
        const branched = direction.clone().applyAxisAngle(axis, .34 + random() * .58)
        next.push({ index, direction: branched, generation: tip.generation + 1 })
      }
    })
    // Resource competition prunes tips while preserving a dense advancing front.
    tips = next
      .sort((a, b) => hash(`${a.index}:hypha`) - hash(`${b.index}:hypha`))
      .slice(0, 66)
  }

  // Hyphal fusion: connect spatially adjacent, non-lineage nodes to form the
  // characteristic redundant transport loops of a living fungal network.
  for (let index = 28; index < points.length; index += 2) {
    let nearest = -1, distance = .14
    const start = Math.max(0, index - 210)
    for (let candidate = start; candidate < index - 9; candidate++) {
      const d = points[index].distanceTo(points[candidate])
      if (d < distance) { nearest = candidate; distance = d }
    }
    if (nearest >= 0 && random() < .48) edges.push([index, nearest])
  }
  return { points, edges }
}

function orient(point: THREE.Vector3, id: string): THREE.Vector3 {
  if (id === 'elegans') return new THREE.Vector3(point.y, point.x, point.z)
  if (id === 'rodent') return new THREE.Vector3(point.x, point.z, point.y)
  if (id === 'macaque' || id === 'human') return new THREE.Vector3(point.y, point.z, point.x)
  return point.clone()
}

function fitToChamber(data: GeometryData, id: string, laneY: number): GeometryData {
  const rotated = data.points.map(p => orient(p, id))
  const box = new THREE.Box3().setFromPoints(rotated)
  const size = box.getSize(new THREE.Vector3())
  const center = box.getCenter(new THREE.Vector3())
  const scale = Math.min(2.62 / Math.max(size.x, .01), .64 / Math.max(size.y, .01))
  const points = rotated.map(p => new THREE.Vector3(
    CHAMBER_X + (p.x - center.x) * scale,
    laneY + (p.y - center.y) * scale,
    (p.z - center.z) * scale * .46,
  ))
  return { points, edges: data.edges }
}

function normalizeForFocus(data: GeometryData, id: string): GeometryData {
  const rotated = data.points.map(point => orient(point, id))
  const box = new THREE.Box3().setFromPoints(rotated)
  const center = box.getCenter(new THREE.Vector3())
  const size = box.getSize(new THREE.Vector3())
  const scale = 2.45 / Math.max(size.x, size.y, size.z, .001)
  return {
    points: rotated.map(point => point.clone().sub(center).multiplyScalar(scale)),
    edges: data.edges,
  }
}

function graphFiringAttributes(data: GeometryData, formId: string): {
  phase: Float32Array; degree: Float32Array; seed: Float32Array
} {
  const adjacency = Array.from({ length: data.points.length }, () => [] as number[])
  data.edges.forEach(([a, b]) => {
    if (a < adjacency.length && b < adjacency.length) {
      adjacency[a].push(b); adjacency[b].push(a)
    }
  })
  const degreeMaximum = Math.max(1, ...adjacency.map(neighbors => neighbors.length))
  const hubs = adjacency
    .map((neighbors, index) => ({ index, degree: neighbors.length }))
    .sort((a, b) => b.degree - a.degree || hash(`${formId}:${a.index}`) - hash(`${formId}:${b.index}`))
    .slice(0, Math.max(1, Math.min(5, Math.floor(data.points.length / 180))))
    .map(item => item.index)
  const hops = new Int32Array(data.points.length).fill(-1)
  const queue = [...hubs]
  hubs.forEach(index => { hops[index] = 0 })
  for (let cursor = 0; cursor < queue.length; cursor++) {
    const current = queue[cursor]
    adjacency[current].forEach(neighbor => {
      if (hops[neighbor] >= 0) return
      hops[neighbor] = hops[current] + 1
      queue.push(neighbor)
    })
  }
  const maximumHop = Math.max(1, ...hops)
  const phase = new Float32Array(data.points.length)
  const degree = new Float32Array(data.points.length)
  const seed = new Float32Array(data.points.length)
  for (let index = 0; index < data.points.length; index++) {
    const graphHop = hops[index] < 0 ? maximumHop * hash(`${formId}:${index}:island`) : hops[index]
    phase[index] = (graphHop / maximumHop + hash(`${formId}:${index}:phase`) * .11) % 1
    degree[index] = Math.log1p(adjacency[index].length) / Math.log1p(degreeMaximum)
    seed[index] = hash(`${formId}:${index}:spark`)
  }
  return { phase, degree, seed }
}

function addTemplate(data: GeometryData, color: string, id = ''): void {
  const pointArray = new Float32Array(data.points.length * 3)
  data.points.forEach((p, i) => p.toArray(pointArray, i * 3))
  const pg = new THREE.BufferGeometry(); pg.setAttribute('position', new THREE.BufferAttribute(pointArray, 3))
  const emphasis = id === 'human' ? .82 : id === 'ai' ? 1.08 : 1
  const dots = new THREE.Points(pg, new THREE.PointsMaterial({ color, size: 1.22 * emphasis, sizeAttenuation: false, transparent: true, opacity: .235 * emphasis, blending: THREE.AdditiveBlending, depthWrite: false }))
  scene.add(dots)
  const lineArray = new Float32Array(data.edges.length * 6)
  data.edges.forEach(([a, b], i) => { data.points[a]?.toArray(lineArray, i * 6); data.points[b]?.toArray(lineArray, i * 6 + 3) })
  const lg = new THREE.BufferGeometry(); lg.setAttribute('position', new THREE.BufferAttribute(lineArray, 3))
  scene.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color, transparent: true, opacity: .071 * emphasis, blending: THREE.AdditiveBlending, depthWrite: false })))
}

function addBandsAndFilaments(): void {
  const lineMaterial = new THREE.LineBasicMaterial({ color: 0xb9d8d4, transparent: true, opacity: .085 })
  for (let i = 0; i <= 8; i++) {
    const y = ROW_TOP + ROW_STEP * .5 - i * ROW_STEP
    const geometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-8.2, y, -.7), new THREE.Vector3(8.2, y, -.7)])
    scene.add(new THREE.Line(geometry, lineMaterial))
  }
  corpus.strips.forEach((_, lane) => {
    const laneY = ROW_TOP - lane * ROW_STEP
    corpus.themes.forEach((theme, topic) => {
      const points: THREE.Vector3[] = []
      for (let j = 0; j < 30; j++) {
        const t = j / 29
        const x = STREAM_START + t * (CHAMBER_X - 1.32 - STREAM_START)
        const spread = (topic / 11 - .5) * .56
        const y = laneY + spread * (1 - t * .72) + Math.sin(t * Math.PI * 3 + topic * 1.91 + lane) * .016
        points.push(new THREE.Vector3(x, y, -.38 + Math.sin(topic * 2.3) * .07))
      }
      const curve = new THREE.CatmullRomCurve3(points)
      const geometry = new THREE.BufferGeometry().setFromPoints(curve.getPoints(60))
      scene.add(new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: theme.color, transparent: true, opacity: .021, blending: THREE.AdditiveBlending, depthWrite: false })))
    })
  })
  const marker = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(CHAMBER_X - 1.48, -4.2, -.3), new THREE.Vector3(CHAMBER_X - 1.48, 4.2, -.3)])
  scene.add(new THREE.Line(marker, new THREE.LineDashedMaterial({ color: 0xa9d9d2, transparent: true, opacity: .1, dashSize: .04, gapSize: .05 })))
}

function hash(value: string): number {
  let h = 2166136261
  for (let i = 0; i < value.length; i++) h = Math.imul(h ^ value.charCodeAt(i), 16777619)
  return (h >>> 0) / 4294967295
}

function associationThreshold(stripId: string): number {
  return {
    gradient: .20,
    fungal: .30,
    elegans: .34,
    drosophila: .34,
    rodent: .36,
    macaque: .36,
    human: .40,
    ai: .40,
  }[stripId] ?? .30
}

function setupTimeline(): void {
  const yearWeights = corpus.timeline.map(frame => Math.pow(Math.max(1, frame.query_count), .19))
  const total = yearWeights.reduce((a, b) => a + b, 0)
  cumulative = [0]
  let sum = 0
  yearWeights.forEach(weight => { sum += weight / total; cumulative.push(sum) })
  cumulative[cumulative.length - 1] = 1
  corpus.papers.forEach(paper => {
    const list = papersByYear.get(paper.year) ?? []
    list.push(paper); papersByYear.set(paper.year, list)
  })
}

function calendarToProgress(year: number, month = 1): number {
  const index = Math.max(0, Math.min(corpus.timeline.length - 1, year - 1976))
  return cumulative[index] + (cumulative[index + 1] - cumulative[index]) * ((month - 1) / 12)
}

function progressToCalendar(value: number): { year: number; fraction: number; frame: TimelineFrame; index: number } {
  let index = 0
  while (index < cumulative.length - 2 && value >= cumulative[index + 1]) index++
  const fraction = (value - cumulative[index]) / Math.max(1e-7, cumulative[index + 1] - cumulative[index])
  return { year: 1976 + index, fraction, frame: corpus.timeline[index], index }
}

function drawPublicationAxis(activeIndex: number): void {
  const canvasElement = $('#publication-axis') as HTMLCanvasElement
  const rect = canvasElement.getBoundingClientRect()
  if (!rect.width || !rect.height) return
  const ratio = Math.min(devicePixelRatio, 2.5)
  canvasElement.width = Math.round(rect.width * ratio)
  canvasElement.height = Math.round(rect.height * ratio)
  const context = canvasElement.getContext('2d')
  if (!context) return
  context.scale(ratio, ratio)
  const width = rect.width, height = rect.height
  const values = corpus.timeline.map(frame => Math.log10(Math.max(frame.query_count, 1)))
  const min = Math.min(...values), max = Math.max(...values)
  context.clearRect(0, 0, width, height)
  corpus.timeline.forEach((frame, index) => {
    const x = cumulative[index] * width
    const barWidth = Math.max(1, (cumulative[index + 1] - cumulative[index]) * width - 1)
    const normalized = (values[index] - min) / Math.max(max - min, 1e-6)
    const barHeight = 3 + normalized * (height - 4)
    const themeCounts = new Map<string, number>()
    ;(papersByYear.get(frame.year) ?? []).forEach(paper => themeCounts.set(paper.theme, (themeCounts.get(paper.theme) ?? 0) + 1))
    const dominant = [...themeCounts].sort((a, b) => b[1] - a[1])[0]?.[0]
    const color = corpus.themes.find(theme => theme.id === dominant)?.color ?? '#78ead8'
    context.globalAlpha = index === activeIndex ? 1 : index < activeIndex ? .42 : .15
    context.fillStyle = color
    context.fillRect(x, height - barHeight, barWidth, barHeight)
  })
  context.globalAlpha = .55
  context.fillStyle = '#dffcf6'
  const markerX = cumulative[Math.min(activeIndex, cumulative.length - 2)] * width
  context.fillRect(markerX, 0, 1, height)
  context.globalAlpha = 1
}

function buildPaperCloud(destinations: Record<string, THREE.Vector3[]>): void {
  const streams: number[] = [], targets: number[] = [], colors: number[] = [], births: number[] = [], strengths: number[] = [], sizes: number[] = []
  const color = new THREE.Color()
  corpus.papers.forEach(paper => {
    const entries = Object.entries(paper.strips).sort((a, b) => b[1] - a[1])
    const top = entries[0]?.[0]
    entries.forEach(([stripId, strength]) => {
      if (strength < associationThreshold(stripId) && stripId !== top) return
      const lane = corpus.strips.findIndex(s => s.id === stripId)
      const targetPool = destinations[stripId]
      if (lane < 0 || !targetPool?.length) return
      const topic = corpus.themes.findIndex(t => t.id === paper.theme)
      const within = hash(`${paper.pmid}:${stripId}`)
      const targetIndex = Math.floor((((topic + within * .9) / corpus.themes.length) % 1) * targetPool.length)
      const target = targetPool[targetIndex]
      const birth = calendarToProgress(paper.year, paper.month)
      const streamX = STREAM_START + birth * (STREAM_END - STREAM_START)
      const laneY = ROW_TOP - lane * ROW_STEP
      const streamY = laneY + (topic / 11 - .5) * .54 + paper.xyz[1] * .035
      streams.push(streamX, streamY, paper.xyz[2] * .12)
      targets.push(target.x, target.y, target.z + .04)
      color.set(corpus.themes[Math.max(0, topic)]?.color ?? '#ffffff')
      colors.push(color.r, color.g, color.b)
      births.push(birth); strengths.push(strength); sizes.push(1.2 + paper.density * 1.9 + strength * 1.4)
    })
  })
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(targets, 3))
  geometry.setAttribute('aStream', new THREE.Float32BufferAttribute(streams, 3))
  geometry.setAttribute('aTarget', new THREE.Float32BufferAttribute(targets, 3))
  geometry.setAttribute('aColor', new THREE.Float32BufferAttribute(colors, 3))
  geometry.setAttribute('aBirth', new THREE.Float32BufferAttribute(births, 1))
  geometry.setAttribute('aStrength', new THREE.Float32BufferAttribute(strengths, 1))
  geometry.setAttribute('aSize', new THREE.Float32BufferAttribute(sizes, 1))
  const material = new THREE.ShaderMaterial({
    vertexShader: PAPER_VERTEX, fragmentShader: PAPER_FRAGMENT, uniforms,
    transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
  })
  scene.add(new THREE.Points(geometry, material))

  // The settled particle inhabits the form; this dim twin remains as its bibliographic fossil.
  const archiveGeometry = new THREE.BufferGeometry()
  archiveGeometry.setAttribute('position', new THREE.Float32BufferAttribute(streams, 3))
  archiveGeometry.setAttribute('aColor', new THREE.Float32BufferAttribute(colors, 3))
  archiveGeometry.setAttribute('aBirth', new THREE.Float32BufferAttribute(births, 1))
  archiveGeometry.setAttribute('aStrength', new THREE.Float32BufferAttribute(strengths, 1))
  archiveGeometry.setAttribute('aSize', new THREE.Float32BufferAttribute(sizes, 1))
  scene.add(new THREE.Points(archiveGeometry, new THREE.ShaderMaterial({
    vertexShader: ARCHIVE_VERTEX, fragmentShader: ARCHIVE_FRAGMENT, uniforms,
    transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
  })))
}

function disposeFocusScene(): void {
  focusRoot.traverse(object => {
    const mesh = object as THREE.Mesh
    if (mesh.geometry) mesh.geometry.dispose()
    const material = (mesh as { material?: THREE.Material | THREE.Material[] }).material
    if (Array.isArray(material)) material.forEach(item => item.dispose())
    else material?.dispose()
  })
  focusRoot.clear()
  focusHalos = []
  focusClusterIds.clear()
  focusPaperCloud = undefined
  focusPaperMetadata = []
  focusBirths = []
  focusLineMaterial = undefined
  hideNodeTooltip()
}

function focusPapersFor(stripId: string): Paper[] {
  return corpus.papers.filter(paper => {
    const entries = Object.entries(paper.strips).sort((a, b) => b[1] - a[1])
    return (paper.strips[stripId] ?? 0) >= associationThreshold(stripId) || entries[0]?.[0] === stripId
  })
}

function buildFocusScene(lane: number): void {
  disposeFocusScene()
  const strip = corpus.strips[lane]
  const data = focusGeometries[strip.id]
  if (!data?.points.length) return

  const templatePositions = new Float32Array(data.points.length * 3)
  data.points.forEach((point, index) => point.toArray(templatePositions, index * 3))
  const pointGeometry = new THREE.BufferGeometry()
  pointGeometry.setAttribute('position', new THREE.BufferAttribute(templatePositions, 3))
  const firingGraph = graphFiringAttributes(data, strip.id)
  pointGeometry.setAttribute('aGraphPhase', new THREE.BufferAttribute(firingGraph.phase, 1))
  pointGeometry.setAttribute('aDegree', new THREE.BufferAttribute(firingGraph.degree, 1))
  pointGeometry.setAttribute('aNodeSeed', new THREE.BufferAttribute(firingGraph.seed, 1))
  focusRoot.add(new THREE.Points(pointGeometry, new THREE.ShaderMaterial({
    vertexShader: FOCUS_TEMPLATE_VERTEX,
    fragmentShader: FOCUS_TEMPLATE_FRAGMENT,
    uniforms: { ...uniforms, uColor: { value: new THREE.Color(strip.color) } },
    transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
  })))

  const linePositions = new Float32Array(data.edges.length * 6)
  data.edges.forEach(([a, b], index) => {
    data.points[a]?.toArray(linePositions, index * 6)
    data.points[b]?.toArray(linePositions, index * 6 + 3)
  })
  const lineGeometry = new THREE.BufferGeometry()
  lineGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3))
  focusLineMaterial = new THREE.LineBasicMaterial({
    color: strip.color, transparent: true, opacity: .13, blending: THREE.AdditiveBlending, depthWrite: false,
  })
  focusRoot.add(new THREE.LineSegments(lineGeometry, focusLineMaterial))

  const targetPool = [...data.points].sort((a, b) => (Math.atan2(a.y, a.x) + a.z * .2) - (Math.atan2(b.y, b.x) + b.z * .2))
  const papers = focusPapersFor(strip.id)
  const positions: number[] = [], origins: number[] = [], colors: number[] = [], births: number[] = [], strengths: number[] = []
  const clusterPoints = new Map<number, THREE.Vector3[]>()
  const color = new THREE.Color()
  papers.forEach(paper => {
    const topic = Math.max(0, corpus.themes.findIndex(theme => theme.id === paper.theme))
    const within = hash(`${paper.pmid}:${strip.id}`)
    const targetIndex = Math.floor((((topic + within * .9) / corpus.themes.length) % 1) * targetPool.length)
    const target = targetPool[targetIndex]
    target.toArray(positions, positions.length)
    const semanticDirection = new THREE.Vector3(
      paper.xyz[0] + (within - .5) * .3,
      paper.xyz[1] + (hash(`${paper.pmid}:orbit`) - .5) * .3,
      paper.xyz[2] + (hash(`${paper.pmid}:depth`) - .5) * .3,
    )
    if (semanticDirection.lengthSq() < .01) semanticDirection.set(1, 0, 0)
    semanticDirection.normalize().multiplyScalar(3.25 + hash(`${paper.pmid}:radius`) * 1.75)
    semanticDirection.toArray(origins, origins.length)
    color.set(corpus.themes[topic]?.color ?? '#ffffff')
    colors.push(color.r, color.g, color.b)
    births.push(calendarToProgress(paper.year, paper.month))
    strengths.push(paper.strips[strip.id] ?? .18)
    const formCluster = paper.form_clusters?.[strip.id]
    if (formCluster != null && formCluster >= 0) {
      const points = clusterPoints.get(formCluster) ?? []
      points.push(target); clusterPoints.set(formCluster, points)
    }
  })
  focusClusterIds.clear()
  const paperGeometry = new THREE.BufferGeometry()
  paperGeometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  paperGeometry.setAttribute('aOrigin', new THREE.Float32BufferAttribute(origins, 3))
  paperGeometry.setAttribute('aColor', new THREE.Float32BufferAttribute(colors, 3))
  paperGeometry.setAttribute('aBirth', new THREE.Float32BufferAttribute(births, 1))
  paperGeometry.setAttribute('aStrength', new THREE.Float32BufferAttribute(strengths, 1))
  focusPaperCloud = new THREE.Points(paperGeometry, new THREE.ShaderMaterial({
    vertexShader: FOCUS_VERTEX, fragmentShader: PAPER_FRAGMENT, uniforms,
    transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
  }))
  focusPaperMetadata = papers
  focusBirths = births
  focusRoot.add(focusPaperCloud)

  clusterPoints.forEach((points, clusterId) => {
    if (points.length < 5) return
    const cluster = corpus.form_clusters?.[strip.id]?.find(item => item.id === clusterId)
    if (!cluster) return
    focusClusterIds.add(clusterId)
    const ordered = [...points]
      .sort((a, b) => (a.x * .71 + a.y * .23 + a.z * .49) - (b.x * .71 + b.y * .23 + b.z * .49))
      .filter((_, index) => index % Math.max(1, Math.floor(points.length / 92)) === 0)
      .slice(0, 96)
    const filaments: number[] = []
    ordered.forEach((point, index) => {
      if (index) {
        point.toArray(filaments, filaments.length)
        ordered[index - 1].toArray(filaments, filaments.length)
      }
      if (index > 5 && index % 4 === 0) {
        point.toArray(filaments, filaments.length)
        ordered[index - 5].toArray(filaments, filaments.length)
      }
    })
    const formationGeometry = new THREE.BufferGeometry()
    formationGeometry.setAttribute('position', new THREE.Float32BufferAttribute(filaments, 3))
    const halo = new THREE.LineSegments(
      formationGeometry,
      new THREE.LineBasicMaterial({ color: cluster.color, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false }),
    )
    halo.userData.clusterId = clusterId
    halo.userData.emergence = cluster.emergence_year
    focusRoot.add(halo); focusHalos.push(halo)
  })
  focusRoot.rotation.set(.06, -.18, 0)
}

function enterFocus(lane: number): void {
  if (!started || lane < 0 || lane >= corpus.strips.length) return
  focusedLane = lane
  focusDistance = focusTargetDistance = 4.2
  focusCamera.position.set(0, 0, focusDistance)
  buildFocusScene(lane)
  const strip = corpus.strips[lane]
  const source = atlas.entries.find(entry => entry.id === strip.id)
  $('#focus-index').textContent = `0${lane + 1} / 08`
  $('#focus-title').textContent = strip.name.toUpperCase()
  $('#focus-cluster-scope').textContent = strip.name.toUpperCase()
  $('#focus-source').textContent = `${source?.kind?.replaceAll('-', ' ').toUpperCase() ?? 'GENERATIVE FORM'} / ${strip.source_short.toUpperCase()}`
  $('#focus-description').textContent = `${focusPapersFor(strip.id).length.toLocaleString()} publications inhabit this three-dimensional semantic body. Drag to orbit; travel through the surface to inspect its interior. Placement is semantic, not anatomical localization.`
  $('#focus-panel').setAttribute('aria-hidden', 'false')
  document.body.classList.add('focus-active')
  lastFormation = -1
}

function exitFocus(): void {
  focusedLane = -1
  dragging = false
  activePointers.clear()
  $('#focus-panel').setAttribute('aria-hidden', 'true')
  document.body.classList.remove('focus-active', 'dragging')
  hideNodeTooltip()
}

function resetFocusView(): void {
  focusTargetDistance = 4.2
  focusRoot.rotation.set(.06, -.18, 0)
}

function updateFocus(calendar: ReturnType<typeof progressToCalendar>, elapsed: number, delta: number): void {
  if (focusedLane < 0) return
  focusDistance += (focusTargetDistance - focusDistance) * Math.min(1, delta * 8)
  focusCamera.position.z = focusDistance
  if (!dragging) focusRoot.rotation.y += delta * .115
  focusRoot.position.y = Math.sin(elapsed * .37) * .025
  const yearlyIntensity = THREE.MathUtils.clamp(
    (Math.log10(Math.max(10, calendar.frame.query_count)) - 4) / 2,
    0, 1,
  )
  let arrivals = 0
  for (const birth of focusBirths) arrivals += Math.exp(-Math.abs(progress - (birth + .036)) * 170)
  const firing = THREE.MathUtils.clamp(Math.sqrt(arrivals) * .08 + yearlyIntensity * .01, 0, 1)
  const response = firing > uniforms.uFire.value ? 11 : 2.4
  uniforms.uFire.value += (firing - uniforms.uFire.value) * Math.min(1, delta * response)
  if (focusLineMaterial) focusLineMaterial.opacity = .078 + uniforms.uFire.value * .026
  focusHalos.forEach(halo => {
    const formId = corpus.strips[focusedLane].id
    const cluster = corpus.form_clusters?.[formId]?.find(item => item.id === halo.userData.clusterId)
    const material = halo.material as THREE.LineBasicMaterial
    const born = cluster && cluster.emergence_year <= calendar.year
    halo.visible = Boolean(born)
    material.opacity = born ? .035 + Math.sin(elapsed * 1.3 + halo.userData.clusterId) * .018 : 0
  })
  const formId = corpus.strips[focusedLane].id
  const present = (corpus.form_clusters?.[formId] ?? []).filter(cluster =>
    cluster.emergence_year <= calendar.year && focusClusterIds.has(cluster.id)
  )
  if (!present.length) {
    $('#focus-cluster-count').textContent = '00'
    $('#focus-cluster-name').textContent = 'PRE-FORMATION FIELD'
    $('#focus-cluster-terms').textContent = 'The semantic graph is accumulating evidence; no stable neighborhood has crossed its density threshold at this date.'
    $('#focus-coherence').style.width = '0%'
    $('#focus-cluster-meta').textContent = `${formId.toUpperCase()}-SPECIFIC HDBSCAN / STABILITY THRESHOLD PENDING`
    return
  }
  const selected = present[Math.floor(elapsed / 4.8) % present.length]
  $('#focus-cluster-count').textContent = String(present.length).padStart(2, '0')
  $('#focus-cluster-name').textContent = selected.label.toUpperCase()
  $('#focus-cluster-terms').textContent = selected.vocabulary.join(' · ').toUpperCase() || selected.theme.toUpperCase()
  $('#focus-coherence').style.width = `${Math.round(selected.coherence * 100)}%`
  $('#focus-coherence').style.background = selected.color
  $('#focus-cluster-meta').textContent = `${formId.toUpperCase()}-SPECIFIC · EMERGENCE ${selected.emergence_year} / ${selected.size.toLocaleString()} PAPERS / COHERENCE ${selected.coherence.toFixed(2)}`
  focusHalos.forEach(halo => {
    const material = halo.material as THREE.LineBasicMaterial
    if (halo.userData.clusterId === selected.id) material.opacity = .17 + Math.sin(elapsed * 2.1) * .04
  })
}

function addAmbientField(): void {
  const count = 950
  const positions = new Float32Array(count * 3)
  for (let i = 0; i < count; i++) {
    positions[i * 3] = (Math.random() - .5) * 17
    positions[i * 3 + 1] = (Math.random() - .5) * 9.2
    positions[i * 3 + 2] = -1 - Math.random() * 2
  }
  const geometry = new THREE.BufferGeometry(); geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  scene.add(new THREE.Points(geometry, new THREE.PointsMaterial({ color: 0x8ecbc4, size: .75, sizeAttenuation: false, opacity: .08, transparent: true, depthWrite: false })))
}

function addPublicationVolumeField(): void {
  const positions: number[] = [], colors: number[] = [], births: number[] = [], strengths: number[] = [], sizes: number[] = []
  const logCounts = corpus.timeline.map(frame => Math.log10(Math.max(1, frame.query_count)))
  const minimum = Math.min(...logCounts), maximum = Math.max(...logCounts)
  const color = new THREE.Color()
  corpus.timeline.forEach((frame, yearIndex) => {
    const yearPapers = papersByYear.get(frame.year) ?? []
    const normalized = (logCounts[yearIndex] - minimum) / Math.max(.001, maximum - minimum)
    const particles = Math.round(54 + normalized * 236)
    for (let index = 0; index < particles; index++) {
      const sample = yearPapers.length
        ? yearPapers[Math.floor(hash(`${frame.year}:${index}:sample`) * yearPapers.length)]
        : corpus.papers[Math.floor(hash(`${frame.year}:${index}:fallback`) * corpus.papers.length)]
      const stripEntries = Object.entries(sample?.strips ?? {}).sort((a, b) => b[1] - a[1])
      const stripId = stripEntries[Math.min(stripEntries.length - 1, Math.floor(hash(`${frame.year}:${index}:lane`) * Math.min(3, stripEntries.length)))]?.[0]
      const lane = Math.max(0, corpus.strips.findIndex(strip => strip.id === stripId))
      const topic = Math.max(0, corpus.themes.findIndex(theme => theme.id === sample?.theme))
      const month = 1 + Math.floor(hash(`${frame.year}:${index}:month`) * 12)
      const birth = calendarToProgress(frame.year, month)
      const x = STREAM_START + birth * (STREAM_END - STREAM_START)
      const y = ROW_TOP - lane * ROW_STEP + (topic / Math.max(1, corpus.themes.length - 1) - .5) * .48
        + (hash(`${frame.year}:${index}:jitter`) - .5) * .075
      positions.push(x, y, -.52 + (hash(`${frame.year}:${index}:z`) - .5) * .22)
      color.set(corpus.themes[topic]?.color ?? '#78ead8')
      colors.push(color.r, color.g, color.b)
      births.push(birth)
      strengths.push(.14 + normalized * .38)
      sizes.push(.32 + normalized * .75)
    }
  })
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
  geometry.setAttribute('aColor', new THREE.Float32BufferAttribute(colors, 3))
  geometry.setAttribute('aBirth', new THREE.Float32BufferAttribute(births, 1))
  geometry.setAttribute('aStrength', new THREE.Float32BufferAttribute(strengths, 1))
  geometry.setAttribute('aSize', new THREE.Float32BufferAttribute(sizes, 1))
  scene.add(new THREE.Points(geometry, new THREE.ShaderMaterial({
    vertexShader: ARCHIVE_VERTEX, fragmentShader: ARCHIVE_FRAGMENT, uniforms,
    transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
  })))
}

function buildInterface(): void {
  const labels = $('#strip-labels')
  labels.innerHTML = corpus.strips.map((strip, i) => `<button type="button" class="strip-label" data-index="0${i + 1}" data-lane="${i}" data-strip="${strip.id}" aria-label="Explore ${strip.name} in 3D"><b>${strip.name}</b><span>${strip.source_short}</span></button>`).join('')
  labels.querySelectorAll<HTMLButtonElement>('.strip-label').forEach(label => {
    label.addEventListener('click', () => enterFocus(Number(label.dataset.lane)))
  })
  const key = $('#topic-key')
  key.innerHTML = corpus.themes.map(theme => `<span class="topic-chip" data-theme="${theme.id}" style="--c:${theme.color}">${theme.name.toUpperCase()}</span>`).join('')
  const queryUniverse = corpus.timeline.reduce((sum, frame) => sum + frame.query_count, 0)
  $('#corpus-count').textContent = `${(queryUniverse / 1_000_000).toFixed(2)}M QUERY UNIVERSE · ${corpus.papers.length.toLocaleString()} EMBEDDED`
}

function updateInterface(calendar: ReturnType<typeof progressToCalendar>, elapsed: number): void {
  const displayYear = Math.min(2026, calendar.year + Math.floor(calendar.fraction))
  $('#year').textContent = String(displayYear)
  $('#publication-count').textContent = `${calendar.frame.query_count.toLocaleString()} MATCHING RECORDS / YEAR`
  $('#timeline').style.setProperty('--progress', `${progress * 100}%`)
  ;($('#timeline') as HTMLInputElement).value = String(progress)
  audio.update(calendar.frame); $('#chord').textContent = audio.label()
  const aiStage = document.querySelector<HTMLElement>('.strip-label[data-strip="ai"] span')
  if (aiStage && learningMachine) aiStage.textContent = learningMachine.stageName
  const yearPapers = papersByYear.get(calendar.year) ?? []
  const themeCounts = new Map<string, number>()
  yearPapers.forEach(p => themeCounts.set(p.theme, (themeCounts.get(p.theme) ?? 0) + 1))
  const activeTheme = [...themeCounts].sort((a, b) => b[1] - a[1])[0]?.[0] ?? corpus.themes[0].id
  const theme = corpus.themes.find(t => t.id === activeTheme) ?? corpus.themes[0]
  $('#topic').textContent = theme.name.toUpperCase()
  const maximumThemeCount = Math.max(1, ...themeCounts.values())
  document.querySelectorAll<HTMLElement>('.topic-chip').forEach(el => {
    const count = themeCounts.get(el.dataset.theme ?? '') ?? 0
    el.classList.toggle('active', el.dataset.theme === activeTheme)
    el.style.setProperty('--weight', String(.18 + count))
    el.style.setProperty('--dominance', String(.18 + .82 * count / maximumThemeCount))
  })
  if (lastAxisYear !== calendar.index) { lastAxisYear = calendar.index; drawPublicationAxis(calendar.index) }
  const frameWithClusters = calendar.frame as TimelineFrame & { new_clusters?: number[]; active_clusters?: number }
  const availableClusters = (corpus.clusters ?? []).filter(cluster => cluster.emergence_year <= calendar.year)
  const newClusterId = frameWithClusters.new_clusters?.[0]
  const formation = (newClusterId != null ? corpus.clusters?.find(cluster => cluster.id === newClusterId) : undefined)
    ?? availableClusters[Math.floor(elapsed / 6.4) % Math.max(1, availableClusters.length)]
  if (formation) {
    $('#formation-name').textContent = formation.label.toUpperCase()
    $('#formation-meta').textContent = `${formation.emergence_year} / ${formation.theme.toUpperCase()} / ${formation.size.toLocaleString()} PAPERS`
    if (formation.id !== lastFormation) {
      lastFormation = formation.id
      const ticker = $('#formation-ticker')
      ticker.classList.remove('flare'); void ticker.offsetWidth; ticker.classList.add('flare')
    }
  } else {
    $('#formation-name').textContent = 'PRE-FORMATION FIELD'
    $('#formation-meta').textContent = `${calendar.year} / DENSITY THRESHOLDS ACCUMULATING`
  }
  const landmark = [...LANDMARKS].reverse().find(item => item.year <= calendar.year) ?? LANDMARKS[0]
  $('#landmark').textContent = `${landmark.year} / ${landmark.label}`
  const cardKey = Math.floor(elapsed / 1.7)
  if (cardKey !== lastCardKey && yearPapers.length) {
    lastCardKey = cardKey
    const available = yearPapers.filter(p => calendar.fraction > (p.month - 1) / 12)
    const paper = (available.length ? available : yearPapers)[cardKey % (available.length || yearPapers.length)]
    const paperTheme = corpus.themes.find(t => t.id === paper.theme)
    $('#paper-year').textContent = `${paper.year} · PMID ${paper.pmid}`
    $('#paper-theme').textContent = paperTheme?.name.toUpperCase() ?? paper.theme.toUpperCase()
    $('#paper-theme').style.color = paperTheme?.color ?? '#78ead8'
    $('#paper-title').textContent = paper.title
    $('#paper-journal').textContent = `${paper.journal || 'PUBMED'} · ${Object.keys(paper.strips).slice(0, 3).join(' / ').toUpperCase()}`
  }
}

async function init(): Promise<void> {
  if (globalThis.__NEURO_CORPUS__ && globalThis.__NEURO_ATLAS__) {
    corpus = globalThis.__NEURO_CORPUS__
    atlas = globalThis.__NEURO_ATLAS__
  } else {
    ;[corpus, atlas] = await Promise.all([
      fetch('./public/data/corpus.json').then(r => r.json()),
      fetch('./public/data/templates.json').then(r => r.json()),
    ])
  }
  setupTimeline(); buildInterface(); addAmbientField(); addBandsAndFilaments(); addPublicationVolumeField()
  corpus.strips.forEach((strip, lane) => {
    const laneY = ROW_TOP - lane * ROW_STEP
    if (strip.id === 'ai') return
    const entry = atlas.entries.find(item => item.id === strip.id)
    let geometry: GeometryData
    if (strip.id === 'gradient') geometry = gradientGeometry()
    else if (strip.id === 'fungal') geometry = fungalGeometry()
    else geometry = { points: decodePoints(entry!), edges: decodeEdges(entry!) }
    focusGeometries[strip.id] = normalizeForFocus(geometry, strip.id)
    const fitted = fitToChamber(geometry, strip.id, laneY)
    addTemplate(fitted, strip.color, strip.id)
    destinations[strip.id] = [...fitted.points].sort((a, b) => (Math.atan2(a.y - laneY, a.x - CHAMBER_X) + a.z * .2) - (Math.atan2(b.y - laneY, b.x - CHAMBER_X) + b.z * .2))
  })
  learningMachine = new LearningMachine(CHAMBER_X, ROW_TOP - 7 * ROW_STEP)
  scene.add(learningMachine.group)
  learningMachine.update(2026, 0)
  destinations.ai = learningMachine.destinations.map(p => p.clone()).sort((a, b) => Math.atan2(a.y - learningMachine.group.position.y, a.x - CHAMBER_X) - Math.atan2(b.y - learningMachine.group.position.y, b.x - CHAMBER_X))
  const aiPoints = learningMachine.focusPoints.map(point => point.clone())
  focusGeometries.ai = normalizeForFocus({ points: aiPoints, edges: learningMachine.focusEdges }, 'ai')
  learningMachine.update(1976, 0)
  buildPaperCloud(destinations)
  $('#loading').classList.add('done')
  requestAnimationFrame(animate)
}

function resize(): void {
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2.5))
  renderer.setSize(innerWidth, innerHeight, false)
  const height = 9.2, width = height * (innerWidth / innerHeight)
  camera.left = -width / 2; camera.right = width / 2; camera.top = height / 2; camera.bottom = -height / 2
  camera.updateProjectionMatrix()
  focusCamera.aspect = innerWidth / innerHeight; focusCamera.updateProjectionMatrix()
  uniforms.uPixelRatio.value = renderer.getPixelRatio()
  if (corpus) drawPublicationAxis(Math.max(0, lastAxisYear))
}

function animate(): void {
  const delta = Math.min(.05, clock.getDelta())
  if (started && playing) {
    artworkElapsed += delta * Math.sqrt(speed)
    if (progress < 1) {
      progress = Math.min(1, progress + delta / (178 / speed))
      codaElapsed = 0
    } else {
      codaElapsed += delta
      if (codaElapsed >= CODA_SECONDS) setPlaying(false)
    }
  }
  const elapsed = artworkElapsed
  uniforms.uTime.value = progress; uniforms.uPulse.value = elapsed
  if (corpus) {
    const calendar = progressToCalendar(progress)
    learningMachine?.update(calendar.year + calendar.fraction, elapsed)
    updateInterface(calendar, elapsed)
    updateFocus(calendar, elapsed, delta)
    audio.tick()
  }
  camera.position.x = Math.sin(elapsed * .067) * .025
  camera.position.y = Math.cos(elapsed * .053) * .018
  renderer.render(scene, camera)
  if (focusedLane >= 0) {
    renderer.autoClear = false
    renderer.clearDepth()
    renderer.render(focusScene, focusCamera)
    renderer.autoClear = true
  }
  requestAnimationFrame(animate)
}

$('#enter').addEventListener('click', () => {
  started = true; playing = true; $('#entry').classList.add('hidden')
  void audio.start().then(() => audio.setPaused(false))
})
function setPlaying(value: boolean): void {
  if (value && progress >= 1 && codaElapsed >= CODA_SECONDS) {
    progress = 0; codaElapsed = 0; lastCardKey = -1; audio.seek()
  }
  playing = value
  $('#play').textContent = playing ? 'PAUSE' : 'PLAY'
  audio.setPaused(!playing)
}
$('#play').addEventListener('click', () => setPlaying(!playing))
$('#timeline').addEventListener('input', event => {
  progress = Number((event.target as HTMLInputElement).value)
  codaElapsed = 0; lastCardKey = -1; audio.seek()
})
const speeds = [.5, 1, 2, 4, 8]
$('#speed').addEventListener('click', () => {
  speed = speeds[(speeds.indexOf(speed) + 1) % speeds.length]
  $('#speed').textContent = `${speed}×`
  audio.setPlaybackSpeed(speed)
})
function toggleSound(): void { audio.setMuted(!audio.isMuted()); $('#sound').textContent = audio.isMuted() ? 'SOUND OFF' : 'SOUND ON' }
$('#sound').addEventListener('click', toggleSound)
async function fullscreen(): Promise<void> { if (!document.fullscreenElement) await document.documentElement.requestFullscreen(); else await document.exitFullscreen() }
$('#full').addEventListener('click', fullscreen)
$('#focus-close').addEventListener('click', exitFocus)

function hideNodeTooltip(): void {
  const tooltip = document.querySelector<HTMLElement>('#node-tooltip')
  if (!tooltip) return
  tooltip.classList.remove('visible')
  tooltip.setAttribute('aria-hidden', 'true')
}

function inspectFocusNode(event: PointerEvent): void {
  if (focusedLane < 0 || dragging || !focusPaperCloud) { hideNodeTooltip(); return }
  const rect = canvas.getBoundingClientRect()
  pointerNdc.set(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1,
  )
  focusRoot.updateMatrixWorld(true)
  focusCamera.updateMatrixWorld(true)
  raycaster.setFromCamera(pointerNdc, focusCamera)
  const intersection = raycaster.intersectObject(focusPaperCloud, false)[0]
  const paperIndex = intersection?.index
  if (paperIndex == null || focusBirths[paperIndex] > progress + .002) { hideNodeTooltip(); return }
  const paper = focusPaperMetadata[paperIndex]
  if (!paper) { hideNodeTooltip(); return }
  const theme = corpus.themes.find(item => item.id === paper.theme)
  const formId = corpus.strips[focusedLane].id
  const formCluster = paper.form_clusters?.[formId]
  const cluster = formCluster != null ? corpus.form_clusters?.[formId]?.find(item => item.id === formCluster) : undefined
  const tooltip = $('#node-tooltip')
  $('#node-tooltip-meta').textContent = `${paper.year} · PMID ${paper.pmid} · ${theme?.name.toUpperCase() ?? paper.theme.toUpperCase()}`
  $('#node-tooltip-meta').style.color = theme?.color ?? '#78ead8'
  $('#node-tooltip-title').textContent = paper.title
  $('#node-tooltip-cluster').textContent = cluster
    ? `${formId.toUpperCase()}-SPECIFIC · ${cluster.label.toUpperCase()} · COHERENCE ${cluster.coherence.toFixed(2)} · ${paper.journal || 'PUBMED'}`
    : `NO STABLE ${formId.toUpperCase()}-SPECIFIC FORMATION · ${paper.journal || 'PUBMED'}`
  const width = Math.min(innerWidth * .2, 380)
  const left = Math.min(event.clientX + 18, innerWidth - width - 18)
  const top = Math.min(event.clientY + 18, innerHeight - 126)
  tooltip.style.left = `${Math.max(12, left)}px`
  tooltip.style.top = `${Math.max(12, top)}px`
  tooltip.classList.add('visible')
  tooltip.setAttribute('aria-hidden', 'false')
}

function laneAtPointer(event: PointerEvent): number {
  const rect = canvas.getBoundingClientRect()
  const nx = ((event.clientX - rect.left) / rect.width) * 2 - 1
  const ny = 1 - ((event.clientY - rect.top) / rect.height) * 2
  const worldHeight = 9.2, worldWidth = worldHeight * (rect.width / rect.height)
  const x = nx * worldWidth * .5
  const y = ny * worldHeight * .5
  if (x < CHAMBER_X - 1.65 || x > CHAMBER_X + 1.65) return -1
  const lane = Math.round((ROW_TOP - y) / ROW_STEP)
  return lane >= 0 && lane < 8 && Math.abs(y - (ROW_TOP - lane * ROW_STEP)) < ROW_STEP * .48 ? lane : -1
}

canvas.addEventListener('pointerdown', event => {
  if (focusedLane < 0) {
    const lane = laneAtPointer(event)
    if (lane >= 0) enterFocus(lane)
    return
  }
  canvas.setPointerCapture(event.pointerId)
  activePointers.set(event.pointerId, { x: event.clientX, y: event.clientY })
  dragging = true
  document.body.classList.add('dragging')
  hideNodeTooltip()
  if (activePointers.size === 2) {
    const points = [...activePointers.values()]
    pinchDistance = Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y)
  }
})
canvas.addEventListener('pointermove', event => {
  if (focusedLane < 0) return
  if (!activePointers.has(event.pointerId)) { inspectFocusNode(event); return }
  const previous = activePointers.get(event.pointerId)!
  activePointers.set(event.pointerId, { x: event.clientX, y: event.clientY })
  if (activePointers.size >= 2) {
    const points = [...activePointers.values()]
    const distance = Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y)
    if (pinchDistance > 0) focusTargetDistance = THREE.MathUtils.clamp(focusTargetDistance * (pinchDistance / Math.max(distance, 1)), .42, 7.5)
    pinchDistance = distance
  } else {
    focusRoot.rotation.y += (event.clientX - previous.x) * .0062
    focusRoot.rotation.x = THREE.MathUtils.clamp(focusRoot.rotation.x + (event.clientY - previous.y) * .0052, -1.45, 1.45)
  }
})
function releasePointer(event: PointerEvent): void {
  activePointers.delete(event.pointerId)
  if (activePointers.size < 2) pinchDistance = 0
  if (!activePointers.size) { dragging = false; document.body.classList.remove('dragging') }
}
canvas.addEventListener('pointerup', releasePointer)
canvas.addEventListener('pointercancel', releasePointer)
canvas.addEventListener('pointerleave', hideNodeTooltip)
canvas.addEventListener('wheel', event => {
  if (focusedLane < 0) return
  event.preventDefault()
  focusTargetDistance = THREE.MathUtils.clamp(focusTargetDistance * Math.exp(event.deltaY * .00105), .42, 7.5)
}, { passive: false })
canvas.addEventListener('dblclick', () => { if (focusedLane >= 0) resetFocusView() })
window.addEventListener('resize', resize)
window.addEventListener('keydown', event => {
  if (event.code === 'Space') { event.preventDefault(); setPlaying(!playing) }
  if (event.key.toLowerCase() === 'm') toggleSound()
  if (event.key.toLowerCase() === 'h') { clean = !clean; document.body.classList.toggle('capture-clean', clean) }
  if (event.key.toLowerCase() === 'f') void fullscreen()
  if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
    progress = THREE.MathUtils.clamp(progress + (event.key === 'ArrowRight' ? .01 : -.01), 0, 1)
    codaElapsed = 0; lastCardKey = -1; audio.seek()
  }
  if (event.key.toLowerCase() === 'r') { progress = 0; codaElapsed = 0; lastCardKey = -1; audio.seek() }
  if (event.key === 'Escape' && focusedLane >= 0) exitFocus()
})

resize()
void init().catch(error => {
  console.error(error)
  $('#loading').innerHTML = `<em>THE FIELD COULD NOT ASSEMBLE · ${String(error)}</em>`
})
