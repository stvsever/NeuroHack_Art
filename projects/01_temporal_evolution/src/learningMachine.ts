import * as THREE from 'three'

const N = 256
const STAGE_YEARS = [1976, 1986, 1998, 2012, 2017, 2021]
export const STAGE_NAMES = ['PERCEPTRON', 'BACKPROPAGATION', 'CONVOLUTION', 'RESIDUAL DEPTH', 'ATTENTION', 'MIXTURE OF EXPERTS']
const STAGE_LAYERS = [
  [128, 128],
  [48, 64, 64, 48, 32],
  [32, 48, 64, 64, 32, 16],
  [32, 32, 32, 32, 32, 32, 32, 32],
  [32, 32, 32, 32, 32, 32, 32, 32],
  [40, 56, 64, 56, 32, 8],
]

function feedForwardEdges(layers: number[]): [number, number][] {
  const edges: [number, number][] = []
  let sourceOffset = 0
  for (let layer = 0; layer < layers.length - 1; layer++) {
    const sourceCount = layers[layer], targetCount = layers[layer + 1]
    const targetOffset = sourceOffset + sourceCount
    for (let source = 0; source < sourceCount; source++) {
      const center = Math.floor(source / Math.max(1, sourceCount - 1) * Math.max(0, targetCount - 1))
      const fan = sourceCount > 48 ? 2 : 3
      for (let branch = 0; branch < fan; branch++) {
        const target = Math.max(0, Math.min(targetCount - 1, center + branch - Math.floor(fan / 2)))
        edges.push([sourceOffset + source, targetOffset + target])
      }
    }
    sourceOffset = targetOffset
  }
  return edges
}

function layered(layers: number[], curve = 0): Float32Array {
  const out = new Float32Array(N * 3)
  let cursor = 0
  for (let l = 0; l < layers.length; l++) {
    const count = layers[l]
    for (let j = 0; j < count && cursor < N; j++, cursor++) {
      const y = count === 1 ? 0 : (j / (count - 1) - .5) * 1.55
      out[cursor * 3] = layers.length === 1 ? 0 : (l / (layers.length - 1) - .5) * 2.55
      out[cursor * 3 + 1] = y + Math.sin(l * 1.7 + j * .8) * curve
      out[cursor * 3 + 2] = Math.sin(j * 2.399 + l) * .28
    }
  }
  while (cursor < N) {
    const source = cursor % Math.max(1, layers.reduce((a, b) => a + b, 0))
    out[cursor * 3] = out[source * 3] + Math.sin(cursor) * .025
    out[cursor * 3 + 1] = out[source * 3 + 1] + Math.cos(cursor * 2.1) * .025
    out[cursor * 3 + 2] = out[source * 3 + 2] + Math.sin(cursor * .7) * .025
    cursor++
  }
  return out
}

function residual(): Float32Array {
  const out = layered([32, 32, 32, 32, 32, 32, 32, 32], .06)
  for (let i = 0; i < N; i++) out[i * 3 + 1] += (i % 2 ? .14 : -.14) * Math.sin(i * .47)
  return out
}

function attention(): Float32Array {
  const out = new Float32Array(N * 3)
  for (let i = 0; i < N; i++) {
    const head = i % 8, token = Math.floor(i / 8)
    const a = head / 8 * Math.PI * 2 + token * .025
    const r = .25 + token / 32 * 1.05
    out[i * 3] = Math.cos(a) * r * 1.25
    out[i * 3 + 1] = Math.sin(a) * r * .68
    out[i * 3 + 2] = Math.sin(token * .7 + head) * .26
  }
  return out
}

function experts(): Float32Array {
  // A legible trained system: input layer → increasingly rich hidden/expert layers → output layer.
  const counts = [40, 56, 64, 56, 32, 8]
  const out = layered(counts, .035)
  let cursor = 0
  counts.forEach((count, layer) => {
    for (let within = 0; within < count; within++, cursor++) {
      if (layer > 0 && layer < counts.length - 1) {
        const expert = within % 4
        out[cursor * 3 + 1] += (expert - 1.5) * .035
        out[cursor * 3 + 2] += (expert - 1.5) * .085
      } else {
        out[cursor * 3 + 2] *= .25
      }
    }
  })
  return out
}

export class LearningMachine {
  readonly group = new THREE.Group()
  readonly destinations: THREE.Vector3[] = []
  readonly focusPoints: THREE.Vector3[] = []
  readonly focusEdges = feedForwardEdges(STAGE_LAYERS[STAGE_LAYERS.length - 1])
  stageName = STAGE_NAMES[0]
  private stages = [
    layered([128, 128]),
    layered([48, 64, 64, 48, 32]),
    layered([32, 48, 64, 64, 32, 16]),
    residual(),
    attention(),
    experts(),
  ]
  private positions = new Float32Array(N * 3)
  private linePositions: Float32Array
  private points: THREE.Points
  private lines: THREE.LineSegments
  private stageEdges = STAGE_LAYERS.map(feedForwardEdges)

  constructor(x: number, y: number) {
    const maximumEdges = Math.max(...this.stageEdges.map(edges => edges.length))
    this.linePositions = new Float32Array(maximumEdges * 6)
    const pg = new THREE.BufferGeometry()
    pg.setAttribute('position', new THREE.BufferAttribute(this.positions, 3))
    const pm = new THREE.PointsMaterial({ color: 0x75ecff, size: 1.2, sizeAttenuation: false, transparent: true, opacity: .16, blending: THREE.NormalBlending, depthWrite: false })
    this.points = new THREE.Points(pg, pm)
    const lg = new THREE.BufferGeometry()
    lg.setAttribute('position', new THREE.BufferAttribute(this.linePositions, 3))
    const lm = new THREE.LineBasicMaterial({ color: 0x58c7e7, transparent: true, opacity: .035, blending: THREE.NormalBlending, depthWrite: false })
    this.lines = new THREE.LineSegments(lg, lm)
    this.group.add(this.lines, this.points)
    this.group.position.set(x, y, .04)
    // The overview has strict horizontal lanes. Compress only the procedural
    // display body's vertical envelope; the undistorted topology is retained
    // separately for the full-screen focus view.
    this.group.scale.set(.73, .38, .73)
    this.update(1976, 0)
    const focusStage = this.stages[this.stages.length - 1]
    for (let i = 0; i < N; i++) {
      this.destinations.push(new THREE.Vector3())
      this.focusPoints.push(new THREE.Vector3(
        focusStage[i * 3],
        focusStage[i * 3 + 1],
        focusStage[i * 3 + 2],
      ))
    }
  }

  update(year: number, pulse: number): void {
    let stage = 0
    while (stage < STAGE_YEARS.length - 1 && year >= STAGE_YEARS[stage + 1]) stage++
    const next = Math.min(stage + 1, this.stages.length - 1)
    const span = next === stage ? 1 : STAGE_YEARS[next] - STAGE_YEARS[stage]
    let mix = next === stage ? 0 : (year - STAGE_YEARS[stage]) / span
    mix = mix * mix * (3 - 2 * mix)
    const a = this.stages[stage], b = this.stages[next]
    for (let i = 0; i < N; i++) {
      const breath = 1 + Math.sin(pulse * 1.7 + i * .37) * .018
      this.positions[i * 3] = (a[i * 3] * (1 - mix) + b[i * 3] * mix) * breath
      this.positions[i * 3 + 1] = (a[i * 3 + 1] * (1 - mix) + b[i * 3 + 1] * mix) * breath
      this.positions[i * 3 + 2] = a[i * 3 + 2] * (1 - mix) + b[i * 3 + 2] * mix
    }
    this.linePositions.fill(0)
    this.stageEdges[stage].forEach(([u, v], i) => {
      this.linePositions.set(this.positions.subarray(u * 3, u * 3 + 3), i * 6)
      this.linePositions.set(this.positions.subarray(v * 3, v * 3 + 3), i * 6 + 3)
    })
    ;(this.points.geometry.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true
    ;(this.lines.geometry.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true
    ;(this.points.material as THREE.PointsMaterial).opacity = .16 + Math.sin(pulse * 2.3) * .012
    this.stageName = STAGE_NAMES[stage]
    if (this.destinations.length) {
      for (let i = 0; i < N; i++) this.destinations[i].set(
        this.group.position.x + this.positions[i * 3] * this.group.scale.x,
        this.group.position.y + this.positions[i * 3 + 1] * this.group.scale.y,
        this.positions[i * 3 + 2] * this.group.scale.z,
      )
    }
  }
}
