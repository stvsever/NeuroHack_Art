export type TimelineFrame = {
  year: number
  query_count: number
  sample_count: number
  centroid: number[]
  dispersion: number
  entropy: number
  velocity: number
  strip_activity: number[]
  interconnectedness?: number
  cumulative_sample_count?: number
  semantic_edges?: number
  active_clusters?: number
}

const NAMES = ['C', 'C♯', 'D', 'E♭', 'E', 'F', 'F♯', 'G', 'A♭', 'A', 'B♭', 'B']
const FIFTHS = [2, 9, 4, 11, 6, 1, 8, 3, 10, 5, 0, 7]
const MODES: Record<string, number[]> = {
  DORIAN: [0, 2, 3, 5, 7, 9, 10],
  LYDIAN: [0, 2, 4, 6, 7, 9, 11],
  AEOLIAN: [0, 2, 3, 5, 7, 8, 10],
}
const MELODY = [0, 2, 4, 5, 4, 2, 1, null, 2, 4, 6, 5, 4, 2, 1, 0, 0, 1, 2, 4, 3, 2, 0, null, 4, 5, 6, 4, 2, 1, 2, 0] as const
const MELODY_LENGTHS = [1.8, .82, .82, 1.65, .82, .82, 1.35, .6, 1.2, .82, 1.5, .82, .82, .82, 1.25, 2.4] as const
const CODA_MELODY = [4, 3, 2, 1, 2, 0, null, 4, 2, 1, 0] as const
const CODA_LENGTHS = [1.15, .9, .9, 1.35, .9, 2.2, .75, 1.1, 1.1, 1.35, 2.25] as const

type HarmonicState = {
  root: number
  mode: string
  bpm: number
  activity: number[]
  interconnectedness: number
  masterGain: number
  arc: number
  coda: number
  endpoint: boolean
  formationDensity: number
  semanticMotion: number
  dispersion: number
}

export class HarmonicEngine {
  private ctx?: AudioContext
  private master?: GainNode
  private reverb?: ConvolverNode
  private reverbSend?: GainNode
  private delay?: DelayNode
  private delayGain?: GainNode
  private tone?: BiquadFilterNode
  private nextBeat = 0
  private beat = 0
  private muted = false
  private paused = false
  private playbackSpeed = 1
  private activeOscillators = new Set<OscillatorNode>()
  private finalePadUntil = 0
  private lastYear = 1976
  private lastMidi = [38, 45, 62, 74, 50, 67, 57, 81]
  private leadMidi = 69
  private melodyStep = 0
  private codaMelodyStep = 0
  private state: HarmonicState = {
    root: 2, mode: 'DORIAN', bpm: 64, activity: Array(8).fill(.2),
    interconnectedness: .18, masterGain: .24, arc: 0, coda: 0, endpoint: false,
    formationDensity: 0, semanticMotion: 0, dispersion: 0,
  }
  private borrowed: number[] = [0, 7, 3, 9]

  async start(): Promise<void> {
    if (this.ctx) {
      if (this.ctx.state === 'suspended') await this.ctx.resume()
      return
    }
    const ctx = new AudioContext()
    this.ctx = ctx
    const master = ctx.createGain()
    master.gain.value = this.state.masterGain
    const compressor = ctx.createDynamicsCompressor()
    compressor.threshold.value = -18
    compressor.knee.value = 16
    compressor.ratio.value = 4
    compressor.attack.value = .02
    compressor.release.value = .55
    const tone = ctx.createBiquadFilter()
    tone.type = 'lowpass'
    tone.frequency.value = 1_900
    tone.Q.value = .48
    master.connect(tone).connect(compressor).connect(ctx.destination)
    this.master = master
    this.tone = tone

    const reverb = ctx.createConvolver()
    reverb.buffer = this.impulse(ctx, 3.8, 2.7)
    const reverbSend = ctx.createGain()
    reverbSend.gain.value = .24
    reverbSend.connect(reverb).connect(master)
    this.reverb = reverb
    this.reverbSend = reverbSend

    const delay = ctx.createDelay(1.5)
    delay.delayTime.value = .375
    const delayGain = ctx.createGain()
    delayGain.gain.value = .18
    delay.connect(delayGain).connect(delay)
    delay.connect(master)
    this.delay = delay
    this.delayGain = delayGain
    this.nextBeat = ctx.currentTime + .08
    if (ctx.state === 'suspended') await ctx.resume()
  }

  private impulse(ctx: AudioContext, seconds: number, decay: number): AudioBuffer {
    const length = Math.floor(ctx.sampleRate * seconds)
    const buffer = ctx.createBuffer(2, length, ctx.sampleRate)
    for (let channel = 0; channel < 2; channel++) {
      const data = buffer.getChannelData(channel)
      let smooth = 0
      for (let i = 0; i < length; i++) {
        smooth = smooth * .86 + (Math.random() * 2 - 1) * .14
        data[i] = smooth * Math.pow(1 - i / length, decay)
      }
    }
    return buffer
  }

  setMuted(value: boolean): void {
    this.muted = value
    if (this.ctx && this.master) {
      this.master.gain.cancelScheduledValues(this.ctx.currentTime)
      this.master.gain.setTargetAtTime(value || this.paused || this.state.endpoint ? 0 : this.state.masterGain, this.ctx.currentTime, .12)
    }
  }

  isMuted(): boolean { return this.muted }

  setPaused(value: boolean): void {
    this.paused = value
    if (!this.ctx || !this.master) return
    this.master.gain.cancelScheduledValues(this.ctx.currentTime)
    this.master.gain.setTargetAtTime(value || this.muted || this.state.endpoint ? 0 : this.state.masterGain, this.ctx.currentTime, value ? .035 : .12)
    if (value) this.stopVoices()
    else this.nextBeat = this.ctx.currentTime + .06
  }

  setPlaybackSpeed(value: number): void {
    this.playbackSpeed = Math.max(.5, value)
    if (this.ctx) this.nextBeat = this.ctx.currentTime + .05
  }

  seek(): void {
    this.stopVoices()
    this.melodyStep = 0
    this.codaMelodyStep = 0
    if (this.ctx) this.nextBeat = this.ctx.currentTime + .06
  }

  private track(oscillator: OscillatorNode): void {
    this.activeOscillators.add(oscillator)
    oscillator.addEventListener('ended', () => this.activeOscillators.delete(oscillator), { once: true })
  }

  private stopVoices(): void {
    this.activeOscillators.forEach(oscillator => {
      try { oscillator.stop() } catch { /* already ended */ }
    })
    this.activeOscillators.clear()
    this.finalePadUntil = 0
  }

  private circularStep(current: number, target: number): number {
    const currentIndex = Math.max(0, FIFTHS.indexOf(current))
    const targetIndex = Math.max(0, FIFTHS.indexOf(target))
    const clockwise = (targetIndex - currentIndex + 12) % 12
    const direction = clockwise === 0 ? 0 : clockwise <= 6 ? 1 : -1
    return FIFTHS[(currentIndex + direction + 12) % 12]
  }

  update(frame: TimelineFrame, exactProgress = (frame.year - 1976) / 50): void {
    const timelineProgress = Math.max(0, Math.min(1, exactProgress))
    const endpoint = timelineProgress >= .9999
    const wasEndpoint = this.state.endpoint
    const wasCadence = this.state.coda >= .72
    const rise = timelineProgress * timelineProgress * (3 - 2 * timelineProgress)
    const climax = Math.max(0, Math.min(1, (timelineProgress - .56) / .36))
    const coda = Math.max(0, Math.min(1, (timelineProgress - .80) / .20))
    const arc = Math.max(0, Math.min(1, rise * .42 + climax * .58 - coda * .18))
    const rootIndex = Math.max(0, Math.min(11, Math.floor(((frame.centroid[0] + 1) * .5) * 12)))
    const targetRoot = coda > .18 ? 2 : FIFTHS[rootIndex]
    let root = this.state.root
    if (frame.year !== this.lastYear) {
      const cadenceYear = (frame.year - 1976) % 3 === 0 || timelineProgress > .8
      if (cadenceYear || coda > .18) root = coda > .64 ? 2 : this.circularStep(root, targetRoot)
      this.lastYear = frame.year
    }
    const mode = coda > .22
      ? 'DORIAN'
      : arc > .74 && frame.entropy > .68 ? 'LYDIAN'
        : frame.dispersion > .56 ? 'DORIAN' : 'AEOLIAN'
    const semanticMotion = Math.max(0, Math.min(1, frame.velocity / .16))
    const formationDensity = Math.max(0, Math.min(1, (frame.active_clusters ?? 0) / 66))
    const bpm = Math.round(54 + frame.entropy * 14 + semanticMotion * 9 + arc * 26 - coda * 16)
    const interconnectedness = Math.max(.08, Math.min(1, frame.interconnectedness ?? (.2 + frame.entropy * .45)))
    const volumeGrowth = Math.max(0, Math.min(1, (Math.log10(Math.max(frame.query_count, 8_000)) - Math.log10(8_000)) / (Math.log10(165_000) - Math.log10(8_000))))
    const masterGain = .15 + volumeGrowth * .18 + arc * .06 - coda * .015
    this.state = {
      root,
      mode, bpm, activity: frame.strip_activity, interconnectedness, masterGain, arc, coda, endpoint,
      formationDensity, semanticMotion, dispersion: frame.dispersion,
    }
    if (endpoint && !wasEndpoint) {
      this.stopVoices()
      if (this.ctx && this.master) {
        this.master.gain.cancelScheduledValues(this.ctx.currentTime)
        this.master.gain.setTargetAtTime(0, this.ctx.currentTime, .006)
        this.master.gain.setValueAtTime(0, this.ctx.currentTime + .035)
      }
    } else if (!endpoint && wasEndpoint) {
      this.melodyStep = 0
      this.codaMelodyStep = 0
      if (this.ctx) this.nextBeat = this.ctx.currentTime + .06
    }
    if (coda >= .72 && !wasCadence) {
      this.codaMelodyStep = 0
      if (this.ctx) this.nextBeat = this.ctx.currentTime + .045
    }
    if (this.ctx && this.master && !this.muted && !this.paused && !endpoint) this.master.gain.setTargetAtTime(masterGain, this.ctx.currentTime, .32)
    if (this.ctx && this.tone) {
      const cutoff = 1_050 + arc * 1_850 + interconnectedness * 1_050 + frame.dispersion * 620 - coda * 420
      this.tone.frequency.setTargetAtTime(cutoff, this.ctx.currentTime, .9)
    }
    if (this.ctx && this.reverbSend) this.reverbSend.gain.setTargetAtTime(.09 + interconnectedness * .17 + formationDensity * .04 + coda * .08, this.ctx.currentTime, .8)
    if (this.ctx && this.delayGain) this.delayGain.gain.setTargetAtTime(.055 + (1 - interconnectedness) * .09 - coda * .025, this.ctx.currentTime, .8)
    if (coda > .18 && !endpoint) this.ensureFinaleField()
  }

  label(): string {
    const movement = this.state.endpoint
      ? 'SILENCE'
      : this.state.coda > .72 ? 'CADENCE'
        : this.state.coda > .18 ? 'CONVERGENCE'
      : this.state.arc > .84 ? 'CLIMAX'
        : this.state.interconnectedness > .72 ? 'CONVERGENCE'
          : this.state.arc > .5 ? 'ASCENT' : 'FORMATION'
    return `${NAMES[this.state.root]} ${this.state.mode} · ${movement} · LINK ${Math.round(this.state.interconnectedness * 100)}% · ${this.state.bpm} BPM`
  }

  tick(): void {
    if (!this.ctx || !this.master || this.ctx.state !== 'running' || this.paused || this.state.endpoint) return
    const seconds = 60 / (this.state.bpm * this.playbackSpeed)
    while (this.nextBeat < this.ctx.currentTime + .12) {
      this.scheduleBeat(this.nextBeat, seconds)
      this.nextBeat += seconds / 2
      this.beat++
    }
  }

  private frequency(midi: number): number { return 440 * Math.pow(2, (midi - 69) / 12) }

  private voiceLead(index: number, pitchClass: number): number {
    const ranges = [[34, 48], [41, 57], [55, 70], [67, 82], [45, 62], [57, 74], [50, 67], [69, 86]]
    const [low, high] = ranges[index]
    const previous = this.lastMidi[index]
    const candidates: number[] = []
    for (let midi = low; midi <= high; midi++) if ((midi % 12 + 12) % 12 === pitchClass) candidates.push(midi)
    const center = (low + high) * .5
    const chosen = candidates.sort((a, b) =>
      (Math.abs(a - previous) + Math.abs(a - center) * .08)
      - (Math.abs(b - previous) + Math.abs(b - center) * .08)
    )[0] ?? Math.round(center)
    this.lastMidi[index] = chosen
    return chosen
  }

  private scheduleBeat(time: number, seconds: number): void {
    const scale = MODES[this.state.mode]
    const phrase = Math.floor(this.beat / 8) % 4
    const progression = this.state.coda > .18 ? [0, 3, 4, 0] : [0, 4, 3, 5]
    const localRoot = progression[phrase]
    const tension = Math.max(0, Math.min(1, (this.state.arc - .62) / .28)) * (1 - this.state.coda)
    const chord = tension > .62 && phrase === 2
      ? [localRoot, localRoot + 1, localRoot + 4, localRoot + 6]
      : [localRoot, localRoot + 2, localRoot + 4, localRoot + 6]
    const harmonicPull = Math.max(Math.pow(this.state.interconnectedness, 1.35), this.state.coda * .985)
    for (let voice = 0; voice < 8; voice++) {
      const activity = Math.max(.04, this.state.activity[voice] ?? .05)
      const density = Math.min(1, .08 + activity * 1.04 + this.state.arc * .2 + this.state.formationDensity * .14 + this.state.coda * .18)
      const subdivision = [8, 6, 4, 3, 2, 2, 1, 2][voice]
      const groundPulse = voice === 0 && this.beat % 4 === 0
      const finalePulse = this.state.coda > .55 && (this.beat + voice) % Math.max(2, subdivision) === 0
      if (!groundPulse && !finalePulse && (this.beat + voice * 3) % subdivision !== 0 && Math.random() > density * .3) continue
      const integrated = ((this.beat * 37 + voice * 17) % 101) / 100 < harmonicPull
      const independent = (voice * 3 + this.beat + phrase) % 7
      const degree = this.state.coda > .72
        ? [0, 4, 2, 0][(this.beat + voice) % 4]
        : integrated ? chord[(this.beat + voice) % chord.length] % 7 : independent
      const pitchClass = (this.state.root + scale[degree]) % 12
      const midi = this.voiceLead(voice, pitchClass)
      if (voice < 7 && this.beat % 4 === 0) this.borrowed[(this.beat + voice) % this.borrowed.length] = degree
      this.voice(voice, this.frequency(midi), time + voice * .006, seconds, activity)
    }
    if (this.beat % 2 === 0) this.scheduleMelody(time + .018, seconds, scale, localRoot)
    if (this.beat % 16 === 0) {
      const bedDegrees = this.state.coda > .55 ? [0, 2, 4] : [chord[0] % 7, chord[1] % 7, chord[2] % 7]
      this.harmonicBed(time, seconds * 8, bedDegrees.map(degree => (this.state.root + scale[degree]) % 12))
    }
  }

  private nearestMelodyMidi(pitchClass: number, previous: number, low = 60, high = 81): number {
    const candidates: number[] = []
    for (let midi = low; midi <= high; midi++) if ((midi % 12 + 12) % 12 === pitchClass) candidates.push(midi)
    return candidates.sort((a, b) => Math.abs(a - previous) - Math.abs(b - previous))[0] ?? previous
  }

  private scheduleMelody(time: number, seconds: number, scale: number[], localRoot: number): void {
    let degree: number | null
    let length: number
    if (this.state.coda > .72) {
      if (this.codaMelodyStep >= CODA_MELODY.length) return
      const step = this.codaMelodyStep++
      degree = CODA_MELODY[step]
      length = CODA_LENGTHS[step]
    } else {
      const step = this.melodyStep++ % MELODY.length
      degree = MELODY[step]
      length = MELODY_LENGTHS[step % MELODY_LENGTHS.length]
      const sparseOpening = this.state.arc < .22 && step % 4 !== 0
      const melodicDensity = Math.min(1, .3 + this.state.formationDensity * .28 + this.state.semanticMotion * .18 + this.state.arc * .22)
      const motifGate = ((step * 29 + Math.floor(this.beat / 32) * 17) % 101) / 100
      if (sparseOpening || motifGate > melodicDensity) return
    }
    if (degree == null) return
    const resolvedDegree = this.state.coda > .72 ? degree : (degree + localRoot) % 7
    const pitchClass = (this.state.root + scale[resolvedDegree]) % 12
    const previous = this.leadMidi
    const midi = this.nearestMelodyMidi(pitchClass, previous)
    this.leadMidi = midi
    const harmonyDegree = (resolvedDegree + 5) % 7
    const harmonyPitch = (this.state.root + scale[harmonyDegree]) % 12
    const harmonyMidi = this.nearestMelodyMidi(harmonyPitch, midi - 4, 52, 72)
    const accent = .68 + this.state.arc * .16 + this.state.interconnectedness * .1 + this.state.formationDensity * .08 + this.state.coda * .14
    this.melodyNote(time, Math.max(.18, seconds * length), midi, harmonyMidi, previous, accent)
  }

  private melodyNote(time: number, duration: number, midi: number, harmonyMidi: number, previousMidi: number, accent: number): void {
    if (!this.ctx || !this.master || !this.reverbSend) return
    const ctx = this.ctx
    const bus = ctx.createGain()
    const filter = ctx.createBiquadFilter()
    const pan = ctx.createStereoPanner()
    const amplitude = (.018 + this.state.arc * .006 + this.state.interconnectedness * .007) * accent
    const attack = Math.min(.075, duration * .18)
    const releaseStart = Math.max(time + attack + .03, time + duration * .68)
    filter.type = 'lowpass'
    filter.frequency.value = 1_450 + this.state.arc * 1_050 + this.state.interconnectedness * 720
    filter.Q.value = .72
    pan.pan.value = -.08 * (1 - this.state.interconnectedness)
    bus.gain.setValueAtTime(.0001, time)
    bus.gain.exponentialRampToValueAtTime(amplitude, time + attack)
    bus.gain.setValueAtTime(amplitude * .82, releaseStart)
    bus.gain.exponentialRampToValueAtTime(.0001, time + duration)
    bus.connect(filter).connect(pan).connect(this.master)
    pan.connect(this.reverbSend)
    if (this.delay) pan.connect(this.delay)

    const glideStart = this.frequency(previousMidi)
    const target = this.frequency(midi)
    ;[
      { type: 'sine' as OscillatorType, gain: 1, detune: -2.4 },
      { type: 'triangle' as OscillatorType, gain: .34, detune: 2.4 },
    ].forEach((part, index) => {
      const oscillator = ctx.createOscillator()
      const partial = ctx.createGain()
      const vibrato = ctx.createOscillator()
      const vibratoDepth = ctx.createGain()
      oscillator.type = part.type
      oscillator.frequency.setValueAtTime(glideStart, time)
      oscillator.frequency.exponentialRampToValueAtTime(target, time + Math.min(.09, attack))
      oscillator.detune.value = part.detune * (1 - this.state.interconnectedness * .72)
      partial.gain.value = part.gain
      vibrato.frequency.value = 4.7 + index * .34
      vibratoDepth.gain.value = 2.4 + this.state.semanticMotion * 2.6 + this.state.arc * 1.1
      vibrato.connect(vibratoDepth).connect(oscillator.detune)
      oscillator.connect(partial).connect(bus)
      this.track(oscillator); this.track(vibrato)
      oscillator.start(time); vibrato.start(time)
      oscillator.stop(time + duration + .04); vibrato.stop(time + duration + .04)
    })

    const harmonyAmount = this.state.coda > .72
      ? .48
      : Math.max(0, (this.state.interconnectedness - .3) / .7) * (.26 + this.state.formationDensity * .12)
    if (harmonyAmount > .015) {
      const harmony = ctx.createOscillator()
      const harmonyGain = ctx.createGain()
      harmony.type = 'sine'
      harmony.frequency.value = this.frequency(harmonyMidi)
      harmonyGain.gain.setValueAtTime(.0001, time + .025)
      harmonyGain.gain.exponentialRampToValueAtTime(amplitude * harmonyAmount, time + attack + .05)
      harmonyGain.gain.exponentialRampToValueAtTime(.0001, time + duration * .94)
      harmony.connect(harmonyGain).connect(filter)
      this.track(harmony)
      harmony.start(time + .025)
      harmony.stop(time + duration + .04)
    }
  }

  private harmonicBed(time: number, duration: number, pitchClasses: number[]): void {
    if (!this.ctx || !this.master || !this.reverbSend) return
    const ctx = this.ctx
    const bus = ctx.createGain()
    const filter = ctx.createBiquadFilter()
    const amplitude = .0035 + this.state.interconnectedness * .0075 + this.state.formationDensity * .004 + this.state.arc * .0025
    filter.type = 'lowpass'
    filter.frequency.value = 780 + this.state.interconnectedness * 1_050
    filter.Q.value = .42
    bus.gain.setValueAtTime(.0001, time)
    bus.gain.exponentialRampToValueAtTime(amplitude, time + Math.min(1.4, duration * .24))
    bus.gain.exponentialRampToValueAtTime(.0001, time + duration)
    bus.connect(filter).connect(this.master)
    filter.connect(this.reverbSend)
    pitchClasses.forEach((pitchClass, index) => {
      const oscillator = ctx.createOscillator()
      const partial = ctx.createGain()
      const midi = 48 + ((pitchClass - 0 + 12) % 12) + (index === 2 ? 12 : 0)
      oscillator.type = index === 0 ? 'sine' : 'triangle'
      oscillator.frequency.value = this.frequency(midi)
      oscillator.detune.value = (index - 1) * 3.5 * (1 - this.state.interconnectedness)
      partial.gain.value = [1, .55, .32][index]
      oscillator.connect(partial).connect(bus)
      this.track(oscillator)
      oscillator.start(time)
      oscillator.stop(time + duration + .05)
    })
  }

  private voice(index: number, frequency: number, time: number, beat: number, activity: number): void {
    if (!this.ctx || !this.master || !this.reverbSend) return
    const ctx = this.ctx
    const out = ctx.createGain()
    const pan = ctx.createStereoPanner()
    const convergence = Math.max(this.state.interconnectedness, this.state.coda * .985)
    pan.pan.value = (-0.78 + index * .22) * (1 - convergence * .62)
    out.connect(pan).connect(this.master)
    out.connect(this.reverbSend)
    const amplitude = .014 + Math.min(.019, activity * .018)

    if (index === 6) {
      // Human: a soft three-part formant cloud.
      const filter = ctx.createBiquadFilter()
      filter.type = 'bandpass'; filter.frequency.value = 900; filter.Q.value = .7
      filter.connect(out)
      ;[-7, 0, 7].forEach((detune, i) => {
        const osc = ctx.createOscillator(); const gain = ctx.createGain()
        osc.type = i === 1 ? 'sine' : 'triangle'; osc.frequency.value = frequency / 2; osc.detune.value = detune * (1 - convergence * .92)
        gain.gain.setValueAtTime(.0001, time); gain.gain.exponentialRampToValueAtTime(amplitude * .46, time + .32)
        gain.gain.exponentialRampToValueAtTime(.0001, time + beat * 3.4)
        osc.connect(gain).connect(filter); this.track(osc); osc.start(time); osc.stop(time + beat * 3.6)
      })
      return
    }

    const carrier = ctx.createOscillator()
    const gain = ctx.createGain()
    const filter = ctx.createBiquadFilter()
    filter.type = index === 1 ? 'lowpass' : index === 5 ? 'bandpass' : 'lowpass'
    filter.frequency.value = [700, 520, 2400, 3400, 1100, 1500, 1800, 4200][index]
    filter.Q.value = [1.2, 4, 2.8, 1.5, 1.1, 2.2, 1, .8][index]
    carrier.type = (['sine', 'triangle', 'sine', 'square', 'triangle', 'sawtooth', 'sine', 'sine'] as OscillatorType[])[index]
    carrier.frequency.setValueAtTime(frequency, time)
    carrier.detune.setValueAtTime((1 - convergence) * Math.sin(index * 2.37 + this.beat) * 12, time)
    carrier.connect(filter).connect(gain).connect(out)

    const durations = [beat * 3.2, beat * 1.4, beat * .75, beat * .22, beat * 2.2, beat * 1.8, beat * 3, beat * .9]
    const duration = durations[index]
    const attack = [0.2, .008, .004, .002, .05, .035, .25, .012][index]
    gain.gain.setValueAtTime(.0001, time)
    gain.gain.exponentialRampToValueAtTime(amplitude, time + attack)
    gain.gain.exponentialRampToValueAtTime(.0001, time + duration)

    if (index === 2 || index === 7) {
      const mod = ctx.createOscillator(); const modGain = ctx.createGain()
      mod.frequency.value = frequency * (index === 7 ? 1.618 : 2.01)
      modGain.gain.value = frequency * (index === 7 ? .28 : .7)
      mod.connect(modGain).connect(carrier.frequency); this.track(mod); mod.start(time); mod.stop(time + duration)
    }
    if (index === 1 && this.delay) out.connect(this.delay)
    if (index === 7) carrier.detune.linearRampToValueAtTime((1 - convergence) * (5 + activity * 9), time + duration)
    this.track(carrier); carrier.start(time); carrier.stop(time + duration + .04)
  }

  private ensureFinaleField(): void {
    if (!this.ctx || !this.master || !this.reverbSend || this.paused || this.muted || this.state.endpoint) return
    const ctx = this.ctx
    const now = ctx.currentTime
    if (now < this.finalePadUntil) return
    const duration = Math.max(.72, 5.5 / this.playbackSpeed)
    this.finalePadUntil = now + duration * .88
    const bus = ctx.createGain()
    const filter = ctx.createBiquadFilter()
    filter.type = 'lowpass'
    filter.Q.value = .7
    filter.frequency.setValueAtTime(520 + this.state.dispersion * 360, now)
    filter.frequency.exponentialRampToValueAtTime(1_450 + this.state.interconnectedness * 1_350, now + duration * .46)
    filter.frequency.exponentialRampToValueAtTime(720 + this.state.coda * 260, now + duration)
    const amplitude = .012 + this.state.coda * .018 + this.state.formationDensity * .006
    bus.gain.setValueAtTime(.0001, now)
    bus.gain.exponentialRampToValueAtTime(amplitude, now + duration * .26)
    bus.gain.setValueAtTime(amplitude * .82, now + duration * .72)
    bus.gain.exponentialRampToValueAtTime(.0001, now + duration)
    bus.connect(filter).connect(this.master)
    filter.connect(this.reverbSend)
    const scale = MODES[this.state.mode]
    let bass = 36 + this.state.root
    while (bass < 38) bass += 12
    while (bass > 49) bass -= 12
    const midis = [bass, bass + scale[4], bass + 12, bass + 12 + scale[2], bass + 12 + scale[4]]
    midis.forEach((midi, index) => {
      const oscillator = ctx.createOscillator()
      const partial = ctx.createGain()
      oscillator.type = index < 2 ? 'sine' : 'triangle'
      oscillator.frequency.value = this.frequency(midi)
      oscillator.detune.setValueAtTime((index - 2) * 4.2 * (1 - this.state.coda), now)
      oscillator.detune.linearRampToValueAtTime(0, now + duration * .72)
      partial.gain.value = [1, .58, .42, .25, .18][index]
      oscillator.connect(partial).connect(bus)
      this.track(oscillator)
      oscillator.start(now)
      oscillator.stop(now + duration + .05)
    })
  }
}
