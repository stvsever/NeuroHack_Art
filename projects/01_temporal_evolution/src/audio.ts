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
}

const NAMES = ['C', 'C♯', 'D', 'E♭', 'E', 'F', 'F♯', 'G', 'A♭', 'A', 'B♭', 'B']
const FIFTHS = [2, 9, 4, 11, 6, 1, 8, 3, 10, 5, 0, 7]
const MODES: Record<string, number[]> = {
  DORIAN: [0, 2, 3, 5, 7, 9, 10],
  LYDIAN: [0, 2, 4, 6, 7, 9, 11],
  AEOLIAN: [0, 2, 3, 5, 7, 8, 10],
}

type HarmonicState = {
  root: number
  mode: string
  bpm: number
  activity: number[]
  interconnectedness: number
  masterGain: number
  arc: number
  coda: number
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
  private codaPadUntil = 0
  private lastYear = 1976
  private lastMidi = [38, 45, 62, 74, 50, 67, 57, 81]
  private state: HarmonicState = {
    root: 2, mode: 'DORIAN', bpm: 64, activity: Array(8).fill(.2),
    interconnectedness: .18, masterGain: .24, arc: 0, coda: 0,
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
      this.master.gain.setTargetAtTime(value || this.paused ? 0 : this.state.masterGain, this.ctx.currentTime, .12)
    }
  }

  isMuted(): boolean { return this.muted }

  setPaused(value: boolean): void {
    this.paused = value
    if (!this.ctx || !this.master) return
    this.master.gain.cancelScheduledValues(this.ctx.currentTime)
    this.master.gain.setTargetAtTime(value || this.muted ? 0 : this.state.masterGain, this.ctx.currentTime, value ? .035 : .12)
    if (value) this.stopVoices()
    else this.nextBeat = this.ctx.currentTime + .06
  }

  setPlaybackSpeed(value: number): void {
    this.playbackSpeed = Math.max(.5, value)
    if (this.ctx) this.nextBeat = this.ctx.currentTime + .05
  }

  seek(): void {
    this.stopVoices()
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
    this.codaPadUntil = 0
  }

  private circularStep(current: number, target: number): number {
    const currentIndex = Math.max(0, FIFTHS.indexOf(current))
    const targetIndex = Math.max(0, FIFTHS.indexOf(target))
    const clockwise = (targetIndex - currentIndex + 12) % 12
    const direction = clockwise === 0 ? 0 : clockwise <= 6 ? 1 : -1
    return FIFTHS[(currentIndex + direction + 12) % 12]
  }

  update(frame: TimelineFrame): void {
    const timelineProgress = Math.max(0, Math.min(1, (frame.year - 1976) / 50))
    const rise = timelineProgress * timelineProgress * (3 - 2 * timelineProgress)
    const climax = Math.max(0, Math.min(1, (timelineProgress - .68) / .24))
    const coda = Math.max(0, Math.min(1, (timelineProgress - .94) / .06))
    const arc = Math.max(0, Math.min(1, rise * .55 + climax * .45 - coda * .34))
    const rootIndex = Math.max(0, Math.min(11, Math.floor(((frame.centroid[0] + 1) * .5) * 12)))
    const targetRoot = coda > .55 ? 2 : FIFTHS[rootIndex]
    let root = this.state.root
    if (frame.year !== this.lastYear) {
      const cadenceYear = (frame.year - 1976) % 3 === 0 || timelineProgress > .86
      if (cadenceYear || coda > .55) root = coda > .78 ? 2 : this.circularStep(root, targetRoot)
      this.lastYear = frame.year
    }
    const mode = coda > .45
      ? 'DORIAN'
      : arc > .74 && frame.entropy > .68 ? 'LYDIAN'
        : frame.dispersion > .56 ? 'DORIAN' : 'AEOLIAN'
    const bpm = Math.round(56 + frame.entropy * 15 + Math.min(.16, frame.velocity) * 54 + arc * 28 - coda * 14)
    const interconnectedness = Math.max(.08, Math.min(1, frame.interconnectedness ?? (.2 + frame.entropy * .45)))
    const volumeGrowth = Math.max(0, Math.min(1, (Math.log10(Math.max(frame.query_count, 8_000)) - Math.log10(8_000)) / (Math.log10(165_000) - Math.log10(8_000))))
    const masterGain = .16 + volumeGrowth * .19 + arc * .065 - coda * .03
    this.state = {
      root,
      mode, bpm, activity: frame.strip_activity, interconnectedness, masterGain, arc, coda,
    }
    if (this.ctx && this.master && !this.muted && !this.paused) this.master.gain.setTargetAtTime(masterGain, this.ctx.currentTime, .45)
    if (this.ctx && this.tone) {
      const cutoff = 1_150 + arc * 2_150 + interconnectedness * 1_250 - coda * 650
      this.tone.frequency.setTargetAtTime(cutoff, this.ctx.currentTime, .9)
    }
    if (this.ctx && this.reverbSend) this.reverbSend.gain.setTargetAtTime(.10 + interconnectedness * .18 + arc * .04 + coda * .08, this.ctx.currentTime, .8)
    if (this.ctx && this.delayGain) this.delayGain.gain.setTargetAtTime(.065 + (1 - interconnectedness) * .10 + coda * .02, this.ctx.currentTime, .8)
    if (frame.year >= 2026 && coda > .72) this.ensureCodaPad()
  }

  label(): string {
    const movement = this.state.coda > .55
      ? 'CODA'
      : this.state.arc > .84 ? 'CLIMAX'
        : this.state.interconnectedness > .72 ? 'CONVERGENCE'
          : this.state.arc > .5 ? 'ASCENT' : 'FORMATION'
    return `${NAMES[this.state.root]} ${this.state.mode} · ${movement} · LINK ${Math.round(this.state.interconnectedness * 100)}% · ${this.state.bpm} BPM`
  }

  tick(): void {
    if (!this.ctx || !this.master || this.ctx.state !== 'running' || this.paused) return
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
    const progression = this.state.coda > .55 ? [0, 3, 4, 0] : [0, 4, 3, 5]
    const localRoot = progression[phrase]
    const tension = Math.max(0, Math.min(1, (this.state.arc - .62) / .28)) * (1 - this.state.coda)
    const chord = tension > .62 && phrase === 2
      ? [localRoot, localRoot + 1, localRoot + 4, localRoot + 6]
      : [localRoot, localRoot + 2, localRoot + 4, localRoot + 6]
    const harmonicPull = Math.pow(this.state.interconnectedness, 1.35)
    for (let voice = 0; voice < 8; voice++) {
      const activity = Math.max(.04, this.state.activity[voice] ?? .05)
      const density = Math.min(1, (.09 + activity * 1.22 + this.state.arc * .24) * (1 - this.state.coda * .32))
      const subdivision = [8, 6, 4, 3, 2, 2, 1, 2][voice]
      const groundPulse = voice === 0 && this.beat % 4 === 0
      if (!groundPulse && (this.beat + voice * 3) % subdivision !== 0 && Math.random() > density * .34) continue
      const integrated = ((this.beat * 37 + voice * 17) % 101) / 100 < Math.min(1, harmonicPull + this.state.coda * .38)
      const independent = (voice * 3 + this.beat + phrase) % 7
      const degree = this.state.coda > .68
        ? [0, 4, 2, 0][(this.beat + voice) % 4]
        : integrated ? chord[(this.beat + voice) % chord.length] % 7 : independent
      const pitchClass = (this.state.root + scale[degree]) % 12
      const midi = this.voiceLead(voice, pitchClass)
      if (voice < 7 && this.beat % 4 === 0) this.borrowed[(this.beat + voice) % this.borrowed.length] = degree
      this.voice(voice, this.frequency(midi), time + voice * .006, seconds, activity)
    }
    if (this.beat % 16 === 0) {
      const bedDegrees = this.state.coda > .55 ? [0, 2, 4] : [chord[0] % 7, chord[1] % 7, chord[2] % 7]
      this.harmonicBed(time, seconds * 8, bedDegrees.map(degree => (this.state.root + scale[degree]) % 12))
    }
  }

  private harmonicBed(time: number, duration: number, pitchClasses: number[]): void {
    if (!this.ctx || !this.master || !this.reverbSend) return
    const ctx = this.ctx
    const bus = ctx.createGain()
    const filter = ctx.createBiquadFilter()
    const amplitude = .0045 + this.state.interconnectedness * .011 + this.state.arc * .004
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
    const amplitude = .022 + Math.min(.028, activity * .025)

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
    if (index === 7) carrier.detune.linearRampToValueAtTime(5 + activity * 9, time + duration)
    this.track(carrier); carrier.start(time); carrier.stop(time + duration + .04)
  }

  private ensureCodaPad(): void {
    if (!this.ctx || !this.master || !this.reverbSend || this.paused || this.muted) return
    const ctx = this.ctx
    const now = ctx.currentTime
    if (now < this.codaPadUntil) return
    const duration = 10
    this.codaPadUntil = now + duration
    const bus = ctx.createGain()
    const filter = ctx.createBiquadFilter()
    filter.type = 'lowpass'
    filter.Q.value = .7
    filter.frequency.setValueAtTime(420, now)
    filter.frequency.exponentialRampToValueAtTime(2_650, now + 4.2)
    filter.frequency.exponentialRampToValueAtTime(680, now + duration)
    bus.gain.setValueAtTime(.0001, now)
    bus.gain.exponentialRampToValueAtTime(.046, now + 3.7)
    bus.gain.exponentialRampToValueAtTime(.034, now + 7.6)
    bus.gain.exponentialRampToValueAtTime(.0001, now + duration)
    bus.connect(filter).connect(this.master)
    filter.connect(this.reverbSend)
    ;[38, 45, 50, 53, 57].forEach((midi, index) => {
      const oscillator = ctx.createOscillator()
      const partial = ctx.createGain()
      oscillator.type = index < 2 ? 'sine' : 'triangle'
      oscillator.frequency.value = this.frequency(midi)
      oscillator.detune.setValueAtTime((index - 2) * 4.5, now)
      oscillator.detune.linearRampToValueAtTime(0, now + 6.2)
      partial.gain.value = [1, .62, .46, .27, .20][index]
      oscillator.connect(partial).connect(bus)
      this.track(oscillator)
      oscillator.start(now)
      oscillator.stop(now + duration + .05)
    })
  }
}
