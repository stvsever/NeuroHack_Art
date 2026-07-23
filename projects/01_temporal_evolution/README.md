# Eight Forms of Intelligence

An interactive, 4K-ready WebGL audiovisual artwork about the growth of neuroscience from 1976 to 2026. It is a fullscreen instrument for performance and screen recording, not a dashboard or conventional web app.

Exactly eight horizontal forms share one time axis and one harmonic system:

1. Basic neural gradient
2. Fungal network
3. *C. elegans*
4. *Drosophila*
5. Rodent
6. Macaque
7. *Homo sapiens*
8. Artificial intelligence

The eighth form is deliberately not a datacenter. It is **the learning machine**: a legible input → hidden/expert → output topology that morphs from perceptron to backpropagation, convolution, residual depth, attention, and finally a mixture of experts.

## Open the artwork

The root [`index.html`](./index.html) is the artwork. It is a genuine offline entry point: CSS, bundled JavaScript, atlas geometry, and the compact corpus load through adjacent classic-script files, so it can be double-clicked without a server.

For live editing:

```bash
npm install
npm run dev
```

Open `http://127.0.0.1:5173`, click **Enter the field**, and use headphones.

Global controls:

- `Space` — play / pause
- `←` / `→` — scrub
- `M` — sound on / off
- `H` — hide all interface for a clean recording
- `F` — fullscreen
- `R` — restart at 1976
- timeline, speed, sound, and fullscreen are also onscreen

Click any form name or anatomical chamber to enter its focused 3D body:

- drag to orbit; the released orientation is preserved
- scroll or pinch to travel through the surface and inspect the interior
- hover an arrived node for title, PMID, topic, journal, formation, and coherence
- double-click to restore the default camera
- `Esc` or **Return to evolution** to restore all eight strips

The publication histogram, year, corpus-volume readout, harmonic state, playback, and sound controls remain available in focus mode.

For a 4K capture, set the viewport or display to 3840 × 2160, enter fullscreen, press `H`, and capture the browser with system audio. Rendering follows the viewport and scales to high-density displays.

## Visual and interaction logic

Every individually embedded paper has two visible lives:

1. A dim **bibliographic fossil** remains in the left-to-right thematic river at its publication date.
2. A brighter particle travels from that river and settles into its form-specific scaffold.

All papers inhabit one shared semantic space. Global UMAP neighborhoods are blended with twelve explicit, non-overlapping neuroscience topic territories:

- molecular & cellular
- development & plasticity
- sensation & perception
- motor & action
- learning & memory
- cognition & language
- social & affective
- sleep & consciousness
- clinical & disease
- networks & computation
- methods & neurotechnology
- brain ↔ AI convergence

Topic color is scientific subject matter. Placement inside an anatomical scaffold is an artistic semantic territory, **not a claim of anatomical localization**.

### Focused neural activity

Entering a form does not freeze the chronology. New literature approaches from a surrounding 3D semantic shell, spirals into its final location, and briefly raises the structure’s firing state. Activity is seeded at high-degree graph hubs and phase-offset by shortest-hop distance through the scaffold’s actual edge graph. Stochastic recruitment keeps the response event-like: it is triggered by arriving publications, not a looping radial wave.

The anatomical or computational scaffold has its own density-balanced structural light: crisp baseline nodes, a low halo, and brighter edges remain visible before any paper arrives. This baseline is independent of dynamic firing, so stronger template legibility does not inflate activation intensity.

Every travelling publication remains in motion until the final instant, then eases into its settled coordinate at 2026. The coda therefore holds a complete bibliography rather than freezing late papers in transit. Archive drift and graph firing decay to zero at the endpoint while the focused specimen may continue its slow horizontal autorotation.

Semantic formations are not hand-drawn bubbles. Theme-conditioned HDBSCAN detects density-stable neighborhoods in the global manifold; cluster-specific MeSH/keyword evidence is scored against a fixed, inspectable neuroscience formation ontology so administrative tags cannot become formation names. Emergence is the first repeatable evidence body, scaled to formation size instead of a late fixed percentile. Entity histories require at least three supporting papers. Before stable separation, the focus view names the historically grounded pioneering evidence phase rather than incorrectly calling decades “pre-formation.”

The overview preserves one global stream of 66 formations. Focus mode performs a second, independent analysis for each entity: a paper needs a strong form association plus explicit entity evidence in its title, MeSH, or keywords before it can shape that entity’s formation history. The packaged focus analyses contain 11 gradient, 9 fungal, 9 *C. elegans*, 15 *Drosophila*, 24 rodent, 13 macaque, 30 human, and 21 AI formations. Papers without a stable entity-specific neighborhood remain visible but cannot lend a global or cross-species label to the focused body.

## Actual structures, without distortion

Anatomies are uniformly scaled and rotated into specimen chambers; axes are never stretched independently.

In the eight-form overview, every compact scaffold has a restrained fixed tilt and visibly turns left to right through a front-biased arc. All eight use the same motion rate and reach an oblique view in about four seconds, while avoiding the anatomies' narrow side profiles preserves their apparent volume. Settled publication particles use the same tilt, pivot, and angular motion, so the semantic body remains registered to the source geometry while revealing its depth. Labels, lanes, and chamber positions do not move.

| Form | Geometry used in the artwork | Status |
|---|---|---|
| Neural gradient | Excitable-wave / electrochemical field | Procedural: a mechanism has no atlas |
| Fungal network | Apical hyphal growth, lateral branching, resource competition, and anastomosis | Procedural: fungi have no canonical neural atlas |
| *C. elegans* | OpenWorm c302 NeuroML neuron morphology set | Source-derived |
| *Drosophila* | 78 FlyWire FAFB neuropil mesh fragments | Source-derived |
| Rodent | Allen Mouse CCFv3 100 µm annotation boundary | Source-derived |
| Macaque | NIMH Macaque Template v2 symmetric 0.5 mm brain mask | Source-derived |
| Human | TemplateFlow MNI152NLin2009cAsym 2 mm brain mask | Source-derived |
| Artificial intelligence | Time-varying feed-forward learning graph, constrained to its overview lane with an undistorted focus topology | Procedural: a trained system has no anatomical atlas |

The compact browser geometry contains 5,000 source-derived points for each biological reference, plus sparse morphology/surface edges. Raw downloads are checksummed during derivation and deleted afterward. Full provenance, source URLs, coordinate spaces, and hashes are in [`assets/template_provenance.json`](./assets/template_provenance.json).

## Corpus, scale, and embedding

- **3.13 million PubMed query matches** across the annual 1976–2026 field drive aggregate publication density, the timeline histogram, and the loudness trajectory
- **8,581 real PubMed records** are individually embedded in the packaged build
- the embedded corpus combines deterministic per-year coverage with targeted gradient, fungal, *C. elegans*, *Drosophila*, macaque, and NeuroAI material
- the dedicated neuroscience-AI query has **77,237 PubMed matches** and contributes a deterministic 1,600-paper target sample; 1,258 papers pass the stricter AI entity-formation rule and 1,809 inhabit the interactive AI body
- abstracts stream through NCBI E-utilities
- every title + abstract + compact MeSH context is embedded with `openai/text-embedding-3-large` through OpenRouter at 256 dimensions
- one global projection: L2 normalization → PCA(32) → UMAP(3) → restrained topic-topology blend
- transparent keyword/MeSH evidence plus embedding similarity provide topic and multi-form assignment; no generative LLM performs classification
- **no abstract text is saved**
- browser-facing files contain no embedding vectors; an ignored, content-addressed float16 vector cache in `.cache/` skips unchanged OpenRouter work

This two-resolution design is deliberate. The artwork does not invent semantic coordinates for millions of unembedded abstracts: aggregate PubMed match counts control mass and intensity, while only embedded papers become inspectable semantic nodes. The embedded corpus is a large, balanced artistic sample rather than an exhaustive bibliometric dataset. The 2026 interval is partial as of 2026-07-22.

Exact source counts, sampling parameters, embedding usage/cost, cache hits, projection settings, clustering parameters, and limitations live in [`data/run_manifest.json`](./data/run_manifest.json).

## Harmonic sonification

The audio is one composition, not eight unrelated sonifications. A shared modal chord field is derived from semantic centroid, dispersion, entropy, year-to-year movement, and the evolving paper-neighbor graph.

The eight voices are:

1. sine-like excitable glissandi
2. woody delayed mycelial plucks
3. glass/FM worm pointillism
4. fast articulated fly pulses
5. warm rodent strings
6. odd-harmonic macaque counterpoint
7. a detuned human formant choir
8. a morphing AI voice that borrows interval motifs from biological voices

Interconnectedness combines semantic-neighbor edges, cross-theme bridges, density, cumulative growth, and active formations. As it increases, deterministic voice-leading pulls all eight timbres toward a shared four-note chord field, detuning narrows, stereo voices cohere, and harmonized response tones become audible. Delays recede as the voices integrate while reverb and a shared low-pass master contour widen the common space. Publication volume raises orchestral density and mastered gain through a compressor, so the late literature feels larger without clipping.

Every musical control has a direct bibliometric or semantic source:

| Literature feature | Sonified parameter |
|---|---|
| Timeline position | Large-scale form, buildup, convergence, and final cadence |
| Annual PubMed query count | Log-compressed master level and orchestral event density |
| Activity of each of the eight forms | The activity of that form’s dedicated timbral voice |
| Global semantic centroid | Target tonal center, reached through restrained circle-of-fifths motion |
| Semantic entropy | Tempo pressure and modal openness |
| Manifold dispersion | Mode choice and master-filter brightness |
| Year-to-year semantic velocity | Tempo acceleration, melodic density, and vibrato energy |
| Number of active semantic formations | Harmonic-bed strength and how completely the recurring melody is stated |
| Interconnectedness and cross-theme bridges | Chord-tone probability, harmony-voice gain, stereo convergence, and removal of detuning |
| Final-timeline convergence | Progressive recruitment of all eight voices into a D-centered, fully aligned harmonic field |

The score lasts for the 178-second data ascent itself. Its final fifth is an extended integration section: the tonal center is gradually pulled toward D, the mode stabilizes as Dorian, all eight form voices are recruited more reliably, a five-part sustained field thickens, and the recurring melody gains a diatonic lower response. The final cadence occupies the last part of the moving timeline and finishes before the playhead reaches 2026.

At the exact final index, all oscillators and the master output stop. The timeline then holds the fully settled visual bibliography in silence for ten seconds before the whole work restarts automatically. Playback speed multiplies the timeline, visual formation clock, and musical scheduler together; seeking kills the old harmonic tail before rebuilding the score from the selected year, and pausing freezes both visual phase and sound rather than letting either timeline drift.

The entry gesture explicitly creates and resumes the browser audio context. If a browser or operating system suspends it, pressing play or the sound control resumes and reschedules the score. The control changes to `AUDIO RETRY` when browser audio cannot start instead of incorrectly reporting that sound is active. The mastered opening level is intentionally audible while the compressor retains headroom for the denser late composition.

## Temporal landmarks

Restrained archival cues punctuate the data chronology: the 1981 visual-system Nobel, backpropagation, early human fMRI, adult hippocampal neurogenesis, optogenetic control, the Human Connectome Project, the 2011 reproducibility reckoning, the BRAIN Initiative and cerebral organoids, attention architectures, and the 2023/2024 larval and adult fly connectomes. They provide historical orientation; cluster emergence and paper motion still come from the corpus.

Selected source anchors: [1981 Nobel](https://www.nobelprize.org/prizes/medicine/1981/summary/), [adult human hippocampal neurogenesis](https://www.nature.com/articles/nm1198_1313), [optogenetic neural control](https://pubmed.ncbi.nlm.nih.gov/16116447/), [Human Connectome Project](https://www.nimh.nih.gov/research/research-funded-by-nimh/research-initiatives/human-connectome-project-hcp), [2011 false-positive psychology](https://pubmed.ncbi.nlm.nih.gov/22006061/), [BRAIN Initiative](https://braininitiative.nih.gov/news-events/blog/brain-director-new-beginnings), [cerebral organoids](https://www.nature.com/articles/nature12517), [larval fly connectome](https://www.nsf.gov/news/scientists-complete-first-map-insect-brain), and [adult fly connectome](https://www.nature.com/articles/s41586-024-07558-y).

## Codebase

```text
index.html                     cinematic offline entry point
artwork.css                    root offline CSS bundle
artwork.js                     root offline IIFE; no module server required
offline-data.js                compact corpus + atlas globals for file:// use
build_offline.mjs              reproducible root-artifact packager
src/main.ts                    WebGL, focus camera, graph firing, timeline
src/learningMachine.ts         feed-forward perceptron → experts topology
src/audio.ts                   harmonic engine, interconnection, eight timbres
src/style.css                  responsive 4K editorial layout
public/data/corpus.json        compact browser corpus; no abstracts/vectors
public/data/templates.json     compact source-derived geometry
pipeline/corpus.py             PubMed, embedding, topic, UMAP pipeline
pipeline/clusters.py           global + eight entity-specific HDBSCAN histories
pipeline/build_templates.py    atlas acquisition, derivation, checksums
assets/template_provenance.json
data/run_manifest.json
```

## Rebuild the scientific data

The OpenRouter key is read from the project-level `.env` outside this artwork. It is never copied here.

```bash
python3 -m pip install -r requirements.txt
python3 -m pipeline.build_templates
python3 build_data.py --papers-per-year 104 --targeted-per-form 350 --targeted-ai 1600 --force
npm run build
```

`pipeline.build_templates` temporarily downloads the selected sources, derives compact reference geometry, records hashes, and removes raw downloads. PubMed searches and embedding vectors are cached by content; the normal non-`--force` command reuses a finished corpus. `--refresh-search` explicitly refreshes NCBI ID pools. A rebuild with new papers incurs an OpenRouter charge only for embedding-cache misses.

## Scientific and artistic limits

- Sampling and automatic labels are unvalidated and must not be used as bibliometric ground truth.
- Rodent publications can include non-mouse species; the Allen mouse CCF is a declared visual scaffold, not a localization assertion.
- FlyWire meshes are neuropil morphology, not every neuron in the FAFB connectome.
- The social-preview image is promotional art and is never used as data or atlas geometry.
- The work visualizes relationships among texts and forms; it does not claim that ideas literally evolved in a linear biological hierarchy.
