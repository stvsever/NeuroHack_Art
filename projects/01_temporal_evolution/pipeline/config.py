from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ASSET_DIR = ROOT / "assets"
BUILD_DIR = ROOT / ".build"
CACHE_DIR = ROOT / ".cache"
DEFAULT_ENV = ROOT.parent.parent / ".env"

START_YEAR = 1976
END_YEAR = 2026
EMBEDDING_MODEL = "openai/text-embedding-3-large"
EMBEDDING_DIMENSIONS = 256

BROAD_QUERY = r'''(
  "Nervous System"[MeSH Terms]
  OR "Neurosciences"[MeSH Terms]
  OR neuron*[Title/Abstract]
  OR neuroscien*[Title/Abstract]
  OR neurobiolog*[Title/Abstract]
  OR "neural circuit*"[Title/Abstract]
  OR brain[Title/Abstract]
  OR cognition[Title/Abstract]
  OR neuroimaging[Title/Abstract]
  OR electrophysiolog*[Title/Abstract]
  OR connectom*[Title/Abstract]
  OR "neural network*"[Title/Abstract]
)
AND hasabstract[text]
NOT (editorial[Publication Type] OR letter[Publication Type]
     OR comment[Publication Type] OR correction[Publication Type])'''

STRIPS = [
    {
        "id": "gradient",
        "name": "Basic neural gradient",
        "source_short": "field model",
        "color": "#68E0D3",
        "anchor": "Elementary biological information propagation: membrane potentials, ion and electrochemical gradients, action potentials, calcium waves, excitable media, chemotaxis, morphogen gradients, reaction diffusion, synaptic concentration gradients, and bioelectric patterning in neural systems.",
        "positive": [
            r"membrane potential", r"ion(?:ic)? gradient", r"electrochemical",
            r"calcium wave", r"action potential", r"excitable (?:media|membrane|cell)",
            r"bioelectric", r"reaction.?diffusion", r"morphogen gradient",
            r"chemotaxis gradient", r"synaptic gradient", r"voltage gradient",
        ],
        "negative": [r"gradient descent", r"mri gradient", r"gradient coil"],
    },
    {
        "id": "fungal",
        "name": "Fungal network",
        "source_short": "mycelial graph",
        "color": "#DDB36C",
        "anchor": "Fungal electrical signalling and computation: mycelial and hyphal networks, fungal electrophysiology, oscillatory activity, nutrient transport, adaptive network growth, mycorrhizal communication, decision making and information processing in fungi.",
        "positive": [
            r"fungal (?:electrical|signal|network|communication)", r"myceli",
            r"hyphal network", r"fungal electrophysi", r"mycorrhizal signal",
            r"neurospora", r"physarum",
        ],
        "negative": [r"antifungal", r"fungal infection", r"candidiasis", r"mycosis"],
    },
    {
        "id": "elegans",
        "name": "C. elegans",
        "source_short": "OpenWorm / c302",
        "color": "#DDE96F",
        "anchor": "Neuroscience of Caenorhabditis elegans: the C. elegans nervous system, connectome, nerve ring, sensory neurons, interneurons, motor neurons, neural circuits, behaviour and whole-organism neural dynamics.",
        "positive": [r"caenorhabditis elegans", r"c\.\s*elegans"],
        "negative": [],
    },
    {
        "id": "drosophila",
        "name": "Drosophila",
        "source_short": "FlyWire FAFB",
        "color": "#F18B72",
        "anchor": "Drosophila melanogaster neuroscience: fruit fly brain, neural circuits, connectome, neuropils, mushroom body, larval or adult fly sensory processing, behaviour, learning and motor control.",
        "positive": [r"drosophila melanogaster", r"drosophila", r"fruit fl(?:y|ies)", r"fly brain"],
        "negative": [r"fly ash", r"flywheel"],
    },
    {
        "id": "rodent",
        "name": "Rodent",
        "source_short": "Allen CCFv3",
        "color": "#C77DFF",
        "anchor": "Rodent neuroscience in mouse, rat, hamster, gerbil or vole: mammalian neural circuits, cortex, hippocampus, thalamus, basal ganglia, cerebellum, brainstem, behaviour, electrophysiology and disease models.",
        "positive": [
            r"\bmice\b", r"\bmouse\b", r"\bmurine\b", r"\brats?\b",
            r"\bhamsters?\b", r"\bgerbils?\b", r"\bvoles?\b", r"rodent",
        ],
        "negative": [],
    },
    {
        "id": "macaque",
        "name": "Macaque",
        "source_short": "NMT v2",
        "color": "#7898FF",
        "anchor": "Macaque neuroscience: rhesus monkey, Macaca mulatta, Macaca fascicularis or cynomolgus macaque brain, cortical organization, perception, attention, decision making, social cognition and neural recording.",
        "positive": [
            r"macaque", r"rhesus (?:monkey|macaque)", r"macaca mulatta",
            r"macaca fascicularis", r"cynomolgus",
        ],
        "negative": [r"marmoset", r"baboon", r"chimpanzee"],
    },
    {
        "id": "human",
        "name": "Homo sapiens",
        "source_short": "MNI152 2009c",
        "color": "#F2A6D7",
        "anchor": "Human neuroscience based on human participants, patients, healthy volunteers, human tissue or human-derived data: neuroimaging, EEG, MEG, intracranial recording, cognition, clinical neuroscience and postmortem brain research.",
        "positive": [
            r"human participants?", r"healthy volunteers?", r"\bpatients?\b",
            r"human (?:brain|tissue|subjects?|cortex|neurons?)", r"postmortem human",
            r"functional magnetic resonance", r"\bfmri\b", r"\bmeg\b",
            r"intracranial eeg", r"electrocorticograph",
        ],
        "negative": [r"human relevance", r"humanized mice"],
    },
    {
        "id": "ai",
        "name": "Artificial intelligence",
        "source_short": "trained topology",
        "color": "#5EC8FF",
        "anchor": "Artificial intelligence in dialogue with neuroscience: machine learning and deep learning applied to neural data, learned artificial neural networks as models of brain and cognition, representation learning, convolutional and recurrent networks, transformers, foundation models, reinforcement learning, neuromorphic systems, brain decoding and comparisons of artificial and biological intelligence.",
        "positive": [
            r"artificial intelligence", r"artificial neural network",
            r"deep neural network", r"deep learning", r"machine learning",
            r"neural networks, computer", r"convolutional neural network",
            r"recurrent neural network", r"representation learning",
            r"reinforcement learning", r"\bneuroai\b", r"neuromorphic",
            r"foundation model", r"large language model",
            r"biologically inspired (?:learning|network|ai)",
            r"brain decoding", r"transformer.*(?:brain|neuron|cort|cognit|percept)",
            r"(?:brain|neuron|cort|cognit|percept).*transformer",
            r"representational similarity.*(?:network|model)",
        ],
        "negative": [
            r"non.neurologic image segmentation", r"industrial fault",
            r"financial forecast", r"wireless network", r"traffic prediction",
        ],
    },
]

THEMES = [
    ("molecular", "Molecular & cellular", "Ion channels, receptors, synapses, neurotransmitters, molecular pathways, cellular physiology and glia.", "#67E8D3"),
    ("development", "Development & plasticity", "Neural development, axon guidance, morphogenesis, critical periods, regeneration, synaptic plasticity and learning-dependent change.", "#B7E46C"),
    ("sensation", "Sensation & perception", "Vision, audition, olfaction, somatosensation, multisensory integration and perceptual inference.", "#FFD166"),
    ("action", "Motor & action", "Motor control, movement, locomotion, action selection, basal ganglia, cerebellum and sensorimotor loops.", "#F6A261"),
    ("memory", "Learning & memory", "Memory formation and retrieval, reinforcement learning, hippocampus, navigation, conditioning and decision making.", "#F47F76"),
    ("cognition", "Cognition & language", "Attention, executive function, language, reasoning, working memory and cognitive control.", "#EF8FC6"),
    ("affect", "Social & affective", "Emotion, motivation, reward, stress, social behaviour, empathy and psychiatric dimensions.", "#C492FF"),
    ("sleep", "Sleep & consciousness", "Sleep, arousal, anesthesia, awareness, conscious perception and global brain states.", "#8797FF"),
    ("clinical", "Clinical & disease", "Neurological and psychiatric disorders, degeneration, injury, biomarkers, treatment and patient outcomes.", "#72B5FF"),
    ("networks", "Networks & computation", "Neural coding, population dynamics, oscillations, connectomes, computation, information and network theory.", "#61D5F5"),
    ("methods", "Methods & neurotechnology", "Neuroimaging, electrophysiology, microscopy, optogenetics, interfaces, stimulation and analytical methods.", "#79DED0"),
    ("neuroai", "Brain ↔ AI convergence", "NeuroAI, artificial neural networks as brain models, representational alignment, neuromorphic systems and machine intelligence.", "#E9F5FF"),
]

CIRCLE_OF_FIFTHS = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]
NOTE_NAMES = ["C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B"]


def hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
