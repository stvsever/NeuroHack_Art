from __future__ import annotations

import gzip
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import HDBSCAN

from .config import DATA_DIR, END_YEAR, ROOT, START_YEAR


GENERIC_MESH = {
    "humans", "animals", "male", "female", "adult", "aged", "middle aged",
    "young adult", "mice", "rats", "brain", "neurons", "research support, non-u.s. gov't",
    "article", "review", "research", "study", "studies", "neuroscience", "neurosciences",
    "child", "child, preschool", "rats, inbred strains", "in vitro techniques",
    "academies and institutes", "surveys and questionnaires", "students", "curriculum",
    "germany", "history, 18th century", "history, 19th century", "history, 20th century",
    "history, 21st century",
}

# These are semantic attractors, not labels assigned to individual papers.
# HDBSCAN still decides which papers form a neighborhood. The ontology only
# prevents bibliographic MeSH artefacts ("Germany", "Curriculum", age groups)
# from naming the detected neighborhood. Its language is deliberately broad,
# neuroscientifically legible, and reproducible without a generative model.
CURATED_FORMATIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("sleep", "Arousal, Homeostasis & Thermoregulation", ("body temperature", "hypothalamus", "homeostasis", "norepinephrine", "intracranial pressure")),
    ("sleep", "Dreaming & Consciousness", ("dreams", "dreaming", "consciousness", "awareness", "rapid eye movement")),
    ("sleep", "Vigilance & Electrophysiology", ("sleepiness", "vigilance", "electroencephalography", "event-related potential", "p-300", "coma")),
    ("sleep", "Sleep-Like States & Arousal Circuits", ("drosophila", "caenorhabditis", "lethargus", "arousal", "mushroom bodies")),
    ("sleep", "Circadian Timing & Sleep Homeostasis", ("circadian", "sleep homeostasis", "suprachiasmatic", "clock genes", "melatonin")),

    ("affect", "Monoamines, Mood & Affect", ("norepinephrine", "serotonin", "hydroxyindoleacetic", "depression, chemical", "monoamine")),
    ("affect", "Emotion, Reward & Social Behavior", ("emotions", "social behavior", "reward", "motivation", "attachment")),
    ("affect", "Psychoanalysis & Affective Theory", ("psychoanalysis", "psychodynamic", "metapsychology", "affective theory")),
    ("affect", "Circadian, Social & Drug-Reward Genetics", ("circadian clocks", "ethanol", "cocaine", "sexual behavior", "drosophila")),
    ("affect", "Stress, Coping & the HPA Axis", ("stress", "coping", "corticotropin", "cortisol", "hypothalamo-hypophyseal")),
    ("affect", "Fear, Threat & Amygdala Circuits", ("fear", "threat", "amygdala", "anxiety", "defensive behavior")),

    ("clinical", "Dopaminergic Therapy & Movement Disorders", ("levodopa", "dopamine antagonists", "parkinson", "3,4-dihydroxyphenylacetic", "catecholamine")),
    ("clinical", "Fungal Pharmacology & Infection", ("antifungal agents", "fungi", "mycoses", "fungal disease", "microbial sensitivity")),
    ("clinical", "Dementia, Atrophy & Cognitive Decline", ("dementia", "alzheimer disease", "atrophy", "neuropathology", "cognitive decline")),
    ("clinical", "Addiction, Psychosis & Neurodevelopment", ("behavior, addictive", "schizophrenia", "substance-related disorders", "antipsychotic", "tourette")),
    ("clinical", "Psychiatric Neuroscience", ("psychiatry", "mental disorders", "neuropsychiatry", "psychopathology", "clinical neuroscience")),
    ("clinical", "Epilepsy & Seizure Networks", ("epilepsy", "seizure", "kindling", "anticonvulsants")),
    ("clinical", "Stroke & Cerebrovascular Injury", ("stroke", "brain ischemia", "cerebrovascular", "ischemic attack")),
    ("clinical", "Pain, Migraine & Nociception", ("pain", "migraine", "nociception", "analgesia")),

    ("development", "Neurogenesis & Early Brain Development", ("animals, newborn", "chick embryo", "stem cells", "cell differentiation", "neurogenesis")),
    ("development", "Fungal Morphogenesis & Sporulation", ("spores, bacterial", "spores, fungal", "mycelium", "streptomyces", "morphogenesis")),
    ("development", "Evolution & Development of Neural Form", ("encephalization", "evolution", "neuroanatomy", "ocular dominance", "comparative anatomy")),
    ("development", "C. elegans Development & Axon Guidance", ("caenorhabditis elegans proteins", "axon pathfinding", "programmed cell death", "genes, helminth", "nerve ring")),
    ("development", "Drosophila Neural Development", ("drosophila proteins", "neural stem cells", "morphogenesis", "pattern formation", "developmental")),
    ("development", "Circuit Assembly & Synaptic Maturation", ("axon guidance", "circuit assembly", "synaptogenesis", "synaptic maturation", "growth cone")),
    ("development", "Organoids & Engineered Neural Tissue", ("organoid", "induced pluripotent", "tissue engineering", "brain organoid")),

    ("molecular", "Monoaminergic Signaling", ("norepinephrine", "serotonin", "dopamine antagonists", "monoamine", "catecholamine")),
    ("molecular", "Neuroimmunohistochemistry", ("antibodies, monoclonal", "immunohistochemistry", "fluorescent antibody", "immunoenzyme", "microtubule-associated")),
    ("molecular", "Neurovascular Injury & Blood–Brain Barrier", ("brain ischemia", "brain edema", "blood-brain barrier", "ischemic attack", "hypoxia")),
    ("molecular", "Fungal Metabolites & Fermentation", ("streptomyces", "penicillium", "fermentation", "anti-bacterial agents", "hydrogen-ion concentration")),
    ("molecular", "Membrane Excitability & Action Potentials", ("membrane potentials", "action potentials", "ion channels", "depolarization", "electrophysiology")),
    ("molecular", "Neuropeptides & Central Receptors", ("receptors, angiotensin", "enkephalins", "neuropeptides", "endorphins", "hypothalamus")),
    ("molecular", "Neuroreceptor Binding & Autoradiography", ("binding, competitive", "bungarotoxins", "affinity labels", "binding sites", "autoradiography")),
    ("molecular", "Glia & Neural Cell Culture", ("cells, cultured", "astrocytes", "glial fibrillary", "cell survival", "glia")),
    ("molecular", "Amyloid, Tau & Proteinopathy", ("amyloid beta", "neurofibrillary tangles", "tau", "alzheimer’s disease", "proteinopathy")),
    ("molecular", "Drosophila Molecular Neurogenetics", ("drosophila melanogaster", "drosophila proteins", "insect proteins", "genetically modified")),
    ("molecular", "C. elegans Molecular Neurobiology", ("caenorhabditis elegans", "sensory receptor cells", "neuropeptides", "c. elegans")),
    ("molecular", "Neuropharmacology & Receptor Systems", ("neurotransmitter agents", "neuropharmacology", "receptors, serotonin", "opiates", "neuroendocrinology")),
    ("molecular", "Origins of Molecular Neuroscience", ("nobel prize", "neural integration", "marine biological laboratory", "autonomic nervous system")),
    ("molecular", "Synapses, Vesicles & Plasticity", ("synaptic vesicles", "synaptic plasticity", "long-term potentiation", "synapses", "neurotransmitter release")),
    ("molecular", "Neuroinflammation & Glial Immunity", ("microglia", "neuroinflammation", "cytokines", "astrocytes", "immune response")),
    ("molecular", "Genomics & Neural Gene Regulation", ("gene expression", "transcriptome", "genomics", "rna sequencing", "epigenetic")),
    ("molecular", "Mitochondria & Neural Metabolism", ("mitochondria", "oxidative stress", "brain metabolism", "atp", "metabolic")),

    ("methods", "Foundational Neurochemical Mapping", ("brain chemistry", "hypothalamus", "microscopy, electron", "catecholamine", "blood flow")),
    ("methods", "Hyphal Imaging & Culture", ("mycelium", "spores, fungal", "fungi", "basidiomycota", "electron microscopic")),
    ("methods", "Neural Recording & Microelectrodes", ("microelectrodes", "electrodes, implanted", "unit-recording", "firing frequencies", "equipment design")),
    ("methods", "Structural Neuroimaging", ("tomography, x-ray computed", "magnetic resonance imaging", "brain scanner", "diagnosis, differential", "intracranial")),
    ("methods", "Stimulation, Evoked Potentials & EMG", ("electromyography", "transcranial magnetic stimulation", "evoked potentials, motor", "electrical stimulation", "nerve conduction")),
    ("methods", "Computational Neuroimaging", ("image processing, computer-assisted", "algorithms", "computer graphics", "imaging, three-dimensional", "image analyzer")),
    ("methods", "Machine Learning for Neural Signals", ("machine learning", "artificial intelligence", "neural decoding", "electroencephalography", "functional magnetic resonance imaging", "brain-computer interface")),
    ("methods", "History of Brain Mapping", ("phrenology", "franz josef gall", "freud", "neuroanatomy", "history")),
    ("methods", "Functional Imaging of Perception", ("visual cortex", "motion perception", "photic stimulation", "functional imaging", "face recognition")),
    ("methods", "Genetic & Optical Circuit Dissection", ("optogenetics", "cell typing", "drosophila", "c. elegans", "genetics to genome")),
    ("methods", "Neuroscience Education & Neuroethics", ("education, medical, undergraduate", "neuroethics", "teaching", "courseware", "writing")),
    ("methods", "Microscopy & Neural Tracing", ("microscopy", "neural tracing", "horseradish peroxidase", "fluorescence", "tract tracing")),
    ("methods", "Neuroinformatics & Open Data", ("databases, factual", "neuroinformatics", "data sharing", "open science", "brain atlas")),

    ("networks", "Excitable Biological Networks", ("biophysics", "action potentials", "membrane potentials", "neurospora", "electrotonic")),
    ("networks", "Primate Visual Circuits", ("visual cortex", "macaca mulatta", "photic stimulation", "macaque", "ganglion cells")),
    ("networks", "Computational Neuroscience & Connectomics", ("computational biology", "cognitive neuroscience", "connectome", "neural networks, computer", "functional anatomy")),
    ("networks", "Cybernetics, Perceptrons & Neural Computation", ("cybernetics", "perceptron", "adaptive systems", "parallel distributed processing", "connectionism", "pattern recognition", "computational model")),
    ("networks", "Whole-Brain Connectomics", ("connectome", "neuroarchitecture", "whole-brain", "central complex", "component placement")),
    ("networks", "Spiking Networks & Neuromorphic Models", ("neuromorphic computing", "spiking neural networks", "cortical neuromorphic", "excitable map-based")),
    ("networks", "Oscillations & Population Dynamics", ("oscillations", "population dynamics", "synchronization", "functional connectivity", "network dynamics")),

    ("memory", "Hippocampal Learning & Avoidance", ("hippocampus", "avoidance learning", "maze learning", "electroshock", "conditioning")),
    ("memory", "Human Memory Systems", ("mental recall", "prefrontal cortex", "memory", "magnetic resonance imaging", "implicit memory")),
    ("memory", "Associative Learning & Decision-Making", ("conditioning, classical", "decision making", "associative learning", "modifiable synapses", "categorization")),
    ("memory", "Drosophila Mushroom-Body Memory", ("mushroom body", "drosophila", "learning", "courtship conditioning", "camp")),
    ("memory", "Neuroscience of Learning & Education", ("problem-based learning", "education", "learning packages", "neurophysiology curriculum", "science of learning")),
    ("memory", "Primate Working Memory & Recognition", ("macaca mulatta", "memory, short-term", "prefrontal cortex", "pattern recognition, visual", "association learning")),
    ("memory", "Memory Consolidation & Replay", ("memory consolidation", "replay", "sharp-wave ripple", "systems consolidation")),
    ("memory", "Spatial Navigation & Cognitive Maps", ("spatial navigation", "place cells", "grid cells", "cognitive map")),

    ("action", "Vertebrate Locomotor Circuits", ("motor activity", "lampreys", "motoneurones", "spinal projections", "locomotor")),
    ("action", "Motor Cortex & Skilled Movement", ("movement", "motor cortex", "hand", "psychomotor performance", "electromyography")),
    ("action", "Invertebrate Sensorimotor Control", ("locomotion", "flight, animal", "drosophila", "caenorhabditis", "visual flight control")),
    ("action", "Basal Ganglia Action Selection", ("basal ganglia", "action selection", "substantia nigra", "striatum")),
    ("action", "Cerebellar Coordination & Timing", ("cerebellum", "coordination", "motor timing", "fastigial")),

    ("sensation", "Visual & Auditory Coding", ("visual perception", "visual cortex", "auditory perception", "auditory cortex", "photic stimulation")),
    ("sensation", "Invertebrate Chemosensation & Optic Flow", ("chemotaxis", "odorants", "optical flow", "drosophila", "caenorhabditis")),
    ("sensation", "Somatosensation, Touch & Pain", ("somatosensory", "touch", "mechanosensation", "nociception")),
    ("sensation", "Olfaction & Chemical Sensing", ("olfaction", "olfactory", "odor", "chemical sensing")),

    ("cognition", "Language & Neurodevelopment", ("language development disorders", "cognition disorders", "neuropsychological tests", "cleft lip", "brain specialization")),
    ("cognition", "Language, Attention & Working Memory", ("language", "linguistics", "attention", "memory, short-term", "temporal lobe")),
    ("cognition", "Executive Control & Decision-Making", ("executive function", "cognitive control", "decision making", "prefrontal cortex")),
    ("cognition", "Conscious Access & Metacognition", ("consciousness", "awareness", "metacognition", "global workspace")),

    ("neuroai", "Embodied & Artificial Intelligence", ("artificial intelligence", "robotics", "intelligence", "philosophy", "cognitive neuroscience")),
    ("neuroai", "Connectionism & Backpropagation", ("connectionism", "backpropagation", "parallel distributed processing", "neural networks, computer", "multilayer perceptron")),
    ("neuroai", "Memristive Synapses & Neuromorphic Hardware", ("semiconductors", "artificial synapse", "memristor", "spintronic", "neuromorphic hardware")),
    ("neuroai", "Deep Learning & Transformers", ("transformer", "deep learning", "neural networks, computer", "artificial neural network", "foundation model")),
    ("neuroai", "Representational Alignment & Predictive Models", ("representational similarity", "encoding model", "predictive coding", "representation learning", "brain score", "neural prediction")),
    ("neuroai", "Reinforcement Learning & Neural Agents", ("reinforcement learning", "agent", "policy learning", "reward prediction")),
    ("neuroai", "Brain–Computer Interfaces & Neural Decoding", ("brain-computer interface", "neural decoding", "brain-machine", "neuroprosthesis")),
)


def _formation_score(search_texts: list[str], keywords: tuple[str, ...]) -> float:
    """Score a detected neighborhood against a transparent semantic attractor."""
    score = 0.0
    for keyword in keywords:
        phrase = keyword.lower()
        matches = sum(phrase in text for text in search_texts)
        if matches:
            # Reward support across papers; longer concepts carry more evidence
            # than single generic tokens.
            score += math.log1p(matches) * (1.0 + min(1.4, len(phrase.split()) * .22))
    return score


def _unique_terms(terms: list[str], limit: int = 5) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for term in terms:
        key = " ".join(term.lower().split())
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(term)
        if len(unique) >= limit:
            break
    return unique


def _emergence_year(years: list[int], *, entity_specific: bool = False) -> int:
    ordered = sorted(years)
    if not ordered:
        return START_YEAR
    # Emergence is the first repeatable body of evidence, not an arbitrary late
    # percentile. Specialist histories can be real while the global field is
    # still sparse, so entity-specific formations use a three-paper floor.
    floor = 3 if entity_specific else 5
    share = .03 if entity_specific else .05
    root_scale = .42 if entity_specific else .62
    evidence_count = max(
        floor,
        math.ceil(len(ordered) * share),
        math.ceil(math.sqrt(len(ordered)) * root_scale),
    )
    return int(ordered[min(len(ordered), evidence_count) - 1])


FORM_CLUSTER_EVIDENCE = {
    "gradient": re.compile(r"action potential|membrane potential|calcium wave|bioelectric|excitable|ionic gradient"),
    "fungal": re.compile(r"fung|mycel|hyph|neurospora|streptomyces|penicill|basidiomyc|mycorrhiz"),
    "elegans": re.compile(r"caenorhabditis|c\.\s*elegans"),
    "drosophila": re.compile(r"drosophila|fruit fl(?:y|ies)|fly brain"),
    "rodent": re.compile(r"\bmouse\b|\bmice\b|\brats?\b|murine|rodent|hamster|gerbil|vole"),
    "macaque": re.compile(r"macaque|macaca|rhesus|cynomolgus"),
    "human": re.compile(r"\bhuman|\bpatients?\b|healthy volunteers?|\bfmri\b|\bmeg\b|intracranial eeg|electrocorticograph"),
    "ai": re.compile(
        r"artificial|deep learning|machine learning|neural network|transformer|"
        r"neuromorphic|neuroai|memrist|perceptron|connectionis|backpropagat|"
        r"representation learning|reinforcement learning|foundation model|large language model"
    ),
}
AI_NEUROSCIENCE_EVIDENCE = re.compile(
    r"brain|neurosci|cognit|neuromorph|synap|neuron|spiking|neural decoding|"
    r"neuroimaging|fmri|electroencephal|brain-computer|brain-machine"
)

FORM_LABEL_REWRITES = {
    "gradient": {
        "Neurogenesis & Early Brain Development": "Developmental Bioelectric Gradients",
        "Motor Cortex & Skilled Movement": "Motor-Circuit Excitability",
        "Functional Imaging of Perception": "Mapping Functional Gradients",
        "Primate Visual Circuits": "Sensory Population Excitability",
        "Primate Working Memory & Recognition": "Learning-Dependent Excitability",
        "Memristive Synapses & Neuromorphic Hardware": "Artificial Excitable Media",
        "Visual & Auditory Coding": "Sensory-Evoked Potentials",
    },
    "fungal": {
        "Development & plasticity · Anti-Bacterial Agents": "Fungal Growth & Antimicrobial Response",
        "Monoamines, Mood & Affect": "Neuroactive Fungal Metabolites",
        "Methods & neurotechnology · Magnetic Resonance Spectroscopy": "Spectroscopy of Fungal Networks",
        "Methods & neurotechnology · Aspergillus flavus": "Aspergillus Growth & Imaging",
        "Microscopy & Neural Tracing": "Microscopy of Hyphae & Mycelia",
        "Methods & neurotechnology · Methods": "Fungal Network Measurement",
        "Molecular & cellular · Agaricales": "Basidiomycete Molecular Signaling",
        "Drosophila Neural Development": "Hyphal Development & Branching",
        "Vertebrate Locomotor Circuits": "Directed Hyphal Growth",
        "Pain, Migraine & Nociception": "Fungal Pathogenicity & Neuroactive Compounds",
    },
    "elegans": {
        "Motor Cortex & Skilled Movement": "C. elegans Motor Circuits",
        "Drosophila Neural Development": "C. elegans Neural Development",
        "Dopaminergic Therapy & Movement Disorders": "Dopaminergic Modulation & Degeneration Models",
        "Drosophila Mushroom-Body Memory": "C. elegans Learning & Memory",
        "Human Memory Systems": "C. elegans Learning & Memory",
    },
    "drosophila": {
        "Fungal Morphogenesis & Sporulation": "Drosophila Neural Morphogenesis",
        "C. elegans Molecular Neurobiology": "Drosophila Molecular Neurobiology",
        "Dementia, Atrophy & Cognitive Decline": "Fly Models of Neurodegeneration",
        "Hippocampal Learning & Avoidance": "Associative Memory Circuits",
        "Fear, Threat & Amygdala Circuits": "Threat & Defensive Behavior",
        "Motor Cortex & Skilled Movement": "Flight & Locomotor Control",
    },
    "rodent": {
        "Development & plasticity · Pregnancy": "Prenatal Rodent Brain Development",
        "Clinical & disease · Phospholipids": "Neural Lipids & Disease Models",
        "Drosophila Neural Development": "Rodent Neural Development",
        "Human Memory Systems": "Rodent Memory Systems",
        "Drosophila Mushroom-Body Memory": "Rodent Learning & Memory",
        "Drosophila Molecular Neurogenetics": "Rodent Molecular Neurogenetics",
        "Invertebrate Sensorimotor Control": "Rodent Sensorimotor Circuits",
    },
    "macaque": {
        "Neurogenesis & Early Brain Development": "Primate Cortical Development",
        "Drosophila Mushroom-Body Memory": "Macaque Learning & Associative Memory",
    },
    "human": {
        "Drosophila Neural Development": "Human Neurodevelopment & Plasticity",
        "Primate Visual Circuits": "Human Visual Networks",
        "Sleep-Like States & Arousal Circuits": "Human Sleep–Wake & Arousal Networks",
        "Evolution & Development of Neural Form": "Human Brain Evolution & Development",
        "Deep Learning & Transformers": "Human–AI Representation Alignment",
    },
    "ai": {
        "Primate Visual Circuits": "Learned Models of Visual Cortex",
        "Foundational Neurochemical Mapping": "Machine Learning for Neural Biomarkers",
        "Vigilance & Electrophysiology": "AI for EEG, Sleep & Vigilance",
        "Fear, Threat & Amygdala Circuits": "Affective Computing & Amygdala Models",
        "Primate Working Memory & Recognition": "Learned Working-Memory Models",
        "Neuroinformatics & Open Data": "AI-Enabled Neuroinformatics",
        "Networks & computation · Cerebellum": "Learned Cerebellar Segmentation",
        "Language, Attention & Working Memory": "Language Models & Cognitive Neuroscience",
        "Human Memory Systems": "Learned Memory Representations",
        "Neurogenesis & Early Brain Development": "Architectural Growth & Plasticity",
        "Visual & Auditory Coding": "Multimodal Representation Learning",
        "Dementia, Atrophy & Cognitive Decline": "AI Models of Neurodegeneration",
        "Microscopy & Neural Tracing": "Computer Vision for Neural Structure",
        "Structural Neuroimaging": "AI for Structural Neuroimaging",
        "Membrane Excitability & Action Potentials": "Spiking & Excitable Computation",
        "Emotion, Reward & Social Behavior": "Affective & Reward Modeling",
        "Drosophila Mushroom-Body Memory": "Artificial Memory & Associative Learning",
        "Evolution & Development of Neural Form": "Architectural Growth & Plasticity",
        "Development & plasticity · synaptic plasticity": "Architectural Growth & Plasticity",
        "Excitable Biological Networks": "Spiking & Excitable Computation",
    },
}


def _belongs_to_form_cluster(paper: dict, form_id: str) -> bool:
    searchable = " · ".join([
        paper.get("title", ""),
        *paper.get("mesh", []),
        *paper.get("keywords", []),
    ]).lower()
    evidence = bool(FORM_CLUSTER_EVIDENCE[form_id].search(searchable))
    if form_id == "ai":
        evidence = evidence and bool(AI_NEUROSCIENCE_EVIDENCE.search(searchable))
    return (
        float(paper.get("strips", {}).get(form_id, 0)) >= .65
        and evidence
    )


def _enrich_form_clusters(payload: dict) -> dict[str, dict]:
    """Detect independent temporal formations inside each of the eight forms.

    The global formations remain untouched. A form-level neighborhood can only
    contain papers that are actually admitted to that form by the same rule the
    WebGL artwork uses.
    """
    papers = payload["papers"]
    xyz = np.asarray([paper["xyz"] for paper in papers], dtype=np.float32)
    theme_ids = [theme["id"] for theme in payload["themes"]]
    theme_colors = {theme["id"]: theme["color"] for theme in payload["themes"]}
    theme_names = {theme["id"]: theme["name"] for theme in payload["themes"]}
    formation_lookup = {label: keywords for _, label, keywords in CURATED_FORMATIONS}
    for paper in papers:
        paper["form_clusters"] = {}

    result: dict[str, list[dict]] = {}
    methods: dict[str, dict] = {}
    for strip in payload["strips"]:
        form_id = strip["id"]
        members = np.asarray(
            [index for index, paper in enumerate(papers) if _belongs_to_form_cluster(paper, form_id)],
            dtype=np.int32,
        )
        raw_clusters: list[dict] = []
        raw_id = 0
        assigned_member_indices: set[int] = set()
        min_sizes: dict[str, int] = {}
        for theme_id in theme_ids:
            theme_indices = members[
                np.asarray([papers[int(index)]["theme"] == theme_id for index in members])
            ]
            if len(theme_indices) < 8:
                continue
            min_size = max(8, min(26, len(theme_indices) // 10))
            min_sizes[theme_id] = min_size
            if len(theme_indices) < min_size * 2:
                theme_labels = np.zeros(len(theme_indices), dtype=np.int32)
            else:
                theme_labels = HDBSCAN(
                    min_cluster_size=min_size,
                    min_samples=max(4, min_size // 4),
                    cluster_selection_method="leaf",
                    cluster_selection_epsilon=0.022,
                    metric="euclidean",
                ).fit_predict(xyz[theme_indices])
                if not (set(map(int, theme_labels)) - {-1}):
                    theme_labels = np.zeros(len(theme_indices), dtype=np.int32)

            for theme_raw_id in sorted(set(map(int, theme_labels)) - {-1}):
                indices = theme_indices[theme_labels == theme_raw_id]
                if len(indices) < 8:
                    continue
                assigned_member_indices.update(map(int, indices))
                term_counts: Counter[str] = Counter()
                search_texts: list[str] = []
                for index in indices:
                    paper = papers[int(index)]
                    terms = list(paper.get("mesh", [])) + list(paper.get("keywords", []))
                    search_texts.append(" · ".join([paper.get("title", ""), *terms]).lower())
                    for term in terms:
                        clean = " ".join(term.strip().split())
                        if clean and clean.lower() not in GENERIC_MESH and len(clean) > 3:
                            term_counts[clean] += 1
                years = [int(papers[int(index)]["year"]) for index in indices]
                centroid = xyz[indices].mean(axis=0)
                distances = np.linalg.norm(xyz[indices] - centroid, axis=1)
                spread = float(np.mean(distances))
                growth = np.bincount(
                    np.asarray(years) - START_YEAR,
                    minlength=END_YEAR - START_YEAR + 1,
                )
                raw_clusters.append({
                    "raw_id": raw_id,
                    "form": form_id,
                    "emergence_year": _emergence_year(years, entity_specific=True),
                    "peak_year": START_YEAR + int(np.argmax(growth)),
                    "size": int(len(indices)),
                    "theme": theme_id,
                    "color": theme_colors[theme_id],
                    "term_counts": term_counts,
                    "search_texts": search_texts,
                    "centroid": [round(float(value), 5) for value in centroid],
                    "spread": round(spread, 5),
                    "coherence": round(float(math.exp(-spread * 1.45)), 4),
                    "growth": list(map(int, growth)),
                    "indices": indices,
                })
                raw_id += 1

        document_frequency: Counter[str] = Counter()
        for item in raw_clusters:
            document_frequency.update({term.lower(): 1 for term in item["term_counts"]})
        candidate_pairs: list[tuple[float, int, str]] = []
        for item in raw_clusters:
            for theme_id, label, keywords in CURATED_FORMATIONS:
                if theme_id != item["theme"]:
                    continue
                score = _formation_score(item["search_texts"], keywords)
                if score > 0:
                    candidate_pairs.append((score, item["raw_id"], label))
        assigned_labels: dict[int, str] = {}
        used_labels: set[str] = set()
        for _, candidate_id, label in sorted(candidate_pairs, reverse=True):
            if candidate_id in assigned_labels or label.lower() in used_labels:
                continue
            assigned_labels[candidate_id] = label
            used_labels.add(label.lower())

        for item in raw_clusters:
            ranked_terms = sorted(
                item.pop("term_counts").items(),
                key=lambda pair: (
                    pair[1] / max(item["size"], 1)
                    * math.log((len(raw_clusters) + 1) / (document_frequency[pair[0].lower()] + 1)),
                    pair[1],
                    -len(pair[0]),
                ),
                reverse=True,
            )
            vocabulary = [term for term, _ in ranked_terms[:8]]
            label = assigned_labels.get(item["raw_id"])
            if not label:
                signature = vocabulary[0] if vocabulary else "Emergent Neighborhood"
                label = f"{theme_names[item['theme']]} · {signature}"
            label = FORM_LABEL_REWRITES.get(form_id, {}).get(label, label)
            if label.lower() in used_labels and assigned_labels.get(item["raw_id"]) != label:
                label = f"{label} / {item['emergence_year']}"
            used_labels.add(label.lower())
            semantic_keywords = formation_lookup.get(label, ())
            supported = [
                keyword for keyword in semantic_keywords
                if any(keyword.lower() in text for text in item["search_texts"])
            ]
            item["vocabulary"] = _unique_terms([*supported[:3], *vocabulary])
            item["label"] = label
            item.pop("search_texts")

        raw_clusters.sort(key=lambda item: (item["emergence_year"], item["theme"], item["label"]))
        clusters: list[dict] = []
        for cluster_id, item in enumerate(raw_clusters):
            for index in item.pop("indices"):
                papers[int(index)]["form_clusters"][form_id] = cluster_id
            item.pop("raw_id")
            item["id"] = cluster_id
            clusters.append(item)
        result[form_id] = clusters
        methods[form_id] = {
            "members": int(len(members)),
            "assigned": int(len(assigned_member_indices)),
            "noise_or_small_theme": int(len(members) - len(assigned_member_indices)),
            "formations": len(clusters),
            "min_cluster_size_by_theme": min_sizes,
        }

    payload["form_clusters"] = result
    payload["form_cluster_method"] = {
        "algorithm": "form-conditioned, theme-stratified HDBSCAN",
        "membership": "Association score ≥ 0.65 plus explicit entity evidence in title, MeSH, or keywords",
        "emergence": "First repeatable entity-specific evidence body: at least three papers, scaled by formation size",
        "space": "shared global topic-shaped UMAP(3), independently density-detected inside each form",
        "scopes": methods,
        "interpretation": "Entity-specific semantic formations, not anatomical regions.",
    }
    return methods


def enrich_clusters(payload: dict) -> dict:
    papers = payload["papers"]
    xyz = np.asarray([paper["xyz"] for paper in papers], dtype=np.float32)
    themes = [theme["id"] for theme in payload["themes"]]
    paper_themes = np.asarray([paper["theme"] for paper in papers])
    labels = np.full(len(papers), -1, dtype=np.int32)
    raw_clusters: list[dict] = []
    next_raw_id = 0
    min_sizes: dict[str, int] = {}
    for theme_id in themes:
        theme_indices = np.flatnonzero(paper_themes == theme_id)
        if not len(theme_indices):
            continue
        min_size = max(14, min(36, len(theme_indices) // 12))
        min_sizes[theme_id] = min_size
        theme_labels = HDBSCAN(
            min_cluster_size=min_size,
            min_samples=max(5, min_size // 5),
            cluster_selection_method="leaf",
            cluster_selection_epsilon=0.018,
            metric="euclidean",
        ).fit_predict(xyz[theme_indices])
        detected = sorted(set(map(int, theme_labels)) - {-1})
        # A theme with no stable split remains one valid semantic formation; it
        # is better represented honestly than discarded as global noise.
        if not detected:
            detected = [0]
            theme_labels = np.zeros(len(theme_indices), dtype=np.int32)
        for theme_raw_id in detected:
            indices = theme_indices[theme_labels == theme_raw_id]
            labels[indices] = next_raw_id
            term_counts: Counter[str] = Counter()
            search_texts: list[str] = []
            for index in indices:
                terms = list(papers[index].get("mesh", [])) + list(papers[index].get("keywords", []))
                search_texts.append(
                    " · ".join([papers[index].get("title", ""), *terms]).lower()
                )
                for term in terms:
                    clean = " ".join(term.strip().split())
                    if clean and clean.lower() not in GENERIC_MESH and len(clean) > 3:
                        term_counts[clean] += 1
            years = [int(papers[index]["year"]) for index in indices]
            centroid = xyz[indices].mean(axis=0)
            distances = np.linalg.norm(xyz[indices] - centroid, axis=1)
            spread = float(np.mean(distances))
            coherence = float(math.exp(-spread * 1.45))
            growth = np.bincount(np.asarray(years) - START_YEAR, minlength=END_YEAR - START_YEAR + 1)
            raw_clusters.append(
                {
                    "raw_id": next_raw_id,
                    "emergence_year": _emergence_year(years),
                    "peak_year": START_YEAR + int(np.argmax(growth)),
                    "size": int(len(indices)),
                    "theme": theme_id,
                    "color": next(item for item in payload["themes"] if item["id"] == theme_id)["color"],
                    "term_counts": term_counts,
                    "search_texts": search_texts,
                    "centroid": [round(float(value), 5) for value in centroid],
                    "spread": round(spread, 5),
                    "coherence": round(coherence, 4),
                    "growth": list(map(int, growth)),
                    "indices": indices,
                }
            )
            next_raw_id += 1

    # TF-IDF-like naming favors vocabulary that is characteristic of one
    # formation over generic terminology repeated throughout neuroscience.
    document_frequency: Counter[str] = Counter()
    for item in raw_clusters:
        document_frequency.update({term.lower(): 1 for term in item["term_counts"]})
    candidate_pairs: list[tuple[float, int, str]] = []
    formation_lookup = {label: keywords for _, label, keywords in CURATED_FORMATIONS}
    for item in raw_clusters:
        for theme_id, label, keywords in CURATED_FORMATIONS:
            if theme_id != item["theme"]:
                continue
            score = _formation_score(item["search_texts"], keywords)
            if score > 0:
                candidate_pairs.append((score, item["raw_id"], label))
    assigned_labels: dict[int, str] = {}
    used_labels: set[str] = set()
    for _, raw_id, label in sorted(candidate_pairs, reverse=True):
        if raw_id in assigned_labels or label.lower() in used_labels:
            continue
        assigned_labels[raw_id] = label
        used_labels.add(label.lower())

    for item in raw_clusters:
        ranked_terms = sorted(
            item.pop("term_counts").items(),
            key=lambda pair: (
                pair[1] / max(item["size"], 1)
                * math.log((len(raw_clusters) + 1) / (document_frequency[pair[0].lower()] + 1)),
                pair[1],
                -len(pair[0]),
            ),
            reverse=True,
        )
        vocabulary = [term for term, _ in ranked_terms[:8]]
        theme = next(theme for theme in payload["themes"] if theme["id"] == item["theme"])
        label = assigned_labels.get(item["raw_id"])
        if not label:
            clean_term = next(
                (term for term in vocabulary if term.lower() not in used_labels),
                "Emergent Neighborhood",
            )
            label = f"{theme['name']} · {clean_term}"
        used_labels.add(label.lower())
        semantic_keywords = formation_lookup.get(label, ())
        supported_keywords = [
            keyword for keyword in semantic_keywords
            if any(keyword.lower() in text for text in item["search_texts"])
        ]
        item["vocabulary"] = _unique_terms([*supported_keywords[:3], *vocabulary])
        item["label"] = label
        item.pop("search_texts")

    raw_clusters.sort(key=lambda item: (item["emergence_year"], item["theme"], item["label"]))
    remap: dict[int, int] = {}
    clusters: list[dict] = []
    for cluster_id, item in enumerate(raw_clusters):
        remap[item["raw_id"]] = cluster_id
        for index in item.pop("indices"):
            papers[int(index)]["cluster"] = cluster_id
        item.pop("raw_id")
        item["id"] = cluster_id
        clusters.append(item)
    for index, raw_id in enumerate(labels):
        if int(raw_id) == -1:
            papers[index]["cluster"] = -1

    payload["clusters"] = clusters
    edge_count = 0
    cross_theme_edges = 0
    density_sum = 0.0
    cumulative_count = 0
    papers_by_year: dict[int, list[int]] = {year: [] for year in range(START_YEAR, END_YEAR + 1)}
    for index, paper in enumerate(papers):
        papers_by_year[int(paper["year"])].append(index)
    for frame in payload["timeline"]:
        year = int(frame["year"])
        frame["new_clusters"] = [cluster["id"] for cluster in clusters if cluster["emergence_year"] == year]
        frame["active_clusters"] = sum(cluster["emergence_year"] <= year for cluster in clusters)
        for index in papers_by_year[year]:
            paper = papers[index]
            cumulative_count += 1
            density_sum += float(paper.get("density", 0.5))
            for neighbor in paper.get("neighbors", []):
                neighbor = int(neighbor)
                if neighbor < index:
                    edge_count += 1
                    cross_theme_edges += int(papers[neighbor]["theme"] != paper["theme"])
        growth = math.log1p(cumulative_count) / max(math.log1p(len(papers)), 1e-6)
        bridge_ratio = cross_theme_edges / max(edge_count, 1)
        mean_density = density_sum / max(cumulative_count, 1)
        formation_ratio = frame["active_clusters"] / max(len(clusters), 1)
        interconnectedness = min(1.0, .30 * growth + .30 * bridge_ratio + .22 * mean_density + .18 * formation_ratio)
        frame["cumulative_sample_count"] = cumulative_count
        frame["semantic_edges"] = edge_count
        frame["cross_theme_bridge_ratio"] = round(bridge_ratio, 5)
        frame["interconnectedness"] = round(interconnectedness, 5)
    payload["cluster_method"] = {
        "algorithm": "theme-conditioned HDBSCAN",
        "space": "global topic-shaped UMAP(3), density-detected within non-redundant semantic domains",
        "min_cluster_size_by_theme": min_sizes,
        "noise_points": int(np.sum(labels == -1)),
        "vocabulary": "cluster-specific MeSH/keyword TF-IDF named through a fixed neuroscience formation ontology",
        "interpretation": "Detected semantic formations, not anatomical regions.",
    }
    _enrich_form_clusters(payload)
    return payload


def enrich_saved_corpus() -> None:
    source = DATA_DIR / "papers.json.gz"
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    enrich_clusters(payload)
    with gzip.open(source, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    public = ROOT / "public" / "data" / "corpus.json"
    public.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    manifest_path = DATA_DIR / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["clustering"] = {
            **payload["cluster_method"],
            "detected_formations": len(payload["clusters"]),
            "entity_specific_formations": {
                form_id: len(clusters) for form_id, clusters in payload.get("form_clusters", {}).items()
            },
            "entity_specific_method": payload.get("form_cluster_method"),
            "interconnectedness_drives_harmonization": True,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Detected {len(payload['clusters'])} semantic formations across {len(payload['papers'])} papers")


if __name__ == "__main__":
    enrich_saved_corpus()
