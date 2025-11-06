
# entity_linking_embedding_advanced.py
# Faster entity linking with multi-strategy search, caching, and batched embeddings.
# Adds: acronym expansion, coordinated mention splitting, species biasing,
# angiotensin routing, generic-label penalties, per-group thresholds, safer tie-breaks,
# PLUS: fast-mode, candidate caching, batched embeddings, early-exit, HTTP pool tuning.

import os, re, json, argparse, hashlib, time, threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from rapidfuzz import fuzz
import pandas as pd
import numpy as np
import requests
import unicodedata

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from openai import OpenAI
    _OPENAI_OK = bool(os.getenv("OPENAI_API_KEY"))
except Exception:
    _OPENAI_OK = False

# ===========================
# Globals tuned at runtime
# ===========================
DEFAULT_TIMEOUT = 8
FAST_MODE = False

# HTTP session (pool sizes can be increased in main())
_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=2)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

def _set_http_pool(pool_size: int):
    global _adapter
    _adapter = requests.adapters.HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, max_retries=2)
    _session.mount("https://", _adapter)
    _session.mount("http://", _adapter)

# ---------- Configuration ----------
DOMAIN_STOP_HARD = {
    "process","pathway","mechanism","function","functions","activity","activities","role",
    "state","states","feature","features","property","properties","factor","factors",
    "level","levels","amount","amounts","concentration","concentrations",
    "production","generation","synthesis","formation",
    "damage","injury","breakdown","leakage","disruption","dysfunction"
}

GREEK_MAP = {
    "α":"alpha","β":"beta","γ":"gamma","δ":"delta","ε":"epsilon","ζ":"zeta",
    "η":"eta","θ":"theta","ι":"iota","κ":"kappa","λ":"lambda","μ":"mu",
    "ν":"nu","ξ":"xi","ο":"omicron","π":"pi","ρ":"rho","σ":"sigma","τ":"tau",
    "υ":"upsilon","φ":"phi","χ":"chi","ψ":"psi","ω":"omega"
}

# ---------- Ontology Groups ----------
ENTITY_GROUPS = {
    "gene_protein": {
        "primary": {"hgnc","pr","so","go"},
        "fallback": {"efo","ncit","mesh","obi","wikidata","uniprot","mygene"}
    },
    "chemical_drug": {
        "primary": {"chebi","rxnorm","chembl","ncit"},
        "fallback": {"mesh","efo","wikidata"}
    },
    "phenotype_sign_symptom": {
        "primary": {"hp","ncit","mesh"},
        "fallback": {"efo","doid","wikidata"}
    },
    "disease_disorder": {
        "primary": {"mondo","doid","efo","ncit","mesh"},
        "fallback": {"icd10","wikidata"}
    },
    "anatomy_tissue_cell": {
        "primary": {"uberon","cl","fma"},
        "fallback": {"ncit","mesh","efo","wikidata"}
    },
    "pathway_process": {
        "primary": {"reactome","go"},
        "fallback": {"ncit","wikidata"}
    },
    "assay_device_procedure": {
        "primary": {"obi","ncit"},
        "fallback": {"efo","mesh","wikidata"}
    },
    "variant_hgvs": {
        "primary": {"so","clinvar","dbsnp"},
        "fallback": {"ncit","wikidata"}
    },
    "organism_taxon": {
        "primary": {"ncbitaxon"},
        "fallback": {"wikidata","mesh"}
    }
}

NOISY_BLOCK = {"gsso","enm","omit","ror"}

# Preferred ontologies / authorities for tie-breaking
PREFERRED_AUTHORITY = {
    "gene_protein": ["NCBIGene","HGNC","PR","UniProt"],
    "chemical_drug": ["CHEBI","RxNorm","ChEMBL","NCIT","MeSH"],
    "disease_disorder": ["MONDO","EFO","DOID","NCIT","MeSH"],
    "phenotype_sign_symptom": ["HP","NCIT","MeSH"],
    "anatomy_tissue_cell": ["UBERON","CL","FMA","NCIT"],
    "pathway_process": ["REACTOME","GO","NCIT"],
    "assay_device_procedure": ["OBI","NCIT","EFO"],
    "variant_hgvs": ["ClinVar","dbSNP","SO"],
    "organism_taxon": ["NCBITaxon","MeSH"]
}

GENERIC_BAD_LABELS = {"activation","expression","receptor","disruption","leakage","dysfunction","invasion","accumulation","breakdown","damage"}

# ---------- Caches ----------
HTTP_CACHE = None
LINK_CACHE = None
EMB_CACHE = None
CAND_CACHE = None

class DiskCache:
    def __init__(self, path: Path, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        self._lock = threading.Lock()
        self._memory = {}
        self._data = {}
        self._dirty = False
        if enabled and self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    def get(self, key: str):
        if not self.enabled:
            return None
        if key in self._memory:
            return self._memory[key]
        with self._lock:
            rec = self._data.get(key)
            if rec and "val" in rec:
                self._memory[key] = rec["val"]
                return rec["val"]
        return None

    def set(self, key: str, val):
        if not self.enabled:
            return
        self._memory[key] = val
        with self._lock:
            self._data[key] = {"ts": time.time(), "val": val}
            self._dirty = True

    def flush(self):
        if self._dirty:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
                self._dirty = False

def get_json_cached(url: str, params: dict = None, headers: dict = None):
    key = hashlib.sha1(f"{url}?{json.dumps(params, sort_keys=True)}|{json.dumps(headers, sort_keys=True)}".encode()).hexdigest()
    if HTTP_CACHE:
        hit = HTTP_CACHE.get(key)
        if hit is not None:
            return hit
    try:
        r = _session.get(url, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
        if not r.ok:
            if HTTP_CACHE:
                HTTP_CACHE.set(key, [])
            return []
        data = r.json()
    except Exception:
        data = []
    if HTTP_CACHE:
        HTTP_CACHE.set(key, data)
    return data

# ---------- Text Processing ----------
def norm_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).strip()
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    for k, v in GREEK_MAP.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def canon_entity(s: str) -> str:
    if not s: return ""
    s = norm_text(s)
    s = re.sub(r'(?<=\w)[\-\_\s]+(?=\w)', '', s)
    s = s.replace("sarscov2", "sars-cov-2").replace("covid19", "covid-19")
    s = re.sub(r'(?<!s|x|z)es$', 'e', s)
    s = re.sub(r's$', '', s)
    s = s.replace("p.", "p.").replace("c.", "c.")
    s = re.sub(r'rs0*', 'rs', s)
    return s

HGVS_RE = re.compile(r'^(?:[cnpgrm]\.)|(?:[A-Z]\d+[A-Z]$)')
RSID_RE  = re.compile(r'^rs\d+$')
TAXON_CUES = {"strain","isolate","lineage","subspecies","variant"}

def tokens(s: str) -> list:
    return re.findall(r"[a-z0-9]+", (s or "").lower())

def is_stop_token(tok: str) -> bool:
    if not tok or len(tok) < 3:
        return True
    return tok in DOMAIN_STOP_HARD

def significant_tokens(s: str) -> list:
    return [t for t in tokens(s) if len(t) >= 3 and not is_stop_token(t)]

def _sig_tokens(s: str) -> set:
    return set(significant_tokens(s))

def _token_cover_frac(q: str, cand_text: str) -> float:
    qT = _sig_tokens(q)
    cT = _sig_tokens(cand_text)
    return 0.0 if not qT else len(qT & cT) / len(qT)

def _head_noun(s: str) -> str:
    toks = re.findall(r"[a-z0-9]+", (s or "").lower())
    return toks[-1] if toks else ""

def strip_generic_tails(s: str) -> str:
    toks = tokens(s)
    if not toks:
        return s
    i = len(toks) - 1
    while i >= 0 and is_stop_token(toks[i]):
        i -= 1
    core = " ".join(toks[:i+1]).strip()
    return core if core else s

def token_coverage(q: str, cand: str) -> float:
    Q = set(significant_tokens(q))
    if not Q:
        return 0.0
    C = set(significant_tokens(cand))
    return len(Q & C) / len(Q)

def jaccard_trigram(a: str, b: str) -> float:
    a_set = {a.lower()[i:i+3] for i in range(len(a.lower())-2)}
    b_set = {b.lower()[i:i+3] for i in range(len(b.lower())-2)}
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / max(1, len(a_set | b_set))

def lexical_score(a: str, b: str) -> float:
    a_norm = canon_entity(a)
    b_norm = canon_entity(b)
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0
    ts  = fuzz.token_set_ratio(a_norm, b_norm)  / 100.0
    tr  = fuzz.token_sort_ratio(a_norm, b_norm) / 100.0
    pr  = fuzz.partial_ratio(a_norm, b_norm)    / 100.0
    jac = jaccard_trigram(a_norm, b_norm)
    cov = token_coverage(a_norm, b_norm)
    boost = 0.10 if (len(a_norm) >= 4 and len(b_norm) >= 4 and (a_norm in b_norm or b_norm in a_norm)) else 0.0
    score = 0.30*ts + 0.25*tr + 0.15*pr + 0.15*jac + 0.15*cov + boost
    return float(np.clip(score, 0.0, 1.0))

# ---------- Acronym Expansion ----------
GREEK = {
    "alpha":"α","beta":"β","gamma":"γ","delta":"δ","epsilon":"ε","zeta":"ζ",
    "eta":"η","theta":"θ","iota":"ι","kappa":"κ","lambda":"λ","mu":"μ",
    "nu":"ν","xi":"ξ","omicron":"ο","pi":"π","rho":"ρ","sigma":"σ",
    "tau":"τ","upsilon":"υ","phi":"φ","chi":"χ","psi":"ψ","omega":"ω"
}
def greek_variants(s: str):
    out = {s}
    for en, gr in GREEK.items():
        out.add(s.replace(en, gr))
        out.add(s.replace(gr, en))
    return out

def _norm_key(s: str):
    return re.sub(r'[\s\-\_]+', '', (s or "").strip()).upper()

ACRO_MAP = {
    "AD": ("Alzheimer's disease", None, "disease"),
    "PD": ("Parkinson's disease", None, "disease"),
    "ALS": ("amyotrophic lateral sclerosis", None, "disease"),
    "FTD": ("frontotemporal dementia", None, "disease"),
    "MS": ("multiple sclerosis", None, "disease"),
    "IBD": ("inflammatory bowel disease", None, "disease"),
    "UC": ("ulcerative colitis", None, "disease"),
    "CD": ("Crohn's disease", None, "disease"),
    "RA": ("rheumatoid arthritis", None, "disease"),
    "SLE": ("systemic lupus erythematosus", None, "disease"),
    "OA": ("osteoarthritis", None, "disease"),
    "COPD": ("chronic obstructive pulmonary disease", None, "disease"),
    "CKD": ("chronic kidney disease", None, "disease"),
    "CHF": ("congestive heart failure", None, "disease"),
    "MI": ("myocardial infarction", None, "disease"),
    "TBI": ("traumatic brain injury", None, "disease"),
    "ICH": ("intracerebral hemorrhage", None, "disease"),
    "AIS": ("acute ischemic stroke", None, "disease"),
    "HCC": ("hepatocellular carcinoma", None, "disease"),
    "NAFLD": ("non-alcoholic fatty liver disease", None, "disease"),
    "NASH": ("non-alcoholic steatohepatitis", None, "disease"),
    "ARDS": ("acute respiratory distress syndrome", None, "disease"),
    "ADEM": ("acute disseminated encephalomyelitis", None, "disease"),
    "ANE": ("acute necrotizing encephalopathy", None, "disease"),
    "HIV": ("human immunodeficiency virus infection", None, "disease"),
    "TB": ("tuberculosis", None, "disease"),
    "COVID19": ("COVID-19", None, "disease"),
    "COVID-19": ("COVID-19", None, "disease"),
    "BBB": ("blood–brain barrier", None, "anatomy"),
    "BCSFB": ("blood–cerebrospinal fluid barrier", None, "anatomy"),
    "CNS": ("central nervous system", None, "anatomy"),
    "PNS": ("peripheral nervous system", None, "anatomy"),
    "PBMC": ("peripheral blood mononuclear cell", None, "cell"),
    "DC": ("dendritic cell", None, "cell"),
    "NK": ("natural killer cell", None, "cell"),
    "NSC": ("neural stem cell", None, "cell"),
    "NPC": ("neural progenitor cell", None, "cell"),
    "OPC": ("oligodendrocyte progenitor cell", None, "cell"),
    "RPE": ("retinal pigment epithelium", None, "cell"),
    "HUVEC": ("human umbilical vein endothelial cell", None, "cell"),
    "BMEC": ("brain microvascular endothelial cell", None, "cell"),
    "HBMEC": ("human brain microvascular endothelial cell", None, "cell"),
    "M1": ("M1 macrophage", None, "cell"),
    "M2": ("M2 macrophage", None, "cell"),
    "TREG": ("regulatory T cell", None, "cell"),
    "ELISA": ("enzyme-linked immunosorbent assay", None, "assay"),
    "RT-PCR": ("reverse transcription polymerase chain reaction", None, "assay"),
    "QPCR": ("quantitative polymerase chain reaction", None, "assay"),
    "WB": ("western blot", None, "assay"),
    "IF": ("immunofluorescence", None, "assay"),
    "IHC": ("immunohistochemistry", None, "assay"),
    "FACS": ("fluorescence-activated cell sorting", None, "assay"),
    "RNA-SEQ": ("RNA sequencing", None, "assay"),
    "SCRNA-SEQ": ("single-cell RNA sequencing", None, "assay"),
    "WGS": ("whole-genome sequencing", None, "assay"),
    "WES": ("whole-exome sequencing", None, "assay"),
    "MRI": ("magnetic resonance imaging", None, "assay"),
    "FMRI": ("functional magnetic resonance imaging", None, "assay"),
    "PET": ("positron emission tomography", None, "assay"),
    "CT": ("computed tomography", None, "assay"),
    "EEG": ("electroencephalography", None, "assay"),
    "MEG": ("magnetoencephalography", None, "assay"),
    "IL-1B": ("interleukin-1 beta", "IL1B", "gene"),
    "IL1B": ("interleukin-1 beta", "IL1B", "gene"),
    "IL-6": ("interleukin-6", "IL6", "gene"),
    "IL6": ("interleukin-6", "IL6", "gene"),
    "TNF": ("tumor necrosis factor", "TNF", "gene"),
    "TNF-A": ("tumor necrosis factor alpha", "TNF", "gene"),
    "IFN-G": ("interferon gamma", "IFNG", "gene"),
    "IFNG": ("interferon gamma", "IFNG", "gene"),
    "TGFB": ("transforming growth factor beta 1", "TGFB1", "gene"),
    "VEGF": ("vascular endothelial growth factor A", "VEGFA", "gene"),
    "EGF": ("epidermal growth factor", "EGF", "gene"),
    "EGFR": ("epidermal growth factor receptor", "EGFR", "gene"),
    "BDNF": ("brain-derived neurotrophic factor", "BDNF", "gene"),
    "APP": ("amyloid precursor protein", "APP", "gene"),
    "MAPT": ("microtubule associated protein tau", "MAPT", "gene"),
    "SNCA": ("alpha-synuclein", "SNCA", "gene"),
    "AT1R": ("angiotensin II receptor type 1", "AGTR1", "gene"),
    "AT2R": ("angiotensin II receptor type 2", "AGTR2", "gene"),
    "ACE2": ("angiotensin-converting enzyme 2", "ACE2", "gene"),
    "ACE": ("angiotensin-converting enzyme", "ACE", "gene"),
    "TMPRSS2": ("transmembrane serine protease 2", "TMPRSS2", "gene"),
    "MTOR": ("mechanistic target of rapamycin", "MTOR", "gene"),
    "AMPK": ("AMP-activated protein kinase", None, "pathway"),
    "NF-KB": ("nuclear factor kappa-light-chain-enhancer of activated B cells", None, "pathway"),
    "ANG I": ("angiotensin I", None, "chemical"),
    "ANG II": ("angiotensin II", None, "chemical"),
    "ANG 1-7": ("angiotensin-(1-7)", None, "chemical"),
    "ANG 1-9": ("angiotensin-(1-9)", None, "chemical"),
    "AΒ": ("amyloid beta", None, "chemical"),
    "DA": ("dopamine", None, "chemical"),
    "5-HT": ("serotonin", None, "chemical"),
    "NE": ("norepinephrine", None, "chemical"),
    "NO": ("nitric oxide", None, "chemical"),
    "ROS": ("reactive oxygen species", None, "chemical"),
    "RNS": ("reactive nitrogen species", None, "chemical"),
    "AE": ("adverse event", None, "clinical"),
    "SAE": ("serious adverse event", None, "clinical"),
    "ADR": ("adverse drug reaction", None, "clinical"),
    "QOL": ("quality of life", None, "clinical"),
    "ICU": ("intensive care unit", None, "clinical"),
    "CRS": ("cytokine release syndrome", None, "clinical"),
    "CSS": ("cytokine storm syndrome", None, "clinical"),
}
ACRO_MAP_NORM = {}
def _build_acro():
    for k, v in ACRO_MAP.items():
        ACRO_MAP_NORM[_norm_key(k)] = v
        for gv in greek_variants(k):
            ACRO_MAP_NORM[_norm_key(gv)] = v
_build_acro()

PAREN_RE = re.compile(r'(?P<long>[A-Za-z][A-Za-z0-9\s\-\u00B5\u03B1-\u03C9]{3,})\s*\(\s*(?P<short>[A-Za-z][A-Za-z0-9\-]{2,15})\s*\)')
REV_PAREN_RE = re.compile(r'(?P<short>[A-Za-z][A-Za-z0-9\-]{2,15})\s*\(\s*(?P<long>[A-Za-z][A-Za-z0-9\s\-\u00B5\u03B1-\u03C9]{3,})\s*\)')

TYPE_HINT_TOKENS = {
    "disease": {"disease","syndrome","infection","carcinoma","arthritis","stroke","encephalopathy"},
    "assay": {"assay","sequencing","imaging","tomography","blot","staining","elisa","pcr","chip"},
    "cell": {"cell","cells","neuron","microglia","macrophage","endothelial","epithelium","fibroblast"},
    "anatomy": {"barrier","brain","cortex","hippocampus","spinal","lung","liver","kidney"},
    "gene": {"protein","receptor","kinase","gene"},
    "chemical": {"peptide","hormone","neurotransmitter","metabolite","angiotensin"},
    "clinical": {"patient","outcome","adverse","hospital","icu","qol","stay"},
    "pathway": {"pathway","signaling","signalling","cascade"},
}

def extract_longform_from_context(acronym: str, context: str):
    if not context:
        return None
    a_key = _norm_key(acronym)
    for m in PAREN_RE.finditer(context):
        if _norm_key(m.group('short')) == a_key:
            return m.group('long').strip()
    for m in REV_PAREN_RE.finditer(context):
        if _norm_key(m.group('short')) == a_key:
            return m.group('long').strip()
    return None

def _score_type_hint(longform: str, hint: str):
    if not hint or hint not in TYPE_HINT_TOKENS:
        return 0
    toks = set(re.findall(r'[a-z]+', longform.lower()))
    return 1 if (toks & TYPE_HINT_TOKENS[hint]) else 0

def expand_acronyms(entity: str, context: str = "") -> List[str]:
    out = []
    key = _norm_key(entity)
    if key in ACRO_MAP_NORM:
        longform, canonical, hint = ACRO_MAP_NORM[key]
        if canonical and canonical.upper() != key:
            out.append(canonical)
        out.append(longform)
    lf = extract_longform_from_context(entity, context or "")
    if lf:
        out.append(lf)
    for v in list(out):
        out.extend(greek_variants(v))
    if key in ACRO_MAP_NORM:
        _, _, hint = ACRO_MAP_NORM[key]
        out = sorted(set([s.strip() for s in out if s.strip()]),
                     key=lambda s: _score_type_hint(s, hint),
                     reverse=True)
    else:
        out = list(dict.fromkeys([s.strip() for s in out if s.strip()]))
    return out

# ---------- Group Guessing ----------
ANGIOTENSIN_RE = re.compile(r'^(ang(iotensin)?)([\s\-\_]?)(ii|iv|i|1-7|1-9)$', re.I)
HINT_TO_GROUP = {
    "disease":  "disease_disorder",
    "assay":    "assay_device_procedure",
    "cell":     "anatomy_tissue_cell",
    "anatomy":  "anatomy_tissue_cell",
    "gene":     "gene_protein",
    "chemical": "chemical_drug",
    "clinical": "phenotype_sign_symptom",
    "pathway":  "pathway_process",
}
def guess_entity_group(mention_raw: str, mention_canon: str, context: str) -> str:
    akey = _norm_key(mention_raw)
    if akey in ACRO_MAP_NORM:
        _, _, hint = ACRO_MAP_NORM[akey]
        if hint and hint in HINT_TO_GROUP:
            return HINT_TO_GROUP[hint]
    if ANGIOTENSIN_RE.match(mention_raw) or ANGIOTENSIN_RE.match(mention_canon):
        return "chemical_drug"
    if RSID_RE.match(mention_canon) or HGVS_RE.match(mention_canon):
        return "variant_hgvs"
    if re.fullmatch(r'[a-z]{2,6}\d{0,3}', mention_canon) and mention_raw.isupper():
        return "gene_protein"
    if mention_canon.endswith("ase") or mention_canon.endswith("protein") or re.fullmatch(r'il\d+', mention_canon):
        return "gene_protein"
    if mention_canon.endswith("cell") or mention_canon.endswith("cells") or "barrier" in mention_canon:
        return "anatomy_tissue_cell"
    if any(s in mention_canon for s in ["mg","mol","mmol","ic50","ec50"]) or any(mention_canon.endswith(suf) for suf in ("ate","ide","ine","ol","one","ium")):
        return "chemical_drug"
    if any(k in mention_canon for k in ["pathway","signaling","signalling","cascade","process"]):
        return "pathway_process"
    if any(k in mention_canon for k in ["assay","elisa","western blot","rt-pcr","pcr","immunostain","flow cytometry","facs","sequencing"]):
        return "assay_device_procedure"
    if "sars-cov-2" in mention_canon or "influenza" in mention_canon or any(k in (context or "").lower() for k in TAXON_CUES):
        return "organism_taxon"
    if any(k in mention_canon for k in ["fever","pain","fatigue","hypoxia","dyspnea","phenotype","abnormality"]):
        return "phenotype_sign_symptom"
    if any(k in mention_canon for k in ["disease","syndrome","infection","cancer","carcinoma","diabetes","covid-19"]):
        return "disease_disorder"
    if any(k in mention_canon for k in ["cell","neuron","tissue","organ","epithelium","endothelium"]):
        return "anatomy_tissue_cell"
    return "disease_disorder"

# ---------- Candidate Helpers ----------
def _cand_text(c: Dict) -> str:
    lab = c.get("label") or ""
    syn = "; ".join((c.get("synonyms") or [])[:8])
    defi = c.get("definition") or ""
    return f"{lab} [SYN] {syn} [DEF] {defi}".strip()

# ---------- Surface Forms Generation ----------
def generate_surface_forms(entity: str, context: str = "", max_forms: int = 9999) -> List[str]:
    """Generate multiple search variants for an entity, with acronym expansion first."""
    forms = set()

    for x in expand_acronyms(entity, context):
        forms.add(x)

    forms.add(entity)
    forms.add(canon_entity(entity))
    forms.add(re.sub(r'(?<=\w)[\-\_\s]+(?=\w)', '', entity))

    forms.add(entity.replace("-", ""))
    forms.add(entity.replace("_", ""))
    forms.add(re.sub(r'[-_]+', ' ', entity))
    forms.add(re.sub(r'[_\s]+', '-', entity))
    forms.add(re.sub(r'[-\s]+', '_', entity))

    forms.add(entity.replace("COVID19", "COVID-19"))
    forms.add(entity.replace("covid19", "covid-19"))

    forms.add(re.sub(r'\s+', '', entity))

    core = strip_generic_tails(entity)
    if core and core != entity:
        forms.add(core)
        forms.add(canon_entity(core))

    no_space = entity.replace(" ", "").replace("-", "")
    if len(no_space) > 3:
        forms.add(no_space)

    camel_split = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', entity)
    if camel_split != entity:
        forms.add(camel_split)

    m = re.match(r'(?i)^(pp|p|phospho)[\s-]*([A-Za-z][A-Za-z0-9\-]+)', entity)
    if m:
        base = m.group(2)
        forms.add(base); forms.add(f"phosphorylated {base}"); forms.add(f"{base} phosphorylation")

    if context:
        for longform, short in re.findall(r'([A-Za-z][A-Za-z0-9\s\-]{2,})\s*\(\s*([A-Z][A-Za-z0-9\-]{1,10})\s*\)', context):
            if _norm_key(entity) == _norm_key(short):   forms.add(longform)
            if _norm_key(entity) == _norm_key(longform): forms.add(short)

    plain = re.sub(r'[_\-]+', ' ', entity).strip()
    if ' ' in plain:
        forms.add(f"\"{plain}\"")
        forms.add(" ".join(w.capitalize() for w in plain.split()))
        forms.add(plain.replace(' ', '-'))
        forms.add(plain.replace(' ', '_'))

    # Dedup by canonical and cap count (fast mode will pass a small max_forms)
    canon_map = {}
    for f in forms:
        f_clean = f.strip()
        if f_clean and len(f_clean) >= 2:
            f_canon = canon_entity(f_clean)
            if f_canon and f_canon not in canon_map:
                canon_map[f_canon] = f_clean

    all_forms = list(canon_map.values())

    # Prioritize simpler forms first (shorter & unquoted earlier)
    def form_priority(f):
        return (f.startswith('"'), len(f))
    all_forms = sorted(all_forms, key=form_priority)

    return all_forms[:max_forms]

# ---------- Embeddings ----------
class Embedder:
    def __init__(self, model: str, enabled: bool):
        self.enabled = bool(enabled and _OPENAI_OK)
        self.model = model
        self.client = OpenAI() if self.enabled else None
        self.cache = {}

    def encode_batch(self, texts: List[str]) -> Dict[str, np.ndarray]:
        if not self.enabled or not texts:
            return {}
        results = {}
        to_fetch = []
        for text in texts:
            text = text.strip()
            if not text:
                continue
            if text in self.cache:
                results[text] = self.cache[text]
            elif EMB_CACHE:
                cached = EMB_CACHE.get(text)
                if cached is not None:
                    v = np.array(cached, dtype=np.float32)
                    self.cache[text] = v
                    results[text] = v
                else:
                    to_fetch.append(text)
            else:
                to_fetch.append(text)
        if to_fetch:
            try:
                for i in range(0, len(to_fetch), 100):
                    batch = to_fetch[i:i+100]
                    r = self.client.embeddings.create(model=self.model, input=batch)
                    for j, text in enumerate(batch):
                        v = np.array(r.data[j].embedding, dtype=np.float32)
                        v = v / (np.linalg.norm(v) + 1e-9)
                        self.cache[text] = v
                        results[text] = v
                        if EMB_CACHE:
                            EMB_CACHE.set(text, v.tolist())
            except Exception as e:
                print(f"Embedding error: {e}")
        return results

    def sim_cached(self, a_text: str, b_text: str) -> float:
        va = self.cache.get(a_text)
        vb = self.cache.get(b_text)
        if va is None or vb is None:
            return 0.0
        return float(np.dot(va, vb))

# ---------- Ontology Sources ----------
def ols_search(term: str, ontos: List[str] = None, rows: int = 20) -> List[Dict]:
    params = {"q": term, "rows": rows}
    if ontos:
        params["ontology"] = ",".join(ontos)
    out = []
    try:
        data = get_json_cached("https://www.ebi.ac.uk/ols4/search", params)
        docs = data.get("response", {}).get("docs", []) if isinstance(data, dict) else []
        for d in docs:
            out.append({
                "label": d.get("label") or "",
                "iri": d.get("iri", ""),
                "obo_id": d.get("obo_id", ""),
                "short_form": d.get("short_form", ""),
                "ontology_name": (d.get("ontology_name") or "").lower(),
                "source": "OLS",
                "synonyms": d.get("synonym", []) or [],
                "definition": " ".join(d.get("description", []) or [])[:300]
            })
    except Exception:
        pass
    if not out:
        try:
            data = get_json_cached("https://www.ebi.ac.uk/ols/api/search", params)
            docs = data.get("response", {}).get("docs", []) if isinstance(data, dict) else []
            for d in docs:
                out.append({
                    "label": d.get("label") or "",
                    "iri": d.get("iri", ""),
                    "obo_id": d.get("obo_id", ""),
                    "short_form": d.get("short_form", ""),
                    "ontology_name": (d.get("ontology_name") or "").lower(),
                    "source": "OLS",
                    "synonyms": d.get("synonym", []) or [],
                    "definition": " ".join(d.get("description", []) or [])[:300]
                })
        except Exception:
            pass
    return out

def bioportal_annotate(term: str, ontos: List[str], apikey: str) -> List[Dict]:
    if not apikey:
        return []
    headers = {"Authorization": f"apikey token={apikey}"}
    params = {
        "text": term,
        "ontologies": ",".join(ontos) if ontos else "",
        "longest_only": "false",
        "whole_word_only": "false",
        "max_level": 0
    }
    data = get_json_cached("https://data.bioontology.org/annotator", params, headers)
    out = []
    for ann in (data or []):
        cls = ann.get("annotatedClass", {}) or {}
        definition = cls.get("definition") or ""
        if isinstance(definition, list):
            definition = (definition[0] if definition else "")[:300]
        else:
            definition = str(definition)[:300]
        onto_link = (cls.get("links", {}) or {}).get("ontology", "")
        onto = onto_link.split("/")[-1].lower() if onto_link else ""
        out.append({
            "label": cls.get("prefLabel") or "",
            "iri": cls.get("@id", ""),
            "obo_id": cls.get("notation") or "",
            "ontology_name": onto,
            "source": "BioPortal",
            "synonyms": (cls.get("synonym", []) or [])[:10],
            "definition": definition
        })
    return out

def mygene_search(term: str, rows: int = 10) -> List[Dict]:
    is_symbol = bool(re.match(r"^[A-Za-z0-9\-]{2,15}$", term))
    q = f"symbol:{term}" if is_symbol else term
    params = {"q": q, "species": "all", "size": rows, "fields": "symbol,name,taxid,entrezgene,alias"}
    data = get_json_cached("https://mygene.info/v3/query", params)
    out = []
    for h in data.get("hits", []) if isinstance(data, dict) else []:
        txt = h.get("name", "") or ""
        taxid = h.get("taxid")
        syns = (h.get("alias", []) or []) if isinstance(h.get("alias"), list) else []
        out.append({
            "label": h.get("symbol") or h.get("name") or term,
            "ontology_name": "mygene",
            "source": "MyGene",
            "synonyms": syns[:10],
            "definition": (txt + (f" taxid:{taxid}" if taxid else ""))[:300],
            "curie": f"NCBIGene:{h.get('entrezgene')}" if h.get("entrezgene") else None,
            "iri": "",
            "taxid": taxid
        })
    return out

def uniprot_search(term: str, rows: int = 10) -> List[Dict]:
    params = {"query": term, "format": "json", "size": rows, "fields": "accession,id,protein_name,gene_names,organism_name,organism_id"}
    data = get_json_cached("https://rest.uniprot.org/uniprotkb/search", params)
    out = []
    for res in data.get("results", []) if isinstance(data, dict) else []:
        acc = res.get("primaryAccession")
        prot_name = ""
        try:
            prot_name = (res.get("proteinDescription", {})
                           .get("recommendedName", {})
                           .get("fullName", {})
                           .get("value", "")) or ""
        except Exception:
            pass
        org = (res.get("organism", {}) or {}).get("scientificName") or ""
        taxid = (res.get("organism", {}) or {}).get("taxonId")
        out.append({
            "label": prot_name or acc or term,
            "ontology_name": "uniprot",
            "source": "UniProt",
            "synonyms": [],
            "definition": (org + (f" taxid:{taxid}" if taxid else ""))[:300],
            "curie": f"UniProtKB:{acc}" if acc else None,
            "iri": "",
            "taxid": taxid
        })
    return out

def wikidata_search(term: str, rows: int = 10) -> List[Dict]:
    params_search = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "type": "item",
        "limit": rows,
        "search": term,
    }
    data = get_json_cached("https://www.wikidata.org/w/api.php", params_search)
    qids = [hit.get("id") for hit in (data.get("search", []) if isinstance(data, dict) else []) if hit.get("id")]
    if not qids:
        return []
    params_entities = {
        "action": "wbgetentities",
        "format": "json",
        "props": "labels|descriptions|aliases",
        "languages": "en",
        "ids": "|".join(qids[:rows])
    }
    entities = get_json_cached("https://www.wikidata.org/w/api.php", params_entities)
    if not isinstance(entities, dict) or "entities" not in entities:
        return []
    out = []
    for qid in qids[:rows]:
        e = entities["entities"].get(qid, {})
        label = (e.get("labels", {}).get("en", {}) or {}).get("value", "") or term
        desc = (e.get("descriptions", {}).get("en", {}) or {}).get("value", "")
        aliases_src = e.get("aliases", {}).get("en", []) or []
        aliases = [a.get("value") for a in aliases_src if a.get("value")]
        out.append({
            "label": label,
            "ontology_name": "wikidata",
            "source": "Wikidata",
            "synonyms": aliases[:15],
            "definition": desc[:300],
            "curie": f"Wikidata:{qid}",
            "iri": f"https://www.wikidata.org/wiki/{qid}",
        })
    return out

# ---------- Candidate Collection (FAST + caching) ----------
def collect_candidates(entity: str, context: str = "") -> List[Dict]:
    """Collect candidates using group-routed, multi-source search, with caching and fast-mode caps."""
    bioportal_key = os.getenv("BIOPORTAL_API_KEY")
    mc = canon_entity(entity)
    group = guess_entity_group(entity, mc, context or "")

    cache_key = f"{group}|{mc}"
    if CAND_CACHE:
        hit = CAND_CACHE.get(cache_key)
        if hit is not None:
            return hit

    primary_ontos  = list(ENTITY_GROUPS.get(group, {}).get("primary", []))
    fallback_ontos = list(ENTITY_GROUPS.get(group, {}).get("fallback", []))
    if group == "chemical_drug":
        primary_ontos = [o for o in primary_ontos if o != "mondo"]

    max_forms = 6 if FAST_MODE else 12
    forms = generate_surface_forms(entity, context, max_forms=max_forms)

    rows_primary  = 12 if FAST_MODE else 20
    rows_fallback = 6  if FAST_MODE else 10
    rows_wikidata = 3  if FAST_MODE else 8  # will be conditionally used

    all_cands: List[Dict] = []

    for form in forms:
        if primary_ontos:
            all_cands += ols_search(form, ontos=[o for o in primary_ontos if o not in NOISY_BLOCK], rows=rows_primary)
        # Only use fallback if primary is still sparse
        if len(all_cands) < 5 and fallback_ontos:
            all_cands += ols_search(form, ontos=[o for o in fallback_ontos if o not in NOISY_BLOCK], rows=rows_fallback)

        # BioPortal (primary first; fallback if still sparse)
        if bioportal_key:
            all_cands += bioportal_annotate(form, primary_ontos, bioportal_key)
            if len(all_cands) < 5 and fallback_ontos:
                all_cands += bioportal_annotate(form, fallback_ontos, bioportal_key)

        # Specialized sources
        if group == "gene_protein":
            all_cands += mygene_search(form, rows=15 if not FAST_MODE else 8)
            if len(form) >= 3:
                all_cands += uniprot_search(form, rows=10 if not FAST_MODE else 6)

        # Defer Wikidata unless still sparse (or fast mode)
        if len(all_cands) < 5:
            all_cands += wikidata_search(form, rows=rows_wikidata)

    # Deduplicate
    seen = set()
    uniq = []
    for c in all_cands:
        label_key = canon_entity(c.get("label", ""))
        onto = (c.get("ontology_name") or "").lower()
        curie = c.get("curie") or c.get("obo_id") or ""
        key = (onto, curie, label_key)
        if key not in seen and label_key:
            seen.add(key)
            uniq.append(c)

    # Broad phrase-first fallback if too few
    if len(uniq) < 5:
        extra_forms = []
        for f in forms[:3]:
            pf = re.sub(r'[_\-]+', ' ', f).strip()
            if ' ' in pf:
                extra_forms.extend([f"\"{pf}\"", pf])

        rescue_forms = (extra_forms + forms[:3])[: (2 if FAST_MODE else 6)]
        for f in rescue_forms:
            uniq += ols_search(f, ontos=None, rows=(40 if not FAST_MODE else 20))
            if bioportal_key:
                uniq += bioportal_annotate(f, [], bioportal_key)

        # Re-dedup
        seen = set()
        uniq2 = []
        for c in uniq:
            label_key = canon_entity(c.get("label", ""))
            onto = (c.get("ontology_name") or "").lower()
            curie = c.get("curie") or c.get("obo_id") or ""
            key = (onto, curie, label_key)
            if key not in seen and label_key:
                seen.add(key)
                uniq2.append(c)
        uniq = uniq2

    if CAND_CACHE:
        CAND_CACHE.set(cache_key, uniq)
    return uniq

# ---------- Scoring ----------
@dataclass
class LinkResult:
    entity: str
    matched_label: Optional[str]
    ontology_id: Optional[str]
    ontology_name: Optional[str]
    iri: Optional[str]
    source: Optional[str]
    confidence: float
    lex: float
    embed_sim: float
    normalized: Optional[str] = None

def score_candidate(q_entity: str, q_canon: str, cand: Dict, group: str, embedder, context: str) -> Dict:
    """Lexical & heuristic scoring only (embedding added separately in bulk)."""
    label = cand.get("label","")
    onto  = (cand.get("ontology_name") or "").lower()
    curie = (cand.get("curie") or cand.get("obo_id") or "")
    iri   = cand.get("iri","")

    lex = lexical_score(q_entity, label)
    akey = _norm_key(q_entity)

    if akey in ACRO_MAP_NORM:
        _, _, hint = ACRO_MAP_NORM[akey]
        if hint == "disease":
            lab_can = canon_entity(label)
            if re.fullmatch(r'[a-z]+[ -]?\d+', lab_can):
                lex -= 0.35
            low = (label or "").lower()
            if re.search(r'\b(adenovirus|virus|serotype|strain)\b', low) and "infection" not in low:
                lex -= 0.25
        if canon_entity(label) == q_canon:
            lex += 0.25
        if hint == "disease":
            longform, canonical_sym, _ = ACRO_MAP_NORM[akey]
            lf_lex = lexical_score(longform, label)
            lf_cov = token_coverage(longform, _cand_text(cand))
            lex += 0.40 * max(lf_lex, lf_cov)
            trap_low = (label or "").lower()
            if (len(q_entity) <= 3 and ("/" in label or
                re.search(r'\b(cell line|clone|passage|aliquot|lot|isolate)\b', trap_low) or
                (len(tokens(label)) <= 2 and len(label) <= 8))):
                lex -= 0.35

    prim = ENTITY_GROUPS.get(group, {}).get("primary", set())
    fall = ENTITY_GROUPS.get(group, {}).get("fallback", set())
    if onto in prim:   lex += 0.20
    if onto in fall:   lex += 0.05
    if onto in NOISY_BLOCK: lex -= 0.30

    taxid = cand.get("taxid")
    deftext = (cand.get("definition") or "") + " " + " ".join(cand.get("synonyms") or [])
    if group == "gene_protein" and cand.get("source") in ("MyGene","UniProt"):
        if (taxid == 9606) or re.search(r'\b(human|homo sapiens|taxid:?9606)\b', deftext, re.I):
            lex += 0.15
        elif (taxid and taxid != 9606) or re.search(r'\b(mouse|rat|drosophila|zebrafish|arabidopsis|murine|taxid:\d+)\b', deftext, re.I):
            lex -= 0.15

    q_tokens = _sig_tokens(q_entity)
    cand_text_for_cover = " ".join([label] + (cand.get("synonyms") or [])[:6] + [cand.get("definition") or ""])
    cover = _token_cover_frac(q_entity, cand_text_for_cover)
    if len(q_tokens) >= 2:
        if cover < 0.50:
            lex -= 0.25
        elif cover < 0.70:
            lex -= 0.10
        head_q = _head_noun(q_entity)
        head_l = _head_noun(label)
        if head_q and head_q != head_l and cover < 0.80:
            lex -= 0.10

    if len(q_tokens) >= 2 and canon_entity(label) in GENERIC_BAD_LABELS:
        lex -= 0.30

    low = label.lower()
    if group == "disease_disorder" and any(k in low for k in ["vaccine","immunization","dose","drug"]):
        lex -= 0.20
    if group == "chemical_drug" and any(k in low for k in ["pathway","infection","syndrome","disease"]):
        lex -= 0.15
    if group != "assay_device_procedure" and onto == "ncit" and any(k in low for k in ["product","cellular therapy","gaia","grade"]):
        lex -= 0.20

    # No embeddings here; added in bulk.
    emb_sim = 0.0
    conf = float(np.clip(lex, 0.0, 1.0))
    return {
        "label": label,
        "ontology_name": onto,
        "curie": curie,
        "iri": iri,
        "source": cand.get("source"),
        "synonyms": cand.get("synonyms", [])[:10],
        "definition": cand.get("definition",""),
        "lex": float(np.clip(lex, 0.0, 1.5)),
        "embed_sim": emb_sim,
        "confidence": conf
    }

def score_candidates_bulk(q_entity: str, q_canon: str, cands: List[Dict],
                          group: str, embedder: Optional[Embedder], context: str,
                          threshold: float) -> List[Dict]:
    """Score candidates with lexical heuristics + optional batched embeddings (fast)."""
    # Pre-score without embeddings & filter noisy ontologies early
    prelim: List[Dict] = []
    for c in cands:
        onto = (c.get("ontology_name") or "").lower()
        if onto in NOISY_BLOCK:
            continue
        prelim.append(score_candidate(q_entity, q_canon, c, group, None, context))

    if not prelim:
        return []

    # Decide if embeddings are necessary (cheap guard)
    top_lex = max(p["confidence"] for p in prelim)
    multiword = len(_sig_tokens(q_entity)) >= 2
    need_embeds = (
        embedder and embedder.enabled and multiword and (top_lex < (threshold + 0.20))
    )

    if not need_embeds:
        prelim.sort(key=lambda x: x["confidence"], reverse=True)
        return prelim

    # Batch embeddings: 1 query + all candidates
    ctx = " ".join((context or "").split()[:40])
    q_text = f"{q_entity} [CTX] {ctx}"
    cand_texts = [_cand_text({"label": p["label"], "synonyms": p.get("synonyms", []), "definition": p.get("definition","")}) for p in prelim]

    embedder.encode_batch([q_text])
    embedder.encode_batch(cand_texts)

    # Map text to vector in cache; then adjust scores
    enhanced: List[Dict] = []
    vq = embedder.cache.get(q_text)
    for p, ct in zip(prelim, cand_texts):
        vc = embedder.cache.get(ct)
        emb = float(np.dot(vq, vc)) if (vq is not None and vc is not None) else 0.0
        p2 = dict(p)
        p2["embed_sim"] = emb
        p2["lex"] = min(1.5, p2["lex"] + ((0.35 if multiword else 0.15) * emb))
        p2["confidence"] = float(np.clip(p2["lex"], 0.0, 1.0))
        enhanced.append(p2)

    enhanced.sort(key=lambda x: x["confidence"], reverse=True)
    return enhanced

def link_entity(entity: str, context: str, embedder: Embedder, threshold: float) -> LinkResult:
    # Cache
    cache_key = canon_entity(entity)
    if LINK_CACHE:
        cached = LINK_CACHE.get(cache_key)
        if cached:
            cached["entity"] = entity
            return LinkResult(**cached)

    # Collect candidates
    candidates = collect_candidates(entity, context)
    if not candidates:
        return LinkResult(entity, None, None, None, None, None, 0.0, 0.0, 0.0)

    q_canon = canon_entity(entity)
    group   = guess_entity_group(entity, q_canon, context or "")

    # Score all candidates (batched embeddings)
    scored = score_candidates_bulk(entity, q_canon, candidates, group, embedder, context, threshold)
    if not scored:
        return LinkResult(entity, None, None, None, None, None, 0.0, 0.0, 0.0)

    best = scored[0]

    # Preferred-authority tie-break within Δ
    prefer = PREFERRED_AUTHORITY.get(group, [])
    if prefer and len(scored) >= 2:
        upper_pref = [p.upper() for p in prefer]
        for cand in scored[:5]:
            cand_ont = (cand.get("ontology_name") or "").upper()
            if group == "gene_protein" and cand_ont in upper_pref:
                if cand["confidence"] >= best["confidence"] - 0.03:
                    best = cand
                    break
            elif cand_ont in upper_pref and cand["confidence"] >= best["confidence"] - 0.02:
                best = cand
                break

    # Coverage floor for multiword queries
    if len(_sig_tokens(entity)) >= 2:
        if _token_cover_frac(entity, best["label"]) < 0.50:
            return LinkResult(entity, None, None, None, None, None, 0.0, 0.0, 0.0)

    # Group-aware thresholds
    base_thr = threshold
    group_thr = {
        "gene_protein": base_thr + 0.10,
        "variant_hgvs": base_thr + 0.05,
        "chemical_drug": base_thr - 0.03,
        "disease_disorder": base_thr,
        "phenotype_sign_symptom": base_thr - 0.05,
        "anatomy_tissue_cell": base_thr + 0.05,
        "assay_device_procedure": base_thr + 0.05,
        "pathway_process": base_thr,
        "organism_taxon": base_thr - 0.05,
    }
    min_thr = float(np.clip(group_thr.get(group, base_thr), 0.40, 0.95))

    # Early exit if clearly confident
    if best["confidence"] >= min_thr + 0.12:
        pass
    else:
        if best["confidence"] < min_thr:
            # Light rescue: phrase-only on primary ontologies
            forms = generate_surface_forms(entity, context, max_forms=3)
            phrase_forms = []
            for f in forms:
                pf = re.sub(r'[_\-]+', ' ', f).strip()
                if ' ' in pf:
                    phrase_forms.extend([f"\"{pf}\"", pf])
            rescue_candidates = []
            for f in phrase_forms[: (2 if FAST_MODE else 4)]:
                rescue_candidates += ols_search(f, ontos=list(ENTITY_GROUPS.get(group, {}).get("primary", [])), rows=(25 if FAST_MODE else 40))
            if rescue_candidates:
                rescue_scored = score_candidates_bulk(entity, q_canon, rescue_candidates, group, embedder, context, threshold)
                if rescue_scored:
                    cand = rescue_scored[0]
                    if cand["confidence"] >= max(min_thr - 0.02, 0.45):
                        best = cand
                    else:
                        return LinkResult(entity, None, None, None, None, None, 0.0, 0.0, 0.0)
            else:
                return LinkResult(entity, None, None, None, None, None, 0.0, 0.0, 0.0)

    # CURIE
    curie = best.get("curie") or best.get("obo_id")
    if not curie and best.get("short_form") and best.get("ontology_name"):
        curie = f"{(best['ontology_name'] or '').upper()}:{best['short_form']}"

    result = LinkResult(
        entity=entity,
        matched_label=best["label"],
        ontology_id=curie,
        ontology_name=best.get("ontology_name"),
        iri=best.get("iri"),
        source=best.get("source"),
        confidence=best["confidence"],
        lex=best["lex"],
        embed_sim=best.get("embed_sim", 0.0)
    )

    if LINK_CACHE:
        LINK_CACHE.set(cache_key, result.__dict__)

    return result

# ---------- Coordinated mention splitting ----------
COORD_SEP_RE = re.compile(r'\s*(?:/|&|,|\band\b)\s*', re.I)
def split_coordinated(entity: str) -> Optional[List[str]]:
    if any(ch in entity for ch in ['/', '&', ',']) or ' and ' in entity.lower():
        parts = [p.strip() for p in COORD_SEP_RE.split(entity) if p.strip()]
        if 1 < len(parts) <= 6 and all(len(p) >= 2 for p in parts):
            return parts
    return None

# ---------- Main Processing ----------
def run_linking(df: pd.DataFrame,
                subject_col: str,
                object_col: str,
                context_col: Optional[str],
                use_embeddings: bool,
                embed_model: str,
                max_workers: int,
                threshold: float,
                verbose: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run entity linking with parallel processing."""

    # Collect unique entities
    entities = set()
    for col in [subject_col, object_col]:
        if col in df.columns:
            entities.update(df[col].dropna().astype(str).tolist())
    entities = sorted(entities)

    if verbose:
        print(f"📊 Processing {len(entities)} unique entities")

    # Build context map (optional)
    contexts: Dict[str, str] = {}
    if context_col and context_col in df.columns:
        subj_ctx = df[[subject_col, context_col]].dropna()
        obj_ctx  = df[[object_col, context_col]].dropna()
        for ent in entities:
            row = subj_ctx[subj_ctx[subject_col] == ent]
            if not row.empty:
                contexts[ent] = str(row.iloc[0][context_col])
                continue
            row = obj_ctx[obj_ctx[object_col] == ent]
            if not row.empty:
                contexts[ent] = str(row.iloc[0][context_col])

    # Precompute embeddings only if not fast (warm cache)
    embedder = Embedder(embed_model, enabled=use_embeddings)

    if embedder.enabled and verbose and not FAST_MODE:
        print("🔢 Precomputing embeddings...")
        sample_entities = entities[:min(50, len(entities))]
        all_labels = set()
        for ent in sample_entities:
            ctx = contexts.get(ent, "")
            cands = collect_candidates(ent, ctx)[:5]
            for c in cands:
                if c.get("label"):
                    all_labels.add(_cand_text(c))
        if all_labels:
            embedder.encode_batch(list(all_labels))
        all_qtexts = []
        for ent in sample_entities:
            ctx = contexts.get(ent, "")
            q_text = f"{ent} [CTX] {' '.join((ctx or '').split()[:40])}"
            all_qtexts.append(q_text)
        embedder.encode_batch(list(all_labels) + all_qtexts)
        print(f"✓ Cached {len(embedder.cache)} embeddings")

    if verbose:
        print(f"🔗 Linking entities (parallel, {max_workers} workers)...")

    rows = []
    success = fail = 0

    def process_entity(ent: str) -> List[Dict]:
        ctx = contexts.get(ent, "")
        results: List[Dict] = []

        parts = split_coordinated(ent)
        if parts:
            for p in parts:
                r = link_entity(p, ctx, embedder, threshold)
                results.append(r.__dict__)
            comp = LinkResult(ent, None, None, None, None, None, 0.0, 0.0, 0.0).__dict__
            results.append(comp)
            return results

        r = link_entity(ent, ctx, embedder, threshold)
        results.append(r.__dict__)
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_entity, ent): ent for ent in entities}
        #iterator = tqdm(as_completed(futures), total=len(entities), desc="Linking") if (tqdm and verbose and not FAST_MODE) else as_completed(futures)
        # show progress regardless of FAST_MODE
        iterator = tqdm(as_completed(futures), total=len(entities), desc="Linking") \
    if (tqdm and verbose) else as_completed(futures)

        for future in iterator:
            try:
                result_list = future.result()
                for result in result_list:
                    rows.append(result)
                    if result["ontology_id"]:
                        success += 1
                    else:
                        fail += 1
            except Exception as e:
                fail += 1
                if verbose:
                    print(f"❌ Error: {e}")

    if verbose:
        total = max(1, len(rows))
        print(f"\n📊 Results: {success} linked, {fail} unlinked ({len(rows)} rows incl. splits)")
        linked_rows = sum(1 for r in rows if r.get("ontology_id"))
        print(f"   Linked rows: {linked_rows}/{len(rows)} ({100*linked_rows/max(1,len(rows)):.1f}%)")

    links = pd.DataFrame(rows)

    # Enrich original triples
    df_enriched = df.copy()

    subj_map = links.add_prefix("subject_")
    df_enriched = df_enriched.merge(
        subj_map,
        left_on=subject_col,
        right_on="subject_entity",
        how="left"
    )
    df_enriched = df_enriched.drop(columns=["subject_entity"], errors="ignore")

    obj_map = links.add_prefix("object_")
    df_enriched = df_enriched.merge(
        obj_map,
        left_on=object_col,
        right_on="object_entity",
        how="left"
    )
    df_enriched = df_enriched.drop(columns=["object_entity"], errors="ignore")

    return links, df_enriched

# ---------- File I/O ----------
def auto_resolve_input_and_outdir(input_arg: Optional[str], outdir_arg: Optional[str]) -> Tuple[Path, Path]:
    import glob
    def find_all_triples(base: Path) -> Optional[Path]:
        for name in ("all_triples.xlsx", "all_triples.csv"):
            p = base / name
            if p.exists():
                return p
        return None
    in_path = None
    if input_arg:
        cand = Path(input_arg)
        if cand.is_dir():
            found = find_all_triples(cand)
            if not found:
                raise FileNotFoundError(f"No all_triples.(xlsx|csv) in: {cand}")
            in_path = found
        else:
            if not cand.exists():
                raise FileNotFoundError(f"Input not found: {cand}")
            in_path = cand
    else:
        candidates = [
            Path("gpt_triples_pdf/all_triples.xlsx"),
            Path("gpt_triples_pdf/all_triples.csv"),
            Path("gpt_triples/all_triples.xlsx"),
            Path("gpt_triples/all_triples.csv"),
        ]
        in_path = next((p for p in candidates if p.exists()), None)
        if not in_path:
            hits = glob.glob("**/all_triples.xlsx", recursive=True)
            hits += glob.glob("**/all_triples.csv", recursive=True)
            if hits:
                hits.sort(key=lambda s: len(Path(s).parts))
                in_path = Path(hits[0])

    if not in_path:
        raise FileNotFoundError(
            "Could not find input. Pass --input FILE or place all_triples.(xlsx|csv) in gpt_triples/ or gpt_triples_pdf/"
        )

    if outdir_arg:
        outdir = Path(outdir_arg)
    else:
        tag_pdf = any("pdf" in part.lower() for part in in_path.parts)
        outdir = Path("entity_linking_results_pdf" if tag_pdf else "entity_linking_results")

    return in_path, outdir

# ---------- CLI ----------
def main():
    global HTTP_CACHE, LINK_CACHE, EMB_CACHE, CAND_CACHE, DEFAULT_TIMEOUT, FAST_MODE

    parser = argparse.ArgumentParser(description="Entity linker (fast, cached, batched-embeddings).")
    parser.add_argument("--input", help="Input CSV/XLSX file or directory")
    parser.add_argument("input_pos", nargs="?", help="(Optional) Positional input")
    parser.add_argument("--subject-col", default="subject", help="Subject column")
    parser.add_argument("--object-col", default="object", help="Object column")
    parser.add_argument("--context-col", default=None, help="Context column")
    parser.add_argument("--outdir", default=None, help="Output directory")
    parser.add_argument("--use-embeddings", action="store_true", help="Use OpenAI embeddings for semantic matching")
    parser.add_argument("--embed-model", default="text-embedding-3-large", help="Embedding model")
    parser.add_argument("--max-workers", type=int, default=10, help="Number of parallel workers")
    parser.add_argument("--threshold", type=float, default=0.60, help="Base minimum confidence threshold")
    parser.add_argument("--cache-dir", default=".cache_linker", help="Cache directory")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache first")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")
    parser.add_argument("--show-unlinked", action="store_true", help="Show unlinked entities")
    parser.add_argument("--fast", action="store_true", help="Speed-first mode: fewer forms/sources, smaller rows, quicker timeouts")
    args = parser.parse_args()

    # Fast mode (CLI or env var LINKER_FAST=1)
    FAST_MODE = bool(args.fast or os.getenv("LINKER_FAST") == "1")

    # Setup caching
    CACHE_DIR = Path(args.cache_dir)
    if args.clear_cache and CACHE_DIR.exists():
        import shutil
        print(f"🗑️  Clearing cache: {CACHE_DIR}")
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    HTTP_CACHE = DiskCache(CACHE_DIR / "http.json", enabled=not args.no_cache)
    LINK_CACHE = DiskCache(CACHE_DIR / "links.json", enabled=not args.no_cache)
    EMB_CACHE  = DiskCache(CACHE_DIR / "embeddings.json", enabled=not args.no_cache)
    CAND_CACHE = DiskCache(CACHE_DIR / "candidates.json", enabled=not args.no_cache)

    # Tune HTTP and timeouts for speed
    if FAST_MODE:
        DEFAULT_TIMEOUT = 4
        _set_http_pool(200)
    else:
        DEFAULT_TIMEOUT = 8
        _set_http_pool(100)

    # Resolve input/output
    raw_input = args.input or args.input_pos
    in_path, outdir = auto_resolve_input_and_outdir(raw_input, args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"🚀 Entity Linker (Multi-Strategy, Fast={FAST_MODE})")
        print(f"📥 Input:  {in_path}")
        print(f"📦 Output: {outdir}")
        print(f"⚙️  Workers: {args.max_workers}")
        print(f"🎯 Base Threshold: {args.threshold}")
        print(f"🔢 Embeddings: {args.use_embeddings}")
        print()

    # Load data
    suffix = in_path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(in_path)
    else:
        df = pd.read_csv(in_path, sep=None, engine="python")

    for col in [args.subject_col, args.object_col]:
        if col not in df.columns:
            print(f"❌ Error: Column '{col}' not found in {list(df.columns)}")
            return

    if not args.quiet:
        print(f"📊 Loaded {len(df)} triples")

    # Run linking
    start_time = time.time()
    links, enriched = run_linking(
        df=df,
        subject_col=args.subject_col,
        object_col=args.object_col,
        context_col=args.context_col,
        use_embeddings=args.use_embeddings,
        embed_model=args.embed_model,
        max_workers=args.max_workers,
        threshold=args.threshold,
        verbose=not args.quiet
    )
    elapsed = time.time() - start_time

    # Flush caches
    if not args.no_cache:
        HTTP_CACHE.flush()
        LINK_CACHE.flush()
        EMB_CACHE.flush()
        CAND_CACHE.flush()

    # Save results
    links_path    = outdir / "entity_links.csv"
    enriched_path = outdir / "triples_enriched.csv"
    links.to_csv(links_path, index=False, encoding="utf-8")
    enriched.to_csv(enriched_path, index=False, encoding="utf-8")

    if not args.quiet:
        print(f"\n✅ Complete in {elapsed:.1f}s")
        print(f"📄 Entity links: {links_path}")
        print(f"📄 Enriched triples: {enriched_path}")

        linked   = links[links["ontology_id"].notna()]
        unlinked = links[links["ontology_id"].isna()]

        if len(linked) > 0:
            print(f"\n📈 Summary:")
            print(f"   Linked: {len(linked)}/{len(links)} ({100*len(linked)/max(1,len(links)):.1f}%)")
            print(f"   Avg confidence: {linked['confidence'].mean():.2f}")
            print(f"   Avg lexical: {linked['lex'].mean():.2f}")
            if args.use_embeddings:
                print(f"   Avg embedding: {linked['embed_sim'].mean():.2f}")

            onto_counts = linked['ontology_name'].value_counts()
            print(f"\n   Top ontologies:")
            for onto, count in onto_counts.head(5).items():
                print(f"      {onto}: {count}")

            print(f"\n   Example matches:")
            for _, row in linked.head(5).iterrows():
                print(f"      {row['entity']} → {row['matched_label']} [{row['ontology_name']}] (conf={row['confidence']:.2f})")

        if args.show_unlinked and len(unlinked) > 0:
            print(f"\n⚠️  Unlinked entities ({len(unlinked)}):")
            for entity in unlinked['entity'].head(20):
                print(f"      {entity}")
            if len(unlinked) > 20:
                print(f"      ... and {len(unlinked)-20} more")

if __name__ == "__main__":
    main()
