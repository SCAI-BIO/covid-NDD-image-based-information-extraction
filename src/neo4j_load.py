from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import unicodedata
from getpass import getpass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import pandas as pd
from neo4j import GraphDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("neo4j_upload_corrected")

# -------------------------
# Utils
# -------------------------

NA_SET = {"", "nan", "none", "unknown", "null"}

def is_na(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and pd.isna(v):
        return True
    if isinstance(v, str) and v.strip().lower() in NA_SET:
        return True
    return False

def coerce_value(v):
    """Coerce values to Neo4j-friendly scalars; lists/dicts -> JSON strings."""
    if is_na(v):
        return None
    if isinstance(v, (int, float, bool)):
        return v
    if isinstance(v, (list, tuple, set, dict)):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)
    return str(v)

def sanitize_label(label: str) -> str:
    s = str(label or "").strip()
    s = re.sub(r"[^0-9A-Za-z_]", "_", s)
    if not s:
        s = "LABEL"
    if re.match(r"^\d", s):
        s = f"L_{s}"
    return s

def sanitize_prop_key(key: str) -> str:
    k = str(key or "").strip().lower()
    k = re.sub(r"[^A-Za-z0-9_]", "_", k)
    if not re.match(r"^[A-Za-z]", k):
        k = "p_" + k
    return k

# -------------------------
# Normalization for keys
# -------------------------

CURIE_PREFIX_UPPER = re.compile(r"^[a-z0-9]+:", re.I)

def norm_curie(x: Optional[str]) -> Optional[str]:
    if is_na(x):
        return None
    s = str(x).strip()
    s = re.sub(r"\s+", "", s)
    if CURIE_PREFIX_UPPER.match(s):
        pfx, rest = s.split(":", 1)
        s = f"{pfx.upper()}:{rest}"
    s = re.sub(r":0+(?=\d+$)", ":", s)
    return s

def norm_iri(x: Optional[str]) -> Optional[str]:
    if is_na(x):
        return None
    s = str(x).strip()
    s = s.rstrip("/")
    s = s.replace("http://", "https://")
    return s

def canon_name(x: str) -> str:
    s = (x or "").strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"(?<=\w)[\-\_\s]+(?=\w)", "", s)
    s = s.replace("covid19", "covid-19").replace("sarscov2", "sars-cov-2")
    return s

def build_entity_key(name: str, curie: Optional[str], iri: Optional[str], raw_id: Optional[str]) -> str:
    c = norm_curie(curie)
    if c:
        return f"curie:{c}"
    i = norm_iri(iri)
    if i:
        return f"iri:{i}"
    if not is_na(raw_id):
        return f"id:{str(raw_id).strip()}"
    return f"name:{canon_name(name)}"

# -------------------------
# IO helpers
# -------------------------

def sniff_read_csv(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(2048)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
            sep = dialect.delimiter
        except csv.Error:
            sep = ","
    return pd.read_csv(path, sep=sep, encoding="utf-8", engine="python")

def read_any(path: Path, sheet: Optional[str | int] = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt", ".pipe", ".psv"}:
        return sniff_read_csv(str(path))
    if suffix in {".xlsx", ".xls"}:
        if sheet is None:
            return pd.read_excel(path, engine="openpyxl")
        return pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    raise ValueError(f"Unsupported file type: {suffix}")

# -------------------------
# Column detection
# -------------------------

def pick_header(lower_map: Dict[str, str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return None

def detect_spo_columns(
    df: pd.DataFrame,
    subject_override: Optional[str],
    predicate_override: Optional[str],
    object_override: Optional[str],
) -> Tuple[str, str, str]:
    lower_map: Dict[str, str] = {str(c).lower().strip(): c for c in df.columns}

    def require_from_override(name: str) -> str:
        low = name.lower().strip()
        if low not in lower_map:
            raise ValueError(f"Override column '{name}' not found in headers: {list(df.columns)}")
        return lower_map[low]

    subject = (
        require_from_override(subject_override)
        if subject_override else
        pick_header(lower_map, ["subject", "subj", "s", "head", "source", "from"])
    )
    predicate = (
        require_from_override(predicate_override)
        if predicate_override else
        pick_header(lower_map, ["predicate", "pred", "p", "relation", "rel", "edge", "link", "property"])
    )
    object_ = (
        require_from_override(object_override)
        if object_override else
        pick_header(lower_map, ["object", "obj", "o", "target", "tail", "to"])
    )
    if not (subject and predicate and object_):
        raise ValueError(
            "Could not detect subject/predicate/object columns.\n"
            f"Found headers: {list(df.columns)}\n"
            "Use --subject_col/--predicate_col/--object_col to specify explicitly."
        )
    return subject, predicate, object_

def detect_type_columns(
    df: pd.DataFrame,
    subj_type_override: Optional[str],
    obj_type_override: Optional[str],
    subj_subtype_override: Optional[str],
    obj_subtype_override: Optional[str],
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    lower_map: Dict[str, str] = {str(c).lower().strip(): c for c in df.columns}

    def opt(name: Optional[str], cands: List[str]) -> Optional[str]:
        if name:
            low = name.lower().strip()
            if low not in lower_map:
                raise ValueError(f"Override column '{name}' not found in headers: {list(df.columns)}")
            return lower_map[low]
        return pick_header(lower_map, cands)

    subject_type = opt(subj_type_override, ["subject_type", "subject type", "stype", "subjectlabel", "subject_label", "s_type"])
    object_type  = opt(obj_type_override, ["object_type", "object type", "otype", "objectlabel", "object_label", "o_type"])
    subject_subtype = opt(subj_subtype_override, ["subject_subtype", "subject subtype", "subjectclass", "subject_class", "s_subtype"])
    object_subtype  = opt(obj_subtype_override, ["object_subtype", "object subtype", "objectclass", "object_class", "o_subtype"])
    return subject_type, object_type, subject_subtype, object_subtype

# -------------------------
# Relation normalization (BEL-compact)
# -------------------------

BEL_RELATIONS = {
    "directlyIncreases", "directlyDecreases",
    "increases", "decreases", "regulates",
    "association",
    "positiveCorrelation", "negativeCorrelation",
    "noEffect",
    "partOf", "hasMember", "hasComponent",
    "localizes", "translocates",
    "biomarkerFor",
}

REL_TO_BEL = {
    "DIRECT_INCREASE": "directlyIncreases",
    "DIRECT_DECREASE": "directlyDecreases",
    "INCREASE": "increases",
    "DECREASE": "decreases",
    "REGULATES": "regulates",
    "ASSOCIATION": "association",
    "CORRELATION_POS": "positiveCorrelation",
    "CORRELATION_NEG": "negativeCorrelation",
    "NO_EFFECT": "noEffect",
    "PART_OF": "partOf",
    "HAS_MEMBER": "hasMember",
    "HAS_COMPONENT": "hasComponent",
    "LOCALIZES": "localizes",
    "TRANSLOCATES": "translocates",
    "BIOMARKER_FOR": "biomarkerFor",
}

EXACT_MAP = {
    "activates": "INCREASE", "upregulates": "INCREASE", "induces": "INCREASE",
    "stimulates": "INCREASE", "enhances": "INCREASE", "promotes": "INCREASE",
    "increases": "INCREASE", "augments": "INCREASE", "boosts": "INCREASE",
    "elevates": "INCREASE", "potentiates": "INCREASE",
    "phosphorylates": "DIRECT_INCREASE", "phosphorylation": "DIRECT_INCREASE",
    "acetylation": "DIRECT_INCREASE", "methylation": "DIRECT_INCREASE",
    "ubiquitination": "DIRECT_INCREASE", "dephosphorylation": "DIRECT_DECREASE",
    "inhibits": "DECREASE", "suppresses": "DECREASE", "represses": "DECREASE",
    "blocks": "DECREASE", "attenuates": "DECREASE", "reduces": "DECREASE",
    "downregulates": "DECREASE", "abrogates": "DECREASE",
    "impairs": "DECREASE", "dampens": "DECREASE",
    "regulates": "REGULATES", "modulates": "REGULATES",
    "binds": "ASSOCIATION", "interacts": "ASSOCIATION", "associates": "ASSOCIATION",
    "complexes with": "ASSOCIATION", "co-localizes": "ASSOCIATION",
    "positively correlates": "CORRELATION_POS",
    "negatively correlates": "CORRELATION_NEG",
    "part of": "PART_OF", "has member": "HAS_MEMBER", "member of": "HAS_MEMBER",
    "has component": "HAS_COMPONENT", "component of": "HAS_COMPONENT",
    "localizes to": "LOCALIZES", "translocates to": "TRANSLOCATES",
    "biomarker for": "BIOMARKER_FOR",
    "no effect": "NO_EFFECT", "no change": "NO_EFFECT", "unchanged": "NO_EFFECT",
}

REGEX_PATTERNS = [
    (re.compile(r"\b(directlyIncreases|directlyDecreases|increases|decreases|"
                r"regulates|association|positiveCorrelation|negativeCorrelation|"
                r"noEffect|partOf|hasMember|hasComponent|localizes|translocates|biomarkerFor)\b", re.I),
     "BEL_PASSTHRU"),
    (re.compile(r"\b(phosphorylat(e|ion)|acetylat(e|ion)|methylat(e|ion)|ubiquitinat(e|ion))\b", re.I), "DIRECT_INCREASE"),
    (re.compile(r"\b(dephosphorylat(e|ion))\b", re.I), "DIRECT_DECREASE"),
    (re.compile(r"\b(activate|up-?regulat(e|ion)|enhanc(e|ement)|promot(e|ion)|induc(e|tion)|stimulat(e|ion)|"
                r"increase(s|d)?|elevat(e|ion|ed|es)|augment(s|ed|ation)?|boost(s|ed)?)\b", re.I), "INCREASE"),
    (re.compile(r"\b(inhibit(s|ed)?|down-?regulat(e|ion)|suppress(es|ion)?|repress(es|ion)?|block(s|ed)?|"
                r"attenuat(e|ion|ed|es)|reduc(e|tion|ed|es)|diminish(es|ed)?|abrogat(e|ion|ed|es)|"
                r"prevent(s|ed)?|impair(s|ed)?|dampen(s|ed)?)\b", re.I), "DECREASE"),
    (re.compile(r"\b(regulat(e|ion)|modulat(e|ion))\b", re.I), "REGULATES"),
    (re.compile(r"\b(positiv(e|ely)\s*correlat(e|ion))\b", re.I), "CORRELATION_POS"),
    (re.compile(r"\b(negativ(e|ely)\s*correlat(e|ion)|anti[- ]?correlat(e|ion))\b", re.I), "CORRELATION_NEG"),
    (re.compile(r"\b(bind(s|ing)?|interact(s|ion)?|complex(es)?|associate(s|d)?|co-?locali[sz]e(s|d)?)\b", re.I), "ASSOCIATION"),
    (re.compile(r"\b(member of|has member)\b", re.I), "HAS_MEMBER"),
    (re.compile(r"\b(component of|has component|subunit of)\b", re.I), "HAS_COMPONENT"),
    (re.compile(r"\b(part of)\b", re.I), "PART_OF"),
    (re.compile(r"\b(locali[sz]e(s|d)? to|acts in|functions in|participates in)\b", re.I), "LOCALIZES"),
    (re.compile(r"\b(translocat(e|ion)|traffick)\b", re.I), "TRANSLOCATES"),
    (re.compile(r"\b(biomarker|marker of|diagnostic for)\b", re.I), "BIOMARKER_FOR"),
    (re.compile(r"\b(no (significant )?effect|no change|does not affect|unchanged)\b", re.I), "NO_EFFECT"),
]

NEGATION_HINT = re.compile(r"\b(no(t)?|doesn'?t|didn'?t|without|lack of|fails? to|fail(ed)? to)\b", re.I)
NO_CHANGE_HINT = re.compile(r"\b(no\s+(significant\s+)?(increase|decrease|change)|unchanged)\b", re.I)

def _to_compact_from_bel_token(token: str) -> Optional[str]:
    t = token.strip()
    if t in BEL_RELATIONS:
        return {v: k for k, v in REL_TO_BEL.items()}[t]
    for bel in BEL_RELATIONS:
        if bel.lower() == t.lower():
            return {v: k for k, v in REL_TO_BEL.items()}[bel]
    return None

def normalize_relation(pred: str) -> tuple[str, str]:
    original = (pred or "").strip()
    p = original
    compact_from_bel = _to_compact_from_bel_token(p)
    if compact_from_bel:
        return compact_from_bel, original
    plow = p.lower()
    for phrase, rel in EXACT_MAP.items():
        if phrase in plow:
            if rel in {"INCREASE", "DIRECT_INCREASE"} and (NO_CHANGE_HINT.search(plow) or NEGATION_HINT.search(plow)):
                if re.search(r"\b(no\s+increase|does\s+not\s+increase|fail(s|ed)?\s+to\s+increase)\b", plow):
                    return "NO_EFFECT", original
                return "DECREASE", original
            if rel in {"DECREASE", "DIRECT_DECREASE"} and (NO_CHANGE_HINT.search(plow) or NEGATION_HINT.search(plow)):
                if re.search(r"\b(no\s+decrease|does\s+not\s+decrease|fail(s|ed)?\s+to\s+decrease)\b", plow):
                    return "NO_EFFECT", original
                return "INCREASE", original
            return rel, original
    for rx, rel in REGEX_PATTERNS:
        m = rx.search(p)
        if not m:
            continue
        if rel == "BEL_PASSTHRU":
            token = m.group(1) if m.groups() else m.group(0)
            compact = _to_compact_from_bel_token(token)
            if compact:
                return compact, original
            continue
        if rel in {"INCREASE", "DIRECT_INCREASE"}:
            if NO_CHANGE_HINT.search(plow) or re.search(r"\b(does\s+not|fail(s|ed)?\s+to)\s+increase\b", plow):
                return "NO_EFFECT", original
            if NEGATION_HINT.search(plow):
                return "DECREASE", original
        if rel in {"DECREASE", "DIRECT_DECREASE"}:
            if NO_CHANGE_HINT.search(plow) or re.search(r"\b(does\s+not|fail(s|ed)?\s+to)\s+decrease\b", plow):
                return "NO_EFFECT", original
            if NEGATION_HINT.search(plow):
                return "INCREASE", original
        return rel, original
    if re.search(r"\b(up\s*regulat|up[- ]?shift|increase|elevat|augment|boost|potentiat)\w*", plow):
        return "INCREASE", original
    if re.search(r"\b(down\s*regulat|decreas|reduc|attenuat|dampen|impair|repress|suppress)\w*", plow):
        return "DECREASE", original
    return "ASSOCIATION", original

# -------------------------
# Uploader
# -------------------------

class Neo4jUploader:
    def __init__(self, uri: str, user: str, password: str, base_label: str = "Entity"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.base_label = sanitize_label(base_label)

    def close(self):
        self.driver.close()

    def ensure_constraints(self):
        with self.driver.session() as session:
            session.execute_write(
                lambda tx: tx.run(
                    f"CREATE CONSTRAINT entity_key_unique IF NOT EXISTS "
                    f"FOR (n:{self.base_label}) REQUIRE n.key IS UNIQUE"
                )
            )
            session.execute_write(
                lambda tx: tx.run(f"CREATE INDEX entity_onto IF NOT EXISTS FOR (n:{self.base_label}) ON (n.ontology_id)")
            )
            session.execute_write(
                lambda tx: tx.run(f"CREATE INDEX entity_iri IF NOT EXISTS FOR (n:{self.base_label}) ON (n.iri)")
            )

    # ---------- NEW: upsert with key promotion ----------
    def _upsert_node_with_promotion(
        self,
        session,
        preferred_key: str,
        alt_keys: List[str],
        name: str,
        props: Dict[str, object],
        labels: List[str],
    ):
        def build_set_labels(var: str, labs: List[str]) -> str:
            labs = [sanitize_label(l) for l in labs if l and l != self.base_label]
            return "".join([f" SET {var}:{l}" for l in dict.fromkeys(labs)])

        set_labels = build_set_labels("n", labels)

        # Strategy (no subqueries, no YIELD):
        # 1) OPTIONAL MATCH preferred and first alt.
        # 2) If neither exists → CREATE preferred.
        # 3) If only alt exists → promote: SET alt.key = preferred.
        # 4) Pick final node: coalesce(pref, alt), or MATCH the newly-created one.
        # 5) Update timestamps/props/labels and RETURN key.
        cypher = f"""
        // 1) Try preferred and alt
        OPTIONAL MATCH (pref:{self.base_label} {{key:$preferred}})
        OPTIONAL MATCH (a:{self.base_label})
        WHERE a.key IN $alts
        WITH pref, collect(a)[0] AS alt, $preferred AS preferred

        // 2) Create new if neither exists
        FOREACH (_ IN CASE WHEN pref IS NULL AND alt IS NULL THEN [1] ELSE [] END |
        CREATE (:{self.base_label} {{key:preferred, name:$name, created_at:timestamp()}})
        )

        // 3) Promote alt → preferred if needed
        FOREACH (_ IN CASE WHEN pref IS NULL AND alt IS NOT NULL THEN [1] ELSE [] END |
        SET alt.key = preferred
        )

        // 4) Choose the node; if we just created one, fetch it
        WITH coalesce(pref, alt) AS n, preferred
        OPTIONAL MATCH (n2:{self.base_label} {{key:preferred}})
        WITH coalesce(n, n2) AS n

        // 5) Update props/labels
        SET n.last_seen = timestamp()
        SET n += $props
        {set_labels}
        RETURN n.key AS key
        """

        rec = session.execute_write(
            lambda tx: tx.run(
                cypher,
                preferred=preferred_key,
                alts=list(dict.fromkeys(alt_keys)),
                name=name,
                props=props,
            ).single()
        )
        return rec["key"]

    # ----------------------------------------------------

    def upload(
        self,
        df: pd.DataFrame,
        s_col: str,
        p_col: str,
        o_col: str,
        stype_col: Optional[str],
        otype_col: Optional[str],
        ssubtype_col: Optional[str],
        osubtype_col: Optional[str],
    ):
        lower_map: Dict[str, str] = {str(c).lower().strip(): c for c in df.columns}
        sid_col   = lower_map.get("subject_id")
        oid_col   = lower_map.get("object_id")
        sonto_col = lower_map.get("subject_ontology_id")
        oonto_col = lower_map.get("object_ontology_id")
        siri_col  = lower_map.get("subject_iri")
        oiri_col  = lower_map.get("object_iri")

        with self.driver.session() as session:
            total = len(df)
            for i, row in df.iterrows():
                s_raw = str(row[s_col]).strip()
                p_raw = str(row[p_col]).strip()
                o_raw = str(row[o_col]).strip()
                if not s_raw or not p_raw or not o_raw:
                    continue
                if s_raw.lower() in NA_SET or o_raw.lower() in NA_SET or p_raw.lower() in NA_SET:
                    continue

                compact_rel, original_pred = normalize_relation(p_raw)
                rel_type = sanitize_label(compact_rel)

                # Labels to add AFTER merge/upsert
                s_labels: List[str] = [self.base_label]
                o_labels: List[str] = [self.base_label]

                def add_label_if_present(container: List[str], value: Optional[str]):
                    if value and not is_na(value):
                        container.append(sanitize_label(str(value)))

                add_label_if_present(s_labels, row.get(stype_col) if stype_col else None)
                add_label_if_present(o_labels, row.get(otype_col) if otype_col else None)
                add_label_if_present(s_labels, row.get(ssubtype_col) if ssubtype_col else None)
                add_label_if_present(o_labels, row.get(osubtype_col) if osubtype_col else None)

                # Node IDs (raw)
                s_id = coerce_value(row.get(sid_col)) if sid_col else None
                o_id = coerce_value(row.get(oid_col)) if oid_col else None
                s_onto = coerce_value(row.get(sonto_col)) if sonto_col else None
                o_onto = coerce_value(row.get(oonto_col)) if oonto_col else None
                s_iri  = coerce_value(row.get(siri_col))  if siri_col  else None
                o_iri  = coerce_value(row.get(oiri_col))  if oiri_col  else None

                # Preferred keys
                s_key_pref = build_entity_key(s_raw, s_onto, s_iri, s_id)
                o_key_pref = build_entity_key(o_raw, o_onto, o_iri, o_id)

                # Alternate keys (used for adoption/promotion)
                def make_alt_keys(name, curie, iri, raw_id):
                    alts = [f"name:{canon_name(name)}"]
                    c = norm_curie(curie)
                    i = norm_iri(iri)
                    if c: alts.append(f"curie:{c}")
                    if i: alts.append(f"iri:{i}")
                    if not is_na(raw_id): alts.append(f"id:{str(raw_id).strip()}")
                    # we'll de-dup and drop preferred later
                    return alts

                s_alts = [k for k in dict.fromkeys(make_alt_keys(s_raw, s_onto, s_iri, s_id)) if k != s_key_pref]
                o_alts = [k for k in dict.fromkeys(make_alt_keys(o_raw, o_onto, o_iri, o_id)) if k != o_key_pref]

                # Node properties (subject_*/object_* → node)
                s_props: Dict[str, object] = {"name": s_raw}
                o_props: Dict[str, object] = {"name": o_raw}
                if s_onto: s_props.setdefault("ontology_id", norm_curie(s_onto))
                if o_onto: o_props.setdefault("ontology_id", norm_curie(o_onto))
                if s_iri:  s_props.setdefault("iri", norm_iri(s_iri))
                if o_iri:  o_props.setdefault("iri", norm_iri(o_iri))
                if s_id and not is_na(s_id): s_props.setdefault("id", str(s_id))
                if o_id and not is_na(o_id): o_props.setdefault("id", str(o_id))

                # Collect prefixed node props
                for col in df.columns:
                    if col in {s_col, p_col, o_col, sid_col, oid_col, sonto_col, oonto_col, siri_col, oiri_col}:
                        continue
                    low = str(col).lower().strip()
                    if low.startswith("subject_"):
                        if low in {"subject_", "subject_id", "subject_ontology_id", "subject_iri"}:
                            continue
                        key = sanitize_prop_key(low.replace("subject_", "", 1))
                        val = coerce_value(row[col])
                        if val is not None:
                            s_props[key] = val
                    elif low.startswith("object_"):
                        if low in {"object_", "object_id", "object_ontology_id", "object_iri"}:
                            continue
                        key = sanitize_prop_key(low.replace("object_", "", 1))
                        val = coerce_value(row[col])
                        if val is not None:
                            o_props[key] = val

                # Relationship props: everything else (non subject_/object_)
                rel_props: Dict[str, object] = {}
                for col in df.columns:
                    if col in {s_col, p_col, o_col, sid_col, oid_col, sonto_col, oonto_col, siri_col, oiri_col}:
                        continue
                    low = str(col).lower().strip()
                    if low.startswith("subject_") or low.startswith("object_"):
                        continue
                    key = sanitize_prop_key(col)
                    val = coerce_value(row[col])
                    if val is not None:
                        rel_props[key] = val

                rel_props.setdefault("subject", s_raw)
                rel_props.setdefault("predicate", p_raw)
                rel_props.setdefault("object", o_raw)
                rel_props["relation"] = compact_rel
                rel_props["bel_relation"] = REL_TO_BEL.get(compact_rel, "association")
                rel_props["original_predicate"] = original_pred

                # ---- upsert nodes with key promotion ----
                s_key_final = self._upsert_node_with_promotion(
                    session, s_key_pref, s_alts, s_raw, s_props, s_labels
                )
                o_key_final = self._upsert_node_with_promotion(
                    session, o_key_pref, o_alts, o_raw, o_props, o_labels
                )

                # ---- MERGE relationship between final nodes ----
                rel_cypher = (
                    f"MATCH (s:{self.base_label} {{key:$skey}}), (o:{self.base_label} {{key:$okey}})\n"
                    f"MERGE (s)-[r:{rel_type}]->(o)\n"
                    f"ON CREATE SET r.created_at = timestamp()\n"
                    f"SET r += $rel_props"
                )
                session.execute_write(
                    lambda tx: tx.run(rel_cypher, skey=s_key_final, okey=o_key_final, rel_props=rel_props)
                )

                if (i + 1) % 500 == 0 or (i + 1) == total:
                    logger.info("Processed %d/%d rows", i + 1, total)

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Upload triples (CSV/TSV/Excel) to Neo4j with stable keys, key promotion, and BEL-style relation normalization."
    )
    ap.add_argument("--file", required=True, help="Path to CSV/TSV/TXT/PIPE or Excel (.xlsx/.xls) file")
    ap.add_argument("--sheet", help="Excel sheet name or 0-based index (only for Excel files)")
    ap.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j Bolt URI")
    ap.add_argument("--user", default="neo4j", help="Neo4j username")
    ap.add_argument("--password", help="Neo4j password (prompted if omitted)")
    ap.add_argument("--node_label", default="Entity", help="Base label for nodes (default: Entity)")
    ap.add_argument("--subject_col")
    ap.add_argument("--predicate_col")
    ap.add_argument("--object_col")
    ap.add_argument("--subject_type_col")
    ap.add_argument("--object_type_col")
    ap.add_argument("--subject_subtype_col")
    ap.add_argument("--object_subtype_col")
    ap.add_argument("--dry_run", action="store_true", help="Only detect columns and show a preview; no DB writes")
    args = ap.parse_args()

    fpath = Path(args.file)
    if not fpath.exists():
        logger.error("Input not found: %s", fpath)
        sys.exit(1)

    sheet_arg: Optional[str | int] = None
    if args.sheet is not None:
        sheet_arg = int(args.sheet) if args.sheet.isdigit() else args.sheet

    df = read_any(fpath, sheet=sheet_arg)

    # Detect columns
    s_col, p_col, o_col = detect_spo_columns(df, args.subject_col, args.predicate_col, args.object_col)
    stype_col, otype_col, ssubtype_col, osubtype_col = detect_type_columns(
        df, args.subject_type_col, args.object_type_col, args.subject_subtype_col, args.object_subtype_col
    )

    before = len(df)
    df = df.dropna(subset=[s_col, p_col, o_col])
    df = df[df[s_col].astype(str).str.strip() != ""]
    df = df[df[p_col].astype(str).str.strip() != ""]
    df = df[df[o_col].astype(str).str.strip() != ""]
    for c in [s_col, p_col, o_col]:
        df = df[~df[c].astype(str).str.strip().str.lower().isin(NA_SET)]
    after = len(df)
    logger.info("Detected columns — subject: '%s', predicate: '%s', object: '%s'", s_col, p_col, o_col)
    if stype_col or otype_col or ssubtype_col or osubtype_col:
        logger.info(
            "Type columns — subject_type: %s | object_type: %s | subject_subtype: %s | object_subtype: %s",
            stype_col, otype_col, ssubtype_col, osubtype_col,
        )
    else:
        logger.info("No type/subtype columns detected; nodes will use only the base label.")
    logger.info("Preview: kept %d of %d rows after basic cleaning", after, before)

    if args.dry_run:
        logger.info("Dry run — no writes will be performed. Columns: %s", list(df.columns))
        return

    password = args.password if args.password is not None else getpass("Enter Neo4j password: ")

    uploader = Neo4jUploader(uri=args.uri, user=args.user, password=password, base_label=args.node_label)
    try:
        uploader.ensure_constraints()
        uploader.upload(df, s_col, p_col, o_col, stype_col, otype_col, ssubtype_col, osubtype_col)
        logger.info("Upload complete.")
    finally:
        uploader.close()

if __name__ == "__main__":
    main()
