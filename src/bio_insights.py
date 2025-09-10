# bio_insights.py

"""
bio_insights.py

Authors: Elizaveta Popova, Negin Babaiha
Institution: University of Bonn, Fraunhofer SCAI
Date: 25/08/2025

Description:
    Generates publication-ready figures and CSV tables for the "Biological insights" subsection.
    Inputs are CSVs exported from Cypher queries; outputs are static figures and top-N tables.
    Optionally connects to Neo4j to render compact subgraphs for the Minocycline case study
    (CBM vs GPT-fulltext vs combined).

Key functionalities:
    1. Loads CSV summaries:
        - covid_ndd_shared_hubs.csv
        - covid_ndd_cytokine_hubs.csv
        - covid_bbb_path_mediators.csv
        - glia_neighbors.csv
    2. Produces stacked horizontal bar charts splitting degrees by source (GPT/CBM/GPT-fulltext).
    3. Produces top mediators on COVID↔BBB paths and astrocyte-neighbor figures.
    4. (Optional) Connects to Neo4j and exports Minocycline star subgraphs with edge styles by source.

Inputs (files):
    data/bio_insights/neo4j_results/
        - covid_ndd_shared_hubs.csv
        - covid_ndd_cytokine_hubs.csv
        - covid_bbb_path_mediators.csv
        - glia_neighbors.csv

Outputs:
    data/bio_insights/outputs/
        - Fig_shared_hubs.png
        - Fig_cytokine_hubs.png
        - BBB_mediators_top10.png
        - Astrocyte_neighbors_top15_noCOVID.png
        - shared_hubs_topN.csv
        - cytokine_hubs_topN.csv
        - (optional) minocycline_cbm.(png|tiff), minocycline_gptfulltext.(png|tiff), minocycline_combined.(png|tiff)

Usage:
    python src/bio_insights.py \
        --input-dir data/bio_insights/neo4j_results \
        --output-dir data/bio_insights/outputs \
        --top-n 20 \
        --run-neo4j \
        --neo4j-uri neo4j://127.0.0.1:7687 \
        --neo4j-user neo4j \
        --neo4j-password YOUR_PASSWORD \
        --neo4j-db neo4j

Requirements:
    - pandas
    - matplotlib
    - numpy
    - (optional for subgraphs) neo4j, networkx
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple, Dict
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional imports only used if --run-neo4j is passed
try:
    from neo4j import GraphDatabase
    import networkx as nx
    from matplotlib.lines import Line2D
except Exception:
    GraphDatabase = None
    nx = None

# ---- Consistent color palette across all figures ----
PALETTE = {
    "CBM": "#3274a1",        # blue
    "GPT": "#e1812c",        # orange
    "GPT-fulltext": "#3a923a" # green
}

# ----------------------------- Helpers ---------------------------------

def ensure_outdir(path: Path) -> None:
    """Create output directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def standardize_columns(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """
    Bring result frames to a common schema:
      - name:              entity/cytokine string
      - deg_total:         total degree
      - deg_cbm:           degree from CBM edges
      - deg_gpt:           degree from GPT edges
      - deg_gpt_fulltext:  degree from GPT-fulltext edges
    """
    df = df.copy()

    # Detect the name column
    for cand in ("entity", "cytokine", "name", "x.name", "x", "mediator", "neighbor"):
        if cand in df.columns:
            df.rename(columns={cand: "name"}, inplace=True)
            break
    else:
        raise ValueError(f"[{kind}] Cannot find a name/identifier column. Got: {df.columns.tolist()}")

    # Ensure numeric degree columns exist
    for col in ("deg_total", "deg_cbm", "deg_gpt", "deg_gpt_fulltext"):
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Optional pretty label for plotting
    df["name_short"] = df["name"].astype(str).str.replace("_", " ")
    return df


def make_hbar(df: pd.DataFrame, title: str, outfile_png: Path, top_n: int, dpi: int) -> pd.DataFrame:
    """
    Horizontal stacked bar chart of deg_total split into GPT / CBM / GPT-fulltext.
    Uses matplotlib defaults (no seaborn, no custom color palette).
    Returns the descending-sorted (top-first) DataFrame used in the plot.
    """
    d = df.sort_values("deg_total", ascending=False).head(top_n).iloc[::-1]  # reverse for barh order

    plt.figure(figsize=(9, max(4, 0.45 * len(d))), dpi=dpi)

    plt.barh(d["name_short"], d["deg_cbm"],
            label="CBM", color=PALETTE["CBM"],
            left=d["deg_gpt"], edgecolor="black", linewidth=0.3)

    plt.barh(d["name_short"], d["deg_gpt"],
         label="GPT (images)", color=PALETTE["GPT"], edgecolor="black", linewidth=0.3)

    plt.barh(d["name_short"], d["deg_gpt_fulltext"],
            label="GPT-fulltext", color=PALETTE["GPT-fulltext"],
            left=d["deg_gpt"] + d["deg_cbm"], edgecolor="black", linewidth=0.3)

    max_val = d["deg_total"].max()
    plt.xlim(0, max_val * 1.1)

    plt.xlabel("Node degree")
    plt.title(title)
    plt.legend()
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outfile_png, dpi=dpi)
    plt.close()

    return d.iloc[::-1]


# ---------------------- Figure Block A: hubs ---------------------------

def figure_shared_and_cytokine_hubs(input_dir: Path, output_dir: Path, top_n: int, dpi: int) -> None:
    """Render shared hubs and cytokine hubs stacked bar charts and export top-N CSVs."""
    shared_path   = input_dir / "covid_ndd_shared_hubs.csv"     # was Q1A.csv
    cytokine_path = input_dir / "covid_ndd_cytokine_hubs.csv"   # was Q1B.csv

    q1a = pd.read_csv(shared_path)
    q1b = pd.read_csv(cytokine_path)

    q1a_std = standardize_columns(q1a, "shared_hubs")
    q1b_std = standardize_columns(q1b, "cytokine_hubs")

    top_q1a = make_hbar(
        q1a_std,
        title=textwrap.fill(
            "Shared Hubs Within 2 Hops of Both COVID and NDD",
            width=60
        ),
        outfile_png=output_dir / "Fig_shared_hubs.tiff",
        top_n=top_n, dpi=dpi,
    )

    top_q1b = make_hbar(
        q1b_std,
        title="Cytokine/Chemokine Hubs Near Both COVID and NDD (degree split by source)",
        outfile_png=output_dir / "Fig_cytokine_hubs.tiff",
        top_n=top_n, dpi=dpi,
    )

    top_q1a.to_csv(output_dir / "shared_hubs_topN.csv", index=False)
    top_q1b.to_csv(output_dir / "cytokine_hubs_topN.csv", index=False)


# ---- Figure Block B: BBB mediators & astrocyte neighbors --------------

def figure_bbb_mediators_and_glia_neighbors(input_dir: Path, output_dir: Path, dpi: int) -> None:
    """Render BBB mediators (stacked barh) and astrocyte neighbors (stacked barh)."""
    # BBB mediators
    a2 = pd.read_csv(input_dir / "covid_bbb_path_mediators.csv")

    # Robust column mapping
    if "mediator" not in a2.columns:
        for cand in ("name", "entity", "node"):
            if cand in a2.columns:
                a2.rename(columns={cand: "mediator"}, inplace=True)
                break
    for col in ("edges_on_paths", "cbm_edges_on_paths", "gpt_edges_on_paths"):
        if col not in a2.columns:
            a2[col] = 0

    a2_sorted = a2.sort_values("edges_on_paths", ascending=False).head(10).copy()

    y = np.arange(len(a2_sorted))
    plt.figure(figsize=(8, 6), dpi=dpi)
    # BBB mediators
    plt.barh(y, a2_sorted["cbm_edges_on_paths"], label="CBM edges", color=PALETTE["CBM"])
    plt.barh(y, a2_sorted["gpt_edges_on_paths"],
         left=a2_sorted["cbm_edges_on_paths"], label="GPT edges", color=PALETTE["GPT"], alpha=0.85)

    plt.yticks(y, a2_sorted["mediator"])
    plt.xlabel("Edges on COVID↔BBB paths")
    plt.title("Top Mediators on COVID↔BBB Paths")
    plt.legend()
    plt.gca().invert_yaxis()
    xmax = (a2_sorted["cbm_edges_on_paths"] + a2_sorted["gpt_edges_on_paths"]).max()
    plt.xlim(0, xmax * 1.1)
    plt.tight_layout()
    plt.savefig(output_dir / "BBB_mediators_top10.png", dpi=dpi)
    plt.savefig(output_dir / "BBB_mediators_top10.tiff", dpi=dpi)
    plt.close()

    # Astrocyte neighbors
    b1 = pd.read_csv(input_dir / "glia_neighbors.csv")

    # Remove obvious COVID anchors
    covid_terms = {"covid-19", "sars-cov-2", "severe acute respiratory syndrome coronavirus 2"}
    name_col = "neighbor" if "neighbor" in b1.columns else b1.columns[0]
    b1 = b1[~b1[name_col].astype(str).str.lower().isin(covid_terms)].copy()

    # Astrocyte anchor only (if present)
    if "glia_anchor" in b1.columns:
        b1 = b1[b1["glia_anchor"].astype(str).str.upper() == "ASTROCYTE ACTIVATION"].copy()

    # Robust column mapping for edge counts
    if "edges_cbm" not in b1.columns:
        b1["edges_cbm"] = b1["cbm_edges"] if "cbm_edges" in b1.columns else 0
    if "edges_gpt" not in b1.columns:
        b1["edges_gpt"] = b1["gpt_edges"] if "gpt_edges" in b1.columns else 0
    if "deg_local" not in b1.columns:
        b1["deg_local"] = b1["edges_cbm"].fillna(0) + b1["edges_gpt"].fillna(0)

    b1_sorted = b1.sort_values("deg_local", ascending=False).head(15).copy()

    y = np.arange(len(b1_sorted))
    plt.figure(figsize=(9, 7), dpi=dpi)
    # Astrocyte neighbors
    plt.barh(y, b1_sorted["edges_cbm"], label="CBM edges", color=PALETTE["CBM"])
    plt.barh(y, b1_sorted["edges_gpt"],
         left=b1_sorted["edges_cbm"], label="GPT edges", color=PALETTE["GPT"], alpha=0.85)
    plt.yticks(y, b1_sorted[name_col])
    plt.xlabel("Number of edges")
    plt.title("Top Neighbors of ASTROCYTE ACTIVATION", wrap=True)
    plt.legend()
    plt.gca().invert_yaxis()
    xmax = (b1_sorted["edges_cbm"] + b1_sorted["edges_gpt"]).max()
    plt.xlim(0, xmax * 1.1)
    plt.tight_layout()
    plt.savefig(output_dir / "Astrocyte_neighbors_top15_noCOVID.png", dpi=dpi)
    plt.savefig(output_dir / "Astrocyte_neighbors_top15_noCOVID.tiff", dpi=dpi)
    plt.close()


# ----------------- Figure Block C: Neo4j star subgraphs ----------------

DRUG = "MINOCYCLINE"
MINI_TARGETS = [
    "viral replication",
    "virus_replication",
    "microglial cell activation",
    "sars-cov2 crossing blood brain barrier",
    "impaired_neural-microglial_communication",
    "hypoxia-induced_neuroinflammation",
    "neuroinflammation",
    "neuronal_and_glial_injury",
]
BENEFICIAL_DOWN = [
    "DECREASE", "INHIBIT", "BLOCK", "REDUCE",
    "ALLEVIATES", "COUNTERACTS", "PREVENTS", "ATTENUATES"
]
ALIASES = {
    "VIRUS_REPLICATION": "Virus \nreplication",
    "VIRAL REPLICATION": "Viral \nreplication",
    "MICROGLIAL CELL ACTIVATION": "Microglial \nactivation",
    "SARS-COV2 CROSSING BLOOD BRAIN BARRIER": "SARS-CoV-2 \ncrossing BBB",
    "IMPAIRED_NEURAL-MICROGLIAL_COMMUNICATION": "Impaired neural– \nmicroglial \ncommunication",
    "HYPOXIA-INDUCED_NEUROINFLAMMATION": "Hypoxia-induced \nneuroinflammation",
    "NEUROINFLAMMATION": "Neuroinflammation",
    "NEURONAL_AND_GLIAL_INJURY": "Neuronal & \nglial injury",
    "MINOCYCLINE": "Minocycline",
}

def neo4j_fetch_edges(driver, db: str, drug: str, sources: List[str]) -> List[Tuple[str, str, Dict]]:
    """Fetch beneficial edges from `drug` to curated targets, restricted by source."""
    cypher = """
    MATCH (c:Chemical {name:$drug})-[r]->(m)
    WHERE any(t IN $mini_targets
              WHERE replace(toLower(m.name),'_',' ')
                    = replace(toLower(t),'_',' '))
      AND type(r) IN $beneficial_down
      AND r.source IN $sources
    RETURN c.name AS drug, m.name AS target, type(r) AS rel, r.source AS source
    """
    rows = []
    with driver.session(database=db) as s:
        for rec in s.run(
            cypher,
            drug=drug,
            mini_targets=MINI_TARGETS,
            beneficial_down=BENEFICIAL_DOWN,
            sources=sources,
        ):
            rows.append((rec["drug"], rec["target"], {"source": rec["source"], "rel": rec["rel"]}))
    return rows


def radial_positions(center: str, targets: List[str], radius: float = 4.5, start_deg: float = 90.0):
    """Place targets evenly on a circle around the center."""
    pos = {center: (0.0, 0.0)}
    n = max(1, len(targets))
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False) + math.radians(start_deg)
    for t, ang in zip(targets, angles):
        pos[t] = (radius * math.cos(ang), radius * math.sin(ang))
    return pos


def build_star(drug: str, edges: List[Tuple[str, str, Dict]]):
    """Create a star-shaped DiGraph with `drug` at the center."""
    G = nx.DiGraph()
    G.add_node(drug)
    targets = []
    for u, v, attr in edges:
        G.add_node(v)
        G.add_edge(u, v, **attr)
        targets.append(v)
    pos = radial_positions(drug, sorted(set(targets)))
    return G, pos


def pretty(name: str) -> str:
    """Human-friendly labels for node captions."""
    return ALIASES.get(name, name.replace("_", " ").title())


def draw_star(G, pos, title: str, out_base: Path, dpi: int) -> None:
    """Render star graph with CBM (solid) vs GPT-fulltext (dashed) edges."""
    SOURCE_STYLE = {
        "CBM": {"color": "#2F80ED", "style": "solid"},
        "GPT-fulltext": {"color": "#EB5757", "style": (0, (4, 2))},
    }

    plt.figure(figsize=(8, 9), dpi=dpi)

    # Nodes
    node_sizes = [5000 if n == DRUG else 5000 for n in G.nodes()]
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color="#A0C4FF",
        alpha=0.65,
        linewidths=0
    )

    # Node labels
    labels = {n: pretty(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

    # Edges by source
    edges_all = list(G.edges(data=True))
    for src, style in SOURCE_STYLE.items():
        es = [(u, v, d) for (u, v, d) in edges_all if d.get("source") == src]
        if not es:
            continue
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v) for (u, v, _) in es],
            width=2.2,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=14,
            edge_color=style["color"],
            style=style["style"],
            alpha=0.95,
            connectionstyle="arc3,rad=0",
            min_source_margin=40,
            min_target_margin=54,
        )
        e_labels = {(u, v): d.get("rel") for (u, v, d) in es}
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=e_labels, font_size=9, rotate=False,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
        )

    # Legend
    legend = [
        Line2D([0], [0], color="#2F80ED", lw=2.5, label="CBM"),
        Line2D([0], [0], color="#EB5757", lw=2.5, label="GPT-fulltext", linestyle=(0, (4, 2))),
    ]
    plt.legend(handles=legend, loc="upper right", frameon=False, fontsize=10)

    plt.title(title, fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_base.with_suffix(".tiff"))
    plt.savefig(out_base.with_suffix(".png"), dpi=dpi)
    plt.close()


def figure_minocycline_subgraphs(output_dir: Path, uri: str, user: str, password: str, db: str, dpi: int) -> None:
    """Build three Minocycline subgraphs: CBM, GPT-fulltext, and Combined."""
    if GraphDatabase is None or nx is None:
        print("[WARN] neo4j/networkx not available — skipping Neo4j subgraph rendering.")
        return

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        edges_cbm   = neo4j_fetch_edges(driver, db, DRUG, ["CBM"])
        edges_gptft = neo4j_fetch_edges(driver, db, DRUG, ["GPT-fulltext"])

        G_cbm, pos_cbm = build_star(DRUG, edges_cbm)
        draw_star(G_cbm, pos_cbm, "Minocycline → beneficial edges • CBM", output_dir / "minocycline_cbm", dpi)

        G_gptft, pos_gptft = build_star(DRUG, edges_gptft)
        draw_star(G_gptft, pos_gptft, "Minocycline → beneficial edges • GPT-fulltext", output_dir / "minocycline_gptfulltext", dpi)

        G_both, pos_both = build_star(DRUG, edges_cbm + edges_gptft)
        draw_star(G_both, pos_both, "Minocycline → beneficial edges • CBM + GPT-fulltext", output_dir / "minocycline_combined", dpi)
    finally:
        driver.close()


# --------------------------------- CLI --------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Create argparse CLI similar in spirit to neo4j_upload.py."""
    p = argparse.ArgumentParser(
        description="Generate figures/tables for the Biological insights subsection; optional Neo4j subgraphs."
    )
    p.add_argument("--input-dir",  default="data/bio_insights/neo4j_results",
                   help="Input directory with CSVs (default: data/bio_insights/neo4j_results)")
    p.add_argument("--output-dir", default="data/bio_insights/outputs",
                   help="Output directory for figures and tables (default: data/bio_insights/outputs)")
    p.add_argument("--top-n", type=int, default=20, help="Top-N items in hub plots (default: 20)")
    p.add_argument("--dpi",   type=int, default=300, help="Figure DPI (default: 300)")

    # Neo4j toggles and credentials
    p.add_argument("--run-neo4j", action="store_true", help="Also render Minocycline subgraphs via Neo4j")
    p.add_argument("--neo4j-uri", default="neo4j://127.0.0.1:7687", help="Neo4j URI (default: neo4j://127.0.0.1:7687)")
    p.add_argument("--neo4j-user", default="neo4j", help="Neo4j user (default: neo4j)")
    p.add_argument("--neo4j-password", default=None, help="Neo4j password")
    p.add_argument("--neo4j-db", default="neo4j", help="Neo4j database name (default: neo4j)")
    return p


def main():
    args = build_parser().parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ensure_outdir(output_dir)

    # Block A: shared hubs + cytokine hubs
    figure_shared_and_cytokine_hubs(input_dir, output_dir, top_n=args.top_n, dpi=args.dpi)

    # Block B: BBB mediators + astrocyte neighbors
    figure_bbb_mediators_and_glia_neighbors(input_dir, output_dir, dpi=args.dpi)

    # Block C: Minocycline subgraphs (Neo4j)
    if args.run_neo4j:
        if args.neo4j_password is None:
            print("[ERROR] --neo4j-password is required when --run-neo4j is set.")
        else:
            figure_minocycline_subgraphs(
                output_dir=output_dir,
                uri=args.neo4j_uri,
                user=args.neo4j_user,
                password=args.neo4j_password,
                db=args.neo4j_db,
                dpi=args.dpi,
            )

    # Final log
    produced = sorted([p.name for p in output_dir.iterdir() if p.is_file()])
    print("Saved outputs to:", output_dir.resolve())
    for f in produced:
        print(" -", f)


if __name__ == "__main__":
    main()


# python src/bio_insights.py --input-dir data/bio_insights/neo4j_results --output-dir data/bio_insights/outputs_1 --top-n 20 --run-neo4j --neo4j-uri neo4j://127.0.0.1:7687 --neo4j-user neo4j --neo4j-password YOUR_PASSWORD --neo4j-db neo4j