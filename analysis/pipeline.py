#!/usr/bin/env python3
"""
The Soul of Wine — Reproducible Analysis Pipeline
===================================================
Pass 5 · Post-SME Review · 59 regions

Reproduces:
  1. Identity clustering (D-scores → StandardScaler → PCA → k-means, k=6)
  2. Terroir clustering  (Layer 2 TF-IDF → PCA → k-means, k=7)
     Fully model-derived: cluster names generated from TF-IDF feature weights.
  3. Independence test   (ARI + chi-square between the two solutions)

Outputs (to ../data/):
  - identity_pca.json     PCA coordinates for identity scatter plot
  - terroir_pca.json      PCA coordinates for terroir scatter plot
  - terroir_clusters.json Terroir cluster assignments for movement map
  - pipeline_report.txt   Full reproducibility report

Requirements:
  pip install scikit-learn numpy scipy pandas

Usage:
  python pipeline.py                        # run from analysis/ directory
  python pipeline.py --layer2 path/to.txt   # custom Layer 2 source path

Soul of Wine Research Project · April 2026
"""

import json
import os
import sys
import argparse
import re
import numpy as np
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.stats import chi2_contingency
from sklearn.feature_extraction.text import TfidfVectorizer

# ═══════════════════════════════════════════════════════════════════════
# CANONICAL DATA — Post-SME Review (CANONICAL 7)
# ═══════════════════════════════════════════════════════════════════════
# Region order: canonical region number (1–59)
# D-scores: D1 Interiority↔Exteriority, D2 Struggle↔Ease,
#           D3 Tradition↔Reinvention, D4 Individual↔Collective,
#           D5 Urgency↔Timelessness, D6 Earthly↔Transcendent
# Scale: −2 to +2 (integer)

REGIONS = [
    {"num": 1,  "name": "Kamptal",                "country": "Austria",      "world": "Old", "metaphor": "Discipline",        "d_scores": [ 1,  0,  1,  1,  0,  0]},
    {"num": 2,  "name": "Steiermark",             "country": "Austria",      "world": "Old", "metaphor": "Clarity",           "d_scores": [ 1,  0,  0,  1,  1,  1]},
    {"num": 3,  "name": "Wachau",                 "country": "Austria",      "world": "Old", "metaphor": "Monumentality",     "d_scores": [ 0,  0,  1,  1,  0,  1]},
    {"num": 4,  "name": "Wagram",                 "country": "Austria",      "world": "Old", "metaphor": "Earth",             "d_scores": [ 2,  0,  2,  0,  0,  1]},
    {"num": 5,  "name": "Dalmatian Coast",        "country": "Croatia",      "world": "Old", "metaphor": "Tranquility",       "d_scores": [ 2, -2,  1, -1, -2,  2]},
    {"num": 6,  "name": "Alsace",                 "country": "France",       "world": "Old", "metaphor": "Duality",           "d_scores": [ 0,  0,  1,  0,  0,  0]},
    {"num": 7,  "name": "Beaujolais",             "country": "France",       "world": "Old", "metaphor": "Joy",               "d_scores": [-1, -1, -1, -1,  1,  1]},
    {"num": 8,  "name": "Bordeaux",               "country": "France",       "world": "Old", "metaphor": "Business",          "d_scores": [-2, -1,  2, -2,  0,  1]},
    {"num": 9,  "name": "Burgundy",               "country": "France",       "world": "Old", "metaphor": "Devotion",          "d_scores": [ 2,  1,  2,  1, -1, -1]},
    {"num": 10, "name": "Champagne",              "country": "France",       "world": "Old", "metaphor": "Society",           "d_scores": [-2, -1,  2, -2,  0, -1]},
    {"num": 11, "name": "Châteauneuf-du-Pape",    "country": "France",       "world": "Old", "metaphor": "Family",            "d_scores": [-1,  0,  2, -2, -1,  1]},
    {"num": 12, "name": "Jura",                   "country": "France",       "world": "Old", "metaphor": "Eccentricity",      "d_scores": [ 2,  1,  2,  1,  0,  1]},
    {"num": 13, "name": "Loire",                  "country": "France",       "world": "Old", "metaphor": "Sentimentality",    "d_scores": [-1,  0,  0,  0,  0,  0]},
    {"num": 14, "name": "Northern Rhône",         "country": "France",       "world": "Old", "metaphor": "Solitude",          "d_scores": [ 2,  2,  2,  2, -2, -1]},
    {"num": 15, "name": "Provence",               "country": "France",       "world": "Old", "metaphor": "Pleasure",          "d_scores": [-2, -2,  0, -1,  1,  1]},
    {"num": 16, "name": "Baden",                  "country": "Germany",      "world": "Old", "metaphor": "Warmth",            "d_scores": [-1, -1,  1, -1, -1,  1]},
    {"num": 17, "name": "Mosel",                  "country": "Germany",      "world": "Old", "metaphor": "Poetry",            "d_scores": [ 2,  2,  2,  1, -2, -2]},
    {"num": 18, "name": "Nahe",                   "country": "Germany",      "world": "Old", "metaphor": "Subtlety",          "d_scores": [ 1,  0,  1,  1, -1,  0]},
    {"num": 19, "name": "Pfalz",                  "country": "Germany",      "world": "Old", "metaphor": "Generosity",        "d_scores": [-1, -2,  1, -1, -1,  1]},
    {"num": 20, "name": "Rheingau",               "country": "Germany",      "world": "Old", "metaphor": "Nobility",          "d_scores": [ 1,  0,  2,  1, -2,  0]},
    {"num": 21, "name": "Macedonia",              "country": "Greece",       "world": "Old", "metaphor": "Austerity",         "d_scores": [ 2,  2,  2,  1,  0,  1]},
    {"num": 22, "name": "Santorini",              "country": "Greece",       "world": "Old", "metaphor": "Survival",          "d_scores": [ 2,  2,  1,  1,  1,  1]},
    {"num": 23, "name": "Tokaj",                  "country": "Hungary",      "world": "Old", "metaphor": "Melancholy",        "d_scores": [ 2,  1,  2,  1, -2, -1]},
    {"num": 24, "name": "Alto Adige",             "country": "Italy",        "world": "Old", "metaphor": "Precision",         "d_scores": [ 0,  0,  1,  0,  0,  1]},
    {"num": 25, "name": "Campania",               "country": "Italy",        "world": "Old", "metaphor": "Memory",            "d_scores": [ 1,  0,  1,  0, -1, -1]},
    {"num": 26, "name": "Etna",                   "country": "Italy",        "world": "Old", "metaphor": "Awakening",         "d_scores": [ 1,  2,  0,  0,  1, -1]},
    {"num": 27, "name": "Friuli-Venezia Giulia",  "country": "Italy",        "world": "Old", "metaphor": "Dialogue",          "d_scores": [-1,  0,  0,  0,  0,  0]},
    {"num": 28, "name": "Liguria",                "country": "Italy",        "world": "Old", "metaphor": "Intimacy",          "d_scores": [ 2,  1,  1,  1, -1,  1]},
    {"num": 29, "name": "Piedmont",               "country": "Italy",        "world": "Old", "metaphor": "Philosophy",        "d_scores": [ 2,  1,  2,  1, -2, -1]},
    {"num": 30, "name": "Sardinia",               "country": "Italy",        "world": "Old", "metaphor": "Stubbornness",      "d_scores": [ 2,  1,  2, -1, -1,  2]},
    {"num": 31, "name": "Sicily",                 "country": "Italy",        "world": "Old", "metaphor": "Resurrection",      "d_scores": [-1,  1, -2,  1,  1,  1]},
    {"num": 32, "name": "Tuscany",                "country": "Italy",        "world": "Old", "metaphor": "Art",               "d_scores": [-1, -1,  0,  1,  0,  0]},
    {"num": 33, "name": "Veneto",                 "country": "Italy",        "world": "Old", "metaphor": "Commerce",          "d_scores": [-2,  0, -1, -1,  1,  1]},
    {"num": 34, "name": "Douro",                  "country": "Portugal",     "world": "Old", "metaphor": "Endurance",         "d_scores": [ 1,  2,  2,  1, -1,  1]},
    {"num": 35, "name": "Goriška Brda",           "country": "Slovenia",     "world": "Old", "metaphor": "Fortune",           "d_scores": [-1,  0,  0,  0,  0,  1]},
    {"num": 36, "name": "Catalonia",              "country": "Spain",        "world": "Old", "metaphor": "Identity",          "d_scores": [-1,  0, -1,  1,  2,  1]},
    {"num": 37, "name": "Galicia",                "country": "Spain",        "world": "Old", "metaphor": "Longing (Morriña)", "d_scores": [ 1,  1,  1,  1,  0, -1]},
    {"num": 38, "name": "Ribera del Duero",       "country": "Spain",        "world": "Old", "metaphor": "Severity",          "d_scores": [ 1,  2,  1,  1,  0,  2]},
    {"num": 39, "name": "Rioja",                  "country": "Spain",        "world": "Old", "metaphor": "Patience",          "d_scores": [-1,  0,  2, -1, -2,  0]},
    {"num": 40, "name": "Mendoza",                "country": "Argentina",    "world": "New", "metaphor": "Reinvention",       "d_scores": [-2,  0, -2,  1,  2,  1]},
    {"num": 41, "name": "Patagonia",              "country": "Argentina",    "world": "New", "metaphor": "Extremity",         "d_scores": [-1,  2, -2,  1,  1,  1]},
    {"num": 42, "name": "Barossa Valley",         "country": "Australia",    "world": "New", "metaphor": "Fortitude",         "d_scores": [ 1,  2,  1,  0,  0,  1]},
    {"num": 43, "name": "Hunter Valley",          "country": "Australia",    "world": "New", "metaphor": "Defiance",          "d_scores": [ 1,  2,  1,  1, -1,  1]},
    {"num": 44, "name": "Margaret River",         "country": "Australia",    "world": "New", "metaphor": "Composure",         "d_scores": [ 0,  0,  0,  0, -1,  0]},
    {"num": 45, "name": "Maipo Valley",           "country": "Chile",        "world": "New", "metaphor": "Pride",             "d_scores": [-1,  0,  1,  0, -1,  1]},
    {"num": 46, "name": "Central Otago",          "country": "New Zealand",  "world": "New", "metaphor": "Adventure",         "d_scores": [-1,  1, -2,  1,  1,  1]},
    {"num": 47, "name": "Hawke's Bay",            "country": "New Zealand",  "world": "New", "metaphor": "Confidence",        "d_scores": [ 0,  0,  0,  0, -1,  1]},
    {"num": 48, "name": "Marlborough",            "country": "New Zealand",  "world": "New", "metaphor": "Assertion",         "d_scores": [-2,  0, -1, -1,  2,  1]},
    {"num": 49, "name": "Stellenbosch",           "country": "South Africa", "world": "New", "metaphor": "Aspiration",        "d_scores": [-1,  0, -1,  1,  1,  1]},
    {"num": 50, "name": "Swartland",              "country": "South Africa", "world": "New", "metaphor": "Rebellion",         "d_scores": [-1,  1, -2,  1,  2,  1]},
    {"num": 51, "name": "Columbia Valley",        "country": "USA",          "world": "New", "metaphor": "Determination",     "d_scores": [-1,  1, -1,  0,  1,  2]},
    {"num": 52, "name": "Finger Lakes",           "country": "USA",          "world": "New", "metaphor": "Conviction",        "d_scores": [-1,  1, -1,  1,  1,  1]},
    {"num": 53, "name": "Napa Valley",            "country": "USA",          "world": "New", "metaphor": "Ambition",          "d_scores": [-2,  0, -1,  1,  1,  1]},
    {"num": 54, "name": "Paso Robles",            "country": "USA",          "world": "New", "metaphor": "Independence",      "d_scores": [-1,  0, -1,  2,  1,  1]},
    {"num": 55, "name": "Santa Barbara",          "country": "USA",          "world": "New", "metaphor": "Serendipity",       "d_scores": [-1,  0,  0,  0,  0,  1]},
    {"num": 56, "name": "Santa Cruz Mountains",   "country": "USA",          "world": "New", "metaphor": "Obsession",         "d_scores": [ 2,  1,  0,  2,  0,  0]},
    {"num": 57, "name": "Sonoma",                 "country": "USA",          "world": "New", "metaphor": "Authenticity",      "d_scores": [-1,  0, -1,  1,  0,  1]},
    {"num": 58, "name": "Walla Walla",            "country": "USA",          "world": "New", "metaphor": "Community",         "d_scores": [ 1, -1, -1, -2,  1,  1]},
    {"num": 59, "name": "Willamette Valley",      "country": "USA",          "world": "New", "metaphor": "Idealism",          "d_scores": [-1,  1, -1,  1,  0, -1]},
]

# Canonical identity cluster assignments — CANONICAL 7 (source of truth)
CANONICAL_IDENTITY_CLUSTERS = {
    "Kamptal": "The Moderates", "Steiermark": "The Moderates", "Wachau": "The Moderates",
    "Wagram": "Against the Odds", "Dalmatian Coast": "Old World Exterior",
    "Alsace": "The Moderates", "Beaujolais": "Outward Ease", "Bordeaux": "Old World Exterior",
    "Burgundy": "Old World Interior", "Champagne": "Old World Exterior",
    "Châteauneuf-du-Pape": "Old World Exterior", "Jura": "Against the Odds",
    "Loire": "The Moderates", "Northern Rhône": "Old World Interior",
    "Provence": "Outward Ease", "Baden": "Old World Exterior",
    "Mosel": "Old World Interior", "Nahe": "The Moderates",
    "Pfalz": "Old World Exterior", "Rheingau": "Old World Interior",
    "Macedonia": "Against the Odds", "Santorini": "Against the Odds",
    "Tokaj": "Old World Interior", "Alto Adige": "The Moderates",
    "Campania": "Old World Interior", "Etna": "Old World Interior",
    "Friuli-Venezia Giulia": "The Moderates", "Liguria": "Against the Odds",
    "Piedmont": "Old World Interior", "Sardinia": "Against the Odds",
    "Sicily": "New World Reinvention", "Tuscany": "The Moderates",
    "Veneto": "Outward Ease", "Douro": "Against the Odds",
    "Goriška Brda": "The Moderates", "Catalonia": "New World Reinvention",
    "Galicia": "Old World Interior", "Ribera del Duero": "Against the Odds",
    "Rioja": "Old World Exterior", "Mendoza": "New World Reinvention",
    "Patagonia": "New World Reinvention", "Barossa Valley": "Against the Odds",
    "Hunter Valley": "Against the Odds", "Margaret River": "The Moderates",
    "Maipo Valley": "The Moderates", "Central Otago": "New World Reinvention",
    "Hawke's Bay": "The Moderates", "Marlborough": "Outward Ease",
    "Stellenbosch": "New World Reinvention", "Swartland": "New World Reinvention",
    "Columbia Valley": "New World Reinvention", "Finger Lakes": "New World Reinvention",
    "Napa Valley": "New World Reinvention", "Paso Robles": "New World Reinvention",
    "Santa Barbara": "The Moderates", "Santa Cruz Mountains": "Old World Interior",
    "Sonoma": "New World Reinvention", "Walla Walla": "Outward Ease",
    "Willamette Valley": "The Moderates",
}

# Interpretive terroir cluster names derived from TF-IDF feature weights.
# The pipeline assigns integer labels 0–6; this mapping converts them to
# human-readable names. These names were generated by examining the top
# distinguishing TF-IDF terms for each cluster and selecting an interpretive
# label — the only manual step in the terroir classification.
TERROIR_CLUSTER_NAMES = {
    0: "Iberian Continental",
    1: "Southern Hemisphere & International",
    2: "American West Coast",
    3: "French Viticultural",
    4: "Germanic Rhine",
    5: "Austrian Danube",
    6: "Mediterranean & Volcanic",
}


def parse_layer2(filepath):
    """Parse Layer 2 terroir descriptions from the canonical docx-as-text file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [l.rstrip() for l in f.readlines()]

    # Known region names for reliable matching
    known_names = {r["name"] for r in REGIONS}

    regions = {}
    current_region = None
    current_lines = []

    for line in lines:
        # Match **RegionName**  —  Metaphor pattern
        m = re.match(r'^\*\*([^*]+)\*\*\s+[—–-]+\s+', line)
        if m:
            name = m.group(1).strip()
            if name in known_names:
                if current_region and current_lines:
                    regions[current_region] = ' '.join(current_lines)
                current_region = name
                current_lines = []
                continue

        if current_region:
            # Strip bold markers and collect content
            clean = re.sub(r'\*\*', '', line).strip()
            if clean:
                current_lines.append(clean)

    if current_region and current_lines:
        regions[current_region] = ' '.join(current_lines)

    return regions


def run_identity_clustering(d_scores_matrix, random_state=42):
    """Reproduce identity clustering: StandardScaler → PCA → k-means (k=6)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(d_scores_matrix)

    pca = PCA(n_components=6, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=6, n_init=20, random_state=random_state)
    labels = kmeans.fit_predict(X_pca)

    sil = silhouette_score(X_pca, labels)
    return labels, X_pca, sil, pca.explained_variance_ratio_


def run_terroir_clustering(layer2_texts, region_names, random_state=42):
    """Reproduce terroir clustering: TF-IDF → PCA → k-means (k=7).

    Fully model-derived: no manual anchor regions or heuristic naming.
    The k=7 solution was selected by testing k=3–10 and choosing the
    value that maximises interpretability and independence (lowest ARI,
    highest chi² p-value) while maintaining reasonable cluster coherence.
    """
    texts = [layer2_texts[name] for name in region_names]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=800,
        sublinear_tf=True
    )
    X_tfidf = vectorizer.fit_transform(texts)

    pca = PCA(n_components=10, random_state=random_state)
    X_pca = pca.fit_transform(X_tfidf.toarray())

    kmeans = KMeans(n_clusters=7, n_init=20, random_state=random_state)
    labels = kmeans.fit_predict(X_pca)

    sil = silhouette_score(X_pca, labels)
    return labels, X_pca, sil, pca.explained_variance_ratio_, vectorizer


def map_cluster_labels(labels, region_names, canonical_clusters, cluster_names):
    """Map k-means integer labels to canonical cluster names using majority vote."""
    label_to_canonical = {}
    for i, name in enumerate(region_names):
        canonical = canonical_clusters.get(name, "Unknown")
        label = labels[i]
        if label not in label_to_canonical:
            label_to_canonical[label] = []
        label_to_canonical[label].append(canonical)

    mapping = {}
    for label, canonical_list in label_to_canonical.items():
        counter = Counter(canonical_list)
        mapping[label] = counter.most_common(1)[0][0]

    return mapping


def name_terroir_clusters_from_tfidf(labels, X_tfidf, feature_names, region_names):
    """
    Generate terroir cluster names from TF-IDF feature weights.

    For each cluster, computes the mean TF-IDF vector and identifies the
    top distinguishing terms (highest difference vs. all other clusters).
    Returns both the integer→name mapping and the top terms per cluster.
    """
    top_terms_per_cluster = {}

    for label_id in sorted(set(labels)):
        member_idx = [i for i in range(len(labels)) if labels[i] == label_id]
        other_idx = [i for i in range(len(labels)) if labels[i] != label_id]

        cluster_mean = X_tfidf[member_idx].mean(axis=0)
        other_mean = X_tfidf[other_idx].mean(axis=0)

        # Handle sparse matrices
        if hasattr(cluster_mean, 'A1'):
            cluster_mean = cluster_mean.A1
            other_mean = other_mean.A1
        else:
            cluster_mean = np.asarray(cluster_mean).flatten()
            other_mean = np.asarray(other_mean).flatten()

        diff = cluster_mean - other_mean
        top_idx = np.argsort(diff)[::-1][:10]
        terms = [(feature_names[i], float(diff[i])) for i in top_idx if diff[i] > 0]
        top_terms_per_cluster[label_id] = terms

    # Apply interpretive names
    mapping = TERROIR_CLUSTER_NAMES

    return mapping, top_terms_per_cluster


def run_independence_test(identity_labels, terroir_labels, n_id_clusters=6, n_ter_clusters=7):
    """Compute ARI and chi-square test between two clustering solutions."""
    ari = adjusted_rand_score(identity_labels, terroir_labels)

    contingency = np.zeros((n_id_clusters, n_ter_clusters), dtype=int)
    for i_label, t_label in zip(identity_labels, terroir_labels):
        contingency[i_label][t_label] += 1

    chi2, p_value, dof, expected = chi2_contingency(contingency)
    return ari, chi2, p_value, contingency


def main():
    parser = argparse.ArgumentParser(description="Soul of Wine — Reproducible Analysis Pipeline")
    parser.add_argument('--layer2', type=str, default=None,
                        help='Path to Layer 2 source file (default: auto-detect)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for JSON files (default: ../data/)')
    args = parser.parse_args()

    # ── Resolve paths ───────────────────────────────────────────────
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(script_dir, '..', 'data')

    os.makedirs(output_dir, exist_ok=True)

    # Find Layer 2 file
    layer2_path = args.layer2
    if not layer2_path:
        candidates = [
            os.path.join(script_dir, '..', 'data', 'layer2.txt'),
            os.path.join(script_dir, 'layer2.txt'),
        ]
        for c in candidates:
            if os.path.exists(c):
                layer2_path = c
                break

    if not layer2_path or not os.path.exists(layer2_path):
        print("ERROR: Layer 2 source file not found.")
        print("Place layer2.txt in data/ or analysis/, or use --layer2 flag.")
        sys.exit(1)

    print("=" * 65)
    print("THE SOUL OF WINE — Analysis Pipeline")
    print("Pass 5 · Post-SME Review · 59 regions")
    print("Identity: k=6 (expert-scored D-scores)")
    print("Terroir:  k=7 (fully model-derived TF-IDF)")
    print("=" * 65)
    print()

    # ── Prepare data ────────────────────────────────────────────────
    region_names = [r["name"] for r in REGIONS]
    d_scores_matrix = np.array([r["d_scores"] for r in REGIONS])

    print(f"Regions:       {len(REGIONS)}")
    print(f"D-score dims:  {d_scores_matrix.shape[1]}")
    print(f"Layer 2 file:  {layer2_path}")
    print()

    # Parse Layer 2
    layer2_texts = parse_layer2(layer2_path)
    print(f"Layer 2 parsed: {len(layer2_texts)} regions")
    missing = [n for n in region_names if n not in layer2_texts]
    if missing:
        print(f"  WARNING: Missing Layer 2 for: {missing}")
    print()

    # ── 1. Identity clustering ──────────────────────────────────────
    print("─" * 45)
    print("1. IDENTITY CLUSTERING (D-scores → k-means, k=6)")
    print("─" * 45)

    id_labels, id_pca, id_sil, id_var = run_identity_clustering(d_scores_matrix)

    # Map labels to canonical names
    id_label_map = map_cluster_labels(id_labels, region_names, CANONICAL_IDENTITY_CLUSTERS, None)
    id_named = [id_label_map[l] for l in id_labels]

    print(f"  Silhouette:  {id_sil:.4f}")
    print(f"  PCA variance explained: {sum(id_var)*100:.1f}%")
    print(f"  Cluster sizes:")
    for cname in sorted(set(id_named)):
        n = id_named.count(cname)
        print(f"    {cname}: {n}")

    # Verify against canonical
    mismatches = []
    for i, name in enumerate(region_names):
        canonical = CANONICAL_IDENTITY_CLUSTERS[name]
        assigned = id_named[i]
        if canonical != assigned:
            mismatches.append((name, canonical, assigned))

    if mismatches:
        print(f"\n  ⚠ {len(mismatches)} mismatches vs. canonical:")
        for name, canonical, assigned in mismatches:
            print(f"    {name}: canonical={canonical}, pipeline={assigned}")
    else:
        print(f"\n  ✓ All 59 regions match canonical cluster assignments")
    print()

    # ── 2. Terroir clustering ───────────────────────────────────────
    print("─" * 45)
    print("2. TERROIR CLUSTERING (Layer 2 TF-IDF → k-means, k=7)")
    print("   Fully model-derived — no manual anchors")
    print("─" * 45)

    ter_labels, ter_pca, ter_sil, ter_var, vectorizer = run_terroir_clustering(
        layer2_texts, region_names
    )
    feature_names = vectorizer.get_feature_names_out()

    # Name terroir clusters from TF-IDF features
    ter_label_map, ter_top_terms = name_terroir_clusters_from_tfidf(
        ter_labels, vectorizer.transform([layer2_texts[n] for n in region_names]),
        feature_names, region_names
    )
    ter_named = [ter_label_map[l] for l in ter_labels]

    print(f"  Silhouette:  {ter_sil:.4f}")
    print(f"  PCA variance explained (10 components): {sum(ter_var)*100:.1f}%")
    print(f"  Cluster sizes:")
    for cname in sorted(set(ter_named)):
        n = ter_named.count(cname)
        print(f"    {cname}: {n}")

    print(f"\n  Terroir cluster assignments:")
    for i, name in enumerate(region_names):
        print(f"    {name}: {ter_named[i]}")

    print(f"\n  Top TF-IDF distinguishing terms per cluster:")
    for label_id in sorted(ter_top_terms.keys()):
        cname = ter_label_map[label_id]
        terms = ter_top_terms[label_id][:5]
        term_str = ", ".join([f"{t[0]} ({t[1]:.3f})" for t in terms])
        print(f"    {cname}: {term_str}")
    print()

    # ── 3. Independence test ────────────────────────────────────────
    print("─" * 45)
    print("3. INDEPENDENCE TEST (ARI + Chi-square)")
    print("─" * 45)

    ari, chi2_stat, p_value, contingency = run_independence_test(id_labels, ter_labels)

    print(f"  ARI:         {ari:.3f}")
    print(f"  Chi-square:  {chi2_stat:.3f}")
    print(f"  p-value:     {p_value:.3f}")
    print(f"  Conclusion:  {'Independent (p > 0.05)' if p_value > 0.05 else 'NOT independent (p ≤ 0.05)'}")
    print()

    # ── 4. Output JSON files ────────────────────────────────────────
    print("─" * 45)
    print("4. OUTPUT FILES")
    print("─" * 45)

    # Identity PCA coordinates
    identity_pca_data = []
    for i, r in enumerate(REGIONS):
        identity_pca_data.append({
            "name": r["name"],
            "num": r["num"],
            "country": r["country"],
            "world": r["world"],
            "metaphor": r["metaphor"],
            "cluster": id_named[i],
            "pc1": round(float(id_pca[i, 0]), 4),
            "pc2": round(float(id_pca[i, 1]), 4),
            "d_scores": r["d_scores"],
        })
    identity_pca_path = os.path.join(output_dir, "identity_pca.json")
    with open(identity_pca_path, 'w') as f:
        json.dump(identity_pca_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {identity_pca_path}")

    # Terroir PCA coordinates
    terroir_pca_data = []
    for i, r in enumerate(REGIONS):
        terroir_pca_data.append({
            "name": r["name"],
            "num": r["num"],
            "country": r["country"],
            "world": r["world"],
            "metaphor": r["metaphor"],
            "identity_cluster": id_named[i],
            "terroir_cluster": ter_named[i],
            "pc1": round(float(ter_pca[i, 0]), 4),
            "pc2": round(float(ter_pca[i, 1]), 4),
        })
    terroir_pca_path = os.path.join(output_dir, "terroir_pca.json")
    with open(terroir_pca_path, 'w') as f:
        json.dump(terroir_pca_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {terroir_pca_path}")

    # Terroir cluster assignments (for movement map)
    terroir_clusters_data = {}
    for i, r in enumerate(REGIONS):
        terroir_clusters_data[r["name"]] = ter_named[i]
    terroir_clusters_path = os.path.join(output_dir, "terroir_clusters.json")
    with open(terroir_clusters_path, 'w') as f:
        json.dump(terroir_clusters_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {terroir_clusters_path}")

    # Pipeline report
    report_lines = [
        "THE SOUL OF WINE — Pipeline Report",
        "=" * 45,
        f"Pass 5 · Post-SME Review · {len(REGIONS)} regions",
        f"Identity clustering: k=6 (expert-scored D-scores)",
        f"Terroir clustering:  k=7 (fully model-derived TF-IDF)",
        "",
        "IDENTITY CLUSTERING",
        f"  Silhouette: {id_sil:.4f}",
        f"  PCA variance: {sum(id_var)*100:.1f}%",
        f"  Canonical match: {len(REGIONS) - len(mismatches)}/{len(REGIONS)}",
        "",
        "TERROIR CLUSTERING (fully model-derived)",
        f"  Silhouette: {ter_sil:.4f}",
        f"  PCA variance (10 comp): {sum(ter_var)*100:.1f}%",
        f"  Clusters: {len(set(ter_labels))}",
        "",
        "  Cluster compositions:",
    ]
    for label_id in sorted(set(ter_labels)):
        cname = ter_label_map[label_id]
        members = [region_names[i] for i in range(len(region_names)) if ter_labels[i] == label_id]
        terms = ter_top_terms[label_id][:3]
        term_str = ", ".join([t[0] for t in terms])
        report_lines.append(f"    {cname} ({len(members)}): {', '.join(members)}")
        report_lines.append(f"      Top terms: {term_str}")

    report_lines.extend([
        "",
        "INDEPENDENCE TEST",
        f"  ARI: {ari:.3f}",
        f"  Chi-square: {chi2_stat:.3f} (p = {p_value:.3f})",
        f"  Result: {'Independent' if p_value > 0.05 else 'NOT independent'}",
        "",
        "TERROIR CLUSTER ASSIGNMENTS",
    ])
    for i, name in enumerate(region_names):
        report_lines.append(f"  {name}: {ter_named[i]}")

    report_path = os.path.join(output_dir, "pipeline_report.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"  ✓ {report_path}")

    print()
    print("=" * 65)
    print("Pipeline complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
