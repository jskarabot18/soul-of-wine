# The Soul of Wine

**Terroir shapes what grapes can become. History, ambition, and culture determine what a region chooses to be.**

An anthropological and sociological study of wine region identity, asking whether the cultural character of a wine region can be predicted from its physical terroir. The answer, measured across 59 regions and 16 countries, is no.

🔗 **[Explore the project →](https://jskarabot18.github.io/soul-of-wine)**

---

## The Question

Wine regions are classified in two fundamentally different ways: by what the land gives them (geology, climate, topography) and by what they have made of themselves (narrative, ambition, historical depth, cultural character). This study asks whether these two modes of classification agree.

## The Finding

They do not. The Adjusted Rand Index between terroir clustering and identity clustering is **−0.001** (chi² p = 0.172) — effectively zero. Knowing a region's terroir cluster provides no information about its identity cluster. The two systems are statistically independent.

The map is not the soul.

## Three Key Findings

1. **Terroir shapes identity, but history modulates the signal.** Physical geography creates the conditions for winemaking, but the relationship to difficulty and tradition most strongly differentiates identity clusters. The land sets constraints; culture interprets them.

2. **The type of terroir predicts the type of identity — as tendency, not law.** Harsh terroir tends to produce identities of struggle; generous terroir tends to produce ease. But the exceptions — Santorini's survival, Beaujolais's joy, Dalmatia's tranquility — are as revealing as the pattern.

3. **Where terroir's constraints are weakest, human ambition fills the void.** New World regions with favourable growing conditions show the highest individual ambition and urgency scores. When the land does not impose an identity, the winemaker constructs one.

## The Six Identity Clusters

| Cluster | n | Character | Representative Regions |
|---------|---|-----------|----------------------|
| **Old World Interior** | 10 | Inward, terroir-defined, historically dense | Burgundy (*Devotion*), Northern Rhône (*Solitude*), Piedmont (*Philosophy*), Mosel (*Poetry*) |
| **Outward Ease** | 5 | Approachable, extroverted, pleasure-forward | Beaujolais (*Joy*), Provence (*Pleasure*), Marlborough (*Assertion*), Veneto (*Commerce*) |
| **New World Reinvention** | 12 | Intentional identity-building, constructed character | Napa Valley (*Ambition*), Swartland (*Rebellion*), Sicily (*Resurrection*), Mendoza (*Reinvention*) |
| **Old World Exterior** | 7 | Old World provenance, outward-facing identity | Bordeaux (*Business*), Champagne (*Society*), Rioja (*Patience*), Châteauneuf-du-Pape (*Family*) |
| **Against the Odds** | 10 | Marginality and struggle as central character | Santorini (*Survival*), Douro (*Endurance*), Jura (*Eccentricity*), Hunter Valley (*Defiance*) |
| **The Moderates** | 15 | Balanced, pluralistic, no single force dominates | Steiermark (*Clarity*), Goriška Brda (*Fortune*), Margaret River (*Composure*), Loire (*Sentimentality*) |

## Methodology

The study employs a dual-layer architecture:

- **Layer 1 — Identity:** 280–320 word anthropological narratives per region, written under strict vocabulary controls (no terroir, grape, or technique terms). Six D-score dimensions (−2 to +2) scored from these narratives by SME: Interiority↔Exteriority, Struggle↔Ease, Tradition↔Reinvention, Individual↔Collective, Urgency↔Timelessness, Earthly↔Transcendent.

- **Layer 2 — Terroir:** Six factual fields per region (climate, soils, principal varieties, winemaking, production structure, historical position). TF-IDF vectorised independently.

- **Clustering:** D-scores standardised, PCA-reduced, k-means (k=6). Terroir clustered separately. ARI tests independence between the two solutions.

- **Post-SME silhouette:** 0.3110 · 13 score changes across 12 regions · ARI vs. pre-SME: 0.909

## Interactive Visualisations

| Tool | Description |
|------|-------------|
| [**Cluster Explorer**](https://jskarabot18.github.io/soul-of-wine/visualizations/cluster-explorer.html) | Radar charts showing the D-score identity profile of each region. Filter by cluster, click any region for its soul metaphor and character summary. |
| [**Cluster Map**](https://jskarabot18.github.io/soul-of-wine/visualizations/cluster-map.html) | Regions projected onto principal components — identity PCA (left) and terroir PCA (right). If terroir determined identity, the two plots would show the same structure. They do not. |
| [**Movement Map**](https://jskarabot18.github.io/soul-of-wine/visualizations/movement-map.html) | Alluvial diagram showing how regions flow from terroir clusters to identity clusters. The criss-crossing bands are the visual proof of independence. |
| [**D-Score Dashboard**](https://jskarabot18.github.io/soul-of-wine/visualizations/d-score-dashboard.html) | The full 59 × 6 score matrix. Colour-coded, sortable, filterable by cluster, country, or world. |

## Research Documents

| Document | Description |
|----------|-------------|
| [**The Map and the Soul**](https://jskarabot18.github.io/soul-of-wine/docs/narrative.pdf) | The full narrative paper — hypothesis, related literature across four academic fields, methodology, results, implications, and a plain-language guide for the wine-focused reader. |
| [**Technical Appendix**](https://jskarabot18.github.io/soul-of-wine/docs/technical.pdf) | Methodology deep-dive — pipeline architecture, D-score system, clustering parameters, SME review change log, and the complete 59 × 6 score matrix with colour coding. |

## Repository Structure

```
soul-of-wine/
├── index.html                          Landing page
├── README.md
├── data/
│   ├── regions.json                    59 regions — scores, clusters, metadata
│   ├── regions.csv                     Same data as CSV
│   ├── identity_pca.json              Identity PCA coordinates
│   ├── terroir_pca.json               Terroir PCA coordinates
│   ├── terroir_clusters.json          Terroir cluster assignments
│   └── pipeline_report.txt            Reproducibility report
├── docs/
│   ├── narrative.pdf                   The Map and the Soul (research paper)
│   └── technical.pdf                   Technical Appendix
├── analysis/
│   └── pipeline.py                     Reproducible analysis pipeline
└── visualizations/
    ├── cluster-explorer.html           Radar chart explorer
    ├── cluster-map.html                PCA scatter plots (identity + terroir)
    ├── movement-map.html               Terroir → Identity alluvial diagram
    └── d-score-dashboard.html          Full score matrix
```

## Reproducing the Analysis

```bash
# Requirements
pip install scikit-learn numpy scipy pandas

# Run from the analysis/ directory
cd analysis
python pipeline.py
```

The pipeline reproduces identity clustering (D-scores → k-means), terroir clustering (Layer 2 TF-IDF → k-means), and the ARI independence test. It outputs PCA coordinates and cluster assignments as JSON files to `data/`.

## Coverage

- **59 regions** across **16 countries**
- **Old World:** 39 regions — Austria (4), Croatia (1), France (10), Germany (5), Greece (2), Hungary (1), Italy (10), Portugal (1), Slovenia (1), Spain (4)
- **New World:** 20 regions — Argentina (2), Australia (3), Chile (1), New Zealand (3), South Africa (2), USA (9)

## Key References

- Ronen, S. & Shenkar, O. (2013). Mapping world cultures. *Journal of International Business Studies*, 44(9), 867–897.
- House, R.J. et al. (2004). *Culture, Leadership, and Organizations: The GLOBE Study of 62 Societies.* Sage.
- Hubert, L. & Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2(1), 193–218.
- Demossier, M. (2018). *Burgundy: The Global Story of Terroir.* Berghahn Books.
- Trubek, A. (2008). *The Taste of Place: A Cultural Journey into Terroir.* University of California Press.

## Author

**Jure Skarabot** · The Soul of Wine Research Project · April 2026

---

*The map is not the soul. The soul has to be built.*
