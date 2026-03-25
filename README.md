
# Social Analytics & Text Analysis Projects
---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Repository structure](#2-repository-structure)
3. [Environment setup](#3-environment-setup)
4. [Supported NLTK corpora](#4-supported-nltk-corpora)
5. [Standard analysis pipeline](#5-standard-analysis-pipeline)
6. [NetworkX integration](#6-networkx-integration)
7. [Zipf's law reference](#7-zipfs-law-reference)
8. [Writing up results](#8-writing-up-results)
9. [Key concepts & glossary](#9-key-concepts--glossary)
10. [References & attribution](#10-references--attribution)

---

## 1. Project overview

This repository contains a suite of text-based social and web analytics projects. Each project applies computational linguistics, network analysis, and statistical modeling to real-world corpora. Projects are designed to be modular — corpus selection, preprocessing, and analysis pipelines are interchangeable across assignments.

> **Shared theme:** All projects in this repository treat text as a network. Documents are nodes, words are nodes, co-occurrences are edges. The analytical framework is consistent whether the corpus is presidential speeches, social media posts, news wire, or literary texts.

---

## 2. Repository structure

```
project-root/
│
├── data/
│   ├── raw/                  # Original corpus files — never modified
│   └── processed/            # Tokenized, lemmatized, cleaned output
│
├── notebooks/                # Jupyter notebooks, one per project step
│   ├── step1_load_corpus.ipynb
│   ├── step2_tokenize.ipynb
│   ├── step3_unique_words.ipynb
│   ├── step4_50pct_coverage.ipynb
│   ├── step5_top200_subgraph.ipynb
│   ├── step6_frequency_chart.ipynb
│   └── step7_zipf_test.ipynb
│
├── src/                      # Reusable Python modules
│   ├── tokenizer.py          # Clean, lemmatize, remove stopwords
│   ├── graph_builder.py      # NetworkX graph construction helpers
│   └── plotter.py            # Frequency and log-log chart functions
│
├── outputs/
│   ├── figures/              # All charts, diagrams, and graph visualizations
│   └── reports/              # Final write-ups and README files
│
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 3. Environment setup

### Python dependencies

```bash
pip install nltk networkx matplotlib scipy numpy pandas jupyter
```

### NLTK corpus downloads

Run once in any Python environment or at the top of your notebook:

```python
import nltk

nltk.download('state_union')   # Presidential SOTU addresses
nltk.download('punkt')         # Sentence tokenizer
nltk.download('punkt_tab')     # Punkt tokenizer tables
nltk.download('stopwords')     # Common stopword list
nltk.download('wordnet')       # Lemmatization dictionary
nltk.download('brown')         # General English baseline corpus
nltk.download('reuters')       # News wire corpus (topic-tagged)
nltk.download('inaugural')     # Presidential inaugural addresses
```

---

## 4. Supported NLTK corpora

All projects are designed to work with any of the following NLTK corpora. **Swap the corpus name in Step 1 and the rest of the pipeline runs unchanged.**

| NLTK corpus | Domain | Size (approx.) | Best for |
|---|---|---|---|
| `state_union` | Political speech | 1.6M chars, 65 docs | Zipf, political vocab, temporal analysis |
| `reuters` | News / finance | 1.3M words, 10,788 docs | Topic modeling, cross-domain Zipf |
| `brown` | General American English | 1M words, 500 docs | Baseline / control corpus |
| `gutenberg` | Classic literature | 18 texts, varied sizes | Author style analysis |
| `inaugural` | Presidential inaugural addresses | 58 speeches, 1789–2021 | Political discourse, historical vocab |

### Loading any corpus

```python
# ── State of the Union (this project) ──────────────────────────────────────
from nltk.corpus import state_union
raw   = state_union.raw()          # full corpus as one string
files = state_union.fileids()      # list of speech filenames
one   = state_union.raw("2000-Clinton.txt")  # single speech

# ── Swap to any other corpus ────────────────────────────────────────────────
# from nltk.corpus import reuters    →  reuters.raw(), reuters.fileids()
# from nltk.corpus import brown      →  brown.raw(),   brown.fileids()
# from nltk.corpus import gutenberg  →  gutenberg.raw('melville-moby_dick.txt')
# from nltk.corpus import inaugural  →  inaugural.raw(), inaugural.fileids()
```

---

## 5. Standard analysis pipeline

Every project follows the same seven-step pipeline. Each step maps to one Jupyter notebook cell and one answerable research question.

| Step | Name | Description | Key output | Maps to Q |
|---|---|---|---|---|
| 1 | Load corpus | Import NLTK corpus, inspect file IDs, build NetworkX document graph | `raw`, `files`, `G` | — |
| 2 | Tokenize & clean | Lowercase, alpha-only filter, lemmatize, remove stopwords | `clean` token list | — |
| 3 | Unique word count | Count total tokens and unique lemmas | `freq` Counter, node set | Q2 |
| 4 | 50% coverage | Accumulate sorted frequencies until ≥ 50% of corpus covered | `half_cover` integer | Q3 |
| 5 | Top-200 subgraph | Extract top-200 words; build co-occurrence edges in NetworkX | `H` subgraph | Q4 |
| 6 | Frequency chart | Plot relative frequency of top-200 words (bar + log–log) | `zipf_top200.png` | Q5 |
| 7 | Zipf test | OLS regression on log–log plot; report α and R² | α, R², verdict | Q6 |
| 8 | Corpus comparison | Compare against Brown baseline; compute per-word divergence | divergence chart | Q7 |

### Complete pipeline code

```python
# ── Step 1: Load ─────────────────────────────────────────────────────────────
import nltk
from nltk.corpus import state_union
import networkx as nx

nltk.download('state_union', quiet=True)
nltk.download('punkt',       quiet=True)
nltk.download('punkt_tab',   quiet=True)
nltk.download('stopwords',   quiet=True)
nltk.download('wordnet',     quiet=True)

raw   = state_union.raw()
files = state_union.fileids()

# Document graph — bipartite: presidents ↔ speeches
G = nx.Graph()
for f in files:
    year, president = f[:4], f[5:].replace('.txt', '')
    G.add_node(f,         node_type='speech',    year=year)
    G.add_node(president, node_type='president')
    G.add_edge(president, f)

print(f"Corpus : {len(raw):,} characters | {len(files)} speeches")
print(f"Graph  : {G.number_of_nodes()} nodes | {G.number_of_edges()} edges")

# ── Step 2: Tokenize & clean ──────────────────────────────────────────────────
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from collections import Counter
import re

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

tokens = word_tokenize(raw.lower())
clean  = [
    lemmatizer.lemmatize(t)
    for t in tokens
    if re.match(r'^[a-z]+$', t)
    and t not in stop_words
    and len(t) > 2
]

# ── Step 3: Unique words (Q2) ─────────────────────────────────────────────────
freq   = Counter(clean)
total  = sum(freq.values())
unique = len(freq)

G_words = nx.Graph()
G_words.add_nodes_from(freq.keys())
nx.set_node_attributes(G_words, freq, 'frequency')

print(f"Total tokens  : {total:,}")
print(f"Unique words  : {unique:,}")
print(f"Type-token ratio: {unique/total:.4f}")

# ── Step 4: 50% coverage (Q3) ────────────────────────────────────────────────
sorted_words = freq.most_common()
cumulative, half_cover = 0, 0
for word, count in sorted_words:
    cumulative += count
    half_cover += 1
    if cumulative >= total / 2:
        break

print(f"Top {half_cover} words cover 50% of {total:,} tokens")

# ── Step 5: Top-200 subgraph (Q4) ────────────────────────────────────────────
from itertools import islice

top200       = freq.most_common(200)
top200_words = [w for w, _ in top200]

H = nx.Graph()
H.add_nodes_from(top200_words)
nx.set_node_attributes(H, {w: c for w, c in top200}, 'frequency')

# Co-occurrence edges (sliding window = 5)
window = 5
for i, word in enumerate(clean):
    if word in H:
        neighbors = list(islice(
            (w for w in clean[i+1:i+window] if w in H), window
        ))
        for neighbor in neighbors:
            if H.has_edge(word, neighbor):
                H[word][neighbor]['weight'] = H[word][neighbor].get('weight', 0) + 1
            else:
                H.add_edge(word, neighbor, weight=1)

print(f"Top-200 subgraph: {H.number_of_nodes()} nodes | {H.number_of_edges()} edges")

# ── Step 6: Frequency chart (Q5) ─────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np

ranks     = list(range(1, 201))
rel_freqs = [count / total for _, count in top200]
degrees   = [H.degree(w) for w, _ in top200]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("State of the Union Corpus — Zipf Analysis", fontsize=13)

# Panel A: bar chart colored by co-occurrence degree
colors = plt.cm.Blues(np.array(degrees) / max(degrees))
axes[0].bar(ranks, [f * 100 for f in rel_freqs], color=colors, width=0.9)
axes[0].set_xlabel('Rank')
axes[0].set_ylabel('Relative Frequency (%)')
axes[0].set_title('Top-200 Words — Relative Frequency')

# Panel B: log-log
axes[1].loglog(ranks, rel_freqs, 'o', ms=4, alpha=0.7, color='#185FA5', label='Observed')
axes[1].set_xlabel('log(Rank)')
axes[1].set_ylabel('log(Relative Frequency)')
axes[1].set_title('Log–Log Plot (Zipf test)')

plt.tight_layout()
plt.savefig('outputs/figures/zipf_top200.png', dpi=150, bbox_inches='tight')

# ── Step 7: Zipf test (Q6) ───────────────────────────────────────────────────
from scipy import stats

log_ranks = np.log(ranks)
log_freqs = np.log(rel_freqs)

slope, intercept, r, p, se = stats.linregress(log_ranks, log_freqs)
alpha     = -slope
r_squared = r ** 2

fitted = np.exp(intercept + slope * log_ranks)
axes[1].loglog(ranks, fitted, 'r--', linewidth=1.5,
               label=f'Zipf fit  α={alpha:.2f}  R²={r_squared:.4f}')
axes[1].legend()
plt.savefig('outputs/figures/zipf_top200.png', dpi=150, bbox_inches='tight')

print(f"Zipf exponent α : {alpha:.4f}  (ideal = 1.0)")
print(f"R²              : {r_squared:.4f}  (ideal = 1.0)")
print(f"Follows Zipf?   : {'Yes' if abs(alpha - 1) < 0.2 and r_squared > 0.95 else 'Approximate'}")

# ── Step 8: Corpus comparison (Q7) ───────────────────────────────────────────
from nltk.corpus import brown

brown_freq  = Counter(
    lemmatizer.lemmatize(w.lower())
    for w in brown.words()
    if re.match(r'^[a-z]+$', w.lower()) and w.lower() not in stop_words
)
brown_total = sum(brown_freq.values())

print(f"\n{'Word':<15} {'Corpus RF':>10} {'Brown RF':>10} {'Ratio':>7}")
print("-" * 46)
for word, count in top200[:20]:
    corp_rf  = count / total
    brown_rf = brown_freq.get(word, 1) / brown_total
    print(f"{word:<15} {corp_rf:>10.5f} {brown_rf:>10.5f} {corp_rf/brown_rf:>6.1f}x")
```

---

## 6. NetworkX integration

NetworkX models the corpus as a series of graphs at increasing levels of granularity.

### Graph types

#### Document graph (Step 1)
Bipartite graph — authors on one side, documents on the other. Edge = "authored by."

```
Purpose  : Corpus-level overview; per-author subgraph extraction
Nodes    : presidents + speech filenames
Edges    : president ──── speech (authored-by)
Metrics  : G.degree(president) = number of speeches
```

#### Word co-occurrence graph (Step 5)
Weighted undirected graph of the top-200 vocabulary.

```
Purpose  : Vocabulary hub detection; collocation analysis
Nodes    : top-200 lemmas  (node attr: frequency)
Edges    : word_A ──── word_B  (edge attr: weight = co-occurrence count)
Metrics  : degree centrality, betweenness, clustering coefficient
```

#### Full word graph (Step 3 — optional)
Node-only graph of the entire vocabulary for summary statistics.

```
Purpose  : Type-token ratio, degree distribution, power-law check
Nodes    : all unique lemmas  (node attr: frequency)
Edges    : none (add co-occurrence edges if needed)
```

> **Web analytics parallel:** Word frequency graphs obey the same power-law degree distribution as web link graphs. The top-200 words are your "hub pages" — a tiny fraction of nodes that absorb most of the traffic. This is why Zipf's Law and web analytics share the same mathematical foundation.

### Useful NetworkX snippets

```python
# Most-connected word (highest degree in co-occurrence graph)
top_hub = max(H.nodes(), key=lambda n: H.degree(n))
print(f"Hub word: {top_hub}  degree={H.degree(top_hub)}")

# Nodes sorted by frequency (node attribute)
by_freq = sorted(H.nodes(data=True), key=lambda x: x[1]['frequency'], reverse=True)

# Subgraph for a single president's speeches
reagan_speeches = [f for f in files if 'Reagan' in f]
reagan_raw  = ' '.join(state_union.raw(f) for f in reagan_speeches)

# Degree distribution (power-law check)
degrees = [d for _, d in H.degree()]
plt.loglog(sorted(degrees, reverse=True), 'o', ms=4)
plt.title("Degree distribution — top-200 co-occurrence graph")
```

---

## 7. Zipf's law reference

### Definition

Zipf's Law states that in any sufficiently large natural language corpus, word frequency is inversely proportional to its rank:

```
frequency(rank)  ∝  1 / rank^α     where α ≈ 1.0
```

On a **log–log plot**, this produces a straight line with slope **−α**.

### How to test it

1. Compute relative frequency for each of the top-200 words: `freq / total_tokens`
2. Plot `log(rank)` on the x-axis and `log(relative_frequency)` on the y-axis
3. Fit OLS linear regression with `scipy.stats.linregress`
4. Report slope (= −α), intercept, and R²
5. Overlay the fitted line on the log–log scatter plot

```python
from scipy import stats
import numpy as np

slope, intercept, r, p, se = stats.linregress(np.log(ranks), np.log(rel_freqs))
alpha     = -slope
r_squared = r ** 2
```

### Interpretation

| α range | R² | Interpretation |
|---|---|---|
| 0.9 – 1.1 | > 0.98 | Excellent fit — strong Zipf behavior |
| 0.8 – 0.9 or 1.1 – 1.2 | 0.93 – 0.98 | Good fit — approximate Zipf behavior |
| > 1.2 | any | Steep slope — domain-specific vocabulary dominates |
| < 0.8 | < 0.90 | Flat slope — unusually uniform vocabulary (e.g., Twitter) |

### Corpus-specific notes

| Corpus | Expected α | Notes |
|---|---|---|
| `state_union` | 1.05 – 1.10 | Political terms inflate top ranks |
| `brown` | 0.95 – 1.05 | Closest to "ideal" Zipf — general English |
| `reuters` | 1.00 – 1.15 | Finance terms cluster at mid-ranks |
| `gutenberg` | 1.05 – 1.20 | Literary vocabulary; author names spike |
| `inaugural` | 1.05 – 1.15 | Similar to state_union; smaller corpus |

---

## 8. Writing up results

Address all seven questions. Map each answer back to a pipeline step.

### Q2 — How many unique words?

```
Define "unique" clearly:
  Option A: raw types (case-sensitive)     → highest count
  Option B: lowercased types               → moderate
  Option C: lemmatized + no stopwords      → lowest, most meaningful ✓ recommended
Report: total tokens, unique words, type–token ratio
```

### Q3 — How many words cover 50% of the corpus?

```
Report: the integer N such that the top-N words account for ≥ 50% of all tokens
Visualize: cumulative coverage curve (x = rank, y = % of corpus covered)
Interpret: small N = strong power-law concentration
```

### Q4 — Top 200 words

```
Report: table of rank | word | count | relative frequency (%)
Include: NetworkX subgraph node/edge counts
```

### Q5 — Frequency graph

```
Required charts:
  1. Bar chart — rank (x) vs relative frequency (y), top 200 words
  2. Log–log scatter — log rank (x) vs log relative frequency (y)
Optional: color bars by NetworkX co-occurrence degree
```

### Q6 — Does it follow Zipf's Law?

```
Report: α, R², p-value
Include: log–log plot with fitted Zipf line annotated
Conclude: Yes / Approximate / No — with explanation
```

### Q7 — How does this corpus differ from "all corpora"?

```
Approach: compare your top-200 relative frequencies against Brown corpus
Flag words with ratio > 3.0x (domain markers)
Common patterns:
  Political speech  → nation, congress, freedom spike
  News/finance      → percent, company, market spike
  Literature        → archaic words, character names spike
  Twitter/social    → flat distribution, slang, truncated Zipf
```

> **Tip:** A word appearing 5x more often in your corpus than in Brown is a domain signature — it tells you something specific about the language community that produced the text.

---

## 9. Key concepts & glossary

| Term | Definition |
|---|---|
| **Corpus** | A structured collection of text documents used as the basis for linguistic analysis |
| **Token** | A single occurrence of a word in the corpus. "The cat sat" = 3 tokens |
| **Type** | A distinct word form. "The cat and the dog" = 4 types (the, cat, and, dog) |
| **Lemma** | Dictionary base form. "running", "runs", "ran" → lemma "run" |
| **Type–token ratio (TTR)** | Types ÷ Tokens. Higher TTR = more diverse vocabulary |
| **Relative frequency** | Word count ÷ total tokens. Enables cross-corpus comparison |
| **Stopword** | High-frequency, low-information word (the, and, of). Removed before semantic analysis |
| **Zipf's Law** | Empirical law: word frequency ∝ 1/rank^α. Holds across virtually all natural language |
| **Zipf exponent (α)** | Slope of the log–log rank–frequency line. α ≈ 1.0 for most natural language |
| **Power law** | A distribution where P(x) ∝ x^−α. Zipf, web link graphs, and social networks all follow power laws |
| **Co-occurrence graph** | Graph where nodes are words and edges connect words that appear near each other |
| **Hub word** | High-degree node in the co-occurrence graph. Equivalent to a hub page in web analytics |
| **Document graph** | Bipartite graph connecting authors to their documents |
| **Degree centrality** | Fraction of nodes a given node is connected to. High = hub |
| **OLS regression** | Ordinary least squares — used here to fit the Zipf line on the log–log plot |
| **Brown corpus** | 1M-word balanced sample of American English (1961). Standard NLP baseline |

---

## 10. References & attribution

- **NLTK** — Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly. https://www.nltk.org
- **Zipf (1949)** — Zipf, G.K. (1949). *Human Behavior and the Principle of Least Effort*. Addison-Wesley.
- **NetworkX** — Hagberg, A., Swart, P., & Chult, D. (2008). Exploring network structure, dynamics, and function using NetworkX. *SciPy Conference Proceedings*.
- **Manning & Schütze** — Manning, C. & Schütze, H. (1999). *Foundations of Statistical Natural Language Processing*. MIT Press.
- **Barabási** — Barabási, A.L. (2016). *Network Science*. Cambridge University Press. http://networksciencebook.com
- **State of the Union corpus** — Available via `nltk.corpus.state_union`. Truman (1945) through G.W. Bush (2006). 65 speeches.

---

*Generated for DATA 620 / Web Analytics · CUNY School of Professional Studies · 2025*
