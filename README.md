# FCA constrained hierarchical clustering
This repository contains theoretical work, along with work-in-progress practical components, on aligning constraint-based hierarchical clusters with formal contexts. 
The foundational work on hierarchical clustering with constraints is presented in ["Hierarchical constraints: Providing structural bias for hierarchical clustering" (2013)](https://link.springer.com/article/10.1007/s10994-013-5397-9).
Formal contexts follow the standard definitions used in Formal Concept Analysis (FCA).


## Code
Use Python 3.11.14 (`gensim` is known to have [issues](https://www.linkedin.com/posts/01mayank_gensim-python-nlp-activity-7393890618575556608-Qh4b) with Python 3.14 as of December 2025).
Create virtual environment and install dependencies:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Lattice from Topic Model (LDA)
You can represent texts via their most representive topic.
This topic can be derived from a Topic Model (here: LDA).
Run `src/topic_model/dataset2lda_topics.py` in order to obtain **topics**.

**Topic Threshold.**
You need to define a `MIN_TOPIC_PROB` which is the threshold that decides whether a topic fits a document or not.
- All topics assigned to a document with probability > `MIN_TOPIC_PROB`are selected.
- If for a document, no topic has an assigned probability > `MIN_TOPIC_PROB`, no topic is returned. 

In order to get a feeling which value is fitting run `src/topic_model/lda_threshold_plots.py`.
You'll obtain plots like the one below which help you make informed choices about the threshold.

![Image: LDA Document-Topic Incidence Density per Threshold on the BankSearch Dataset](results/lda/incidence_density_vs_threshold.svg)

Given this plot, a reasonable threshold would `0.07`, because it ensures the document-topic incidence is sparse (i.e.,  incidence density just above `0.2`) and in terms of the elbow criterium.


**LDA Missing Documents (Update after reviewer feedback).**
The BankSearch dataset has some documents which are empty (e.g., `E0621.txt` of category `C`) or contain only floating numbers (`G0531.txt`of category `Astronomy`).
To ensure comparability for later analysis, we created a multi-step fallback system for texts which produce empty tokens from propocessing (`src/topic_model/dataset2lda_topics.py`):
1. if tokens are empty but html text could be extracted, use html instead of tokens
2. if both tokens and html text are empty, but plain text exists, use plain text instead of tokens
3. if 1. and 2. are empty, use "" instead of tokens

We included logging of error types and found that 27 documents has to fallback (1.) to using html text, while one document had to use an empty string (3.) as fallback.
Employing this strategy all 11,000 documents can be assigned topics, with average number of topics per document of 2.407.

---

**Document-Topic Incidence.**
Now, we've created a json file with the information which document has which topics, which are represented by which words.
However, as of now, we only need the document-topic incidence.
Thus, run `src/topic_model/document_representation.py` to create a document-topic **context** json file.

**Comparison to Ground-truth.**
Next, we want to compare the topics from the topic model to the ground truth.
Hence, run `src/experiments/print_stats.py`, which will produce a csv file that below.

![Image: Comparison of the ground truth document-topic incidence and the topic model document-topic incidence of the BankSearch Dataset](resources/banksearch/fca_contexts_comparison_stats.svg)

**Iceberg lattice.**
We use the resulting document-topic context to extract topic hierarchies using iceberg lattices via the TITANIC algorithm.
Iceberg lattices contain only concepts `(A,B)` whose intent has a support higher or equal to `min_supp`.
Intuively, remaining concepts are document-topic pairs, whose topics are representative for at least `min_supp` $\times 100 \%$ of the documents in the corpus. 
To get a feeling for the choice of `min_supp` we run `src/experiments/plot_concepts_vs_support.py`, resulting in the following plot.

![Image: Number of concepts in the iceberg lattice for different min_supp values on the BankSearch Dataset.](resources/banksearch/topic_model/plots/concepts_vs_support.svg)

We find that the `min_supp` value should be no higher then `0.15`, otherwise less than `10` concepts remain.
We choose `0.05`/`0.1` and obtain `24`/`13` concepts.

Using this knowledge run the `src/experiments/run_clj_file.py` file after adjusting the `min_supp` value accordingly.
This generates a `.cxt` and an `.edn` file containing the iceberg context and icebergs concepts, respectively.
The resulting lattice is plotted and saved.


### Lattice from MLB Constraints (BankSearch "ground truth")
The MLB constraints have the format `x,y,z` where `x` and `y` have to be meregd before `z`.
If there is no explicit topic id for any of `x`, `y` or `z`, it is the union of its children. 
For instance, if `x` and `y` have explicit names, but `z` has not, `z=x,y`; leading to: `x,y, x,y`

**Document-Topic Context.** 
The ground-truth *topic hierarchy* by Bade et al. is stored in `resources/banksearch/ground_truth/category_hierarchy.json`.
Each document is assigned exactly one topic, which naturally creates topic-equivalence classes.
We represent each *equivalence class*, consisting of all documents of that topic, by one representative document (cf. mapping in `resources/banksearch/ground_truth/mlb_banksearch_equivalence_classes.json`).
The dataframe containing ground-truth document-topic *context* (document ids as objects, topics as attributes) is constructed via `convert_documents_to_vectors` in `src/topic_model/document_representation.py` called automatically upon initialization of `BankSearchGroundTruthExtractor`.

**MLB constraints.**
We extract MLB constraints on (i) topic-level and (ii) document ID-level (pruned; i.e., only on topic-equivalence classes) using the `BankSearchGroundTruthExtractor` by running `src/constraints/extractor.py`.
MLB constraints (x, y, z) are constructed such that (1) all clusters containing x and z also contain y and (2) there exists a cluster containing x and y, but not z.

**Pruned Document ID-level Context**
In order to build the document ID-topic context based on the pruned document ID equivalence classes' MLB constraints, run `src/attribute_exploration/triple_exploration.py`.
This will create `mlb.cxt`.

**Expanded Context**
Run `src/attribute_exploration/expand_mlb_cxt_equivalence.py`to expand the `mlb.cxt` to contain pruned document ID-level MLB constraints to contains their equivalence class members.
The result is saved to `resources/banksearch/ground_truth/mlb_expanded.cxt`.


## Comparison of MLB and topic model context

Given the context's concepts as `.edn` files, we can compare the concepts of the topic model iceberg context with the 
concepts of 
the MLB context.
Both Hasse diagrams are plotted and saved running `src/context_comparison/run_clj_file.py` and are shown below.
Additional statistics are also generated when running that file.

Topic Model Iceberg Lattice of the BankSearch Dataset with min-support of `0.05`/`0.1`(TODO):
![Iceberg Concept Lattice of the BankSearch Dataset with min-support of 0.05.](resources/banksearch/topic_model/banksearch_0.05_iceberg.svg)

Concepts Lattice of the MLB context on the BankSearch Dataset:
![mlb_expanded_lattice.svg](resources/banksearch/ground_truth/mlb_expanded.svg)

# Comparison of the two lattices
Run `src/context_comparison/compare_concepts.py` (creates and saves coherence context), `src/context_comparison/run_clj_file.py` (statisticy about both input contexts) and `src/context_comparison/explore_cxt.py` (finds ground-truth equivalence and concept extent alignments).

`compare_concepts.py` produces (among other things) a heatmap of the similarity between the concepts of the two 
contexts, which is shown below. Similarity is calculated via the Jaccard similarity of the extents of the concepts, which are sets of documents.


![heatmap.svg](results/context_comparison/CONCEPT_SIM/heatmap.svg)

Heatmap of Jaccard similarity between both contexts (based on shared concept extents).

![Coherence-Lattice](results/context_comparison/CONCEPT_SIM/coherence.svg)


![Consistency-Lattice](results/context_comparison/intersection.svg)