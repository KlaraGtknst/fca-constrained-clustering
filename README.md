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

### Topic Model
You can represent texts via their most representive topic.
This topic can be derived from a Topic Model (here: LDA).
Run `dataset2lda_topics.py` in order to obtain topics.
You need to define a `MIN_TOPIC_PROB` which is the threshold that decides whether a topic fits a document or not.
In order to get a feeling which value is fitting run `lda_threshold_plots.py`.
You'll obtain plots like the one below which help you make informed choices about the threshold.
![LDA Document-Topic Incidence Density per Threshold on the BankSearch Dataset](https://file%2B.vscode-resource.vscode-cdn.net/Users/klara/Developer/fca-constrained-clustering/results/lda/incidence_density_vs_threshold.svg)
Given this plot, a reasonable threshold would `0.3`, because it ensures the document-topic incidence is sparse.


### Contexts
Contexts are saved as .json file.
Run `document_representation.py` to obtain contexts.
Ensure relevant files (e.g., `banksearch_lda_topics.json` for Topic Model approach) exist beforehand.