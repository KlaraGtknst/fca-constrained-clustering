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