import os
from pathlib import Path
import re
import json
from bs4 import BeautifulSoup

import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim import corpora
from gensim.models import LdaModel


# -----------------------------
# Configuration
# -----------------------------
# Ground truth: total number of topics in BankSearch dataset is 1000
NUM_TOPICS_CORPUS = 1000  # number of latent topics
# since in ground truth each document has exactly one topic, we extract the same number of dominant topics
N_TOPICS_PER_DOC = 1  # number of dominant topics to extract per document
N_TOP_WORDS_PER_TOPIC = 10
MIN_TOKEN_LENGTH = 3


# -----------------------------
# Preprocessing utilities
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def extract_id_and_html(text: str) -> tuple[str, str]:
    """
    Extract document ID and HTML content from BankSearch format.
    """
    id_match = re.search(r"^ID=(.+)$", text, re.MULTILINE)
    html_match = re.search(r"HTML=\n?(.*)", text, re.DOTALL)

    if not id_match or not html_match:
        print(f"Warning: could not extract ID or HTML from document. {text[:30]}...")
        return "", ""

    doc_id = id_match.group(1).strip()
    html = html_match.group(1)

    return doc_id, html


def clean_html(html: str) -> str:
    """
    Remove HTML tags, scripts, styles.
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    return text


def preprocess_text(text: str):
    """
    Tokenize, lowercase, remove stopwords, lemmatize.
    """
    text = text.lower()
    tokens = re.findall(r"[a-z]+", text)

    cleaned = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) >= MIN_TOKEN_LENGTH
    ]

    return cleaned


# -----------------------------
# Dataset loading
# -----------------------------
def load_banksearch_dataset(dataset_path: str):
    documents = []
    doc_ids = []
    error_counter = {}

    for filename in os.listdir(dataset_path):
        if not filename.endswith(".txt"):
            error_counter["no-txt"] = error_counter.get("no-txt", 0) + 1
            continue

        file_path = os.path.join(dataset_path, filename)

        with open(file_path, "r", encoding="latin1", errors="ignore") as f:
            raw_text = f.read()

        doc_id, html = extract_id_and_html(raw_text)
        if doc_id == "":
            error_counter["no-id"] = error_counter.get("no-id", 0) + 1
            continue

        plain_text = clean_html(html)
        tokens = preprocess_text(plain_text)

        if tokens:
            doc_ids.append(doc_id)
            documents.append(tokens)
        else:
            error_counter["no-tokens"] = error_counter.get("no-tokens", 0) + 1

    print(f"Skipped files due to errors or invalid format (reason: counts): {error_counter}")
    return doc_ids, documents


# -----------------------------
# LDA pipeline
# -----------------------------
def run_lda(doc_ids, documents):
    dictionary = corpora.Dictionary(documents)
    dictionary.filter_extremes(no_below=5, no_above=0.5)

    corpus = [dictionary.doc2bow(doc) for doc in documents]

    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS_CORPUS,
        random_state=42,
        passes=10,
        alpha="auto",
        per_word_topics=False,
    )

    return lda, corpus, dictionary


def extract_document_topics(lda, corpus, doc_ids, n_topics_per_doc=N_TOPICS_PER_DOC, n_top_words_per_topic=N_TOP_WORDS_PER_TOPIC):
    """
    For each document:
    - find the dominant topics
    - extract top words for those topics
    Returns a dict: {doc_id: {topic_id: [top_words]}}
    """
    result = {}

    for doc_id, bow in zip(doc_ids, corpus):
        # Get topic probabilities and pick top N topics
        topic_probs = lda.get_document_topics(bow)
        # Sort topics by probability descending and take top n_topics_per_doc
        dominant_topics = sorted(topic_probs, key=lambda x: x[1], reverse=True)[:n_topics_per_doc]
        dominant_topic_ids = [topic for topic, _ in dominant_topics]

        # Build dictionary {topic_id: [top_words]}
        words_only = {
            topic: [word for word, _ in lda.show_topic(topic, n_top_words_per_topic)]
            for topic in dominant_topic_ids
        }

        result[doc_id] = words_only

    return result


# -----------------------------
# Main
# -----------------------------
def main(dataset_path, output_file):
    print("Loading dataset...")
    doc_ids, documents = load_banksearch_dataset(dataset_path)

    print(f"Loaded {len(documents)} documents")

    print("Training LDA...")
    lda, corpus, dictionary = run_lda(doc_ids, documents)

    print("Extracting document-topic mappings...")
    doc_topic_map = extract_document_topics(lda, corpus, doc_ids)

    print(f"Saving output to {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(doc_topic_map, f, indent=2)

    print("Done.")


if __name__ == "__main__":

    dataset_path = Path("resources/Dataset")
    assert dataset_path.exists(), "No input dataset exits, ensure to download it from http://lib.stat.cmu.edu/datasets/."
    output_file = Path("resources/banksearch2topics")

    output_file.mkdir(parents=True, exist_ok=True)
    output_file = output_file / "banksearch_lda_topics.json"

    main(dataset_path, output_file)
