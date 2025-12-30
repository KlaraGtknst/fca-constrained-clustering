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
NUM_TOPICS = 20  # number of latent topics
TOP_WORDS_PER_TOPIC = 10
MIN_TOKEN_LENGTH = 3


# -----------------------------
# Preprocessing utilities
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def extract_id_and_html(text: str):
    """
    Extract document ID and HTML content from BankSearch format.
    """
    id_match = re.search(r"^ID=(.+)$", text, re.MULTILINE)
    html_match = re.search(r"HTML=\n(.+)", text, re.DOTALL)

    if not id_match or not html_match:
        return None, None

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

    for filename in os.listdir(dataset_path):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(dataset_path, filename)

        with open(file_path, "r", encoding="latin1", errors="ignore") as f:
            raw_text = f.read()

        doc_id, html = extract_id_and_html(raw_text)
        if doc_id is None:
            continue

        plain_text = clean_html(html)
        tokens = preprocess_text(plain_text)

        if tokens:
            doc_ids.append(doc_id)
            documents.append(tokens)

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
        num_topics=NUM_TOPICS,
        random_state=42,
        passes=10,
        alpha="auto",
        per_word_topics=False,
    )

    return lda, corpus, dictionary


def extract_document_topics(lda, corpus, doc_ids):
    """
    For each document:
    - find the dominant topic
    - extract top words for that topic
    """
    result = {}

    for doc_id, bow in zip(doc_ids, corpus):
        topic_probs = lda.get_document_topics(bow)
        dominant_topic = max(topic_probs, key=lambda x: x[1])[0]

        topic_words = lda.show_topic(dominant_topic, TOP_WORDS_PER_TOPIC)
        words_only = [word for word, _ in topic_words]

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
