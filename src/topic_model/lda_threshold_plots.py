import os
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless backend for saving figures
import matplotlib.pyplot as plt

from dataset2lda_topics import (
    load_banksearch_dataset,
    run_lda,
    NUM_TOPICS_CORPUS,
    MIN_TOPIC_PROB,
)


# -----------------------------
# Configuration
# -----------------------------
DEFAULT_THRESHOLDS = np.arange(0.0, 1.0, 0.05)


# -----------------------------
# Plot utilities
# -----------------------------
def compute_topic_probability_matrix(lda, corpus, doc_ids):
    """
    Build a dense probability matrix with rows as documents and columns as topics.
    Each cell contains the topic probability for that document.
    """
    doc_count = len(doc_ids)
    topic_count = NUM_TOPICS_CORPUS
    probs = np.zeros((doc_count, topic_count), dtype=float)

    for i, bow in enumerate(corpus):
        # minimum_probability=0 ensures all topics appear with a probability
        topic_probs = lda.get_document_topics(bow, minimum_probability=0)
        for topic_id, prob in topic_probs:
            probs[i, topic_id] = prob
    print("Computed topic probability matrix of shape:", probs.shape)
    return probs


def average_topics_per_threshold(prob_matrix, thresholds):
    """
    Compute average number of topics per document for each threshold.
    Ensures at least one topic is counted per document by using a floor of 1.
    """
    averages = []
    for threshold in thresholds:
        counts = (prob_matrix >= threshold).sum(axis=1)
        counts = np.maximum(counts, 1)
        averages.append(counts.mean())
    print(f"Average topics per document for thresholds {thresholds}: {averages}")
    return averages


def plot_average_topics(thresholds, averages, output_path):
    """
    Plot average number of topics per document as a function of threshold.
    """
    plt.figure(figsize=(7, 4))
    plt.plot(thresholds, averages, marker="o")
    plt.xlabel("MIN_TOPIC_PROB threshold")
    plt.ylabel("Average topics per document")
    plt.title("Average Topics per Document vs Threshold")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()
    print(f"Saved average topics per document plot to {output_path}")


def plot_incidence_density(prob_matrix, doc_ids, thresholds, output_dir):
    """
    For each threshold, create a binary incidence matrix (1 if prob >= threshold else 0)
    and plot its density as a heatmap. Documents are rows and topics are columns.
    """
    topic_ids = list(range(NUM_TOPICS_CORPUS))

    for threshold in thresholds:
        incidence = (prob_matrix >= threshold).astype(int)

        plt.figure(figsize=(10, 6))
        plt.imshow(incidence, aspect="auto", interpolation="nearest", cmap="Greys")
        plt.colorbar(label="Incidence (0/1)")
        plt.xlabel("Topic")
        plt.ylabel("Document")
        plt.title(f"Incidence Density (threshold={threshold})")

        # Keep topic labels; only show document labels if there are few docs
        plt.xticks(topic_ids, [f"T{t}" for t in topic_ids])
        if len(doc_ids) <= 50:
            plt.yticks(range(len(doc_ids)), doc_ids)
        else:
            plt.yticks([])

        output_path = Path(output_dir) / f"incidence_density_threshold_{threshold}.svg"
        plt.tight_layout()
        plt.savefig(output_path, format="svg")
        plt.close()
        print(f"Saved incidence density plot for threshold {threshold} to {output_path}")


# -----------------------------
# Main
# -----------------------------
def main(
    dataset_path=Path("resources/Dataset"),
    output_dir=Path("results/lda"),
    thresholds=DEFAULT_THRESHOLDS,
):
    """
    Train LDA on the BankSearch dataset and generate:
    1) Average topics per document vs threshold.
    2) Incidence density heatmaps for each threshold.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            "No input dataset exists; download it from http://lib.stat.cmu.edu/datasets/."
        )

    os.makedirs(output_dir, exist_ok=True)

    doc_ids, documents = load_banksearch_dataset(str(dataset_path))
    lda, corpus, _dictionary = run_lda(doc_ids, documents)

    prob_matrix = compute_topic_probability_matrix(lda, corpus, doc_ids)

    averages = average_topics_per_threshold(prob_matrix, thresholds)
    avg_plot_path = Path(output_dir) / "avg_topics_per_threshold.svg"
    plot_average_topics(thresholds, averages, avg_plot_path)

    plot_incidence_density(prob_matrix, doc_ids, thresholds, output_dir)


if __name__ == "__main__":
    main()
