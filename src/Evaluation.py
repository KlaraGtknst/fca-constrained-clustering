import matplotlib.pyplot as plt
import nltk
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import numpy as np

nltk.download('punkt_tab')
nltk.download('reuters')
nltk.download('stopwords')

stop_words = stopwords.words('english')

def load_dataset():
    documents = reuters.fileids()
    train_docs_id = [doc for doc in documents if doc.startswith('train')]
    sel_train_docs_id = train_docs_id[:100]
    print(f'sel_train_docs_id: {sel_train_docs_id[0]}')
    test_docs_id = [doc for doc in documents if doc.startswith('test')]
    sel_test_docs_id = test_docs_id[:100]
    print(f'sel_test_docs_id: {sel_test_docs_id[0]}')

    train_docs = [reuters.raw(doc_id) for doc_id in sel_train_docs_id]
    print(f'train_docs: {train_docs[0]}')
    test_docs = [reuters.raw(doc_id) for doc_id in sel_test_docs_id]

    train_labels = [reuters.categories(doc_id) for doc_id in sel_train_docs_id]
    print(f'train_labels: {train_labels[0]}')
    test_labels = [reuters.categories(doc_id) for doc_id in sel_test_docs_id]
    return train_docs, test_docs, train_labels, test_labels

def tokenize(text):
    min_length = 3
    words = [word.lower() for word in word_tokenize(text) if word.lower() not in stop_words]
    tokens = [PorterStemmer().stem(word) for word in words]
    p = re.compile('[a-zA-Z]+')
    return [token for token in tokens if p.match(token) and len(token) >= min_length]

def main():
    train_docs, test_docs, train_labels, test_labels = load_dataset()
    vectorizer = TfidfVectorizer(stop_words=stop_words, tokenizer=tokenize)
    vectorised_train = vectorizer.fit_transform(train_docs)
    vectorised_test = vectorizer.transform(test_docs)
    # Compute similarity matrix (train vs test)
    similarity_matrix = cosine_similarity(vectorised_train, vectorised_test)

    # Example: Similarity of first train doc to all test docs
    first_train_similarities = similarity_matrix[0]
    most_similar_test_idx = first_train_similarities.argmax()
    print(f"Most similar test doc index: {most_similar_test_idx}, score: {first_train_similarities[most_similar_test_idx]:.4f}")

    distance_matrix = 1 - similarity_matrix  # Convert to distance [web:21]
    np.fill_diagonal(distance_matrix, 0)   # Ensure zero diagonal

    # Method 1: Sklearn (flat clusters)
    n_clusters = 10
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='average'
    )
    test_clusters = clusterer.fit_predict(distance_matrix)

    print(f"Cluster assignments: {test_clusters[:10]}")  # First 10 docs [web:21]

    # Visualize
    plt.scatter(distance_matrix[:, 0], distance_matrix[:, 1], c=test_clusters)
    plt.show()

if __name__ == '__main__':
    main()