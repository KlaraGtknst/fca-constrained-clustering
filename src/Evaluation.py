import matplotlib.pyplot as plt
import nltk
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
import re
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from owlready2 import get_ontology


nltk.download('punkt_tab')
nltk.download('reuters')
# The Reuters-21578 dataset is a multi-label text classification benchmark with 21,578 newswire articles,
# commonly using the ModApte split of 7,769 training and 3,019 testing documents across 90 categories
# appear in both sets.

nltk.download('stopwords')

stop_words = stopwords.words('english')

def load_dataset():
    documents = reuters.fileids()
    train_docs_id = [doc for doc in documents if doc.startswith('train')]
    # only for testing purposes, choose the first 100 documents
    sel_train_docs_id = train_docs_id
    print(f'sel_train_docs_id: {sel_train_docs_id[0]}')
    test_docs_id = [doc for doc in documents if doc.startswith('test')]
    # only for testing purposes, choose the first 100 documents
    sel_test_docs_id = test_docs_id
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
    categories = reuters.categories()
    print(f'categories: {categories}')
    # Each document has zero or more topic categories, accessed as lists for multi-label structure.
    doc_ids = reuters.fileids()  # Example from 'acq' category
    doc_categories = reuters.categories(doc_ids[0])  # e.g., ['acq', 'money-fx']
    print(f'doc_categories: {doc_categories}')

    # create a binary matrix (documents x categories)
    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform([reuters.categories(doc_id) for doc_id in doc_ids])
    # Shape: (7769, 90), 1 if document belongs to category

    # object_names = [str(doc.split('\n')[0]) for doc in train_docs]
    object_names = [f'{doc_id}: {reuters.raw(doc_id).split('\n')[0]}' for doc_id in doc_ids]
    # object_names = doc_ids
    attribute_names = [str(c) for c in reuters.categories()]

    write_cxt(
        "../resources/reuters.cxt",
        "Reuters training labels",
        train_labels,
        object_names,
        attribute_names,
    )

    # plot binary matrix
    plt.figure(figsize=(20, 40))  # Tall figure for 7769 rows

    sns.heatmap(train_labels,
                cmap='binary_r',  # Reverse binary: black=1, white=0
                cbar_kws={'label': 'Label Presence'},
                xticklabels=True, yticklabels=False)  # Hide y-ticks for readability
    plt.xlabel('Categories')
    plt.ylabel('Training Documents')
    plt.title('Reuters-21578 Full Training Binary Label Matrix')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


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

    # TODO: export burmeister format (1-2 wochen)
    # TODO: ontology (29.12.)
        # coverage DMOZ vgl. ggf. Wordnet, Wikidata -> wir nehmen nur eine 1 ontologie. Weil wir nur eine constraint teilmenge aus
        # aus einem externen Quelle ben?tigen
        #
        # TODO: hierachische constraints liste aus DMOZ

    # 09:15 - 10:15: 29.12.25

    # the categories in the Reuters-21578 dataset do not have a built-in hierarchical structure or formal
    # classification system within the standard dataset distribution. They consist of flat lists across five
    # groups—EXCHANGES, ORGS, PEOPLE, PLACES, and TOPICS—with TOPICS being the primary group used for text
    # classification research, typically reduced to 90 categories in the Mod-Apte split. While some research
    # papers impose an external three-level hierarchy on the topics for experiments (e.g., adding a root
    # category to leaf nodes), this is not part of the original dataset.

    # No single pre-built ontology perfectly matches Reuters-21578's flat categories (e.g., TOPICS like
    # 'acq', 'earn', 'grain'), but the DMOZ Open Directory Project ontology (now archived) is widely
    # suitable for imposing a news/business hierarchy on them. It features a multi-level structure
    # (e.g., Top > News > Business > Mergers & Acquisitions) that aligns with Reuters topics via
    # semantic mapping, used in hierarchical text classification benchmarks.

    # WordNet
    # News-Specific Taxonomies

    # Load DMOZ or WordNet ontology (example with WordNet via RDF)
    onto = get_ontology("http://purl.org/dc/terms/").load()  # Adapt to DMOZ RDF
    with onto:
        for cat in categories:
        # Map via similarity (e.g., using embeddings or rules)
            parent = onto.search(label=cat.upper())  # Pseudo-mapping
            print(f"{cat} -> {parent}")

def write_cxt(filename: str, context_name: str, train_labels, object_names: [str], attribute_names: [str]):
    n_objects, n_attributes = train_labels.shape
    """
    filename: output filename
    """

    with open(filename, "w", encoding="utf-8") as f:
        # Header
        f.write("B\n")
        f.write(f"{context_name}\n")
        f.write(f"{n_objects}\n")
        f.write(f"{n_attributes}\n")
        f.write("\n")

        # Object names
        for obj in object_names:
            f.write(f"{obj}\n")

        # Attribute names
        for attr in attribute_names:
            f.write(f"{attr}\n")

        # Incidence relation (X / .)
        for i in range(n_objects):
            row = train_labels[i]
            line = "".join("X" if val == 1 else "." for val in row)
            f.write(f"{line}\n")

if __name__ == '__main__':
    main()