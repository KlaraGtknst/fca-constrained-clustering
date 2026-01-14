from pathlib import Path
import json
import numpy as np
import pandas as pd
from fcapy import context


class DocumentRepresenter:
    def __init__(self, json_path: str = "resources/banksearch2topics/banksearch_lda_topics.json") -> None:
        self.path2topics = Path(json_path)
        assert self.path2topics.exists(), f"Path does not exist: {self.path2topics}"

        # Load the JSON mapping
        with open(self.path2topics, "r") as f:
            self.topic_dict = json.load(f)

        # Extract unique topics
        self.all_topics = sorted(
            {topic_id for doc_topics in self.topic_dict.values() for topic_id in doc_topics.keys()},
            key=int
        )

        # Mapping: topic -> topic words
        self.topic_words = {
            topic_id: topic_words
            for doc_topics in self.topic_dict.values()
            for topic_id, topic_words in doc_topics.items()
        }

        # Build document vectors
        self.doc_vectors = self.convert_documents_to_vectors()

    def convert_documents_to_vectors(self) -> pd.DataFrame:
        """
        Converts each document into a boolean vector indicating which topics it has.
        Returns a DataFrame: rows = documents, columns = topics, values = 0/1
        """
        data = {}
        for doc_id, topics in self.topic_dict.items():
            vector = [int(topic_id in topics) for topic_id in self.all_topics]
            data[doc_id] = vector

        df = pd.DataFrame.from_dict(data, orient="index", columns=self.all_topics)
        return df

    def to_fca_context(self) -> context.FormalContext:
        """
        Converts the document-topic boolean vectors to an FCA context.
        Returns an fcapy.Context object.
        """
        # fcapy expects a dataframe with rows=objects, cols=attributes, values=True/False
        bool_df = self.doc_vectors.astype(bool)
        c = context.FormalContext.from_pandas(bool_df)
        return c

    def save_fca_context(self, path: str):
        """
        Converts to FCA context and saves it as a CSV for later use.
        """
        c = self.to_fca_context()
        df = context.FormalContext.to_pandas( c)
        print(f"FCA context saved to {path}")

    def display_topic_words(self, topic_id: str):
        """
        Display the words associated with a given topic_id
        """
        words = self.topic_words.get(topic_id)
        if words:
            print(f"Topic {topic_id}: {', '.join(words)}")
        else:
            print(f"No words found for topic {topic_id}")


# Example usage
if __name__ == "__main__":
    dr = DocumentRepresenter()
    print(dr.doc_vectors.head())        # document-topic boolean vectors
    dr.display_topic_words("65")        # show topic words
    fca_ctx = dr.to_fca_context()       # convert to FCA structure
    dr.save_fca_context("fca_context.csv")
