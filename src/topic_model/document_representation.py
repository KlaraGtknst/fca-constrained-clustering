from pathlib import Path
import json
import re
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from fcapy import context
from fcapy.lattice import ConceptLattice
from fcapy.visualizer import LineVizNx
import seaborn as sns


class DocumentRepresentaterBase:
    def __init__(self) -> None:
        self.doc_vectors = pd.DataFrame()

    def convert_documents_to_vectors(self) -> pd.DataFrame:
        """
        Converts each document into a boolean vector indicating which topics it has.
        Returns a DataFrame: rows = documents, columns = topics, values = 0/1
        """
        raise NotImplementedError()

    def to_fca_context(self) -> context.FormalContext:
        """
        Converts the document-topic boolean vectors to an FCA context.
        Returns an fcapy.Context object.
        """
        if self.doc_vectors.empty:
            self.doc_vectors = self.convert_documents_to_vectors()
        # fcapy expects a dataframe with rows=objects, cols=attributes, values=True/False
        bool_df = self.doc_vectors.astype(bool)
        c = context.FormalContext.from_pandas(bool_df)
        return c

    def save_fca_context(self, path: Path):
        """
        Converts to FCA context and saves it as a CSV for later use.
        """
        if self.doc_vectors.empty:
            self.doc_vectors = self.convert_documents_to_vectors()
        c = self.to_fca_context()
        df = context.FormalContext.to_pandas(c)
        df.to_json(path, orient="split", compression="infer", index=True)
        print(f"FCA context saved to {path}")


class DocumentTopicModelRepresenter(DocumentRepresentaterBase):
    def __init__(
        self, json_path: str = "resources/banksearch2topics/banksearch_lda_topics.json"
    ) -> None:
        super().__init__()
        self.path2topics = Path(json_path)
        assert self.path2topics.exists(), f"Path does not exist: {self.path2topics}"

        # Load the JSON mapping
        with open(self.path2topics, "r") as f:
            self.topic_dict = json.load(f)

        # Extract unique topics
        self.all_topics = sorted(
            {
                topic_id
                for doc_topics in self.topic_dict.values()
                for topic_id in doc_topics.keys()
            },
            key=int,
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

    def display_topic_words(self, topic_id: str):
        """
        Display the words associated with a given topic_id
        """
        words = self.topic_words.get(topic_id)
        if words:
            print(f"Topic {topic_id}: {', '.join(words)}")
        else:
            print(f"No words found for topic {topic_id}")

    def display_ctx(self, K, dataset_name: str = "Topic Model"):
        fig, axs = plt.subplots(1, 2, figsize=(15, 10))

        ax = axs[0]
        sns.heatmap(
            K.to_pandas(),
            cmap="Greens",
            alpha=0.5,
            ax=ax,
            cbar=False,
            annot=K.to_pandas().replace(True, "✓").replace(False, ""),
            fmt="",
            annot_kws={"fontsize": 14},
        )
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_xticklabels(
            [
                (
                    lbl.get_text().replace(" ", "\n")
                    if "cotyledon" not in lbl.get_text()
                    else lbl.get_text().replace("cotyledon", "-\ncoty-\nledon")
                )
                for lbl in ax.get_xticklabels()
            ],
            rotation=0,
        )
        ax.set_title("Tabular view", size=24, ha="left", x=0.00)

        ax = axs[1]

        viz = LineVizNx(node_label_font_size=14)

        lattice = ConceptLattice.from_context(K)
        viz.draw_concept_lattice(
            lattice,
            ax=ax,
            # flg_drop_bottom_concept=True,
        )

        leg = plt.legend(
            title="Color coding",
            title_fontproperties={
                "size": "14",
            },
            fontsize=12,
            loc=(0.2, 0),
        )  #'lower center')

        ax.set_title("Line and color diagram", size=24, ha="left", x=0.0)
        ax.set_xlim(-0.8, 0.8)

        plt.suptitle(f'"{dataset_name}" data representations', size=28, ha="left", x=0.07)

        plt.subplots_adjust(wspace=5, top=0.25)
        plt.tight_layout()
        savepath = Path("results/topic_model")
        savepath.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath / f"{dataset_name.lower().replace(' ', '_')}_representation_comparison.png")
        plt.show()


class DocumentGroundTruthRepresenter(DocumentRepresentaterBase):
    def __init__(self, data_path: str = "resources/Dataset") -> None:
        super().__init__()
        self.path2groundtruth = Path(data_path)
        assert (
            self.path2groundtruth.exists()
        ), f"Path does not exist: {self.path2groundtruth}"
        self.save_path = Path("resources/banksearch")

    def convert_documents_to_vectors(self) -> pd.DataFrame:
        """
        Converts each document into a boolean vector indicating which ground truth labels it has.
        Returns a DataFrame: rows = documents, columns = labels, values = 0/1
        """
        # iterate over all files in the ground truth directory
        self.ground_truth_dict = {}
        # extract labels using RegEx: DATASET=
        dataset_re = re.compile(r"^DATASET=(.+)$", re.MULTILINE)
        self.ground_truth_dict = {}
        all_labels = set()
        for doc_path in self.path2groundtruth.glob("*.txt"):
            with doc_path.open("r", errors="ignore") as f:
                head = f.read(4096)
            match = dataset_re.search(head)
            if not match:
                continue
            if dataset_re.search(head, match.end()):
                raise ValueError(f"Multiple DATASET= labels in {doc_path}")
            label = match.group(1).strip()
            if not label:
                continue
            doc_id = doc_path.stem
            self.ground_truth_dict[doc_id] = label
            all_labels.add(label)
        self.all_labels = sorted(all_labels)
        data = {}
        for doc_id, doc_label in self.ground_truth_dict.items():
            vector = [int(label == doc_label) for label in self.all_labels]
            data[doc_id] = vector

        df = pd.DataFrame.from_dict(data, orient="index", columns=self.all_labels)
        return df


# Example usage
if __name__ == "__main__":
    dr = DocumentTopicModelRepresenter()
    print(dr.doc_vectors.head())  # document-topic boolean vectors
    dr.display_topic_words("65")  # show topic words
    fca_ctx = dr.to_fca_context()  # convert to FCA structure
    dr.display_ctx(fca_ctx)
    savepath = Path("resources/banksearch")
    savepath.mkdir(True, exist_ok=True)
    dr.save_fca_context(savepath / "fca_topic_model_context.json")

    # dgt = DocumentGroundTruthRepresenter()
    # savepath = Path("resources/banksearch")
    # savepath.mkdir(True, exist_ok=True)
    # dgt.save_fca_context(savepath / "fca_gt_context.json")
    # print(dgt.doc_vectors.head())  # document-ground truth boolean vectors
