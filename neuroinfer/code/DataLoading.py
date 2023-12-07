import numpy as np
import pandas as pd
from scipy import sparse


def load_data_and_create_dataframe(npz_file, metadata_file, vocab_file):
    """
    Load sparse data, metadata, and vocabulary to create a DataFrame.

    Parameters:
    - npz_file (str): The file path to the NumPy compressed sparse row matrix (.npz) containing feature data (tfidf data).
    - metadata_file (str): The file path to the metadata file, i.e., papers, (in tabular format) associated with the feature data.
    - vocab_file (str): The file path to the vocabulary file containing cognitive labels.

    Returns:
    - feature_df (pd.DataFrame): The DataFrame created by combining sparse feature data, metadata, and vocabulary. rows are the papers and columns the TFIDF for each entry
    - feature_names (list): list with the vocabulary words loaded from the cognitive labels
    """
    # Load sparse data
   

    feature_data_sparse = sparse.load_npz(npz_file)
    feature_data = feature_data_sparse.todense()

    # Load metadata
    metadata_df = pd.read_table(metadata_file)
    ids = metadata_df["id"].tolist()

    # Load vocabulary
    feature_names = np.genfromtxt(vocab_file, dtype=str, delimiter="\t").tolist()

    # Create DataFrame
    feature_df = pd.DataFrame(index=ids, columns=feature_names, data=feature_data)

    return feature_df, feature_names
