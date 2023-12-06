# MainScript.py
import argparse
from neuroinfer.code.BayesianAnalysis import run_bayesian_analysis
from neuroinfer.code.UserInputs import get_user_inputs
from neuroinfer.code.DataLoading import load_data_and_create_dataframe

if __name__ == "__main__":
    """
    $ python MainScript.py /path/to/your/npz_tfidf_data/file.npz /path/to/your/metadata_paper/file.tsv /path/to/your/vocabulary/file.txt

    """
    parser = argparse.ArgumentParser(description="Load data and create a DataFrame.")
    parser.add_argument("npz_file",
                        nargs='?',
                        default="../data/features7.npz",
                        help="Path to the NPZ file containing features data (tfidf data).")
    parser.add_argument("metadata_file",
                        nargs='?',
                        default="../data/metadata7.tsv",
                        help="Path to the metadata file (papers in the database).")
    parser.add_argument("vocab_file",
                        nargs='?',
                        default="../data/vocabulary7.txt",
                        help="Path to the vocabulary file (cognitive labels).")

    args = parser.parse_args()

    # Call the function with command line arguments
    result_df, feature_names = load_data_and_create_dataframe(args.npz_file, args.metadata_file, args.vocab_file)

    cog_list, prior_list, x_target, y_target, z_target, radius = get_user_inputs(feature_names)

    print("Selected coordinates: " + str(x_target) + "; " + str(y_target) + "; " + str(z_target) + ". Radius: " + str(
        radius) + " mm.")

    run_bayesian_analysis(cog_list, prior_list, x_target, y_target, z_target, radius, result_df)
