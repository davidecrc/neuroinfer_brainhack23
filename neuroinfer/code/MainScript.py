# MainScript.py
import argparse
import pickle
import pandas as pd
import os

from neuroinfer.code.BayesianAnalysis import run_bayesian_analysis_router
from neuroinfer.code.UserInputs import get_user_inputs
from neuroinfer.code.DataLoading import load_data_and_create_dataframe
from neuroinfer import RESULTS_FOLDER, DATA_FOLDER


"""
Main Script- executes Bayesian analysis given NPZ, metadata, and vocabulary files based on user-defined parameters. 
It parses command-line arguments for paths to NPZ, metadata, and vocabulary files.
These files are loaded to create a DataFrame. 
Users input parameters for Bayesian analysis, defining regions of interest.
It ensures a "results" folder exists and saves analysis results using pickle, named according to the selected area. 


INPUTS
1) Command-line arguments:
Path to an NPZ file containing TF-IDF features data.
Path to a metadata file containing information about papers in a database.
Path to a file containing vocabulary (cognitive labels).

2)User input for Bayesian analysis parameters:
cog_list: List of cognitive labels.
prior_list: List of prior probabilities.
x_target, y_target, z_target: Coordinates of the target area (for non-spherical regions).
radius: Radius of the target area (for spherical regions).
area: Selected area for analysis.

OUTPUTS
Results DataFrame (result_df) from the loaded data and created DataFrame.
Output of the Bayesian analysis (results_all_rois). It has been saved inside run_bayesian_analysis_router.

"""

if __name__ == "__main__":
    """
    $ python MainScript.py /path/to/your/npz_tfidf_data/file.npz /path/to/your/metadata_paper/file.tsv /path/to/your/vocabulary/file.txt
    """

    parser = argparse.ArgumentParser(description="Load data and create a DataFrame.")
    parser.add_argument(
        "npz_file",
        nargs="?",
        # default="../data/features7.npz",
        default=DATA_FOLDER / "features7.npz",
        help="Path to the NPZ file containing features data (tfidf data).",
    )
    parser.add_argument(
        "metadata_file",
        nargs="?",
        # default="../data/metadata7.tsv",
        default=DATA_FOLDER / "metadata7.tsv",
        help="Path to the metadata file (papers in the database).",
    )
    parser.add_argument(
        "vocab_file",
        nargs="?",
        # default="../data/vocabulary7.txt",
        default=DATA_FOLDER / "vocabulary7.txt",
        help="Path to the vocabulary file (comemorgnitive labels).",
    )

    args = parser.parse_args()

    print(args.npz_file)

    # Call the function with command line arguments
    result_df, feature_names = load_data_and_create_dataframe(
        args.npz_file, args.metadata_file, args.vocab_file
    )

    cog_list, prior_list, x_target, y_target, z_target, radius, area = get_user_inputs(
        feature_names
    )
    # cog_list, prior_list, area, radius = get_user_inputs(feature_names)

    try:
        print("Selected area: " + str(area) + "Radius: " + str(radius) + " mm.")
    except:
        print(
            "Selected coordinates: "
            + str(x_target)
            + "; "
            + str(y_target)
            + "; "
            + str(z_target)
            + ". Radius: "
            + str(radius)
            + " mm."
        )

    results_all_rois = run_bayesian_analysis_router(
        cog_list, area, prior_list, x_target, y_target, z_target, radius, result_df
    )

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    # Save the pickle file in the "results" folder if coordinates (the area one is saved in run_bayesian_analysis_area)
    if pd.isnull(area):
        # Use the coordinates to name the pickle file
        pickle_file_name = (
            f"results_BHL_coordinates_x{x_target}_y{y_target}_z{z_target}.pickle"
        )
        pickle_file_path = RESULTS_FOLDER / pickle_file_name
        with open(pickle_file_path, "wb") as file:
            pickle.dump(results_all_rois, file)
        print(f"Results saved to: {pickle_file_path}")
