# MainScript.py
import argparse
from neuroinfer.code.BayesianAnalysis import run_bayesian_analysis_router
from neuroinfer.code.UserInputs import get_user_inputs
from neuroinfer.code.DataLoading import load_data_and_create_dataframe
import os

"""
Main Script- executes Bayesian analysis given NPZ, metadata, and vocabulary files based on user-defined parameters. 

This script loads data from NPZ (TF-IDF features), metadata (TSV), and vocabulary (TXT) files, 
then performs a Bayesian analysis using user-provided parameters.

Inputs:
- NPZ file: Contains TF-IDF feature data
- Metadata (TSV) file: Describes the papers in the database.
- Vocabulary (TXT) file: Contains labels used in the analysis.
- User inputs: Cognitive regions of interest (cog_list), priors, and target coordinates (x, y, z) or an area.

Outputs:
- A pickle file (.pickle) with Bayesian analysis results, named based on either:
  - Target coordinates (if no area is provided), or
  - Analysis area (handled in a separate function).

Steps:
1. Parse file paths from command-line arguments or use default paths.
2. Load data into a DataFrame and extract feature names.
3. Collect user inputs (coordinates, area, priors) for analysis.
4. Run Bayesian analysis based on inputs.
5. Save results as a pickle file in the 'results' folder.
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
