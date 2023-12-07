# MainScript.py
import argparse
from BayesianAnalysis import *
from UserInputs import get_user_inputs
from DataLoading import load_data_and_create_dataframe
import pickle

if __name__ == "__main__":
    """
    $ python MainScript.py /path/to/your/npz_tfidf_data/file.npz /path/to/your/metadata_paper/file.tsv /path/to/your/vocabulary/file.txt

    """

    # Get the directory of the current Python script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the parent directory as global path 
    global_path = os.path.dirname(script_directory)
    data_path = os.path.join(global_path, "data")  # Path to the saved_result folder

    parser = argparse.ArgumentParser(description="Load data and create a DataFrame.")
    parser.add_argument("npz_file",
                        nargs='?',
                        #default="../data/features7.npz",
                        default=data_path+"/features7.npz",
                        help="Path to the NPZ file containing features data (tfidf data).")
    parser.add_argument("metadata_file",
                        nargs='?',
                        #default="../data/metadata7.tsv",
                        default=data_path+"/metadata7.tsv",
                        help="Path to the metadata file (papers in the database).")
    parser.add_argument("vocab_file",
                        nargs='?',
                        #default="../data/vocabulary7.txt",
                        default=data_path+"/vocabulary7.txt",
                        help="Path to the vocabulary file (cognitive labels).")

    args = parser.parse_args()

    print(args.npz_file)

    # Call the function with command line arguments
    result_df, feature_names = load_data_and_create_dataframe(args.npz_file, args.metadata_file, args.vocab_file)
    
    cog_list, prior_list, x_target, y_target, z_target, radius,area = get_user_inputs(feature_names)
    #cog_list, prior_list, area, radius = get_user_inputs(feature_names)

    try:
        print("Selected area: " + str(area) +  "Radius: " + str(radius) + " mm.")
    except:
        print("Selected coordinates: " + str(x_target) + "; " + str(y_target) + "; " + str(z_target) + ". Radius: " + str(
        radius) + " mm.")

    results_all_rois=run_bayesian_analysis_router(cog_list, area, prior_list, x_target, y_target, z_target, radius, result_df)
    # Save the results using pickle
    # Construct the path to the "results" folder
    script_directory = os.path.dirname(os.path.abspath(__file__))
    global_path = os.path.dirname(script_directory)
    results_folder_path = os.path.join(global_path, "results")

    # Ensure the "results" folder exists; create it if not
    #os.makedirs(results_folder_path, exist_ok=True)

    # Save the pickle file in the "results" folder
    pickle_file_path = os.path.join(results_folder_path, f'results_BHL_area{area}.pickle')

    with open(pickle_file_path, 'wb') as file:
        pickle.dump(results_all_rois, file)

    print(f"Results saved to: {pickle_file_path}")
