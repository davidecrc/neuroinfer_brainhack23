import os
import numpy as np
import pandas as pd
import pickle

from neuroinfer.code.run_bayesian import run_bayesian_analysis_coordinates, run_bayesian_analysis_area

'''
The script orchestrates Bayesian analysis on brain data  ensuring efficiency and insightful interpretation through statistical summaries and graphical representations.
It offers two analysis pathways:
-`run_bayesian_analysis_area`: conducts analysis over defined brain regions, leveraging coordinates from an atlas.
-`run_bayesian_analysis_coordinates`: performs analysis at specified coordinates.
The pathway is choosen from the function run_bayesian_analysis_router based on the input (coordinates or area)

The script calculates various statistics including likelihood, prior, posterior, and Bayesian Factor (BF).
It plots BF distributions and computes basic statistical metrics. 

'''


def run_bayesian_analysis_router(cog_list, area, prior_list, x_target, y_target, z_target, radius, result_df):
    cm = input(
        "Specify the Bayesian confirmation measure you want to use in the analysis (type one of the following letter):\n"
        "'a' for Bayes' Factor;\n"
        "'b' for difference measure;\n"
        "'c' for ratio measure.\n")

    script_directory = os.path.dirname(os.path.abspath(__file__))
    global_path = os.path.dirname(script_directory)
    results_folder_path = os.path.join(global_path, "results")

    if not pd.isnull(area) and pd.isnull(x_target) and pd.isnull(y_target) and pd.isnull(z_target):
        # Call run_bayesian_analysis_area if area is not nan and coordinates are nan
        results=run_bayesian_analysis_area(cog_list, prior_list, area, radius, result_df, cm)

        # Save results_dict to a pickle file
        file_path = os.path.join(results_folder_path, f"results_area_{area}_{cog_list}.pkl")

        with open(file_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Results saved to: {file_path}")

    elif not np.isnan(x_target) and not np.isnan(y_target) and not np.isnan(z_target) and np.isnan(area):
        # Call run_bayesian_analysis_coordinates if coordinates are not nan and area is nan
        results=run_bayesian_analysis_coordinates(cog_list, prior_list, x_target, y_target, z_target, radius, result_df, cm)

        pickle_file_name = f'results_BHL_coordinates_x{x_target}_y{y_target}_z{z_target}.pickle'
        pickle_file_path = os.path.join(results_folder_path, pickle_file_name)
        with open(pickle_file_path, 'wb') as file:
            pickle.dump(results, file)
        print(f"Results saved to: {pickle_file_path}")

    elif np.isnan(area):
        # Print a message asking to check the input value if area is nan
        print("Please check the input value for 'area'.")
    else:
        # Handle the case where none of the conditions are met
        print("Invalid combination of input values. Please check.")

