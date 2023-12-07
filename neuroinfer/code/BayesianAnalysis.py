# Standard Library Imports
import os
import time
from argparse import ArgumentParser

# Third-party Library Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nilearn import image
from scipy import sparse
import json

# Local Module Imports
from neuroinfer.code.SummaryReport import save_summary_report

def run_bayesian_analysis_router(cog_list, area, prior_list, x_target, y_target, z_target, radius, result_df):
    cm = input(
        "Specify the Bayesian confirmation measure you want to use in the analysis (type one of the following letter):\n"
        "'a' for Bayes' Factor;\n"
        "'b' for difference measure;\n"
        "'c' for ratio measure.\n")
    if area != 99 and x_target == 99 and y_target == 99 and z_target == 99:
        # Call run_bayesian_analysis_area if area is 99 and x_target, y_target, z_target are 99
        run_bayesian_analysis_area(cog_list, prior_list, area, radius, result_df,cm)
    elif x_target != 99 and y_target != 99 and z_target != 99 and area == 99:
        # Call run_bayesian_analysis_coordinates if x_target, y_target, z_target are non-99 and area is 99
        run_bayesian_analysis_coordinates(cog_list, prior_list, x_target, y_target, z_target, radius, result_df,cm)
    elif area == 99:
        # Print a message asking to check the input value if area is 99
        print("Please check the input value for 'area'.")
    else:
        # Handle the case where none of the conditions are met
        print("Invalid combination of input values. Please check.")


def run_bayesian_analysis_coordinates(cog_list, prior_list, x_target, y_target, z_target, radius, feature_df,cm):
    """
    Perform Bayesian (confirmation) analysis using specified parameters.

    Parameters:
    - cog_list (list): List of cognitive labels.
    - prior_list (list): List of priors corresponding to cognitive labels.
    - x_target (float): X-coordinate target for analysis.
    - y_target (float): Y-coordinate target for analysis.
    - z_target (float): Z-coordinate target for analysis.
    - radius (float): Radius for spatial analysis.
    - feature_df (pd.DataFrame): DataFrame containing feature data for analysis.

    Returns:
    - Tuple: A tuple containing results of the Bayesian analysis.
    """
    frequency_threshold = 0.01
    t = time.time()
    cog_all, prior_all, ids_cog_nq_all, intersection_cog_nq_all, intersection_not_cog_nq_all = [], [], [], [], []
    lik_cog_nq_all, lik_not_cog_nq_all, lik_ratio_nq_all, post_cog_nq_all, df_nq_all, rm_nq_all = [], [], [], [], [], []


    feature_names = feature_df.columns

    # Get the directory of the current Python script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the parent directory as global path 
    global_path = os.path.dirname(script_directory)
    data_path = os.path.join(global_path, "data")  # Path to the saved_result folder
    

    for q, cog in enumerate(cog_list):
        prior = prior_list[q]

        if cog not in feature_names:
            print(f'Please check the correct spelling of: "{cog}"; it is not in the NeuroSynth list. '
                  'In the meanwhile, the script goes to the next *cog*, if any.')
        else:
            cog_all.append(cog)
            prior_all.append(prior)
            print(f'Processing "{cog}" (step {q + 1} out of {len(cog_list)})')

            ids_cog_nq = [feature_df[cog].index[i] for i in range(len(feature_df[cog])) if
                          feature_df[cog].iloc[i] > frequency_threshold]

            ids_cog_nq_all.append(ids_cog_nq)

            


            dt_papers_nq = pd.read_csv(data_path+'/data-neurosynth_version-7_coordinates.tsv', sep='\t')
            paper_id_nq = dt_papers_nq["id"]
            list_activations_nq = [paper_id_nq[i] for i in range(len(dt_papers_nq)) if
                                   x_target - radius < int(float(dt_papers_nq["x"].iloc[i])) < x_target + radius and
                                   y_target - radius < int(float(dt_papers_nq["y"].iloc[i])) < y_target + radius and
                                   z_target - radius < int(float(dt_papers_nq["z"].iloc[i])) < z_target + radius]

            total_activations_nq = len(list_activations_nq)
            intersection_cog_nq = len(set(ids_cog_nq) & set(list_activations_nq))
            intersection_not_cog_nq = total_activations_nq - intersection_cog_nq

            intersection_cog_nq_all.append(intersection_cog_nq)
            intersection_not_cog_nq_all.append(intersection_not_cog_nq)

            total_database_nq = len(set(dt_papers_nq["id"]))
            total_not_cog_nq = total_database_nq - len(set(ids_cog_nq))

            lik_cog_nq = round(intersection_cog_nq / len(ids_cog_nq), 3)
            lik_cog_nq_all.append(lik_cog_nq)

            lik_not_cog_nq = round((intersection_not_cog_nq / total_not_cog_nq) + 0.001, 3)
            lik_not_cog_nq_all.append(lik_not_cog_nq)

            lik_ratio_nq = round((lik_cog_nq / lik_not_cog_nq), 3) if cm in {"a", "c"} else None
            lik_ratio_nq_all.append(lik_ratio_nq) if lik_ratio_nq is not None else None

            post_cog_nq = round(((lik_cog_nq * prior) / (lik_cog_nq * prior + lik_not_cog_nq * (1 - prior))), 3)
            post_cog_nq_all.append(post_cog_nq)

            df_nq = post_cog_nq - prior if cm == "b" else None
            df_nq_all.append(df_nq) if df_nq is not None else None

            rm_nq = post_cog_nq / prior if cm == "c" else None
            rm_nq_all.append(rm_nq) if rm_nq is not None else None

    df_columns = ['cog', 'Likelihood']
    if cm == "a":
        df_columns.extend(['BF', 'Prior', 'Posterior'])
        data_all = list(zip(cog_all, lik_cog_nq_all, lik_ratio_nq_all, prior_all, post_cog_nq_all))
        sort_column = 'BF'
    elif cm == "b":
        df_columns.extend(['Difference', 'Prior', 'Posterior'])
        data_all = list(zip(cog_all, lik_cog_nq_all, df_nq_all, prior_all, post_cog_nq_all))
        sort_column = 'Difference'
    elif cm == "c":
        df_columns.extend(['Ratio', 'Prior', 'Posterior'])
        data_all = list(zip(cog_all, lik_cog_nq_all, rm_nq_all, prior_all, post_cog_nq_all))
        sort_column = 'Ratio'

    df_data_all = pd.DataFrame(data_all, columns=df_columns)
    df_data_all_sorted = df_data_all.sort_values(sort_column, ascending=False)
    print(df_data_all_sorted)

    elapsed = time.time() - t
    print(f'Time in min: {round(elapsed / 60, 2)}')

    #local_path = input("Insert a path to save a summary report (.txt) of the analysis (start and end path with '/'): ").encode('utf-8').decode('utf-8')
    #save_summary_report(local_path, df_data_all, cm, cog_all, prior_all, x_target, y_target, z_target, radius)

    #return local_path, df_data_all, cm, cog_all, prior_all, x_target, y_target, z_target, radius
    return df_data_all, cm, cog_all, prior_all, x_target, y_target, z_target, radius

def get_distance(coord1, coord2):
    #print(np.linalg.norm(np.array(coord1) - np.array(coord2)))
    return np.linalg.norm(np.array(coord1) - np.array(coord2))




def plot_distribution_statistics(result_all):
    """
    Plot distribution of the Bayesian Factor (BF) values for the area.

    Parameters:
    - result (list or array): List or array containing Bayesian Factor (BF) values.

    Returns:
    - None
    """
    # Convert the result to a NumPy array if it's not already
    #all_results = [result for result, _ in result_with_coordinates]
    df_data_all, cm, cog_all, prior_all, x_target, y_target, z_target, radius=result_all
    result_array = np.array(cm)
    print(len(result_array))
    print(result_array)

    # Create a histogram
    plt.hist(result_array, bins=20, edgecolor='black', alpha=0.7)
    plt.title('BF Distribution for the Area')
    plt.xlabel('BF Values')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('BF Distribution')
    plt.show()

    # Print basic statistics
    print(f"\nStatistics:")
    print(f"  - Mean: {np.mean(result_array)}")
    print(f"  - Median: {np.median(result_array)}")
    print(f"  - Standard Deviation: {np.std(result_array)}")
    print(f"  - Minimum: {np.min(result_array)}")
    print(f"  - Maximum: {np.max(result_array)}")


def get_atlas_coordinates_json(json_path):
    with open(json_path, 'r') as infile:
        coordinates_atlas = json.load(infile)
    #for key, coordinates in coordinates_atlas.items():
        #print(f"Key: {key}, Number of Coordinates: {len(coordinates)}")

    return coordinates_atlas

def run_bayesian_analysis_area(cog_list, prior_list, area, radius, feature_df,cm):
    """
    Perform Bayesian analysis in a specified area using coordinates from the atlas.

    Parameters:
    - cog_list (list): List of cognitive labels.
    - prior_list (list): List of priors corresponding to cognitive labels.
    - area (str): Name of the brain area.
    - radius (float): Original radius for spatial analysis.
    - feature_df (pd.DataFrame): DataFrame containing feature data for analysis.

    Returns:
    - results: list of BF and coordinates.
    """
    # Extract coordinates of the specified area from the atlas
     # Get the script's directory using Path

    # Set the global path
    # Get the directory of the current Python script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    global_path = os.path.dirname(script_directory)
    print('Global path: ',global_path)
    data_path = os.path.join(global_path, "data")  # Path to the saved_result folder
    print('data path: ',data_path)

    t_area = time.time()


    # Load coordinates from the JSON file
    json_path = data_path+ "/coord_label_all_harv_ox.json"
    coordinates_atlas = get_atlas_coordinates_json(json_path)

    # Extract coordinates specific to the provided area
    coordinates_area = coordinates_atlas.get(str(area), [])
    print('len(coordinates_area) for this area: ' , len(coordinates_area))


    # Rescale the radius to be between 2 and 4
    rescaled_radius = max(2, min(radius, 4))
    
    # Initialize an empty list to store results and coordinates
    result_all = []
    coord=[]
    
    # Iterate through the coordinates_area
    for i in range(len(coordinates_area)-1): #-1 in order to have the last coordinates_area[i+1]) #TODO improve
        print(f'Calculating cm for the {i+1} ROI out of {len(coordinates_area)}, {((i+1)/len(coordinates_area))*100}% ')
        x_target, y_target, z_target = coordinates_area[i]
        coord.append([x_target, y_target, z_target])
        #print(coord)
        #print (i)
        if i==0:
            result = run_bayesian_analysis_coordinates(cog_list, prior_list, x_target, y_target, z_target, rescaled_radius, feature_df,cm)
            # Append a tuple containing the BF value and the coordinates to result_with_coordinates
            result_all.append(result)
        #df_data_all, cm, cog_all, prior_all, x_target, y_target, z_target, radius

        # Check if next_coord is distant > rescaled_radius * 0.98 from all previous coordinates
        
        #if all(get_distance((x_target, y_target, z_target), coordinate) > rescaled_radius * 0.98 for coordinate in coord):
        if get_distance((x_target, y_target, z_target), coordinates_area[i+1]) > rescaled_radius: #TODO upload checking all the distance from the ROIS
            print(get_distance((x_target, y_target, z_target), coordinates_area[i+1]))    
            # Call run_bayesian_analysis_coordinates with the current coordinates
            result = run_bayesian_analysis_coordinates(cog_list, prior_list, x_target, y_target, z_target, rescaled_radius, feature_df,cm)
            # Append a tuple containing the BF value and the coordinates to result_with_coordinates
            result_all.append(result)

    elapsed_area = time.time() - t_area
    print(f'Time in min: {round(elapsed_area / 60, 2)}')

    # Plot distribution and statistics of the Bayesian Factor (BF) for the area
    #plot_distribution_statistics(result_with_coordinates)

    return result_all