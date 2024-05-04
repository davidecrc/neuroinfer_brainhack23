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
from neuroinfer import PKG_FOLDER, DATA_FOLDER
import pickle

'''
Script to run_bayesian_analysis_coordinates

The script orchestrates Bayesian analysis on brain data  ensuring efficiency and insightful interpretation through statistical summaries and graphical representations.
It offers two analysis pathways:
-`run_bayesian_analysis_area`: conducts analysis over defined brain regions, leveraging coordinates from an atlas.
-`run_bayesian_analysis_coordinates`: performs analysis at specified coordinates.

The script calculates various statistics including likelihood, prior, posterior, and Bayesian Factor (BF).
It uses the functions get_atlas_coordinates_json e get_distance to obtain the coordinates in an atlas and the distance between pair of coordinates.
'''

def calculate_z(posterior, prior):
    """
    Calculate the Z-measure based on posterior and prior probabilities.

    Args:
        posterior (float): Posterior probability.
        prior (float): Prior probability.

    Returns:
        float: Z-measure indicating the significance of the difference between posterior and prior probabilities.
    """
    # Ensure that the denominator is not zero
    if prior == 0:
        return float('inf') if posterior > 0 else 0

    # Calculate Z-measure based on whether posterior is greater than or equal to prior
    if posterior >= prior:
        z = (posterior - prior) / (1 - prior)
    else:
        z = (posterior - prior) / prior

    return z


def run_bayesian_analysis_coordinates(cog_list, prior_list, x_target, y_target, z_target, radius, feature_df, cm, xyz_coords, dt_papers_nq_id_np, nb_unique_paper):
    frequency_threshold = 0.05
    t = time.time()
    cog_all, prior_all, ids_cog_nq_all, intersection_cog_nq_all, intersection_not_cog_nq_all = [], [], [], [], []
    lik_cog_nq_all, lik_not_cog_nq_all, lik_ratio_nq_all, post_cog_nq_all, df_nq_all, rm_nq_all, z_measure_all = [], [], [], [], [], [], []

    for q, cog in enumerate(cog_list):
        prior = prior_list[q]

        if cog not in feature_df.columns:  # features names
            print(f'Please check the correct spelling of: "{cog}"; it is not in the NeuroSynth list. ')
            return None

        else:
            cog_all.append(cog)
            prior_all.append(prior)
            print(f'Processing "{cog}" (step {q + 1} out of {len(cog_list)})')

            ids_cog_nq = feature_df.index[feature_df[cog] > frequency_threshold].tolist()
            ids_cog_nq_all.append(ids_cog_nq)            
            center = np.array([x_target, y_target, z_target])
            
            distances = np.linalg.norm(xyz_coords - center, axis=1) #check operation applicability
            # Use the distances to filter the DataFrame
            list_activations_nq = dt_papers_nq_id_np[distances <= radius]

            total_activations_nq = len(list_activations_nq)
            intersection_cog_nq = len(set(ids_cog_nq) & set(list_activations_nq))
            intersection_not_cog_nq = total_activations_nq - intersection_cog_nq

            intersection_cog_nq_all.append(intersection_cog_nq)
            intersection_not_cog_nq_all.append(intersection_not_cog_nq)

            total_not_cog_nq = nb_unique_paper - len(ids_cog_nq)

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

            z_measure_nq = calculate_z(post_cog_nq, prior) if cm == "d" else None
            z_measure_all.append(z_measure_nq) if z_measure_nq is not None else None

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
    elif cm == "d":
        df_columns.extend(['Z-measure', 'Prior', 'Posterior'])
        data_all = list(zip(cog_all, lik_cog_nq_all, z_measure_all, prior_all, post_cog_nq_all))
        sort_column = 'Z-measure'

    df_data_all = pd.DataFrame(data_all, columns=df_columns)
    print(df_data_all)

    elapsed = time.time() - t
    print('time elapsed:', elapsed)

    results = {
        "df_data_all": df_data_all,
        "cm": cm,
        "cog_list": cog_all,
        "prior_list": prior_all,
        "x_target": x_target,
        "y_target": y_target,
        "z_target": z_target,
        "radius": radius,
    }
    return results


def get_distance(coord1, coord2):
    return np.linalg.norm(np.array(coord1) - np.array(coord2))


def get_atlas_coordinates_json(json_path):
    with open(json_path, 'r') as infile:
        coordinates_atlas = json.load(infile)

    return coordinates_atlas


def run_bayesian_analysis_area(cog_list, prior_list, mask, radius, feature_df,cm,dt_papers_nq_id_list, nb_unique_paper, xyz_coords):
    """
    Perform Bayesian analysis in a specified area using coordinates from the atlas.

    Parameters:
    - cog_list (list): List of cognitive labels.
    - prior_list (list): List of priors corresponding to cognitive labels.
    - mask (boolean): volume to consider
    - radius (float): Original radius for spatial analysis.
    - feature_df (pd.DataFrame): DataFrame containing feature data for analysis.

    Returns:
    - results: list of BF and coordinates.
    """

    t_area = time.time()

    coordinates_area = np.where(mask == 1)
    coordinates_area = [[coordinates_area[0][a], coordinates_area[1][a], coordinates_area[2][a]] for a in range(len(coordinates_area[0]))] #list of coordinated in the mask with value 1
    
    # Initialize an empty list to store results and coordinates
    result_all = []
    coord=[]
    
    # Iterate through the coordinates_area
    for i in range(len(coordinates_area)-1): #-1 in order to have the last coordinates_area[i+1]) #TODO improve
        
        x_target, y_target, z_target = coordinates_area[i]
        coord.append([x_target, y_target, z_target])
        if i==0:
            results = run_bayesian_analysis_coordinates(cog_list, prior_list, x_target, y_target, z_target,
                                                        radius, feature_df, cm,xyz_coords,dt_papers_nq_id_list, nb_unique_paper)
            #print(results_dict[i])           
            # Append a tuple containing the BF value and the coordinates to result_with_coordinates
            result_all.append(results)

        # Check if next_coord is distant > rescaled_radius * 0.98 from all previous coordinates

        # if all(get_distance((x_target, y_target, z_target), coordinate) > rescaled_radius * 0.98 for coordinate in coord):
        if get_distance((x_target, y_target, z_target), coordinates_area[i + 1]) > radius:  # TODO upload checking all the distance from the ROIS
            print(f'Calculating cm for the {i + 1} ROI out of {len(coordinates_area)}, {((i + 1) / len(coordinates_area)) * 100:.2f}%')

            if i % 10 == 0:
                #create the folder if not existing
                os.makedirs(PKG_FOLDER / 'neuroinfer' / '.tmp', exist_ok=True)                
                with open(PKG_FOLDER / 'neuroinfer' / '.tmp' / 'processing_progress', 'w') as f_tmp:
                    f_tmp.write(str((i + 1) / len(coordinates_area)))
            print(get_distance((x_target, y_target, z_target), coordinates_area[i + 1]))
            # Call run_bayesian_analysis_coordinates with the current coordinates
            results = run_bayesian_analysis_coordinates(cog_list, prior_list, x_target, y_target, z_target, radius, feature_df, cm, xyz_coords, dt_papers_nq_id_list, nb_unique_paper)
            # Append a tuple containing the BF value and the coordinates to result_with_coordinates
            result_all.append(results)

    elapsed_area = time.time() - t_area
    print(f'Time in min: {round(elapsed_area / 60, 2)}')

    print(result_all)
    

    return result_all

#crea un volume con stesso volume del template
#man mano che si scorrono le coordinate si mette a 0 l'intorno di quella coordinat con in input il raggio (occhio a considerare padding). Quando si considera la coordinata successiva, ripete il calcolo del BF solo se e' diversa da 0 
#cosi' da non ripetere il calcolo del BF per coordinate vicine
