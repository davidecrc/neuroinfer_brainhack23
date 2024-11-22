import os
import time
import numpy as np
import pandas as pd
import json
from neuroinfer import PKG_FOLDER
from neuroinfer.code.python.utils import send_progress

"""
Script to run_bayesian_analysis_coordinates

The script orchestrates Bayesian analysis on brain data  ensuring efficiency and insightful interpretation through
statistical summaries and graphical representations.
It offers two analysis pathways:
-`run_bayesian_analysis_area`: conducts analysis over defined brain regions, leveraging coordinates from an atlas.
-`run_bayesian_analysis_coordinates`: performs analysis at specified coordinates.

The script calculates various statistics including likelihood, prior, posterior, and Bayesian Factor (BF).
It uses the functions get_atlas_coordinates_json e get_distance to obtain the coordinates in an atlas and the distance between pair of coordinates.

"""


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
        return float("inf") if posterior > 0 else 0

    # Calculate Z-measure based on whether posterior is greater than or equal to prior
    if posterior >= prior:
        z = (posterior - prior) / (1 - prior)
    else:
        z = (posterior - prior) / prior

    return z


def run_bayesian_analysis_coordinates(
    cog_list,
    prior_list,
    x_target,
    y_target,
    z_target,
    radius,
    feature_df,
    cm,
    xyz_coords,
    dt_papers_nq_id_np,
    nb_unique_paper,
):
    t = time.time()
    frequency_threshold = 0.05
    (
        cog_all,
        prior_all,
        ids_cog_nq_all,
        intersection_cog_nq_all,
        intersection_not_cog_nq_all,
    ) = ([], [], [], [], [])
    (
        lik_cog_nq_all,
        lik_not_cog_nq_all,
        lik_ratio_nq_all,
        post_cog_nq_all,
        df_nq_all,
        rm_nq_all,
        z_measure_all,
    ) = ([], [], [], [], [], [], [])

    for q, cog in enumerate(cog_list):
        prior = prior_list[q]

        if cog not in feature_df.columns:  # features names
            print(
                f'Please check the correct spelling of: "{cog}"; it is not in the NeuroSynth list. '
            )
            return None

        else:
            cog_all.append(cog)
            prior_all.append(prior)
            # print(f'Processing "{cog}" (step {q + 1} out of {len(cog_list)})')

            ids_cog_nq = feature_df.index[
                feature_df[cog] > frequency_threshold
            ].tolist()
            ids_cog_nq_all.append(ids_cog_nq)
            center = np.array([x_target, y_target, z_target], dtype=np.float32)

            distances = np.linalg.norm(xyz_coords - center, axis=1)

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

            lik_not_cog_nq = round(
                (intersection_not_cog_nq / total_not_cog_nq) + 0.001, 3
            )
            lik_not_cog_nq_all.append(lik_not_cog_nq)

            lik_ratio_nq = (
                round((lik_cog_nq / lik_not_cog_nq), 3) if cm in {"a", "c"} else None
            )
            lik_ratio_nq_all.append(lik_ratio_nq) if lik_ratio_nq is not None else None

            post_cog_nq = round(
                (
                    (lik_cog_nq * prior)
                    / (lik_cog_nq * prior + lik_not_cog_nq * (1 - prior))
                ),
                3,
            )
            post_cog_nq_all.append(post_cog_nq)

            df_nq = post_cog_nq - prior if cm == "b" else None
            df_nq_all.append(df_nq) if df_nq is not None else None

            rm_nq = post_cog_nq / prior if cm == "c" else None
            rm_nq_all.append(rm_nq) if rm_nq is not None else None

            z_measure_nq = calculate_z(post_cog_nq, prior) if cm == "d" else None
            z_measure_all.append(z_measure_nq) if z_measure_nq is not None else None

    df_columns = ["cog", "Likelihood"]
    if cm == "a":
        df_columns.extend(["BF", "Prior", "Posterior"])
        data_all = list(
            zip(cog_all, lik_cog_nq_all, lik_ratio_nq_all, prior_all, post_cog_nq_all)
        )
        sort_column = "BF"

    elif cm == "b":
        df_columns.extend(["Difference", "Prior", "Posterior"])
        data_all = list(
            zip(cog_all, lik_cog_nq_all, df_nq_all, prior_all, post_cog_nq_all)
        )

    elif cm == "c":
        df_columns.extend(["Ratio", "Prior", "Posterior"])
        data_all = list(
            zip(cog_all, lik_cog_nq_all, rm_nq_all, prior_all, post_cog_nq_all)
        )

    elif cm == "d":
        df_columns.extend(["Z-measure", "Prior", "Posterior"])
        data_all = list(
            zip(cog_all, lik_cog_nq_all, z_measure_all, prior_all, post_cog_nq_all)
        )

    df_data_all = pd.DataFrame(data_all, columns=df_columns)
    # toLog(df_data_all)

    # elapsed = time.time() - t
    # toLog("time elapsed:", elapsed)

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
    with open(json_path, "r") as infile:
        coordinates_atlas = json.load(infile)

    return coordinates_atlas


def is_visited(current_index, visited_indices, radius):
    """
    Check if a coordinate has been visited.

    Args:
        current_coord (tuple): Tuple containing x, y, and z coordinates.
        visited_coords (ndarray): 3D array of visited coordinates.

    Returns:
        bool: True if the coordinate has been visited, False otherwise.
    """
    X, Y, Z = get_cubic_mesh(current_index, radius, visited_indices)

    distances = np.linalg.norm(
        np.array([X - current_index[0], Y - current_index[1], Z - current_index[2]]),
        axis=0,
    )

    within_indices = distances <= radius

    # Extract the coordinates within the radius and convert to global coordinates in visited_indices
    X_within = X[within_indices]
    Y_within = Y[within_indices]
    Z_within = Z[within_indices]

    # Check if any of the within-radius global coordinates are visited
    visited_within_radius = visited_indices[X_within, Y_within, Z_within]

    # Return True if any of these within-radius coordinates are visited
    return np.any(visited_within_radius)


def update_visited_coord(visited_coords, x_norm, y_norm, z_norm, padding):
    """
    Update the array of visited coordinates with the given normalized coordinates,
    marking a spherical region of 1s around the specified point.

    Args:
        visited_coords (ndarray): 3D array of visited coordinates.
        x_norm (int): Normalized x-coordinate.
        y_norm (int): Normalized y-coordinate.
        z_norm (int): Normalized z-coordinate.
        padding (int): Radius of the spherical region to be updated around the coordinates.

    Returns:
        None
    """
    X, Y, Z = get_cubic_mesh([x_norm, y_norm, z_norm], padding, visited_coords)

    # Compute the Euclidean distance from the center (x_norm, y_norm, z_norm)
    distances = np.linalg.norm(np.array([X - x_norm, Y - y_norm, Z - z_norm]), axis=0)

    within_indices = distances <= padding
    # Extract the coordinates within the radius and convert to global coordinates in visited_indices
    X_within = X[within_indices]
    Y_within = Y[within_indices]
    Z_within = Z[within_indices]

    # Mark points within the spherical radius (padding) as visited
    visited_coords[X_within, Y_within, Z_within] = 1

    return visited_coords


def get_cubic_mesh(coords, half_edge, control_dims):
    x, y, z = coords

    x_min = max(0, x - half_edge)
    x_max = min(control_dims.shape[0], x + half_edge + 1)
    y_min = max(0, y - half_edge)
    y_max = min(control_dims.shape[1], y + half_edge + 1)
    z_min = max(0, z - half_edge)
    z_max = min(control_dims.shape[2], z + half_edge + 1)

    # Create a meshgrid of the local coordinates
    X, Y, Z = np.meshgrid(
        np.arange(x_min, x_max),
        np.arange(y_min, y_max),
        np.arange(z_min, z_max),
        indexing="ij",
    )

    # Convert mesh grid coordinates to integer type
    X = X.astype(int)
    Y = Y.astype(int)
    Z = Z.astype(int)

    return X, Y, Z


def normalize_coord(x_target, y_target, z_target, coord_ranges):
    """
    Normalize target coordinates based on specified coordinate ranges.

    Args:
        x_target (int): Target x-coordinate.
        y_target (int): Target y-coordinate.
        z_target (int): Target z-coordinate.
        coord_ranges (dict): Dictionary containing coordinate range information.

    Returns:
        tuple: Normalized x, y, and z coordinates.
    """
    xnorm = x_target - coord_ranges["xmin"]
    ynorm = y_target - coord_ranges["ymin"]
    znorm = z_target - coord_ranges["zmin"]
    return xnorm, ynorm, znorm


def run_bayesian_analysis_area(
    cog_list,
    prior_list,
    mask,
    radius,
    feature_df,
    cm,
    dt_papers_nq_id_list,
    nb_unique_paper,
    xyz_coords,
):
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

    coordinates_area_list = np.where(mask == 1)
    coordinates_area = [
        [
            coordinates_area_list[0][a],
            coordinates_area_list[1][a],
            coordinates_area_list[2][a],
        ]
        for a in range(len(coordinates_area_list[0]))
    ]  # list of coordinated in the mask with value 1

    # Initialize an empty list to store results and coordinates
    result_all = []

    padding = min(1, int(radius * 2 / 3))

    # Initialize visited_coord array
    visited_coord = np.zeros(
        (
            np.max(coordinates_area_list[0]) - np.min(coordinates_area_list[0]) + 1,
            np.max(coordinates_area_list[1]) - np.min(coordinates_area_list[1]) + 1,
            np.max(coordinates_area_list[2]) - np.min(coordinates_area_list[2]) + 1,
        )
    )

    coord_ranges = {
        "xmax": max([a[0] for a in coordinates_area]),
        "xmin": min([a[0] for a in coordinates_area]),
        "ymax": max([a[1] for a in coordinates_area]),
        "ymin": min([a[1] for a in coordinates_area]),
        "zmax": max([a[2] for a in coordinates_area]),
        "zmin": min([a[2] for a in coordinates_area]),
    }

    # Iterate through the coordinates_area
    for i in range(len(coordinates_area)):
        x_target, y_target, z_target = coordinates_area[i]
        x_norm, y_norm, z_norm = normalize_coord(
            x_target, y_target, z_target, coord_ranges
        )
        if is_visited((x_norm, y_norm, z_norm), visited_coord, radius):
            continue

        if i % 50 == 0:
            send_progress((i + 1) / len(coordinates_area))
            print(
                f"Calculating cm for the {i + 1} ROI out of {len(coordinates_area)}, {((i + 1) / len(coordinates_area)) * 100:.2f}%"
            )
        results = run_bayesian_analysis_coordinates(
            cog_list,
            prior_list,
            x_target,
            y_target,
            z_target,
            radius,
            feature_df,
            cm,
            xyz_coords,
            dt_papers_nq_id_list,
            nb_unique_paper,
        )
        result_all.append(results)

        visited_coord = update_visited_coord(
            visited_coord, x_norm, y_norm, z_norm, padding
        )

        # Append a tuple containing the BF value and the coordinates to result_with_coordinates
        result_all.append(results)

    os.remove(PKG_FOLDER / "neuroinfer" / ".tmp" / "processing_progress.txt")
    # elapsed_area = time.time() - t_area
    # toLog(f"Time in min: {round(elapsed_area / 60, 2)}")
    return result_all
