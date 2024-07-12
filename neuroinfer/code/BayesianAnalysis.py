import os
import numpy as np
import pandas as pd
import pickle

from neuroinfer.code.run_bayesian import (
    run_bayesian_analysis_coordinates,
    run_bayesian_analysis_area,
)
from neuroinfer import TEMPLATE_FOLDER, DATA_FOLDER, RESULTS_FOLDER
from neuroinfer.code.utils import generate_nifti_mask
from nilearn.image import coord_transform

atlas_path = (
    TEMPLATE_FOLDER
    / "atlases"
    / "HarvardOxford-cort-maxprob-thr25-2mm.nii.gz"
)

"""
The script orchestrates Bayesian analysis on brain data  ensuring efficiency and insightful interpretation through statistical summaries and graphical representations.
It offers two analysis pathways:
-`run_bayesian_analysis_area`: conducts analysis over defined brain regions, leveraging coordinates from an atlas.
-`run_bayesian_analysis_coordinates`: performs analysis at specified coordinates.
The pathway is choosen from the function run_bayesian_analysis_router based on the input (coordinates or area)

The script calculates various statistics including likelihood, prior, posterior, and Bayesian Factor (BF).
It plots BF distributions and computes basic statistical metrics. 

"""


def run_bayesian_analysis_router(
    cog_list, area, prior_list, x_target, y_target, z_target, radius, result_df
):
    cm = input(
        "Specify the Bayesian confirmation measure you want to use in the analysis (type one of the following letter):\n"
        + "'a' for Bayes' Factor;\n"
        + "'b' for difference measure;\n"
        + "'c' for ratio measure.\n"
        + "'d' for z measure.\n"
    )

    # Check if RESULTS_FOLDER exists, if not, create it

    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    mask, affine = generate_nifti_mask(area, atlas_path)
    affine_inv = np.linalg.inv(
        affine
    )  # inverse of the affine matrix to convert from MNI coordinates tp voxel coordinates

    dt_papers_nq_id_list, nb_unique_paper, xyz_coords = load_or_calculate_variables(
        DATA_FOLDER, affine_inv
    )

    if (
        not pd.isnull(area)
        and pd.isnull(x_target)
        and pd.isnull(y_target)
        and pd.isnull(z_target)
    ):
        results = run_bayesian_analysis_area(
            cog_list,
            prior_list,
            mask,
            radius,
            result_df,
            cm,
            dt_papers_nq_id_list,
            nb_unique_paper,
            xyz_coords,
        )

        save_results(
            results, RESULTS_FOLDER / f"results_area_cm_{cm}_{area}_{cog_list}.pkl"
        )

    elif (
        not np.isnan(x_target)
        and not np.isnan(y_target)
        and not np.isnan(z_target)
        and np.isnan(area)
    ):
        # Call run_bayesian_analysis_coordinates if coordinates are not nan and area is nan
        x_target, y_target, z_target = coord_transform(
            x_target, y_target, z_target, affine_inv
        )
        results = run_bayesian_analysis_coordinates(
            cog_list,
            prior_list,
            x_target,
            y_target,
            z_target,
            radius,
            result_df,
            cm,
            xyz_coords,
            dt_papers_nq_id_list,
            nb_unique_paper,
        )

        pickle_file_name = f"results_BHL_coordinates_cm_{cm}_x{x_target}_y{y_target}_z{z_target}.pickle"

        save_results(results, RESULTS_FOLDER / pickle_file_name)

    elif np.isnan(area):
        # Print a message asking to check the input value if area is nan
        print("Please check the input value for 'area'.")
    else:
        # Handle the case where none of the conditions are met
        print("Invalid combination of input values. Please check.")


def save_results(results, filename):
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to: {filename}")


def load_or_calculate_variables(data_path, affine_inv):
    dt_papers_nq_path = os.path.join(data_path, "dt_papers_nq.pkl")
    xyz_coords_path = os.path.join(data_path, "xyz_coords.pkl")

    # Check if the files exist
    if os.path.exists(dt_papers_nq_path) and os.path.exists(xyz_coords_path):
        # Load dt_papers_nq
        with open(dt_papers_nq_path, "rb") as f:
            dt_papers_nq = pickle.load(f)

        # Load xyz_coords
        with open(xyz_coords_path, "rb") as f:
            xyz_coords = pickle.load(f)
    else:
        # Calculate and save dt_papers_nq
        dt_papers_nq = pd.read_csv(
            os.path.join(data_path, "data-neurosynth_version-7_coordinates.tsv"),
            sep="\t",
        )

        # Calculate the Euclidean distance from each point to the center of the sphere
        mni_coords = dt_papers_nq[
            ["x", "y", "z"]
        ].values  # assuming this is a list of triplets

        # Convert MNI coordinates to voxel coordinates for each coordinate
        xyz_coords = [coord_transform(a[0], a[1], a[2], affine_inv) for a in mni_coords]

        # Save dt_papers_nq
        with open(dt_papers_nq_path, "wb") as f:
            pickle.dump(dt_papers_nq, f)

        # Save xyz_coords
        with open(xyz_coords_path, "wb") as f:
            pickle.dump(xyz_coords, f)

    return dt_papers_nq["id"].to_numpy(), dt_papers_nq["id"].nunique(), xyz_coords
