import pickle

import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt

from neuroinfer import DATA_FOLDER, TEMPLATE_FOLDER, RESULTS_FOLDER
from neuroinfer.code.python.BayesianAnalysis import load_or_calculate_variables
from neuroinfer.code.python.DataLoading import load_data_and_create_dataframe
from neuroinfer.code.python.generate_nifti_heatmap import generate_nifti_bf_heatmap
from neuroinfer.code.python.run_bayesian import run_bayesian_analysis_area
from neuroinfer.code.python.utils import (
    get_combined_overlays,
    create_hist,
    generate_nifti_mask,
)

plt.switch_backend("Agg")
atlas_path = TEMPLATE_FOLDER / "atlases"


def create_mask_region(atlas, brain_region, smooth_factor):
    """
    Create a NIfTI mask for a specified list of brain regions.

    Parameters:
    - brain_region: The ID or label of the brain region for which the mask will be generated.

    Output:
    - Returns a response dictionary indicating the status and success message.

    Description:
    - The function uses the 'generate_nifit_mask' function to create a NIfTI mask for the specified brain region.
    - 'generate_nifit_mask' generates and saves the mask to a file.
    - The mask is generated based on the specified brain region and a pre-defined atlas image path.
    - The function then returns a response dictionary indicating the success of the mask generation.
    """
    print(brain_region)
    smooth_factor = np.int32(smooth_factor)
    print(atlas_path / atlas)

    # Use the 'generate_nifit_mask' function to create the NIfTI mask for the specified brain region
    mask_3d, affine = generate_nifti_mask(
        brain_region, atlas_path / atlas, smooth_factor
    )
    tot_vx = np.sum(mask_3d)

    print(type(mask_3d))
    print(type(mask_3d[1, 1, 1]))
    print(mask_3d.shape)

    # Check if the '.tmp/' directory exists, if not, create it
    if not os.path.isdir(".tmp/"):
        os.mkdir(".tmp/")

    # Save the modified atlas image as 'mask.nii.gz' in the '.tmp/' directory
    nifti_image = nib.Nifti1Image(mask_3d, affine)
    nib.save(nifti_image, ".tmp/mask.nii.gz")

    # Construct a response indicating success
    response = {
        "status": "success",
        "message": "Mask generated successfully",
        "totvx": tot_vx,
    }

    return response


def update_overlay(combination_bool, file_list):
    output_names = []
    for index, bool_list in enumerate(combination_bool):
        print(bool_list)
        print(file_list)
        output_names.append(os.path.join(".tmp/", f"overlay_{index}.nii.gz"))
        get_combined_overlays(bool_list, file_list, output_names[index])

    response = {
        "status": "success",
        "message": output_names,
    }

    return response


def main_analyse_and_render(data):
    """
    Generate a bar plot based on input data and return the plot as a base64-encoded image.

    Parameters:
    - data (dict): A dictionary containing input data for generating the plot.

    Output:
    - Returns a response dictionary with a base64-encoded image of the generated plot.

    Description:
    - The function extracts necessary data from the input using 'parse_input_args' function.
    - It generates a NIfTI mask for the selected brain region using 'generate_nifit_mask' function.
    - It then generates a bar plot using matplotlib with brain region and corresponding priors.
    - The plot is saved to a BytesIO object, which is an in-memory binary stream.
    - The BytesIO object allows saving the plot image without writing it to a file on disk.
    - The saved image in the BytesIO object is then converted to base64 encoding.
    - Base64 encoding is a binary-to-text encoding scheme that represents binary data (the image in this case)
      as a string of ASCII characters. This makes it suitable for embedding in HTML and other text-based formats.
    - The base64-encoded image data is included in the response dictionary.

    """

    # Extracting data from the form using 'parse_input_args'
    cog_list, prior_list, x_target, y_target, z_target, radius, brain_region = (
        parse_input_args(data)
    )

    # Generating a NIfTI mask for the selected region
    mask, affine = generate_nifti_mask(
        data["brainRegion"], atlas_path / data["atlas"], data["smooth"]
    )
    affine_inv = np.linalg.inv(affine)

    # Perform Bayesian analysis to obtain coordinates (coords) and Bayesian factor values (bf)
    # The run_bayesian_analysis function takes parameters such as brain_region, words, radius, and priors,
    # and returns the computed coordinates and Bayesian factor values.
    result_df, _ = load_data_and_create_dataframe(
        DATA_FOLDER / "features7.npz",
        DATA_FOLDER / "metadata7.tsv",
        DATA_FOLDER / "vocabulary7.txt",
    )
    dt_papers_nq_id_list, nb_unique_paper, xyz_coords = load_or_calculate_variables(
        DATA_FOLDER, affine_inv
    )
    xyz_coords = np.array(xyz_coords, dtype=np.float32)
    result_dict = run_bayesian_analysis_area(
        cog_list,
        prior_list,
        mask,
        radius,
        result_df,
        "a",
        dt_papers_nq_id_list,
        nb_unique_paper,
        xyz_coords,
    )
    result_dict[0].update(
        {
            "atlas": atlas_path / data["atlas"],
            "radius": radius,
            "cog_list": cog_list,
            "mask": mask,
        }
    )
    with open(
        RESULTS_FOLDER / f"results_area_cm_{brain_region}_{cog_list}.pkl", "wb"
    ) as f:
        pickle.dump(result_dict, f)

    # Generate a NIfTI heatmap using the coordinates and Bayesian factors
    # The generate_nifti_bf_heatmap function utilizes the coordinates and Bayesian factors
    # to generate a heatmap and saves it as a NIfTI file. This heatmap visually represents
    # the spatial distribution of the Bayesian factor values in the specified brain region.
    overlay_results, filenames = generate_nifti_bf_heatmap(
        result_dict, atlas_path / data["atlas"], radius, cog_list, mask
    )

    img_base64 = create_hist(overlay_results, cog_list)

    # Send the base64 encoded image data as a response
    response = {
        "num_slices": len(filenames),
        "status": "success",
        "message": filenames,
        "image": img_base64,
        "max_value": np.max(overlay_results),
    }

    return response


def load_results(filename):
    with open(RESULTS_FOLDER / filename, "rb") as f:
        loaded_dict = pickle.load(f)

    overlay_results, filenames = generate_nifti_bf_heatmap(
        loaded_dict,
        atlas_path / os.path.basename(loaded_dict[0]["atlas"]),
        loaded_dict[0]["radius"],
        loaded_dict[0]["cog_list"],
        loaded_dict[0]["mask"],
    )

    img_base64 = create_hist(overlay_results, loaded_dict[0]["cog_list"])

    # Send the base64 encoded image data as a response
    response = {
        "num_slices": len(filenames),
        "status": "success",
        "message": filenames,
        "image": img_base64,
        "max_value": np.max(overlay_results),
        "words": loaded_dict[0]["cog_list"],
    }
    return response


def parse_input_args(data):
    """
    Parse input data for an analysis tool.

    Parameters:
    - data (dict): A dictionary containing input parameters for the main script.

    Output:
    - Returns parsed values for words, priors, x, y, z, radius, and brain_region.

    Description:
    - The function takes a dictionary 'data' as input, containing various parameters for analysis.
    - It extracts values from the dictionary, performs type conversion checks, and raises exceptions for invalid inputs.
    - The function returns the parsed values for words, priors, x, y, z, radius, and brainRegion.
    """

    # Extract values from the dictionary
    brain_region = data.get("brainRegion")
    radius = data.get("radius")
    x = data.get("x")
    y = data.get("y")
    z = data.get("z")
    words = data.get("words")
    words = words.split(",")
    priors = data.get("probabilities")
    priors = priors.split(",")

    # Type conversion and validation checks
    if (len(brain_region) > 0) and (str(type(brain_region)) != "int"):
        try:
            brain_region = [np.int32(a) for a in brain_region]
        except ValueError:
            raise Exception(f"Cannot convert {brain_region} to an int")

    if (len(x) > 0) and (str(type(x)) != "float"):
        try:
            x = float(x)
        except ValueError:
            raise Exception(f"Cannot convert {x} to a float")

    if (len(y) > 0) and (str(type(y)) != "float"):
        try:
            y = float(y)
        except ValueError:
            raise Exception(f"Cannot convert {y} to a float")

    if (len(z) > 0) and (str(type(z)) != "float"):
        try:
            z = float(z)
        except ValueError:
            raise Exception(f"Cannot convert {z} to a float")

    if (len(radius) > 0) and (str(type(radius)) != "float"):
        try:
            radius = float(radius)
        except ValueError:
            raise Exception(f"Cannot convert {radius} to a float")

    # Convert 'words' and 'priors' to lists and validate types
    if type(words) != list:
        raise Exception(f"{words} of type {type(words)} is not a list")

    if type(priors) != list:
        raise Exception(f"{priors} of type {type(priors)} is not a list")

    # Convert 'priors' to a list of integers
    try:
        priors = [np.float32(p) for p in priors]
    except ValueError:
        raise Exception(f"Cannot convert {priors} to a list of float")

    # Validate lengths of 'priors' and 'words'
    if (len(priors) > 1) and (len(priors) != len(words)):
        raise Exception(
            f"Illegal prior length {len(priors)} with words length {len(words)}"
        )

    # Return the parsed values
    return words, priors, x, y, z, radius, brain_region


if __name__ == "__main__":
    data = dict()
    data["brainRegion"] = ["47"]
    data["smooth"] = 1
    data["radius"] = "3"
    data["x"] = "0"
    data["y"] = "0"
    data["z"] = "0"
    data["words"] = "face,children,true"
    data["probabilities"] = "0.5,0.5,0.5"
    main_analyse_and_render(data)
