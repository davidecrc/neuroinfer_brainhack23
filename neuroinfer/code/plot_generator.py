from io import BytesIO
import base64
from neuroinfer.code.BayesianAnalysis import run_bayesian_analysis
import numpy as np
import datetime
import nibabel as nib
from nilearn import image
import os
import matplotlib.pyplot as plt
plt.switch_backend('Agg')


def create_mask_region(brain_region):
    """
    Create a NIfTI mask for a specified brain region.

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

    # Use the 'generate_nifit_mask' function to create the NIfTI mask for the specified brain region
    generate_nifit_mask(brain_region, './templates/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz')

    # Construct a response indicating success
    response = {
        'status': 'success',
        'message': 'Mask generated successfully',
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
    cog_list, prior_list, x_target, y_target, z_target, radius, brain_region = parse_input_args(data)

    # Generating a NIfTI mask for the selected region
    generate_nifit_mask(brain_region, './templates/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz')

    # Perform Bayesian analysis to obtain coordinates (coords) and Bayesian factor values (bf)
    # The run_bayesian_analysis function takes parameters such as brain_region, words, radius, and priors,
    # and returns the computed coordinates and Bayesian factor values.
        # [coords, bf] = run_bayesian_analysis(brain_region, words, radius, priors)

    # Create a histogram of the obtained coordinates and Bayesian factors
        # results = create_hist(coords, bf, atlas_target_path)

    # Generate a NIfTI heatmap using the coordinates and Bayesian factors
    # The generate_nifti_bf_heatmap function utilizes the coordinates and Bayesian factors
    # to generate a heatmap and saves it as a NIfTI file. This heatmap visually represents
    # the spatial distribution of the Bayesian factor values in the specified brain region.
        # generate_nifti_bf_heatmap(coords, bf)

    # Creating a bar plot using matplotlib
    plt.bar(brain_region, [float(p) for p in prior_list])

    # Save the plot to a BytesIO object
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)

    # Convert the image in the BytesIO object to base64 encoding
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')

    # Send the base64 encoded image data as a response
    response = {
        'status': 'success',
        'message': 'Plot generated successfully',
        'image': img_base64
    }

    return response


def generate_nifit_mask(region_id, atlas_target_path):
    """
    Generate a NIfTI mask based on a specified region ID in an atlas image.

    Parameters:
    - region_id (int): The ID of the region for which the mask will be generated.
    - atlas_target_path (str): The file path to the NIfTI atlas image.

    Output:
    - Saves the generated mask as 'mask.nii.gz' in the '.tmp/' directory.
    - The function does not return anything explicitly, as it modifies files in place

    Description:
    - The function loads the atlas image, creates a mask with values set to 1000
      where the atlas data is equal to the specified region_id, and saves the
      modified atlas image as a NIfTI file in the '.tmp/' directory.
    """

    # Convert region_id to integer
    region_id = np.int32(region_id)

    # Load atlas image
    atlas_img = image.load_img(atlas_target_path)

    # Convert atlas image data to 32-bit integer
    atlas_data = np.asarray(atlas_img.get_fdata(), dtype=np.int32)

    # Create a mask and set values to 1000 where atlas_data is equal to region_id
    mask = np.zeros(atlas_data.shape)
    mask[atlas_data == region_id] = 1000

    # convert to Nifti1Image by using the original affine transformation as reference
    modified_atlas_img = nib.Nifti1Image(mask, atlas_img.affine)

    # Check if the '.tmp/' directory exists, if not, create it
    if not os.path.isdir('.tmp/'):
        os.mkdir('.tmp/')

    # Save the modified atlas image as 'mask.nii.gz' in the '.tmp/' directory
    nib.save(modified_atlas_img, '.tmp/mask.nii.gz')

    return


def create_hist(coords, bf):
    results = 0
    return results


def generate_nifti_bf_heatmap(coords, bf, atlas_target_path):
    """
    Generate a NIfTI heatmap based on sorted coordinates and bayesian factors/measurements.

    Parameters:
    - coords (list or array): List or array of coordinates where the heatmap will be generated.
    - bf (list or array): Measurements for one metric. it is used to set the values for the heatmap.
    - atlas_target_path (str): The file path to the NIfTI atlas image for reference data shape.

    Output:
    - Saves the generated heatmap as 'overlay_results.nii.gz' in the '.tmp/' directory with a timestamp.
    - Function does not return anything explicitly, as it modifies files in place

    Description:
    - The function takes coordinates, a custom function (bf), and an atlas image as input.
    - It creates a NIfTI heatmap by applying the custom function to each coordinate and
      sets the corresponding values in the overlay_results array.
    - The resulting heatmap is saved as a NIfTI file in the '.tmp/' directory with a timestamp.
    """

    # Load atlas image to get the reference data shape
    reference_data_shape = image.load_img(atlas_target_path)
    reference_data_shape = np.asarray(reference_data_shape.get_fdata(), dtype=np.int32)

    # Initialize an array for overlay results with zeros of the same shape as reference data
    overlay_results = np.zeros(reference_data_shape.shape)

    # Iterate through the given coordinates and apply the measurements to populate overlay_results
    for j, coord in enumerate(coords):
        overlay_results[coord] = bf[j]

    # Create a Nifti1Image using the overlay_results and the affine transformation from reference data
    overlay_results_img = nib.Nifti1Image(overlay_results, reference_data_shape.affine)

    # Check if the '.tmp/' directory exists, if not, create it
    if not os.path.isdir('.tmp/'):
        os.mkdir('.tmp/')

    # Generate a timestamp for file naming and save the heatmap in the '.tmp/' folder
    time_results = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    nib.save(overlay_results_img, ".tmp/" + time_results + "overlay_results.nii.gz")

    return


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
    brain_region = data.get('brainRegion')
    radius = data.get('radius')
    x = data.get('x')
    y = data.get('y')
    z = data.get('z')
    words = data.get('words')
    words = words.split(',')
    priors = data.get('probabilities')
    priors = priors.split(',')

    # Type conversion and validation checks
    if (len(brain_region) > 0) and (str(type(brain_region)) != "int"):
        try:
            brain_region = np.int32(brain_region)
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
        priors = [np.int32(p) for p in priors]
    except ValueError:
        raise Exception(f"Cannot convert {priors} to a list of int")

    # Validate lengths of 'priors' and 'words'
    if (len(priors) > 1) and (len(priors) != len(words)):
        raise Exception(f"Illegal prior length {len(priors)} with words length {len(words)}")

    # Return the parsed values
    return words, priors, x, y, z, radius, brain_region
