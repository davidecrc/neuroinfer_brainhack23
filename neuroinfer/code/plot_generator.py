from io import BytesIO
import base64
from neuroinfer.code.BayesianAnalysis import run_bayesian_analysis
import numpy as np
import datetime
from pathlib import Path
import nibabel as nib
from nilearn import image
import os
import matplotlib.pyplot as plt
plt.switch_backend('Agg')


def createMaskRegion(brainRegion):
    generate_nifit_mask(brainRegion, './templates/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz')
    response = {
        'status': 'success',
        'message': 'Mask generated successfully',
    }
    return response


def generate_plot(data):
    # Extracting data from the form:
    cog_list, prior_list, x_target, y_target, z_target, radius, brainRegion = parse_input_ars(data)

    # generating a nifti mask for the selected region
    generate_nifit_mask(brainRegion, './templates/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz')

    # [coords, bf] = run_bayesian_analysis(brainRegion, words, radius, priors)
    # results = create_hist(coords, bf, atlas_target_path)
    # generate_nifti_bf_heatmap(coords, bf)

    plt.bar(brainRegion, [float(p) for p in prior_list])

    # Save the plot to a BytesIO object
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)

    # Convert the image to base64 for embedding in HTML
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


def parse_input_ars(data):
    brainRegion = data.get('brainRegion')
    radius = data.get('radius')
    x = data.get('x')
    y = data.get('y')
    z = data.get('z')
    words = data.get('words')
    words = words.split(',')
    priors = data.get('probabilities')
    priors = priors.split(',')
    if (len(brainRegion)>0) & (str(type(brainRegion)) != "int"):
        try:
            brainRegion = np.int32(brainRegion)
        except ValueError:
            raise Exception(f"Cannot convert {brainRegion} to a int")
    if (len(x)>0) & (str(type(x)) != "float"):
        try:
            x = float(x)
        except ValueError:
            raise Exception(f"Cannot convert {x} to a float")
    if (len(y)>0) & (str(type(y)) != "float"):
        try:
            y = float(y)
        except ValueError:
            raise Exception(f"Cannot convert {y} to a float")
    if (len(z)>0) & (str(type(z)) != "float"):
        try:
            z = float(z)
        except ValueError:
            raise Exception(f"Cannot convert {z} to a float")
    if (len(radius)>0) & (str(type(radius)) != "float"):
        try:
            radius = float(radius)
        except ValueError:
            raise Exception(f"Cannot convert {radius} to a float")
    if type(words) != list:
        raise Exception(f"{words} of type {type(words)} is not a list")
    if type(priors) != list:
        raise Exception(f"{priors} of type {type(priors)} is not a list")
    try:
        priors = [np.int32(p) for p in priors]
    except ValueError:
        raise Exception(f"Cannot convert {priors} to a list of int")
    if (len(priors) > 1) & (len(priors) != len(words)):
        raise Exception(f"Illegal prior length {len(priors)} with words length {len(priors)}")
    return words, priors, x, y, z, radius, brainRegion
