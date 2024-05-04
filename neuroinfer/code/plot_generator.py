from io import BytesIO
import base64
import nilearn
import numpy as np
import datetime
import nibabel as nib
from nilearn import image
import os
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

from neuroinfer import DATA_FOLDER, TEMPLATE_FOLDER
from neuroinfer.code.DataLoading import load_data_and_create_dataframe
from neuroinfer.code.run_bayesian import run_bayesian_analysis_area

plt.switch_backend('Agg')
atlas_path = TEMPLATE_FOLDER / 'atlases' / 'HarvardOxford' / 'HarvardOxford-cort-maxprob-thr25-2mm.nii.gz'


def create_mask_region(brain_region, smooth_factor):
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

    # Use the 'generate_nifit_mask' function to create the NIfTI mask for the specified brain region
    mask_3d, affine = generate_nifit_mask(brain_region, atlas_path, smooth_factor)

    print(type(mask_3d))
    print(type(mask_3d[1,1,1]))
    print(mask_3d.shape)

    # Check if the '.tmp/' directory exists, if not, create it
    if not os.path.isdir('.tmp/'):
        os.mkdir('.tmp/')

    # Save the modified atlas image as 'mask.nii.gz' in the '.tmp/' directory
    nifti_image = nib.Nifti1Image(mask_3d, affine)
    nib.save(nifti_image, '.tmp/mask.nii.gz')

    # Construct a response indicating success
    response = {
        'status': 'success',
        'message': 'Mask generated successfully',
    }

    return response


def update_overlay(combination_bool, file_list):
    output_names = []
    for index, bool_list in enumerate(combination_bool):
        output_names.append(os.path.join('.tmp/', f"overlay_{index}.nii.gz"))
        get_combined_overlays(bool_list, file_list, output_names[index])

    response = {
        'status': 'success',
        'message': output_names,
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
    mask, affine = generate_nifit_mask(data['brainRegion'], atlas_path, data['smooth'])
    affine_inv = np.linalg.inv(affine)

    # Perform Bayesian analysis to obtain coordinates (coords) and Bayesian factor values (bf)
    # The run_bayesian_analysis function takes parameters such as brain_region, words, radius, and priors,
    # and returns the computed coordinates and Bayesian factor values.
    result_df, _ = load_data_and_create_dataframe(DATA_FOLDER / "features7.npz",
                                                  DATA_FOLDER / "metadata7.tsv",
                                                  DATA_FOLDER / "vocabulary7.txt")
    result_dict = run_bayesian_analysis_area(cog_list, prior_list, mask, affine_inv, radius, result_df, 'a')

    # Generate a NIfTI heatmap using the coordinates and Bayesian factors
    # The generate_nifti_bf_heatmap function utilizes the coordinates and Bayesian factors
    # to generate a heatmap and saves it as a NIfTI file. This heatmap visually represents
    # the spatial distribution of the Bayesian factor values in the specified brain region.
    overlay_results, filenames = generate_nifti_bf_heatmap(result_dict, atlas_path, radius, cog_list)

    hist_bins = np.histogram_bin_edges(overlay_results.flatten(), bins=30)
    num_items = overlay_results.shape[-1]
    for i in range(num_items):
        flattened_data = overlay_results[..., i].flatten()
        plt.hist(flattened_data, bins=hist_bins, label=cog_list[i])

    # Add labels and legend
    plt.xlabel('BF')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')

    # Save the plot to a BytesIO object
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)

    # Convert the image in the BytesIO object to base64 encoding
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')

    # Send the base64 encoded image data as a response
    response = {
        'num_slices': len(filenames),
        'status': 'success',
        'message': filenames,
        'image': img_base64
    }

    return response


def generate_nifit_mask(region_id, atlas_target_path, smooth_factor=0):
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

    # Check if region_id is a list, if not, convert it to a list

    if type(region_id) != list:
        region_id = [region_id]    
    # Convert region_id to integer
    region_id = [int(i) for i in region_id]
   

    # Load atlas image
    atlas_img = image.load_img(atlas_target_path)

    # Convert atlas image data to 32-bit integer
    atlas_data = np.asarray(atlas_img.get_fdata())

    # Create a mask and set values to 1000 where atlas_data is equal to region_id
    mask = np.zeros(atlas_data.shape)
    for current_id in region_id:
        mask[atlas_data == current_id] = 1

    # convert to Nifti1Image by using the original affine transformation as reference

    if smooth_factor > 0:
        mask = smooth_mask(mask, smooth_factor)
    return mask, atlas_img.affine


def create_hist(coords, bf):
    results = 0
    return results


def generate_nifti_bf_heatmap(result_dict, atlas_target_path, radius, cog_list):
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
    bf = []
    coords = []
    for sphere in result_dict:
        bf.append(sphere['df_data_all'].BF)
        coords.append([sphere['x_target'],
                       sphere['y_target'],
                       sphere['z_target'],
                       ])

    # Load atlas image to get the reference data shape
    reference_data_shape_nifti = image.load_img(atlas_target_path)
    reference_data_shape = np.asarray(reference_data_shape_nifti.get_fdata(), dtype=np.int32)

    # Initialize an array for overlay results with zeros of the same shape as reference data
    overlay_results = np.zeros([*reference_data_shape.shape, len(bf[0])])
    counter = np.zeros([*reference_data_shape.shape, len(bf[0])])

    # Iterate through the given coordinates and apply the measurements to populate overlay_results
    vx_size = np.abs(reference_data_shape_nifti.affine[0, 0])
    vx_radius = np.ceil(radius/vx_size)

    for j, coord in enumerate(coords):
        sphere_coords = get_sphere_coords([int(coord[0]), int(coord[1]), int(coord[2])], vx_radius, overlay_results)
        for sc_i in range(sphere_coords[0].shape[0]):
            overlay_results[sphere_coords[0][sc_i], sphere_coords[1][sc_i], sphere_coords[2][sc_i], :] += bf[j] #TODO: this depends on the type of analysis done!
            for cog_counter in range(len(result_dict[0]['cog_list'])):
                if bf[j][cog_counter] == 0:
                    continue
                counter[sphere_coords[0][sc_i], sphere_coords[1][sc_i], sphere_coords[2][sc_i], cog_counter] += 1

    overlay_results = overlay_results/np.where(counter == 0, 1, counter)

    # Check if the '.tmp/' directory exists, if not, create it
    if not os.path.isdir('.tmp/'):
        os.mkdir('.tmp/')

    overlay_results_img = []
    filename = []
    # Create a Nifti1Image using the overlay_results and the affine transformation from reference data
    for i in range(overlay_results.shape[-1]):
        overlay_results_img.append(nib.Nifti1Image(overlay_results[:, :, :, i], reference_data_shape_nifti.affine))

        # Generate a timestamp for file naming and save the heatmap in the '.tmp/' folder
        time_results = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
        filename.append(".tmp/" + time_results + "init_results_" + str(i) + ".nii.gz")
        nib.save(overlay_results_img[i], filename[i])

    return overlay_results, filename


def get_sphere_coords(coords, vx_radius, overlay_results):
    volume_shape = overlay_results.shape
    X, Y, Z = np.meshgrid(np.arange(volume_shape[0]),
                          np.arange(volume_shape[1]),
                          np.arange(volume_shape[2]), indexing='ij')

    distances = np.sqrt((X - coords[0])**2 + (Y - coords[1])**2 + (Z - coords[2])**2)

    template_3D = distances <= vx_radius

    template_indices = np.nonzero(template_3D)

    return template_indices


def get_combined_overlays(bool_list, file_list, out_file):
    print(bool_list)
    print(file_list)
    print(out_file)
    file_nifti = None

    for index, to_load in enumerate(bool_list):
        if to_load:
            image_nifti_current = nib.load(file_list[index])
            if file_nifti is None:
                file_nifti = np.asarray(image_nifti_current.get_fdata())
            else:
                current_img = np.asarray(image_nifti_current.get_fdata())
                file_nifti = np.maximum(file_nifti, current_img)

    overlay_results_img = nib.Nifti1Image(file_nifti, image_nifti_current.affine)
    nib.save(overlay_results_img, out_file)


def smooth_mask(mask_3d, n):
    # Define the size of the kernel based on the smoothing factor
    kernel_size = 2 * n + 1

    # Define the smoothing kernel for averaging
    kernel = np.ones((kernel_size, kernel_size, kernel_size)) / (kernel_size ** 3)

    # Apply convolution to perform smoothing
    smoothed_mask = convolve(mask_3d.astype(float), kernel)

    # Convert back to binary mask
    smoothed_mask = (smoothed_mask > 0.5).astype(float)

    return smoothed_mask


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
        raise Exception(f"Illegal prior length {len(priors)} with words length {len(words)}")

    # Return the parsed values
    return words, priors, x, y, z, radius, brain_region


if __name__ == '__main__':
    data = dict()
    data['brainRegion'] = ["47"]
    data['smooth'] = 1
    data['radius'] = "3"
    data['x'] = "0"
    data['y'] = "0"
    data['z'] = "0"
    data['words'] = "face,children,true"
    data['probabilities'] = "0.5,0.5,0.5"
    main_analyse_and_render(data)
