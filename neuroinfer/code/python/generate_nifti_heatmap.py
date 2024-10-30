import datetime
import os

import nibabel as nib
import numpy as np
from nilearn import image

from neuroinfer import PKG_FOLDER
from neuroinfer.code.python.utils import get_sphere_coords, send_progress


def generate_nifti_bf_heatmap(result_dict, atlas_target_path, radius, cog_list, mask):
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
        bf.append(sphere["df_data_all"].BF)
        coords.append(
            [
                sphere["x_target"],
                sphere["y_target"],
                sphere["z_target"],
            ]
        )

    # Load atlas image to get the reference data shape
    reference_data_shape_nifti = image.load_img(atlas_target_path)
    reference_data_shape = np.asarray(
        reference_data_shape_nifti.get_fdata(), dtype=np.int32
    )

    # Initialize an array for overlay results with zeros of the same shape as reference data
    overlay_results = np.zeros([*reference_data_shape.shape, len(bf[0])])
    counter = np.zeros([*reference_data_shape.shape, len(bf[0])])

    # Iterate through the given coordinates and apply the measurements to populate overlay_results
    vx_size = np.abs(reference_data_shape_nifti.affine[0, 0])
    vx_radius = np.ceil(radius / vx_size)

    num_words = len(result_dict[0]["cog_list"])

    for j, coord in enumerate(coords):
        sphere_coords = get_sphere_coords(
            [int(coord[0]), int(coord[1]), int(coord[2])], vx_radius, overlay_results
        )

        sc_0, sc_1, sc_2 = sphere_coords[0], sphere_coords[1], sphere_coords[2]
        sc_length = sc_0.shape[0]  # Store shape once
        bf_j = bf[j]  # Cache bf[j] for fewer accesses

        for sc_i in range(sc_length):
            overlay_results[sc_0[sc_i], sc_1[sc_i], sc_2[sc_i], :] += bf_j  # TODO: this depends on the type of analysis done!
            for cog_counter in range(num_words):
                if bf[j][cog_counter] != 0:
                    counter[
                        sphere_coords[0][sc_i],
                        sphere_coords[1][sc_i],
                        sphere_coords[2][sc_i],
                        cog_counter,
                    ] += 1
        if j % 300 == 0:
            send_progress(str((j + 1) / len(coords)))

    overlay_results = overlay_results / np.where(counter == 0, 1, counter)
    overlay_results = overlay_results * np.repeat(
        mask[:, :, :, np.newaxis], overlay_results.shape[-1], 3
    )

    # Check if the '.tmp/' directory exists, if not, create it
    if not os.path.isdir(".tmp/"):
        os.mkdir(".tmp/")

    overlay_results_img = []
    filename = []
    # Create a Nifti1Image using the overlay_results and the affine transformation from reference data
    for i in range(overlay_results.shape[-1]):
        overlay_results_img.append(
            nib.Nifti1Image(
                overlay_results[:, :, :, i], reference_data_shape_nifti.affine
            )
        )

        # Generate a timestamp for file naming and save the heatmap in the '.tmp/' folder
        time_results = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
        filename.append(".tmp/" + time_results + "init_results_" + str(i) + ".nii.gz")
        nib.save(overlay_results_img[i], filename[i])

    return overlay_results, filename
