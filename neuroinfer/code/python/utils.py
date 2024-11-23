import base64
import os
from io import BytesIO

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from nilearn import image
from scipy.ndimage import convolve
from sklearn.mixture import GaussianMixture

from neuroinfer import PKG_FOLDER


def smooth_mask(mask_3d, n):
    """
    Smooths a 3D binary mask using a convolutional kernel with size based on the smoothing factor `n`.
    The resulting mask is returned as a smoothed binary array.
    """

    # Define the size of the kernel based on the smoothing factor
    kernel_size = 2 * n + 1

    # Define the smoothing kernel for averaging
    kernel = np.ones((kernel_size, kernel_size, kernel_size)) / (kernel_size**3)

    # Apply convolution to perform smoothing
    smoothed_mask = convolve(mask_3d.astype(float), kernel)
    print(smoothed_mask.shape)
    print(smoothed_mask.dtype)
    print(mask_3d.shape)
    print(mask_3d.dtype)

    # Convert back to binary mask
    smoothed_mask = (mask_3d + smoothed_mask > 0.4).astype(float)
    print(smoothed_mask.shape)

    return smoothed_mask


def get_combined_overlays(bool_list, file_list, out_file):
    """
    Combines multiple NIfTI images (3D brain scans) into a single overlay by taking the
    maximum value at each voxel across the selected images. Saves the combined overlay as a NIfTI file.
    """
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


def get_sphere_coords(coords, vx_radius, overlay_results):
    """
    Finds all voxel indices within a spherical region around the given coordinates in a 3D volume.
    The radius of the sphere is defined by `vx_radius`, and the volume is based on `overlay_results`.
    """
    volume_shape = overlay_results.shape

    vx_radius = int(vx_radius)
    # Define bounds within vx_radius to avoid generating full meshgrid
    x_min, x_max = max(0, coords[0] - vx_radius), min(volume_shape[0], coords[0] + vx_radius + 1)
    y_min, y_max = max(0, coords[1] - vx_radius), min(volume_shape[1], coords[1] + vx_radius + 1)
    z_min, z_max = max(0, coords[2] - vx_radius), min(volume_shape[2], coords[2] + vx_radius + 1)

    # Create grids within this smaller bounding box
    # X, Y, Z = np.ogrid[x_min:x_max, y_min:y_max, z_min:z_max]
    ## X, Y, Z = np.meshgrid[np.arange(x_min,x_max), np.arange(y_min,y_max), np.arange(z_min,z_max)]
    # Generate ranges for the bounding box
    x_range = np.arange(x_min, x_max)
    y_range = np.arange(y_min, y_max)
    z_range = np.arange(z_min, z_max)

    # Calculate squared distances efficiently using broadcasting
    x_diff = (x_range - coords[0])[:, None, None]
    y_diff = (y_range - coords[1])[None, :, None]
    z_diff = (z_range - coords[2])[None, None, :]

    # Squared distances to avoid using np.sqrt
    squared_distances2 = x_diff**2 + y_diff**2 + z_diff**2
    # Calculate squared distances to avoid using np.sqrt
    # squared_distances2 = (X - coords[0]) ** 2 + (Y - coords[1]) ** 2 + (Z - coords[2]) ** 2

    # Mask where the squared distance is within the square of the radius
    template_3D = squared_distances2 <= (vx_radius ** 2)

    # Get the indices within the sphere and adjust for bounding box offset
    template_indices = np.nonzero(template_3D)
    sphere_coords = (template_indices[0] + x_min,
                     template_indices[1] + y_min,
                     template_indices[2] + z_min)

    return sphere_coords


def create_hist(overlay_results, cog_list):
    """
    This function creates histograms of non-zero values in the `overlay_results` data
    for each cognitive region in `cog_list` and fits a Gaussian Mixture Model (GMM)
    to these values, visualizing both the histogram and the GMM fit.

    Inputs:
    - overlay_results: A 2D array where each column corresponds to data from a cognitive region.
    - cog_list: A list of cognitive region labels, used for the plot legend.

    Steps:
    1. Reshape `overlay_results` into a 2D array where rows are samples and columns are the different regions.
    2. Calculate the optimal number of bins for the histograms based on the data size.
    3. For each cognitive region:
    - Filter non-zero values from `overlay_results` for that region.
    - Fit a Gaussian Mixture Model (GMM) with 3 components to the non-zero data.
    - Plot the histogram of the data with corresponding GMM curve for visual comparison.
    4. Customize the plot with labels, a legend, and different colors for each region.
    5. Save the plot as a PNG image in memory (using a BytesIO object) and encode it as a base64 string.
    6. Return the base64-encoded image for further use (e.g., displaying it on a web interface).
    """

    overlay_results = np.reshape(overlay_results, (-1, overlay_results.shape[-1]))
    num_items = overlay_results.shape[-1]
    nbins = min(int(len(overlay_results.flatten()) / num_items / 20), 200)
    hist_bins = np.histogram_bin_edges(overlay_results.flatten(), bins=nbins)
    plt.figure()

    color = ["#FF0000", "#00FF00", "#0000FF", "#00AAFF", "#AA00FF", "#FFAA00"]

    for i in range(num_items):
        try:
            nonzeros_results = overlay_results[overlay_results[:, i] != 0, i]

            # Fit GMM
            gmm = GaussianMixture(n_components=3)
            gmm = gmm.fit(X=np.expand_dims(nonzeros_results, 1))

            # Evaluate GMM
            gmm_x = np.linspace(0, np.max(nonzeros_results), nbins)
            gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))

            plt.hist(
                nonzeros_results,
                bins=hist_bins,
                label=cog_list[i],
                density=True,
                color=color[i],
                alpha=0.3,
            )
            plt.plot(gmm_x, gmm_y, color=color[i], lw=2)
        except ValueError:
            pass

    # Add labels and legend
    plt.xlabel("BF")
    plt.ylabel("Frequency")
    plt.legend(loc="upper right")

    # Save the plot to a BytesIO object
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)

    # Convert the image in the BytesIO object to base64 encoding
    img_base64 = base64.b64encode(img_buffer.read()).decode("utf-8")
    return img_base64


def generate_nifti_mask(region_id, atlas_target_path, smooth_factor=0):
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

    for j in range(int(smooth_factor)):
        mask = smooth_mask(mask, 1)

    return mask, atlas_img.affine


def send_progress(prop):
    # create the folder if not existing
    os.makedirs(PKG_FOLDER / "neuroinfer" / ".tmp", exist_ok=True)
    with open(
        PKG_FOLDER / "neuroinfer" / ".tmp" / "processing_progress.txt", "w"
    ) as f_tmp:
        f_tmp.write(str(prop))
