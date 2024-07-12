import base64
from io import BytesIO

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from nilearn import image
from scipy.ndimage import convolve
from sklearn.mixture import GaussianMixture


def smooth_mask(mask_3d, n):
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
    volume_shape = overlay_results.shape
    X, Y, Z = np.meshgrid(
        np.arange(volume_shape[0]),
        np.arange(volume_shape[1]),
        np.arange(volume_shape[2]),
        indexing="ij",
    )

    distances = np.sqrt(
        (X - coords[0]) ** 2 + (Y - coords[1]) ** 2 + (Z - coords[2]) ** 2
    )

    template_3D = distances <= vx_radius

    template_indices = np.nonzero(template_3D)

    return template_indices


def create_hist(overlay_results, cog_list):
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
