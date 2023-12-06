import numpy as np
import nilearn
import brainiak


def extract_roi(data, roi, affine):
    mni2vx_mat = np.linalg.inv(affine)
    center_vx = nilearn.image.coord_transform(
        roi.coords[0], roi.coords[1], roi.coords[2], mni2vx_mat
    )
    radius_vx = roi.radius / 2
    cubic_roi = data[
        int(center_vx[0] - radius_vx) : int(center_vx[0] + radius_vx),
        int(center_vx[1] - radius_vx) : int(center_vx[1] + radius_vx),
        int(center_vx[2] - radius_vx) : int(center_vx[2] + radius_vx),
    ]
    center_cube = [a / 2 for a in cubic_roi.shape]
    center_cube = np.asarray(center_cube[:3])
    for x in range(cubic_roi.shape[0]):
        for y in range(cubic_roi.shape[1]):
            for z in range(cubic_roi.shape[2]):
                distances = np.sqrt(
                    np.sum((np.asarray([x, y, z]) - center_cube) ** 2, axis=0)
                )
                if distances > radius_vx:
                    cubic_roi[x, y, z] = 0

    return cubic_roi, center_vx



def browse_nifti_with_spherical_roi(image_path, radius):

    # Load the NIfTI image
    print('loading image...', end=' ')
    nifti_img = image.load_img(image_path)
    data = nifti_img.get_fdata()
    print('loaded')

    # Define the voxel dimensions
    voxel_size = np.array(nifti_img.header.get_zooms())
    if len(np.unique(voxel_size[:3])) != 1:
        raise Exception('error, different resolution among the axes')
    voxel_size = voxel_size[0]

    # Calculate the radius in physical units
    radius = radius/voxel_size
    hop_size = int(radius//2)

    # Get the image dimensions
    image_shape = data.shape

    # Calculate the number of ROIs in each dimension
    num_rois = np.ceil(np.array(image_shape[:3]) / hop_size).astype(int)

    # Initialize an empty list to store the ROIs
    rois = []
    rois_cubic = []

    # Generate the spherical ROIs
    for i in range(num_rois[0]):
        print(f"external for: {i+1}/{num_rois[0]}", end='\r')
        for j in range(num_rois[1]):
            for k in range(num_rois[2]):
                # Calculate the center coordinates of the current ROI
                center = np.array([i, j, k]) * hop_size

                # Generate a sphere mask for the current ROI
                x_coords = np.arange(image_shape[0])[:, None, None] - center[0]
                y_coords = np.arange(image_shape[1])[None, :, None] - center[1]
                z_coords = np.arange(image_shape[2])[None, None, :] - center[2]

                squared_distance = x_coords ** 2 + y_coords ** 2 + z_coords ** 2
                roi_mask = squared_distance <= radius ** 2
                roi_data_flatten = data[roi_mask, :]

                max_coord = np.max(np.where(roi_mask), axis=1)
                min_coord = np.min(np.where(roi_mask), axis=1)
                roi_mask_cubic = roi_mask[min_coord[0]: max_coord[0]+1, min_coord[1]: max_coord[1]+1, min_coord[2]: max_coord[2]+1]
                roi_data_cubic = data[min_coord[0]: max_coord[0]+1, min_coord[1]: max_coord[1]+1, min_coord[2]: max_coord[2]+1]
                roi_data_cubic[~roi_mask_cubic, :] = 0

                # Store the ROI data in the list
                if np.sum(roi_data_flatten[:, 1] != 0) > 10:
                    rois.append(roi_data_flatten)
                    rois_cubic.append(roi_data_cubic)

                #hmm_sim = brainiak.eventseg.event.EventSegment(10)
                #hmm_sim.fit(D.T)

    return rois


if __name__ == '__main__':
    rois = browse_nifti_with_spherical_roi('sub-25_2mnipreproc.nii.gz', 10)
    z=0
    pass