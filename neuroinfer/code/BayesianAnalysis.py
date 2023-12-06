import pandas as pd
import time

from code.SummaryReport import save_summary_report


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



def run_bayesian_analysis(cog_list, prior_list, x_target, y_target, z_target, radius, feature_df):
    """
    Perform Bayesian (confirmation) analysis using specified parameters.

    Parameters:
    - local_path (str): The local path for storing results.
    - df_data_all (pd.DataFrame): The DataFrame containing all data for analysis.
    - cm (Optional[object]): Custom Bayesian confirmation measure for analysis.
                             Default is None, which triggers the default configuration.

    Returns:
    - result (object): The result of the Bayesian analysis.
    """
    frequency_threshold = 0.01
    t = time.time()
    cog_all = []
    prior_all = []
    ids_cog_nq_all = []
    intersection_cog_nq_all = []
    intersection_not_cog_nq_all = []
    lik_cog_nq_all = []
    lik_not_cog_nq_all = []
    lik_ratio_nq_all = []
    post_cog_nq_all = []
    df_nq_all = []
    rm_nq_all = []

    cm = input(
        "Specify the Bayesian confirmation measure you want to use in the analysis (type one of the following letter):\n" +
        "'a' for Bayes' Factor;\n" +
        "'b' for difference measure;\n" +
        "'c' for ratio measure.\n")

    feature_names = feature_df.columns

    for q in range(0, len(cog_list)):
        cog = cog_list[q].strip()
        prior = prior_list[q]

        if cog not in feature_names:
            print(
                f'Please check the correct spelling of: "{cog}"; it is not in the NeuroSynth list. In the meanwhile, the script goes to the next *cog*, if any.')
        else:
            cog_all.append(cog)
            prior_all.append(prior)
            print(f'Processing "{cog}" (step {q + 1} out of {len(cog_list)})')
            ids_cog_nq = []

            for i in range(0, len(feature_df[cog])):
                if feature_df[cog].iloc[i] > frequency_threshold:
                    ids_cog_nq.append(feature_df[cog].index[i])

            ids_cog_nq_all.append(ids_cog_nq)

            dt_papers_nq = pd.read_csv(
                '../data/data-neurosynth_version-7_coordinates.tsv', sep='\t')
            paper_id_nq = dt_papers_nq["id"]
            paper_x_nq = dt_papers_nq["x"]
            paper_y_nq = dt_papers_nq["y"]
            paper_z_nq = dt_papers_nq["z"]

            list_activations_nq = [paper_id_nq[i] for i in range(len(paper_id_nq)) if
                                   x_target - radius < int(float(paper_x_nq[i])) < x_target + radius and
                                   y_target - radius < int(float(paper_y_nq[i])) < y_target + radius and
                                   z_target - radius < int(float(paper_z_nq[i])) < z_target + radius]
            
            # Calculate ROIs based on target coordinates and radius
            rois = browse_nifti_with_spherical_roi('sub-25_2mnipreproc.nii.gz', radius)

            # Example: Perform some operation with the generated ROIs
            for roi in rois:
                # Perform operations using the generated ROI

                total_activations_nq = len(list_activations_nq)
                intersection_cog_nq = len(ids_cog_nq) & total_activations_nq
                intersection_not_cog_nq = total_activations_nq - intersection_cog_nq

                intersection_cog_nq_all.append(intersection_cog_nq)
                intersection_not_cog_nq_all.append(intersection_not_cog_nq)

                total_database_nq = len(set(paper_id_nq))
                total_not_cog_nq = total_database_nq - len(ids_cog_nq)

                lik_cog_nq = round(intersection_cog_nq / len(ids_cog_nq), 3)
                lik_cog_nq_all.append(lik_cog_nq)

                lik_not_cog_nq = round((intersection_not_cog_nq / total_not_cog_nq) + 0.001, 3)
                lik_not_cog_nq_all.append(lik_not_cog_nq)

                lik_ratio_nq = round((lik_cog_nq / lik_not_cog_nq), 3) if cm in {"a", "c"} else None
                lik_ratio_nq_all.append(lik_ratio_nq) if lik_ratio_nq is not None else None

                post_cog_nq = round(((lik_cog_nq * prior) / (lik_cog_nq * prior + lik_not_cog_nq * (1 - prior))), 3)
                post_cog_nq_all.append(post_cog_nq)

                df_nq = post_cog_nq - prior if cm == "b" else None #difference
                df_nq_all.append(df_nq) if df_nq is not None else None

                rm_nq = post_cog_nq / prior if cm == "c" else None #ratio measure
                rm_nq_all.append(rm_nq) if rm_nq is not None else None

        df_columns = ['cog', 'Likelihood']
        if cm == "a":
            df_columns.extend(['BF', 'Prior', 'Posterior'])
            data_all = list(zip(cog_all, lik_cog_nq_all, lik_ratio_nq_all, prior_all, post_cog_nq_all))
            sort_column = 'BF'
        elif cm == "b":
            df_columns.extend(['Difference', 'Prior', 'Posterior'])
            data_all = list(zip(cog_all, lik_cog_nq_all, df_nq_all, prior_all, post_cog_nq_all))
            sort_column = 'Difference'
        elif cm == "c":
            df_columns.extend(['Ratio', 'Prior', 'Posterior'])
            data_all = list(zip(cog_all, lik_cog_nq_all, rm_nq_all, prior_all, post_cog_nq_all))
            sort_column = 'Ratio'

        df_data_all = pd.DataFrame(data_all, columns=df_columns)
        df_data_all_like = df_data_all.sort_values('Likelihood', ascending=False) 
        df_data_all_sorted = df_data_all.sort_values(sort_column, ascending=False)
        print(df_data_all_sorted)

        elapsed = time.time() - t
        print(f'Time in min: {round(elapsed / 60, 2)}')

        local_path = input(
            "Insert a path to save a summary report (.txt) of the analysis (start and end path with '/'): ").encode(
            'utf-8').decode('utf-8')
        save_summary_report(local_path, df_data_all, cm, cog_all, prior_all, x_target, y_target, z_target, radius)

    return local_path, df_data_all, cm, cog_all, prior_all, x_target, y_target, z_target, radius



import numpy as np
import pandas as pd
import time
from nilearn import image  # Import the necessary modules

# Place the definition of extract_roi function here

# Define the run_bayesian_analysis function with modifications
def run_bayesian_analysis(cog_list, prior_list, x_target, y_target, z_target, radius, feature_df):
    # Previous code remains unchanged...

    # Inside the loop for Bayesian analysis
    for q in range(0, len(cog_list)):
        # Previous code remains unchanged...

        # Calculate ROIs based on target coordinates and radius
        rois = browse_nifti_with_spherical_roi('sub-25_2mnipreproc.nii.gz', radius)

        # Example: Perform some operation with the generated ROIs
        for roi in rois:
            end
            # Perform operations using the generated ROI
            # You can use these ROIs within your Bayesian analysis here

# Previous code remains unchanged...

    # Remaining code remains unchanged