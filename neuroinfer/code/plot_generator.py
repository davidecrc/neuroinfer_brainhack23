import matplotlib.pyplot as plt
from io import BytesIO
import base64
from neuroinfer.code.BayesianAnalysis import run_bayesian_analysis
import numpy as np
import datetime
from pathlib import Path

plt.switch_backend('Agg')
import nibabel as nib
from nilearn import image
import os

def generate_plot(data):
    # Extracting data from the form
    brainRegion = data.get('brainRegion')
    radius = data.get('radius')
    x = data.get('x')
    y = data.get('y')
    z = data.get('z')
    words = data.get('words')
    priors = data.get('probabilities')

    # generating a nifti mask fro the selected region
    generate_nifit_mask(brainRegion, './templates/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr25-2mm.nii.gz')

    #[coords, bf] = run_bayesian_analysis(brainRegion, words, radius, priors)
    #results = create_hist(coords, bf)
    #generate_nifti_bf(coords, bf)

    plt.bar(words.split(','), [float(p) for p in priors.split(',')])

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
    region_id = np.int32(region_id)
    atlas_img = image.load_img(atlas_target_path)
    atlas_data = np.asarray(atlas_img.get_fdata(), dtype=np.int32)
    mask = np.zeros(atlas_data.shape)
    mask[atlas_data == region_id] = 1000

    modified_atlas_img = nib.Nifti1Image(mask, atlas_img.affine)
    if not os.path.isdir('.tmp/'):
        os.mkdir('.tmp/')
    nib.save(modified_atlas_img, '.tmp/mask.nii.gz')
    return


def generate_nifti_bf(coords, bf, atlas_target_path):
    atlas_img = image.load_img(atlas_target_path)
    atlas_data = np.asarray(atlas_img.get_fdata(), dtype=np.int32)
    bf_map = np.zeros(atlas_data.shape)
    bf_map[coords]=bf
    bf_map_img = nib.Nifti1Image(bf_map, atlas_img.affine)

    if not os.path.isdir('.tmp/'):
        os.mkdir('.tmp/')
    filename = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
    nib.save(bf_map_img, Path('.tmp') / filename)
    return