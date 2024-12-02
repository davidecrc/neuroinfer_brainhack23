"""
This script extracts coordinates from a specified brain atlas and saves them to a JSON file.

Functions:
    extract_coordinates_from_atlas(atlas):
        Extracts coordinates from the given atlas.
        Args:
            atlas (dict): A dictionary containing atlas data with "labels" and "maps".
        Returns:
            dict: A dictionary where keys are label indices and values are lists of coordinates.

    save_coordinates_to_json(coord_label_all, output_path):
        Saves the extracted coordinates to a JSON file.
        Args:
            coord_label_all (dict): A dictionary of coordinates.
            output_path (str): The file path where the JSON file will be saved.

    main():
        Main function to fetch the atlas, extract coordinates, and save them to a JSON file.
"""

import json
import nilearn
import numpy as np
from nilearn import datasets, image


def extract_coordinates_from_atlas(atlas):
    atlas_labels = atlas["labels"]
    atlas_regions = nilearn.image.get_data(atlas["maps"])
    niimg = datasets.load_mni152_template()

    coord_label_all = {}

    for n_labho in range(len(atlas_labels)):
        selected = np.where(atlas_regions == n_labho)
        coord_label = []

        for a in range(len(selected[0])):
            i, j, k = image.coord_transform(
                selected[0][a], selected[1][a], selected[2][a], niimg.affine
            )
            coord_label.append([int(i), int(j), int(k)])

        coord_label_all[n_labho] = coord_label

    return coord_label_all


def save_coordinates_to_json(coord_label_all, output_path):
    with open(output_path, "w") as outfile:
        json.dump(coord_label_all, outfile)


def main():
    atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    coordinates = extract_coordinates_from_atlas(atlas)
    # Get the script's directory using Path
    script_directory = Path(__file__).resolve().parent
    output_path = script_directory / "data" / "coord_label_all_harv_ox.json"
    save_coordinates_to_json(coordinates, output_path)


if __name__ == "__main__":
    main()
