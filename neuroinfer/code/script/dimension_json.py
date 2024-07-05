import numpy as np
import json
from BayesianAnalysis import get_atlas_coordinates_json
import pandas as pd

import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image
from scipy import sparse
import argparse

# Set the global path
# Get the directory of the current Python script
script_directory = os.path.dirname(os.path.abspath(__file__))
global_path = os.path.dirname(script_directory)
print("Global path: ", global_path)
data_path = os.path.join(global_path, "data")  # Path to the saved_result folder
print("data path: ", data_path)


# Load coordinates from the JSON file
json_path = data_path + "/coord_label_all_harv_ox.json"
coordinates_atlas = get_atlas_coordinates_json(json_path)

# Extract coordinates specific to the provided area
area = 47
coordinates_area = coordinates_atlas.get(str(area), [])
print("len(coordinates_area) for this area: ", len(coordinates_area))
