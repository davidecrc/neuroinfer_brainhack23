import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image
from scipy import sparse
import argparse

# Set the global path
# Get the directory of the current Python script
script_directory = os.path.dirname(os.path.abspath(__file__))
# Navigate to the parent directory as global path 
global_path = os.path.dirname(script_directory)
print('Global path: ',global_path)
data_path = os.path.join(global_path, "data")  # Path to the saved_result folder
print('data path: ',data_path)

'''
#img_array = np.load(data_path+'/features7/data.npy')

img_array = np.load(os.path.join(data_path, 'features7', 'data.npy'))

# Convert the array to uint8
img_array_uint8 = (img_array * 255).astype(np.uint8)

# Create Image object
im = Image.fromarray(img_array_uint8)

# Save the image
im.save(os.path.join(global_path, 'images', 'features7_data.png'))
'''

parser = argparse.ArgumentParser(description="Load data and create a DataFrame.")
parser.add_argument("npz_file",
                    nargs='?',
                    default=os.path.join(data_path, 'features7.npz'), #"../data/features7.npz",
                    help="Path to the NPZ file containing features data (tfidf data).")
args = parser.parse_args()

# Call the function with command line arguments
#result_df, feature_names = load_data_and_create_dataframe(args.npz_file, args.metadata_file, args.vocab_file)


# Load sparse data
npz_file = args.npz_file# np.load(os.path.join(data_path, 'features7.npz'))

feature_data_sparse = sparse.load_npz(npz_file)
feature_data = feature_data_sparse.todense()
print('feature_data_sparse')
print(feature_data_sparse)
print(feature_data_sparse.shape)