import pickle
import os

script_directory = os.path.dirname(os.path.abspath(__file__))
global_path = os.path.dirname(script_directory)
print("Global path: ", global_path)
data_path = os.path.join(global_path, "results")  # Path to the saved_result folder
print("data path: ", data_path)


# Load the data from the pickle file
# results_area_{area}_.pkl
# with open(data_path+'/results_BHL_area47.pickle', 'rb') as file:

script_directory = os.path.dirname(os.path.abspath(__file__))
global_path = os.path.dirname(script_directory)
results_folder_path = os.path.join(global_path, "results")

with open(data_path + "/results_area_47_.pkl", "rb") as file:
    # with open( '/home/saram/PhD/neuroinfer_brainhack23/neuroinfer/results/results_area_47_.pkl') as f:
    loaded_results = pickle.load(file, encoding="utf-8")

# Print the loaded data
print("Loaded Results:")
print(loaded_results)

# Print the size of the loaded data
print(f"Size of results_BHL_area47.pickle: {len(loaded_results)}")
