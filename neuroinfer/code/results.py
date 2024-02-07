import pickle
import os

script_directory = os.path.dirname(os.path.abspath(__file__))
global_path = os.path.dirname(script_directory)
print('Global path: ',global_path)
data_path = os.path.join(global_path, "results")  # Path to the saved_result folder
print('data path: ',data_path)


# Load the data from the pickle file
with open(data_path+'/results_BHL_area4.pickle', 'rb') as file:
    loaded_results = pickle.load(file)

# Print the loaded data
print("Loaded Results:")
print(loaded_results)

# Print the size of the loaded data
print(f"Size of results_BHL_area47.pickle: {len(loaded_results)}")



# Extract all the cm values
cm_values = [result[1] for result in loaded_results]

# Print all the cm values
print("All cm values:")
print(cm_values)
