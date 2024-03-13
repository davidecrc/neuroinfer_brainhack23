import numpy as np

def get_user_inputs(feature_names):
    """
    Collect user inputs (area, radius, cog_list) for Bayesian analysis parameters.

    Returns:
    - area (float): Name of the area of interest (AOI), from a menu  associated with a numerical number 0-49 .
    - radius (float): Radius of the Region of Interest (ROI) in millimeters.
    - cog_list (list): List of cognitive terms for analysis.
    - prior_list (list): List of prior probabilities associated with each cognitive term.

    """
    cog_list = [str(x) for x in input("Enter cog(s) here (if more than one separate them with a comma): ").split(",")]

    prior_list = []
    for cog in cog_list:
        cog = cog.strip()
        if cog in feature_names:
            prior_one = float(input(f"The prior probability is the probability of a certain hypothesis (here the occurrence of a term as proxy of the engagement of a cognitive process or function) before some evidence is taken into account. Enter a prior between 0.001 and 1.000 associated with {cog}: "))
            while prior_one < 0.001 or prior_one > 1.000:
                prior_one = float(input(f"Please enter again a prior between 0.0 and 1 associated with {cog}: "))
            prior_list.append(prior_one)
        else:
            print(f'Please check the correct spelling of: "{cog}"; it is not in the NeuroSynth list.')

    for i in range(len(prior_list)):
        print(f"The analysis will be run on: {cog_list[i]} with prior probability: {prior_list[i]}.")

    x_target, y_target, z_target = np.nan, np.nan, np.nan
    area = np.nan

    coordinates_or_area = input("Do you want to insert coordinates (c) or the area (a)? ")
    if coordinates_or_area.lower() == 'c':
        x_target = float(input("Enter brain coordinate x (sagittal plane; based on NeuroSynth, range: -90 / +90): "))
        y_target = float(input("Enter brain coordinate y (coronal plane; based on NeuroSynth, range: -126 / +90): "))
        z_target = float(input("Enter brain coordinate z (horizontal plane; based on NeuroSynth, range: -72 / +108): "))
    elif coordinates_or_area.lower() == 'a':
        area = input("Choose an area from the list: ")

    radius = float(input("Enter ROI radius here (in mm): "))
    while radius < 0 or radius > 1000:
        radius = float(input("Please enter again radius here (in mm; range: 0 / 1000 mm): "))

    return cog_list, prior_list, x_target, y_target, z_target, radius, area
