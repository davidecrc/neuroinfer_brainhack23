def get_user_inputs(feature_names):
    """
    Collect user inputs for Bayesian analysis parameters.

    Returns:
    - cog_list (list): List of cognitive terms for analysis.
    - prior_list (list): List of prior probabilities associated with each cognitive term.
    - x_target (float): Brain coordinate x (sagittal plane) for the analysis.
    - y_target (float): Brain coordinate y (coronal plane) for the analysis.
    - z_target (float): Brain coordinate z (horizontal plane) for the analysis.
    - radius (float): Radius of the Region of Interest (ROI) in millimeters.

    """
    cog_list = [str(x) for x in input("Enter cog(s) here (if more than one separate them with a comma): ").split(",")]

    prior_list = []
    for cog in cog_list:
        cog = cog.strip()
        if cog in feature_names:
            prior_one = float(input(f"Enter a prior between 0.001 and 1.000 associated to {cog}: "))
            while prior_one < 0.001 or prior_one > 1.000:
                prior_one = float(input(f"Please enter again a prior between 0.0 and 1 associated to {cog}: "))
            prior_list.append(prior_one)
        else:
            print(f'Please check the correct spelling of: "{cog}"; it is not in the NeuroSynth list.')

    for i in range(len(prior_list)):
        print(f"The analysis will be run on: {cog_list[i]} with prior probability: {prior_list[i]}.")

    x_target = float(input("Enter brain coordinate x (sagittal plane; based on NeuroSynth, range: -90 / +90): "))
    while x_target < -90 or x_target > 90:
        x_target = float(input("Please enter again brain coordinate x (sagittal plane; range: -90 / +90):"))

    y_target = float(input("Enter brain coordinate y (coronal plane; based on NeuroSynth, range: -126 / +90): "))
    while y_target < -126 or y_target > 90:
        y_target = float(input("Please enter again brain coordinate y (coronal plane; range: -126 / +90):"))

    z_target = float(input("Enter brain coordinate z (horizontal plane; based on NeuroSynth, range: -72 / +108): "))
    while z_target < -72 or z_target > 108:
        z_target = float(input("Please enter again brain coordinate z (horizontal plane; range: -72 / +108):"))

    radius = float(input("Enter ROI radius here (in mm): "))
    while radius < 0 or radius > 1000:
        radius = float(input("Please enter again radius here (in mm; range: 0 / 1000 mm):"))

    return cog_list, prior_list, x_target, y_target, z_target, radius
