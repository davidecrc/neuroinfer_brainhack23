from datetime import datetime


def save_summary_report(
    local_path,
    df_data_all,
    cm,
    cog_all,
    prior_all,
    x_target,
    y_target,
    z_target,
    radius,
):
    """
    Save a summary report of the NeuroInfer analysis.

    Parameters:
    - local_path (str): The local path for storing the summary report file.
    - df_data_all (pd.DataFrame): The DataFrame containing analysis results.
    - cm (str): Bayesian confirmation measure type ('a', 'b', or 'c').
    - cog_all (list): List of cognitive processes analyzed.
    - prior_all (list): List of prior probabilities associated with each cognitive process.
    - x_target (float): Brain coordinate x (sagittal plane) for the analysis.
    - y_target (float): Brain coordinate y (coronal plane) for the analysis.
    - z_target (float): Brain coordinate z (horizontal plane) for the analysis.
    - radius (float): Radius of the Region of Interest (ROI) in millimeters.

    Returns:
    - None

    """
    now = datetime.now()

    with open(local_path + "neuroinfer_summary" + str(now) + ".txt", "w") as f:
        f.write(
            "NeuroInfer -- Summary report of the analysis\n\n\n"
            + "Database update: 2018;\n\n"
            + "Analysis: Date & Time: "
            + str(now)
            + ";\n\n"
        )

    with open(local_path + "neuroinfer_summary" + str(now) + ".txt", "w") as f:
        for u in range(0, len(cog_all)):
            f.write(
                "Cognitive process analyzed: "
                + str(cog_all[u])
                + ";\n"
                + "Prior probability associated to "
                + str(cog_all[u])
                + ": "
                + str(prior_all[u])
                + ";\n\n"
            )

    with open(local_path + "neuroinfer_summary" + str(now) + ".txt", "w") as f:
        f.write(
            "Brain coordinates [x;y;z]: "
            + str(x_target)
            + "; "
            + str(y_target)
            + "; "
            + str(z_target)
            + ";\n\n"
            + "Radius (in mm): "
            + str(radius)
            + ";\n\n"
        )

    with open(local_path + "neuroinfer_summary" + str(now) + ".txt", "w") as f:
        if cm == "a":
            f.write(
                "Confirmation measure: Bayes' Factor; \n\n" + str(df_data_all) + "\n\n"
            )
        elif cm == "b":
            f.write(
                "Confirmation measure: Difference measure; \n\n"
                + str(df_data_all)
                + "\n\n"
            )
        elif cm == "c":
            f.write(
                "Confirmation measure: Ratio measure; \n\n" + str(df_data_all) + "\n\n"
            )
    f.close()
