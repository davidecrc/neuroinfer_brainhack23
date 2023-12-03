import pandas as pd
import time

from code.SummaryReport import save_summary_report


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

            df_nq = post_cog_nq - prior if cm == "b" else None
            df_nq_all.append(df_nq) if df_nq is not None else None

            rm_nq = post_cog_nq / prior if cm == "c" else None
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