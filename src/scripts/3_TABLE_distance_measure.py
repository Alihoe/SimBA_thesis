import subprocess
from os.path import realpath, dirname
import pandas as pd
from evaluation.utils import get_map_5


def run():

    filepath = realpath(__file__)
    dir_of_file = dirname(filepath)
    parent_dir_of_file = dirname(dir_of_file)
    parent_parent_dir_of_file = dirname(parent_dir_of_file)
    repo_path = parent_parent_dir_of_file

    data_names = ["clef_2020_checkthat_2_english",
                  "clef_2021_checkthat_2a_english",
                  "clef_2022_checkthat_2a_english",
                  "clef_2021_checkthat_2b_english",
                  "clef_2022_checkthat_2b_english"]

    columns = ["braycurtis",
               "canberra",
               "cosine",
               "euclidean"]

    table_dict = {"datasets": [
        "ct 2020 2a",
        "ct 2021 2a",
        "ct 2022 2a",
        "ct 2021 2b",
        "ct 2022 2b"]}

    for column in columns:
        table_dict[column] = []

    for idx, data_name in enumerate(data_names):

        print(data_name)

        dataset_path = repo_path + "/data/" + data_name
        data_name_queries = dataset_path + "/queries.tsv"
        data_name_targets = dataset_path + "/corpus"
        data_name_gold = dataset_path + "/gold.tsv"

        for column in columns:
            this_data_name = data_name + "/" + column
            this_dataset_path = repo_path + "/data/" + this_data_name

            print(this_data_name)

            subprocess.call(["python",
                             repo_path + "/src/re_ranking/re_ranking.py",
                             data_name_queries,
                             data_name_targets,
                             this_data_name,
                             this_data_name,
                             this_data_name,
                             column,
                             "50",
                             "--ranking_only",
                             '-sentence_embedding_models', "all-mpnet-base-v2",
                             "sentence-transformers/sentence-t5-base",
                             "Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
                             '-lexical_similarity_measures', "similar_words_ratio"
                             ])

            data_name_pred = this_dataset_path + "/pred_qrels.tsv"

            subprocess.call(["python", repo_path + "/evaluation/scorer/evaluator.py",
                             this_data_name,
                             data_name_gold,
                             data_name_pred])

            score = float(get_map_5(this_dataset_path))

            table_dict[column].append(score)

    scores_df = pd.DataFrame.from_dict(table_dict, dtype=str)
    column_format = "l"
    for _ in range(len(columns)):
        column_format = column_format + "|c"

    with open(dir_of_file + "/output/3_TABLE_distance_measure.txt", 'w') as f:
        print(scores_df.style.format_index(axis=1, formatter="{}".format).hide(
            axis=0).format(precision=3).to_latex(column_format=column_format, position="!htbp",
                                                 label="table:distance_measure",
                                                 caption="Comparison of Distance Measures.",
                                                 multirow_align="t", multicol_align="r"), file=f)
    print(scores_df.style.format_index(axis=1, formatter="{}".format).hide(
        axis=0).format(precision=3).to_latex(column_format=column_format, position="!htbp",
                                             label="table:distance_measure",
                                             caption="Comparison of Distance Measures.",
                                             multirow_align="t", multicol_align="r"))


if __name__ == "__main__":
    run()