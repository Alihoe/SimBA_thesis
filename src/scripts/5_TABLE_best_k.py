import os
import subprocess
from os.path import realpath, dirname
import pandas as pd
import numpy as np
from evaluation.utils import get_ndcg_10, get_map_5, get_recall


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

    columns = ["ct 2020 2a",
                    "ct 2021 2a",
                      "ct 2022 2a",
                     "ct 2021 2b",
                      "ct 2022 2b",
               "all"]

    variants = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 100, 500, 1000]

    table_dict = {"variants": variants}

    all_dataset_scores = {}
    for variant in variants:
        all_dataset_scores[variant] = []

    for column in columns:
        table_dict[column] = []

    for idx, data_name in enumerate(data_names):
        column = columns[idx]

        print(data_name)

        dataset_path = repo_path + "/data/" + data_name
        data_name_queries = dataset_path + "/queries.tsv"
        data_name_targets = dataset_path + "/corpus"
        data_name_gold = dataset_path + "/gold.tsv"

        for k in variants:

            subprocess.call(["python",
                             repo_path + "/src/candidate_retrieval/retrieval.py",
                             data_name_queries,
                             data_name_targets,
                             data_name,
                             data_name,
                             "braycurtis",
                              str(k),
                             '-sentence_embedding_models', "princeton-nlp/sup-simcse-roberta-large",
                             "multi-qa-mpnet-base-dot-v1",
                             ])

            subprocess.call(["python",
                             repo_path + "/src/re_ranking/re_ranking.py",
                             data_name_queries,
                             data_name_targets,
                             data_name,
                             data_name,
                             data_name,
                             "braycurtis",
                             "50",
                             '-sentence_embedding_models', "all-mpnet-base-v2",
                             "sentence-transformers/sentence-t5-base",
                             "princeton-nlp/unsup-simcse-roberta-large",
                             '-lexical_similarity_measures', "similar_words_ratio"
                             ])

            data_name_pred = dataset_path + "/pred_qrels.tsv"

            subprocess.call(["python", repo_path + "/evaluation/scorer/evaluator.py",
                             data_name,
                             data_name_gold,
                             data_name_pred])

            score = float(get_map_5(repo_path + "/data/" + data_name))

            table_dict[column].append(score)
            all_dataset_scores[k].append(score)

    for variant, scores in all_dataset_scores.items():
        average_score = round(np.mean(scores), 3)
        table_dict["all"].append(average_score)

    scores_df = pd.DataFrame.from_dict(table_dict, dtype=str)
    column_format = "l"
    for _ in range(len(columns)):
        column_format = column_format + "|c"

    with open(dir_of_file + "/output/3_TABLE_best_k.txt", 'w') as f:
        print(scores_df.style.format_index(axis=1, formatter="{}".format).hide(
            axis=0).format(precision=3).to_latex(column_format=column_format, position="!htbp",
                                                 label="table:k",
                                                 caption="Influence on Number of Retrieved Documents and Final Score.",
                                                 multirow_align="t", multicol_align="r"), file=f)
    print(scores_df.style.format_index(axis=1, formatter="{}".format).hide(
        axis=0).format(precision=3).to_latex(column_format=column_format, position="!htbp",
                                             label="table:k",
                                             caption="Influence on Number of Retrieved Documents and Final Score.",
                                             multirow_align="t", multicol_align="r"))


if __name__ == "__main__":
    run()