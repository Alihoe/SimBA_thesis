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

    data_names = ["scifact",
                  "nf",
                  "arguana",
                  "scidocs",
                  "cqa_dupstack_programmers",
                  "cqa_dupstack_english"]

    columns = ["SimBA"]

    table_dict = {"datasets": ["scifact",
                  "nf",
                  "arguana",
                  "scidocs",
                  "cqa dupstack programmers"
                  "cqa dupstack english"]}

    table_dict["SimBA"] = []

    for idx, data_name in enumerate(data_names):

        print(data_name)

        dataset_path = repo_path + "/data/" + data_name
        data_name_queries = dataset_path + "/queries.tsv"
        data_name_targets = dataset_path + "/corpus"
        data_name_gold = dataset_path + "/gold.tsv"

        subprocess.call(["python",
                         repo_path + "/src/re_ranking/re_ranking.py",
                         data_name_queries,
                         data_name_targets,
                         data_name,
                         data_name,
                         data_name,
                         "cosine",
                         "50",
                         '--ranking_only',
                         '-sentence_embedding_models', "all-mpnet-base-v2",
                         "Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
                         "sentence-transformers/sentence-t5-base",
                         '-lexical_similarity_measures', "similar_words_ratio"
                         ])

        data_name_pred = dataset_path + "/pred_qrels.tsv"

        subprocess.call(["python", repo_path + "/evaluation/scorer/evaluator.py",
                         data_name,
                         data_name_gold,
                         data_name_pred])

        map = float(get_map_5(repo_path + "/data/" + data_name))
        ndcg = float(get_ndcg_10(repo_path + "/data/" + data_name))

        table_dict["SimBA"].append(map)
        table_dict["SimBA"].append(ndcg)

    scores_df = pd.DataFrame.from_dict(table_dict, dtype=str)
    column_format = "l"
    for _ in range(len(columns)):
        column_format = column_format + "|c"

    with open(dir_of_file + "/output/4_TABLE_other_ir_tasks.txt", 'w') as f:
        print(scores_df.style.format_index(axis=1, formatter="{}".format).hide(
            axis=0).format(precision=3).to_latex(column_format=column_format, position="!htbp",
                                                 label="table:other_ir_tasks",
                                                 caption="MAP@5 and NDCG@10 for other IR Tasks.",
                                                 multirow_align="t", multicol_align="r"), file=f)


if __name__ == "__main__":
    run()