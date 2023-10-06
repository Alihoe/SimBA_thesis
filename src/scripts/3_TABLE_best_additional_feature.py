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

    variants = ["SimBA (Semantic + Lexical)",
               "Lexical Only",
               "Semantic Only",
               "Sequence Only",
               "Levenshtein Only",
               "Synonym Only",
               "NE Only",
               "Semantic + Sequence",
               "Semantic + Levenshtein",
               "Semantic + Synonym",
               "Semantic + NE"]

    table_dict = {"variants": variants}

    all_dataset_scores = {}
    for variant in variants:
        all_dataset_scores[variant] = []

    for column in columns:
        table_dict[column] = []

    for idx, data_name in enumerate(data_names):
        column = columns[idx]

        print(data_name)

        # SimBA
        print("SimBA")

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
                         '-sentence_embedding_models', "all-mpnet-base-v2", "sentence-transformers/sentence-t5-base",
                         "Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
                         '-lexical_similarity_measures', "similar_words_ratio"
                         ])

        data_name_pred = dataset_path + "/pred_qrels.tsv"

        subprocess.call(["python", repo_path + "/evaluation/scorer/evaluator.py",
                         data_name,
                         data_name_gold,
                         data_name_pred])

        score = float(get_map_5(repo_path + "/data/" + data_name))
        print(score)

        table_dict[column].append(score)
        all_dataset_scores["SimBA (Semantic + Lexical)"].append(score)

        # Lexical Only
        print("Lexical Only")

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
                         '-lexical_similarity_measures', "similar_words_ratio"
                         ])

        subprocess.call(["python", repo_path + "/evaluation/scorer/evaluator.py",
                         data_name,
                         data_name_gold,
                         data_name_pred])

        score = float(get_map_5(repo_path + "/data/" + data_name))

        table_dict[column].append(score)
        all_dataset_scores["Lexical Only"].append(score)

        # Semantic Only
        print("Semantic Only")

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
                         '-sentence_embedding_models', "all-mpnet-base-v2", "sentence-transformers/sentence-t5-base",
                         "Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit"
                         ])

        subprocess.call(["python", repo_path + "/evaluation/scorer/evaluator.py",
                         data_name,
                         data_name_gold,
                         data_name_pred])

        score = float(get_map_5(repo_path + "/data/" + data_name))

        table_dict[column].append(score)
        all_dataset_scores["Semantic Only"].append(score)

        # Sequence Only
        print("Sequence Only")

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
                         '-string_similarity_measures', "sequence_matching"
                         ])

        data_name_pred = dataset_path + "/pred_qrels.tsv"

        subprocess.call(["python", repo_path + "/evaluation/scorer/evaluator.py",
                         data_name,
                         data_name_gold,
                         data_name_pred])

        score = float(get_map_5(repo_path + "/data/" + data_name))

        table_dict[column].append(score)
        all_dataset_scores["Sequence Only"].append(score)

        # Levenshtein Only
        print("Levenshtein Only")

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
                         '-string_similarity_measures', "levenshtein"
                         ])

        subprocess.call(["python", repo_path + "/evaluation/scorer/evaluator.py",
                         data_name,
                         data_name_gold,
                         data_name_pred])

        score = float(get_map_5(repo_path + "/data/" + data_name))

        table_dict[column].append(score)
        all_dataset_scores["Levenshtein Only"].append(score)

        # Synonym Only
        print("Synonym Only")

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
                         '-referential_similarity_measures', "synonym_similarity"
                         ])

        subprocess.call(["python", repo_path + "/evaluation/scorer/evaluator.py",
                         data_name,
                         data_name_gold,
                         data_name_pred])

        score = float(get_map_5(repo_path + "/data/" + data_name))

        table_dict[column].append(score)
        all_dataset_scores["Synonym Only"].append(score)

        # NE Only
        print("NE Only")

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
                         '-referential_similarity_measures', "ne_similarity"
                         ])

        subprocess.call(["python", repo_path + "/evaluation/scorer/evaluator.py",
                         data_name,
                         data_name_gold,
                         data_name_pred])

        score = float(get_map_5(repo_path + "/data/" + data_name))

        table_dict[column].append(score)
        all_dataset_scores["NE Only"].append(score)

        # Semantic + Sequence
        print("Semantic + Sequence")

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
                         '-sentence_embedding_models', "all-mpnet-base-v2", "sentence-transformers/sentence-t5-base",
                         "Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
                         '-string_similarity_measures', "sequence_matching"
                         ])

        subprocess.call(["python", repo_path + "/evaluation/scorer/evaluator.py",
                         data_name,
                         data_name_gold,
                         data_name_pred])

        score = float(get_map_5(repo_path + "/data/" + data_name))

        table_dict[column].append(score)
        all_dataset_scores["Semantic + Sequence"].append(score)

        # Semantic + Levenshtein
        print("Semantic + Levenshtein")

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
                         '-sentence_embedding_models', "all-mpnet-base-v2", "sentence-transformers/sentence-t5-base",
                         "Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
                         '-string_similarity_measures', "levenshtein"
                         ])

        subprocess.call(["python", repo_path + "/evaluation/scorer/evaluator.py",
                         data_name,
                         data_name_gold,
                         data_name_pred])

        score = float(get_map_5(repo_path + "/data/" + data_name))

        table_dict[column].append(score)
        all_dataset_scores["Semantic + Levenshtein"].append(score)

        # Semantic + Synonym
        print("Semantic + Synonym")

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
                         '-sentence_embedding_models', "all-mpnet-base-v2", "sentence-transformers/sentence-t5-base",
                         "Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
                         '-referential_similarity_measures', "synonym_similarity"
                         ])

        subprocess.call(["python", repo_path + "/evaluation/scorer/evaluator.py",
                         data_name,
                         data_name_gold,
                         data_name_pred])

        score = float(get_map_5(repo_path + "/data/" + data_name))

        table_dict[column].append(score)
        all_dataset_scores["Semantic + Synonym"].append(score)

        # Semantic + NE
        print("Semantic + NE")

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
                         '-sentence_embedding_models', "all-mpnet-base-v2", "sentence-transformers/sentence-t5-base",
                         "Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
                         '-referential_similarity_measures', "ne_similarity"
                         ])

        subprocess.call(["python", repo_path + "/evaluation/scorer/evaluator.py",
                         data_name,
                         data_name_gold,
                         data_name_pred])

        score = float(get_map_5(repo_path + "/data/" + data_name))

        table_dict[column].append(score)
        all_dataset_scores["Semantic + NE"].append(score)

    for variant, scores in all_dataset_scores.items():
        average_score = round(np.mean(scores), 3)
        table_dict["all"].append(average_score)

    scores_df = pd.DataFrame.from_dict(table_dict, dtype=str)
    column_format = "l"
    for _ in range(len(columns)):
        column_format = column_format + "|c"

    with open(dir_of_file + "/output/3_TABLE_best_additional_feature.txt", 'w') as f:
        print(scores_df.style.format_index(axis=1, formatter="{}".format).hide(
            axis=0).format(precision=3).to_latex(column_format=column_format, position="!htbp",
                                                 label="table:additional",
                                                 caption="MAP@5 for Combination of Semantic Features with Additional Features.",
                                                 multirow_align="t", multicol_align="r"), file=f)
    print(scores_df.style.format_index(axis=1, formatter="{}".format).hide(
        axis=0).format(precision=3).to_latex(column_format=column_format, position="!htbp",
                                             label="table:additional",
                                             caption="MAP@5 for Combination of Semantic Features with Additional Features.",
                                             multirow_align="t", multicol_align="r"))


if __name__ == "__main__":
    run()