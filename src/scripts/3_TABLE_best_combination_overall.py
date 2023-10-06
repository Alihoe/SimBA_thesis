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


    features = ["https://tfhub.dev/google/universal-sentence-encoder/4",
                "all-mpnet-base-v2",
                "multi-qa-mpnet-base-dot-v1",
                "all-distilroberta-v1",
                "sentence-transformers/sentence-t5-base",
                "Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit"]

    names = {"https://tfhub.dev/google/universal-sentence-encoder/4": "USE",
             "all-mpnet-base-v2": "all-mpnet-base-v2",
            "multi-qa-mpnet-base-dot-v1": "multi-qa-mpnet-base-dot-v1",
             "all-distilroberta-v1": "all-distilroberta-v1",
             "sentence-transformers/sentence-t5-base": "ST5",
             "Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit": "SGPT"}

    sim_features = []
    for idx_1, feature_1 in enumerate(features):
        for idx_2, feature_2 in enumerate(features[idx_1+1:]):
            combo = set([feature_1, feature_2])
            if combo not in sim_features:
                sim_features.append(combo)
            for idx_3, feature_3 in enumerate(features[idx_1+2:]):
                combo = set([feature_1, feature_2, feature_3])
                if combo not in sim_features:
                    sim_features.append(combo)
                for idx_4, feature_4 in enumerate(features[idx_1+3:]):
                    combo = set([feature_1, feature_2, feature_3, feature_4])
                    if combo not in sim_features:
                        sim_features.append(combo)
                    for idx_5, feature_5 in enumerate(features[idx_1+4:]):
                        combo = set([feature_1, feature_2, feature_3, feature_4, feature_5])
                        if combo not in sim_features:
                            sim_features.append(combo)
                        for idx_6, feature_6 in enumerate(features[idx_1 + 5:]):
                            combo = set([feature_1, feature_2, feature_3, feature_4, feature_5, feature_6])
                            if combo not in sim_features:
                                sim_features.append(combo)

    sim_features = sorted([list(set) for set in sim_features], key=len)
    variants = []
    for feature in sim_features:
        if len(feature) == 2:
            variants.append(names[feature[0]] + " + " +names[feature[1]])
        if len(feature) == 3:
            variants.append(names[feature[0]] + " + " + names[feature[1]] + " + " + names[feature[2]])
        if len(feature) == 4:
            variants.append(names[feature[0]] + " + " + names[feature[1]] + " + " + names[feature[2]] +
                                             " + " + names[feature[3]])
        if len(feature) == 5:
            variants.append(names[feature[0]] + " + " + names[feature[1]] +
                                                 " + " + names[feature[2]] +
                                                 " + " + names[feature[3]] + " + " +
                                                 names[feature[4]])
        if len(feature) == 6:
            variants.append(names[feature[0]] + " + " + names[feature[1]] +
                         " + " + names[feature[2]] +
                         " + " + names[feature[3]] + " + " +
                         names[feature[4]] + " + " + names[feature[5]])

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

        for sim_idx, sim_feature in enumerate(sim_features):

            sim_feature_name = variants[sim_idx]
            print(sim_feature_name)

            n_feature = len(sim_feature)
            if n_feature == 2:

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
                                 '-sentence_embedding_models', sim_feature[0], sim_feature[1],
                                 '-lexical_similarity_measures', "similar_words_ratio"
                                 ])

            elif n_feature == 3:
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
                                 '-sentence_embedding_models', sim_feature[0], sim_feature[1], sim_feature[2],
                                 '-lexical_similarity_measures', "similar_words_ratio"
                                 ])

            if n_feature == 4:
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
                                 '-sentence_embedding_models', sim_feature[0], sim_feature[1], sim_feature[2], sim_feature[3],
                                 '-lexical_similarity_measures', "similar_words_ratio"
                                 ])

            if n_feature == 5:
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
                                 '-sentence_embedding_models', sim_feature[0], sim_feature[1], sim_feature[2],
                                 sim_feature[3], sim_feature[4],
                                 '-lexical_similarity_measures', "similar_words_ratio"
                                 ])
            if n_feature == 6:
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
                                 '-sentence_embedding_models', sim_feature[0], sim_feature[1], sim_feature[2],
                                 sim_feature[3], sim_feature[4], sim_feature[5],
                                 '-lexical_similarity_measures', "similar_words_ratio"
                                 ])

            data_name_pred = dataset_path + "/pred_qrels.tsv"

            subprocess.call(["python", repo_path + "/evaluation/scorer/evaluator.py",
                             data_name,
                             data_name_gold,
                             data_name_pred])

            score = float(get_map_5(repo_path + "/data/" + data_name))

            table_dict[column].append(score)
            all_dataset_scores[sim_feature_name].append(score)

    for variant, scores in all_dataset_scores.items():
        average_score = round(np.mean(scores), 3)
        table_dict["all"].append(average_score)

    scores_df = pd.DataFrame.from_dict(table_dict, dtype=str)
    column_format = "l"
    for _ in range(len(columns)):
        column_format = column_format + "|c"

    with open(dir_of_file + "/output/3_TABLE_best_combination.txt", 'w') as f:
        print(scores_df.style.format_index(axis=1, formatter="{}".format).hide(
            axis=0).format(precision=3).to_latex(column_format=column_format, position="!htbp",
                                                 label="table:combo",
                                                 caption="Highest Scores for Combinations of Semantic Features and Lexical Feature.",
                                                 multirow_align="t", multicol_align="r"), file=f)

    print(scores_df.style.format_index(axis=1, formatter="{}".format).hide(
        axis=0).format(precision=3).to_latex(column_format=column_format, position="!htbp",
                                             label="table:combo",
                                             caption="Highest Scores for Combinations of Semantic Features and Lexical Feature..",
                                             multirow_align="t", multicol_align="r"))



if __name__ == "__main__":
    run()