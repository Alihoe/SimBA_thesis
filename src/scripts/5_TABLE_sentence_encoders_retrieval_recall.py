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
                  "clef_2022_checkthat_2b_english"
                  ]

    columns = ["ct 2020 2a",
               "ct 2021 2a",
               "ct 2022 2a",
               "ct 2021 2b",
               "ct 2022 2b",
               "all"]

    features = [
               "infersent",
               "https://tfhub.dev/google/universal-sentence-encoder/4",
               "all-mpnet-base-v2",
               "multi-qa-mpnet-base-dot-v1",
               "all-distilroberta-v1",
               "princeton-nlp/unsup-simcse-roberta-large",
               "princeton-nlp/sup-simcse-roberta-large",
                "sentence-transformers/sentence-t5-base",
                "Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit"]

    names = {"infersent": "InferSent",
             "https://tfhub.dev/google/universal-sentence-encoder/4": "USE",
             "all-mpnet-base-v2": "all-mpnet-base-v2",
             "all-distilroberta-v1": "all-distilroberta-v1",
             "multi-qa-mpnet-base-dot-v1": "multi-qa-mpnet-base-dot-v1",
             "princeton-nlp/unsup-simcse-roberta-large": "UnsupSimCSE",
             "princeton-nlp/sup-simcse-roberta-large": "SupSimCSE",
             "sentence-transformers/sentence-t5-base": "ST5",
             "Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit": "SGPT"}

    sim_features = []
    for idx_1, feature_1 in enumerate(features):
        for idx_2, feature_2 in enumerate(features[idx_1+1:]):
            combo = set([feature_1, feature_2])
            if combo not in sim_features:
                sim_features.append(combo)
    sim_features = [list(set) for set in sim_features]
    features = [[f] for f in features]
    features.extend(sim_features)

    encoders = []
    for feature in features:
        if len(feature) == 1:
            encoders.append(names[feature[0]])
        elif len(feature) == 2:
            encoders.append("avg.: " + names[feature[0]] + " + " + names[feature[1]])
            encoders.append("union: " + names[feature[0]] + " + " + names[feature[1]])

    table_dict = {"sentence encoder": encoders}

    for similarity_feature in features:
        print(similarity_feature)
        print(len(similarity_feature))

    all_dataset_scores = {}
    for encoder in encoders:
        all_dataset_scores[encoder] = []

    for column in columns:
        table_dict[column] = []

    for idx, data_name in enumerate(data_names):

        column = columns[idx]

        print(data_name)

        dataset_path = repo_path + "/data/" + data_name
        data_name_queries = dataset_path + "/queries.tsv"
        data_name_targets = dataset_path + "/corpus"
        data_name_gold = dataset_path + "/gold.tsv"

        own_idx = 0
        for similarity_feature in features:
            encoder = encoders[own_idx]
            own_idx = own_idx + 1

            if len(similarity_feature) == 2:
                similarity_feature_1 = similarity_feature[0]
                similarity_feature_2 = similarity_feature[1]

                if "/" or ":" or "." in str(similarity_feature_1):
                    similarity_feature_1 = str(similarity_feature_1).replace("/", "_").replace(":", "_").replace(".", "_")

                if "/" or ":" or "." in str(similarity_feature_2):
                    similarity_feature_2 = str(similarity_feature_2).replace("/", "_").replace(":", "_").replace(".", "_")

                subprocess.call(["python",
                                 repo_path + "/src/candidate_retrieval/retrieval.py",
                                 data_name_queries,
                                 data_name_targets,
                                 data_name,
                                 data_name + "/" + similarity_feature_1 + "+" + similarity_feature_2,
                                 "braycurtis",
                                 "50",
                                 '-sentence_embedding_models', similarity_feature_1, similarity_feature_2
                                 ])

                subprocess.call(["python", repo_path + "/evaluation/scorer/recall_evaluator.py",
                                 data_name + "/" + similarity_feature_1 + "+" + similarity_feature_2,
                                 data_name_gold])

                recall = float(get_recall(repo_path + "/data/" + data_name + "/" + similarity_feature_1 + "+" + similarity_feature_2))

                table_dict[column].append(recall)
                all_dataset_scores[encoder].append(recall)

                encoder = encoders[own_idx]
                own_idx = own_idx + 1

                subprocess.call(["python",
                                 repo_path + "/src/candidate_retrieval/retrieval.py",
                                 data_name_queries,
                                 data_name_targets,
                                 data_name,
                                 data_name + "/" + similarity_feature_1 + "&" + similarity_feature_2,
                                 "braycurtis",
                                 "50",
                                 '--union_of_top_k_per_feature',
                                 '-sentence_embedding_models', similarity_feature_1, similarity_feature_2
                                 ])

                subprocess.call(["python", repo_path + "/evaluation/scorer/recall_evaluator.py",
                                 data_name + "/" + similarity_feature_1 + "&" + similarity_feature_2,
                                 data_name_gold])

                recall = float(get_recall(repo_path + "/data/" + data_name + "/" + similarity_feature_1 + "&" + similarity_feature_2))

                table_dict[column].append(recall)
                all_dataset_scores[encoder].append(recall)

            else:

                similarity_feature = similarity_feature[0]

                if "/" or ":" or "." in str(similarity_feature):
                    similarity_feature = str(similarity_feature).replace("/", "_").replace(":", "_").replace(".", "_")
                else:
                    similarity_feature = similarity_feature

                subprocess.call(["python",
                                 repo_path + "/src/candidate_retrieval/retrieval.py",
                                 data_name_queries,
                                 data_name_targets,
                                 data_name,
                                 data_name + "/" + similarity_feature,
                                 "braycurtis",
                                 "50",
                                 '-sentence_embedding_models', similarity_feature
                                 ])

                subprocess.call(["python", repo_path + "/evaluation/scorer/recall_evaluator.py",
                                 data_name + "/" + similarity_feature,
                                 data_name_gold])

                recall = float(get_recall(repo_path + "/data/" + data_name + "/" + similarity_feature))

                # if recall > highest_recall:
                #     highest_recall = recall

                table_dict[column].append(recall)
                all_dataset_scores[encoder].append(recall)


    all_scores = []
    for encoder, scores in all_dataset_scores.items():
        average_score = round(np.mean(scores), 3)
        all_scores.append(average_score)

    table_dict["all"] = all_scores

    print(table_dict)

    scores_df = pd.DataFrame.from_dict(table_dict, dtype=str)

    column_format = "l"
    for _ in range(len(columns)):
        column_format = column_format + "|c"

    with open(dir_of_file + "/output/3_TABLE_sentence_encoder_retrieval_recall.txt", 'w') as f:
        print(scores_df.style.format_index(axis=1, formatter="{}".format).hide(
            axis=0).format(precision=3).to_latex(column_format=column_format, position="!htbp",
                                                 label="table:sentence_encoder_retrieval_recall",
                                                 caption="Recall after Using Sentence Encoders for Retrieval.",
                                                 multirow_align="t", multicol_align="r"), file=f)

    print(scores_df.style.format_index(axis=1, formatter="{}".format).hide(
        axis=0).format(precision=3).to_latex(column_format=column_format, position="!htbp",
                                             label="table:sentence_encoder_retrieval_recall",
                                             caption="Recall after Using Sentence Encoders for Retrieval.",
                                             multirow_align="t", multicol_align="r"))


if __name__ == "__main__":
    run()