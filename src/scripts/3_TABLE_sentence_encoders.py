import os
import subprocess
from os.path import realpath, dirname
import pandas as pd
import numpy as np
from evaluation.utils import get_ndcg_10, get_map_5


def run():

    filepath = realpath(__file__)
    dir_of_file = dirname(filepath)
    parent_dir_of_file = dirname(dir_of_file)
    parent_parent_dir_of_file = dirname(parent_dir_of_file)
    repo_path = parent_parent_dir_of_file

    similarity_features = [
                           "infersent",
                           "https://tfhub.dev/google/universal-sentence-encoder/4",
                           "all-mpnet-base-v2",
                           "multi-qa-mpnet-base-dot-v1",
                           "all-distilroberta-v1",
                           "princeton-nlp/unsup-simcse-roberta-large",
                           "princeton-nlp/sup-simcse-roberta-large",
                            "sentence-transformers/sentence-t5-base",
                            "Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit"
        ]

    data_names = [
        "clef_2020_checkthat_2_english",
                 "clef_2021_checkthat_2a_english",
                  "clef_2022_checkthat_2a_english",
                 "clef_2021_checkthat_2b_english",
                  "clef_2022_checkthat_2b_english"]

    columns = ["\makecell{ct 2020\\\\tweets}",
               "\makecell{ct 2021\\\\tweets}",
               "\makecell{ct 2022\\\\tweets}",
               "\makecell{ct 2021\\\political}",
               "\makecell{ct 2022\\\political}",
               "all"]

    table_dict = {"sentence encoder": [
        "Infersent GloVe",
        "USE",
        "all-mpnet-base-v2",
        " multi-qa-mpnet-base-dot-v1",
        "all-distil-roberta-v1",
        "UnsupSimCSE",
        "SupSimCSE",
        "ST5",
        "SGPT"
        ]}

    sentence_encoder_scores = {}
    for sim_feature in similarity_features:
        sentence_encoder_scores[sim_feature] = []

    for column in columns:
        table_dict[column] = []

    for idx, data_name in enumerate(data_names):

        column = columns[idx]

        print(data_name)

        dataset_path = repo_path + "/data/" + data_name
        data_name_queries = dataset_path + "/queries.tsv"
        data_name_targets = dataset_path + "/corpus"
        data_name_gold = dataset_path + "/gold.tsv"

        for own_idx, similarity_feature in enumerate(similarity_features):
            print(similarity_feature)

            if "/" or ":" or "." in str(similarity_feature):
                similarity_feature_path_name = str(similarity_feature).replace("/", "_").replace(":", "_").replace(".", "_")
            else:
                similarity_feature_path_name = similarity_feature

            data_name_pred = dataset_path + "/" + similarity_feature_path_name + "/pred_qrels.tsv"
            #
            # if not os.path.isfile(data_name_pred):
            subprocess.call(["python",
                             repo_path + "/src/re_ranking/re_ranking.py",
                             data_name_queries,
                             data_name_targets,
                             data_name,
                             data_name,
                             data_name + "/" + similarity_feature_path_name,
                             "cosine",
                             "50",
                             '--ranking_only',
                             '-sentence_embedding_models', similarity_feature])

            data_name_results = dataset_path + "/" + similarity_feature_path_name + "/results.tsv"

            # if not os.path.isfile(data_name_results):

            print("Evaluation Scores for dataset " + data_name + "/" + similarity_feature)
            subprocess.call(["python", repo_path + "/evaluation/scorer/evaluator.py",
                             data_name + "/" + similarity_feature_path_name,
                             data_name_gold,
                             data_name_pred])

            map_5 = get_map_5(repo_path + "/data/" + data_name + "/" + similarity_feature_path_name)
            table_dict[column].append(float(map_5))
            sentence_encoder_scores[similarity_feature].append(float(map_5))


    for variant, scores in sentence_encoder_scores.items():
        average_score = round(np.mean(scores), 3)
        table_dict["all"].append(average_score)


    scores_df = pd.DataFrame.from_dict(table_dict, dtype=str)

    column_format = "l"
    for _ in range(len(columns)):
        column_format = column_format + "|c"

    with open(dir_of_file + "/output/3_TABLE_sentence_encoder_scores.txt", 'w') as f:
        print(scores_df.style.format_index(axis=1, formatter="{}".format).hide(
            axis=0).format(precision=3).to_latex(column_format=column_format, position="!htbp",
                                                 label="table:sentence_encoder_scores",
                                                 caption="Scores of Sentence Encoders per Dataset.",
                                                 multirow_align="t", multicol_align="r"), file=f)

    print(scores_df.style.format_index(axis=1, formatter="{}".format).hide(
        axis=0).format(precision=3).to_latex(column_format=column_format, position="!htbp",
                                             label="table:sentence_encoder_scores",
                                             caption="Scores of Sentence Encoders per Dataset.",
                                             multirow_align="t", multicol_align="r"))


if __name__ == "__main__":
    run()