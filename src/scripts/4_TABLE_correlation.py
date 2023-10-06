import os
import subprocess
from os.path import realpath, dirname
import pandas as pd
import numpy as np
from evaluation.utils import get_map_5


def run():

    filepath = realpath(__file__)
    dir_of_file = dirname(filepath)
    parent_dir_of_file = dirname(dir_of_file)
    parent_parent_dir_of_file = dirname(parent_dir_of_file)
    repo_path = parent_parent_dir_of_file
    data_path = repo_path + "/data/"

    similarity_features = [
                            ('-sentence_embedding_models', "infersent"),
                            ('-sentence_embedding_models', "https://tfhub.dev/google/universal-sentence-encoder/4"),
                            ('-sentence_embedding_models', "all-mpnet-base-v2"),
                            ('-sentence_embedding_models', "multi-qa-mpnet-base-dot-v1"),
                            ('-sentence_embedding_models', "all-distilroberta-v1"),
                            ('-sentence_embedding_models', "princeton-nlp/unsup-simcse-roberta-large"),
                            ('-sentence_embedding_models', "princeton-nlp/sup-simcse-roberta-large"),
                            ('-sentence_embedding_models', "sentence-transformers/sentence-t5-base"),
                            ('-sentence_embedding_models', "Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit"),
                            ('-string_similarity_measures', "levenshtein"),
                            ('-string_similarity_measures', "sequence_matching"),
                            ('-lexical_similarity_measures', "similar_words_ratio"),
                            ('-referential_similarity_measures', "ne_similarity"),
                            ('-referential_similarity_measures', "synonym_similarity"),
                            ]

    skip = 0


    for own_idx, similarity_feature in enumerate(similarity_features[skip:]):

        table_sim_features = [
            "Infersent GloVe",
            "USE",
            "all-mpnet-base-v2",
            "multi-qa-mpnet-base-dot-v1",
            "all-distil-roberta-v1",
            "Unsup Sim CSE",
            "Sup Sim CSE",
            "ST5",
            "SGPT",
            "Levenshtein Similarity",
            "Sequence Matching",
            "Similar Words Ratio",
            "NE Similarity",
            "Synonym Similarity"]

        print(similarity_feature)
        own_idx = own_idx + skip

        similarity_feature_category = similarity_feature[0]
        similarity_feature_name = similarity_feature[1]
        cleaned_similarity_feature_name = str(similarity_feature_name).replace("/", "_").replace(":", "_").replace(".", "_")
        output_path = dir_of_file + "/output/4_" + cleaned_similarity_feature_name+".txt"

        data_names = ["clef_2020_checkthat_2_english", "clef_2021_checkthat_2a_english",
                      "clef_2022_checkthat_2a_english",
                      "clef_2021_checkthat_2b_english", "clef_2022_checkthat_2b_english",
                      ]

        performances_for_datasets = {}
        performances_two_similarity_scores_for_datasets = {}

        correlations = []

        for data_name in data_names:

            dataset_path = repo_path + "/data/" + data_name
            data_name_queries = dataset_path + "/queries.tsv"
            data_name_targets = dataset_path + "/corpus"
            data_name_gold = dataset_path + "/gold.tsv"

            # See how this score correlates with other similarity scores for this document

            correlation_file = dataset_path + "/" + data_name + "_correlation.tsv"

            subprocess.call(["python",
                             repo_path + "/src/candidate_retrieval/retrieval.py",
                             data_name_queries,
                             data_name_targets,
                             data_name,
                             data_name,
                             "cosine",
                             "50",
                             "--correlation_analysis",
                             '-sentence_embedding_models',
                             "infersent",
                             "https://tfhub.dev/google/universal-sentence-encoder/4",
                             "all-mpnet-base-v2",
                             "multi-qa-mpnet-base-dot-v1",
                             "all-distilroberta-v1",
                             "princeton-nlp/unsup-simcse-roberta-large",
                             "princeton-nlp/sup-simcse-roberta-large",
                             "sentence-transformers/sentence-t5-base",
                             "Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
                             '-string_similarity_measures', "levenshtein", "sequence_matching",
                             '-lexical_similarity_measures', "similar_words_ratio",
                             '-referential_similarity_measures', "ne_similarity", "synonym_similarity"])

            this_dataset_correlations = pd.read_csv(dataset_path + "/" + data_name + "_correlation.tsv", sep='\t')
            this_sim_score_correlations = this_dataset_correlations[similarity_feature_name]
            correlations.append(this_sim_score_correlations)

            # Collect predictions for this dataset with specific similarity score

            if "/" or ":" or "." in str(similarity_feature_name):
                similarity_feature_path_name = str(similarity_feature_name).replace("/", "_").replace(":", "_").replace(".", "_")

            data_name_pred = dataset_path + "/" + similarity_feature_path_name + "/pred_qrels.tsv"

            if not os.path.isfile(data_name_pred):
                subprocess.call(["python",
                                 repo_path + "/src/re_ranking/re_ranking.py",
                                 data_name_queries,
                                 data_name_targets,
                                 data_name,
                                 data_name + "/" + similarity_feature_path_name,
                                 data_name + "/" + similarity_feature_path_name,
                                 "cosine",
                                 "50",
                                 '--ranking_only',
                                 similarity_feature_category, similarity_feature_name])

            data_name_results = dataset_path + "/" + similarity_feature_path_name + "/results.tsv"

            if not os.path.isfile(data_name_results):

                print("Evaluation Scores for dataset " + data_name + "/" + similarity_feature_path_name)
                subprocess.call(["python", repo_path + "/evaluation/scorer/evaluator.py",
                                 data_name + "/" + similarity_feature_path_name,
                                 data_name_gold,
                                 data_name_pred])

            map_5 = get_map_5(repo_path + "/data/" + data_name + "/" + similarity_feature_path_name)
            performances_for_datasets[data_name] = float(map_5)

            # Look how similarity score performs with another similarity score for this dataset

            performances_two_similarity_scores_for_datasets[data_name] = {}
            all_other_similarity_features = []

            for second_similarity_feature in similarity_features:
                if second_similarity_feature != similarity_feature:
                    second_similarity_feature_category = second_similarity_feature[0]
                    second_similarity_feature_name = second_similarity_feature[1]

                    if "/" or ":" or "." in str(second_similarity_feature_name):
                        second_similarity_feature_path_name = str(second_similarity_feature_name).replace("/", "_").replace(":", "_")\
                            .replace(".", "_")

                    all_other_similarity_features.append(second_similarity_feature_name)

                    this_data_name = data_name + "/" + similarity_feature_path_name + "/" + second_similarity_feature_path_name

                    data_name_pred = data_path + this_data_name + "/pred_qrels.tsv"

                    if not os.path.isfile(data_name_pred):

                        if second_similarity_feature_category != similarity_feature_category:
                            subprocess.call(["python",
                                             repo_path + "/src/re_ranking/re_ranking.py",
                                             data_name_queries,
                                             data_name_targets,
                                             data_name,
                                             this_data_name,
                                             this_data_name,
                                             "cosine",
                                             "50",
                                             '--ranking_only',
                                             similarity_feature_category, similarity_feature_name,
                                             second_similarity_feature_category, second_similarity_feature_name])
                        else:
                            subprocess.call(["python",
                                             repo_path + "/src/re_ranking/re_ranking.py",
                                             data_name_queries,
                                             data_name_targets,
                                             data_name,
                                             this_data_name,
                                             this_data_name,
                                             "cosine",
                                             "50",
                                             '--ranking_only',
                                             similarity_feature_category, similarity_feature_name, second_similarity_feature_name])

                    data_name_results = data_path + this_data_name + "/results.tsv"

                    if not os.path.isfile(data_name_results):
                        print("Evaluation Scores for dataset " + this_data_name)
                        subprocess.call(["python", repo_path + "/evaluation/scorer/evaluator.py",
                                         this_data_name,
                                         data_name_gold,
                                         data_name_pred])

                    map_5 = get_map_5(data_path + this_data_name)
                    performances_two_similarity_scores_for_datasets[data_name][second_similarity_feature_name] = float(map_5)

        sorted_performances_for_datasets = dict(sorted(performances_for_datasets.items(), key=lambda x:x[1], reverse=True))

        mean_correlations = np.mean(correlations, axis=0).tolist()
        del mean_correlations[own_idx]

        means_of_perf_diff_second_sim_feature = []
        sim_features_perform_diff = {}

        for sim_feature in all_other_similarity_features:
            sim_features_perform_diff[sim_feature] = []

        for dataset, performances in performances_two_similarity_scores_for_datasets.items():
            for sim_feature in all_other_similarity_features:
                sim_features_perform_diff[sim_feature].append(performances[sim_feature] - sorted_performances_for_datasets[dataset])

        for sim_feature in all_other_similarity_features:
            means_of_perf_diff_second_sim_feature.append(np.mean(sim_features_perform_diff[sim_feature], axis=0))

        ensemble_df = pd.DataFrame(columns=['Similarity Feature', 'Correlation', 'Improvement'])
        table_sim_features.pop(own_idx)
        ensemble_df['Similarity Feature'] = table_sim_features
        ensemble_df['Correlation'] = mean_correlations
        ensemble_df['Improvement'] = means_of_perf_diff_second_sim_feature

        print(ensemble_df.style.format_index(axis=1, formatter="${}$".format).hide(
            axis=0).format(precision=3).to_latex(column_format="c|c|c", position="!htbp",
                                                 label="table:ensemble_" + similarity_feature_name,
                                                 caption="Correlation of "+similarity_feature_name + " with other features.",
                                                 multirow_align="t", multicol_align="r"))
        with open("output/4_TABLE_correlation_" + similarity_feature_path_name + ".txt", 'w') as f:
            print(ensemble_df.style.format_index(axis=1, formatter="${}$".format).hide(
                axis=0).format(precision=3).to_latex(column_format="c|c|c", position="h",
                                                     label="table:ensemble_" + similarity_feature_name,
                                                     caption="Correlation of "+similarity_feature_name + " with other features.",
                                                     multirow_align="t", multicol_align="r"), file=f)


if __name__ == "__main__":
    run()