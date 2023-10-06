import subprocess
import numpy as np
from os.path import realpath, dirname

import pandas as pd

from src.create_similarity_features.lexical_similarity import get_lexical_similarity
from src.create_similarity_features.string_similarity import match_sequences, levenshtein_sim
from src.utils import get_queries, get_targets


def run():

    filepath = realpath(__file__)
    dir_of_file = dirname(filepath)
    parent_dir_of_file = dirname(dir_of_file)
    parents_parent_dir_of_file = dirname(parent_dir_of_file)
    repo_path = parents_parent_dir_of_file
    data_path = repo_path + "/data/"

    data_names = ["clef_2020_checkthat_2_english",
                  "clef_2021_checkthat_2a_english",
                  "clef_2022_checkthat_2a_english",
                  "clef_2021_checkthat_2b_english",
                  "clef_2022_checkthat_2b_english"]

    columns = ["\makecell{ct 2020\\\\tweets}", "\makecell{ct 2021\\\\tweets}", "\makecell{ct 2022\\\\tweets}",
               "\makecell{ct 2021\\\political}", "\makecell{ct 2022\\\political}"]

    table_dict = {}
    table_dict["dataset"] = ['\# queries', '\# targets',
                             'avg. query length',
                             'avg. target length',
                             'avg. lexical overlap',
                             'avg. sequence matches',
                             'avg. levenshtein similarity']

    for column in columns:
        table_dict[column] = []

    for idx, data_name in enumerate(data_names):

        print(data_name)
        column = columns[idx]

        dataset_path = data_path + data_name
        data_name_queries = dataset_path + "/queries.tsv"
        data_name_targets = dataset_path + "/corpus"
        data_name_gold = dataset_path + "/gold.tsv"

        subprocess.call(["python", repo_path + "/evaluation/evaluate_datasets.py",
                         data_name,
                         data_name_queries,
                         data_name_targets
                         ])

        dataset_features = pd.read_csv(data_path + data_name + "/dataset_analysis.tsv", sep='\t')
        table_dict[column].append(int(dataset_features['\# queries']))
        table_dict[column].append(int(dataset_features['\# targets']))
        table_dict[column].append(int(dataset_features['\makecell{avg.\\\query\\\length}']))
        table_dict[column].append(int(dataset_features['\makecell{avg.\\target\\\length}']))

        gold_df = pd.read_csv(data_name_gold, sep='\t', dtype=str, names=["query_id", "0", "target_id", "score"])
        query_ids = gold_df["query_id"]
        target_ids = gold_df["target_id"]
        id_pairs = zip(query_ids, target_ids)
        queries = get_queries(data_name_queries)
        targets = get_targets(data_name_targets)
        id_text_pairs = {pair: (queries[pair[0]], targets[pair[1]]) for pair in id_pairs}

        lexical_similarity_scores = [get_lexical_similarity(text_pair[0], text_pair[1]) for text_pair in id_text_pairs.values()]
        avg_lexical_similarity_score = int(np.mean(lexical_similarity_scores))
        sequence_similarity_scores = [match_sequences(text_pair[0], text_pair[1]) for text_pair in id_text_pairs.values()]
        avg_sequence_similarity_score = int(np.mean(sequence_similarity_scores))
        levenshtein_similarity_scores = [levenshtein_sim(text_pair[0], text_pair[1]) for text_pair in id_text_pairs.values()]
        avg_levenshtein_similarity_score = int(np.mean(levenshtein_similarity_scores))

        table_dict[column].append(avg_lexical_similarity_score)
        table_dict[column].append(avg_sequence_similarity_score)
        table_dict[column].append(avg_levenshtein_similarity_score)

    comparison_df = pd.DataFrame.from_dict(table_dict, dtype=str)

    column_format = "l"
    for _ in range(len(columns)):
        column_format = column_format + "|c"

    with open("output/2_TABLE_dataset_comparison.txt", 'w') as f:
        print(comparison_df.style.format_index(axis=1, formatter="${}$".format).
              hide(axis=0).format(precision=3).to_latex(column_format=column_format, position="!htbp",
                                                    label="table:2_dataset_comparison",
                                                    caption="\\textbf{Comparison of Verified Claim Retrieval Datasets.} \\textit{Lexical overlap}, \\textit{sequence matches} and \\textit{levenshtein} refer to the lexical similarity and the similarity of string spans and characters of two text sequences. Their computation is explained in detail in \\ref{similarity_types}. Two identical string sequences would have a value of 100 in all those categories.",
                                                    multirow_align="t", multicol_align="r"), file=f)
    print(comparison_df.style.format_index(axis=1, formatter="${}$".format).
          hide(axis=0).format(precision=3).to_latex(column_format=column_format, position="!htbp",
                                                    label="table:2_dataset_comparison",
                                                    caption="\\textbf{Comparison of Verified Claim Retrieval Datasets.} \\textit{Lexical overlap}, \\textit{sequence matches} and \\textit{levenshtein} refer to the lexical similarity and the similarity of string spans and characters of two text sequences. Their computation is explained in detail in \\ref{similarity_types}. Two identical string sequences would have a value of 100 in all those categories.",
                                                    multirow_align="t", multicol_align="r"))


if __name__ == "__main__":
    run()