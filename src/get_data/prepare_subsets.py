import os
import pathlib
import shutil
from os.path import realpath, dirname

from src.create_similarity_features.lexical_similarity import tokenize_and_filter_out_stop_words
from src.create_similarity_features.referential_similarity import get_named_entities_of_sentence, get_text_synonyms
from src.create_similarity_features.string_similarity import match_sequences, levenshtein_sim
from src.get_data import DATA_PATH
import pandas as pd

from src.utils import get_queries, get_targets

filepath = realpath(__file__)
dir_of_file = dirname(filepath)
parent_dir_of_file = dirname(dir_of_file)
parent_parent_dir_of_file = dirname(parent_dir_of_file)
parents_parent_parent_dir_of_file = dirname(parent_parent_dir_of_file)
repo_path = parents_parent_parent_dir_of_file


def group_dataset(dataset_name):
    gold_path = DATA_PATH + dataset_name + "/gold.tsv"
    query_path = DATA_PATH + dataset_name + "/queries.tsv"
    target_path = DATA_PATH + dataset_name + "/corpus"

    gold_df = pd.read_csv(gold_path, sep='\t', dtype=str, names=["query_id", "0", "target_id", "score"])
    query_df = pd.read_csv(query_path, sep='\t', dtype=str, names=["query_id", "query_text"])
    target_df_orig = pd.read_csv(target_path, sep='\t', dtype=str, names=["target_id", "target_text_1"])#, "target_text_2"])
    #print(target_df_orig)
    target_df = pd.DataFrame(columns=['target_id', 'target_text'])
    target_df["target_id"] = target_df_orig["target_id"]
    target_df["target_text"] = target_df_orig["target_text_1"]#target_df_orig.target_text_1.str.cat(target_df_orig.target_text_2, sep=" ")
    gold_df = gold_df.drop_duplicates()
    gold_text_df = gold_df[["query_id", "target_id"]].merge(query_df, on="query_id", how="left").merge(target_df, on="target_id", how="left")
    queries = gold_text_df["query_text"].tolist()
    targets = list(map(str, gold_text_df["target_text"].tolist()))
    target_ids = gold_text_df["target_id"].tolist()

    #gold_text_df.loc[gold_text_df["target_text"]]

    query_tokens = [set(tokenize_and_filter_out_stop_words(query)) for query in queries]
    target_tokens = [set(tokenize_and_filter_out_stop_words(target)) for target in targets]
    lexical_overlaps = [len(query_tokens[i].intersection(target_tokens[i])) for i in range(len(gold_text_df))]
    gold_text_df["lexical_similarity"] = lexical_overlaps

    gold_text_df = gold_text_df.sort_values("lexical_similarity")
    scores = [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 100]
    for idx, score in enumerate(scores[:-1]):
        group_df = gold_text_df[(gold_text_df["lexical_similarity"] <= scores[idx + 1]) & (gold_text_df["lexical_similarity"] > score)].copy()
        group_df["0"] = str(0)
        group_df["score"] = str(1)
        group_gold_df = group_df[["query_id", "0", "target_id", "score"]]
        group_query_df = group_df[["query_id", "query_text"]]
        group_query_df = group_query_df.drop_duplicates()
        lexical_output_path = DATA_PATH + "lexical/" + dataset_name + "/" + str(score)
        pathlib.Path(lexical_output_path).mkdir(parents=True, exist_ok=True)
        lexical_gold_path = lexical_output_path + "/gold.tsv"
        lexical_query_path = lexical_output_path + "/queries.tsv"
        group_gold_df.to_csv(lexical_gold_path, sep='\t', header=False, index=False)
        group_query_df.to_csv(lexical_query_path, sep='\t', header=False, index=False)



def analyse_subsets(dataset_name):
    similarity_features = ["ne"]#["levenshtein", "sequence_matching", "synonym", "lexical"]
    features_dict = {}
    for feature in similarity_features:
        output_path = DATA_PATH + feature + "/" + dataset_name
        n_queries = []
        for file in os.listdir(output_path):
            gold_path = output_path + "/" + file + "/gold.tsv"
            gold_df = pd.read_csv(gold_path, sep='\t', dtype=str, names=["query_id", "0", "target_id", "score"])
            n_queries.append(len(gold_df))
        features_dict[feature] = n_queries
    return features_dict


def analyse_subsets_lexical(dataset_name):
    features_dict = {}
    output_path = DATA_PATH + "lexical/" + dataset_name
    for file in os.listdir(output_path):
        gold_path = output_path + "/" + file + "/gold.tsv"
        gold_df = pd.read_csv(gold_path, sep='\t', dtype=str, names=["query_id", "0", "target_id", "score"])
        features_dict[file] = len(gold_df)
    return features_dict


def print_subset_examples_lexical():

    dataset_names = ["clef_2020_checkthat_2_english",
                     "clef_2021_checkthat_2a_english",
                     "clef_2022_checkthat_2a_english",
                     "clef_2021_checkthat_2b_english",
                     "clef_2022_checkthat_2b_english"]

    input_claims_highest_lexical = {}
    verified_claims_highest_lexical = {}
    
    input_claims_lowest_lexical = {}
    verified_claims_lowest_lexical = {}

    for dataset_name in dataset_names:

        query_path = DATA_PATH + dataset_name + "/queries.tsv"
        target_path = DATA_PATH + dataset_name + "/corpus"
        
        lexical_output_path = DATA_PATH + "lexical/" + dataset_name
        highest_lexical_output_path = lexical_output_path + "/8"
        lowest_lexical_output_path = lexical_output_path + "/-1"
        
        highest_lexical_gold_df = pd.read_csv(highest_lexical_output_path + "/gold.tsv", sep='\t', dtype=str, names=["query_id", "0", "target_id", "score"])
        highest_lexical_gold_sample = highest_lexical_gold_df.sample(1, random_state=2).values[0]
        highest_lex_query_id = highest_lexical_gold_sample[0]
        highest_lex_target_id = highest_lexical_gold_sample[2]
        highest_lex_query_text = get_queries(query_path)[highest_lex_query_id]
        highest_lex_target_text = get_targets(target_path)[highest_lex_target_id]
        input_claims_highest_lexical[dataset_name] = highest_lex_query_text
        verified_claims_highest_lexical[dataset_name] = highest_lex_target_text
        
        lowest_lexical_gold_df = pd.read_csv(lowest_lexical_output_path + "/gold.tsv", sep='\t', dtype=str, names=["query_id", "0", "target_id", "score"])
        lowest_lexical_gold_sample = lowest_lexical_gold_df.sample(1, random_state=2).values[0]
        lowest_lex_query_id = lowest_lexical_gold_sample[0]
        lowest_lex_target_id = lowest_lexical_gold_sample[2]
        lowest_lex_query_text = get_queries(query_path)[lowest_lex_query_id]
        lowest_lex_target_text = get_targets(target_path)[lowest_lex_target_id]
        input_claims_lowest_lexical[dataset_name] = lowest_lex_query_text
        verified_claims_lowest_lexical[dataset_name] = lowest_lex_target_text

    print("\\begin{figure}[hbtp!]")
    print("\\begin{quote}")

    print("\\textbf{Input Claim ct 2020 2a:} \\textit{"+input_claims_highest_lexical["clef_2020_checkthat_2_english"]+"}\\\\")
    print("\\textbf{Verified Claim ct 2020 2a:} \\textit{"+verified_claims_highest_lexical["clef_2020_checkthat_2_english"]+"}\\\\")
    print("\hline")

    print("\\textbf{Input Claim ct 2021 2a:} \\textit{"+input_claims_highest_lexical["clef_2021_checkthat_2a_english"]+"}\\\\")
    print("\\textbf{Verified Claim ct 2021 2a:} \\textit{"+verified_claims_highest_lexical["clef_2021_checkthat_2a_english"]+"}\\\\")
    print("\hline")

    print("\\textbf{Input Claim ct 2022 2a:} \\textit{"+input_claims_highest_lexical["clef_2022_checkthat_2a_english"]+"}\\\\")
    print("\\textbf{Verified Claim ct 2022 2a:} \\textit{"+verified_claims_highest_lexical["clef_2022_checkthat_2a_english"]+"}\\\\")
    print("\hline")

    print("\\textbf{Input Claim ct 2021 2b:} \\textit{"+input_claims_highest_lexical["clef_2021_checkthat_2b_english"]+"}\\\\")
    print("\\textbf{Verified Claim ct 2021 2b:} \\textit{"+verified_claims_highest_lexical["clef_2021_checkthat_2b_english"]+"}\\\\")
    print("\hline")

    print("\\textbf{Input Claim ct 2022 2b:} \\textit{"+input_claims_highest_lexical["clef_2022_checkthat_2b_english"]+"}\\\\")
    print("\\textbf{Verified Claim ct 2022 2b:} \\textit{"+verified_claims_highest_lexical["clef_2022_checkthat_2b_english"]+"}\\\\")
    print("\hline")

    print("\end{quote}")
    print("\caption{Example of true pairs with high lexical overlap.}")
    print("\end{figure}")

    print()

    print("\\begin{figure}[hbtp!]")
    print("\\begin{quote}")

    print("\\textbf{Input Claim ct 2020 2a:} \\textit{"+input_claims_lowest_lexical["clef_2020_checkthat_2_english"]+"}\\\\")
    print("\\textbf{Verified Claim ct 2020 2a:} \\textit{"+verified_claims_lowest_lexical["clef_2020_checkthat_2_english"]+"}\\\\")
    print("\hline")

    print("\\textbf{Input Claim ct 2021 2a:} \\textit{"+input_claims_lowest_lexical["clef_2021_checkthat_2a_english"]+"}\\\\")
    print("\\textbf{Verified Claim ct 2021 2a:} \\textit{"+verified_claims_lowest_lexical["clef_2021_checkthat_2a_english"]+"}\\\\")
    print("\hline")

    print("\\textbf{Input Claim ct 2022 2a:} \\textit{"+input_claims_lowest_lexical["clef_2022_checkthat_2a_english"]+"}\\\\")
    print("\\textbf{Verified Claim ct 2022 2a:} \\textit{"+verified_claims_lowest_lexical["clef_2022_checkthat_2a_english"]+"}\\\\")
    print("\hline")

    print("\\textbf{Input Claim ct 2021 2b:} \\textit{"+input_claims_lowest_lexical["clef_2021_checkthat_2b_english"]+"}\\\\")
    print("\\textbf{Verified Claim ct 2021 2b:} \\textit{"+verified_claims_lowest_lexical["clef_2021_checkthat_2b_english"]+"}\\\\")
    print("\hline")

    print("\\textbf{Input Claim ct 2022 2b:} \\textit{"+input_claims_lowest_lexical["clef_2022_checkthat_2b_english"]+"}\\\\")
    print("\\textbf{Verified Claim ct 2022 2b:} \\textit{"+verified_claims_lowest_lexical["clef_2022_checkthat_2b_english"]+"}\\\\")
    print("\hline")

    print("\end{quote}")
    print("\caption{Example of true pairs with no lexical overlap.}")
    print("\end{figure}")


def print_lowest_lexical_subset():

    dataset_names = ["clef_2020_checkthat_2_english",
                     "clef_2021_checkthat_2a_english",
                     "clef_2022_checkthat_2a_english",
                     "clef_2021_checkthat_2b_english",
                     "clef_2022_checkthat_2b_english"]

    input_claims_lowest_lexical = {}
    verified_claims_lowest_lexical = {}

    for dataset_name in dataset_names:
        query_path = DATA_PATH + dataset_name + "/queries.tsv"
        target_path = DATA_PATH + dataset_name + "/corpus"

        lexical_output_path = DATA_PATH + "lexical/" + dataset_name
        lowest_lexical_output_path = lexical_output_path + "/-1"

        lowest_lexical_gold_df = pd.read_csv(lowest_lexical_output_path + "/gold.tsv", sep='\t', dtype=str,
                                             names=["query_id", "0", "target_id", "score"])
        lowest_lex_query_ids = lowest_lexical_gold_df["query_id"].values
        lowest_lex_target_ids = lowest_lexical_gold_df["target_id"].values
        queries = get_queries(query_path)
        targets = get_targets(target_path)
        lowest_lex_query_texts = [queries[query_id] for query_id in lowest_lex_query_ids]
        lowest_lex_target_texts = [targets[target_id] for target_id in lowest_lex_target_ids]
        input_claims_lowest_lexical[dataset_name] = lowest_lex_query_texts
        verified_claims_lowest_lexical[dataset_name] = lowest_lex_target_texts
        for idx, query in enumerate(lowest_lex_query_texts):
            print(query)
            print(lowest_lex_target_texts[idx])
            print("---------------")

#print_lowest_lexical_subset()

#print_subset_examples_lexical()


# dataset_names = ["clef_2020_checkthat_2_english",
#                  "clef_2021_checkthat_2a_english",
#                  "clef_2022_checkthat_2a_english",
#                  "clef_2021_checkthat_2b_english",
#                  "clef_2022_checkthat_2b_english"]

dataset_names = ["climate-fever", "fever", "scifact"]

# shutil.rmtree(DATA_PATH + "levenshtein")
# shutil.rmtree(DATA_PATH + "sequence_matching")
# shutil.rmtree(DATA_PATH + "synonym")
# shutil.rmtree(DATA_PATH + "lexical")
# shutil.rmtree(DATA_PATH + "ne")
for dataset_name in dataset_names:
    group_dataset(dataset_name)
#
# for dataset_name in dataset_names:
#     print(analyse_subsets(dataset_name))




