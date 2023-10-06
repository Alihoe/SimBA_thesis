from random import random, seed, choice
from os.path import realpath, dirname
import pandas as pd
from src.utils import get_queries, get_targets


def run():

    seed(a=1234567)

    filepath = realpath(__file__)
    dir_of_file = dirname(filepath)
    parent_dir_of_file = dirname(dir_of_file)
    parents_parent_dir_of_file = dirname(parent_dir_of_file)
    repo_path = parents_parent_dir_of_file
    data_path = repo_path + "/data/"

    data_names = ["clef_2022_checkthat_2a_english", "clef_2022_checkthat_2b_english"]

    input_claims = {}
    verified_claims = {}

    for data_name in data_names:

        dataset_path = data_path + data_name
        data_name_queries = dataset_path + "/queries.tsv"
        data_name_targets = dataset_path + "/corpus"
        data_name_gold = dataset_path + "/gold.tsv"

        gold_df = pd.read_csv(data_name_gold, sep='\t', dtype=str, names=["query_id", "0", "target_id", "score"])
        query_ids = gold_df["query_id"]
        target_ids = gold_df["target_id"]
        id_pairs = zip(query_ids, target_ids)
        queries = get_queries(data_name_queries)
        targets = get_targets(data_name_targets)
        id_text_pairs = {pair: (queries[pair[0]], targets[pair[1]]) for pair in id_pairs}
        rand_key = choice(list(id_text_pairs))
        input_claims[data_name] = id_text_pairs[rand_key][0]
        verified_claims[data_name] = id_text_pairs[rand_key][1]
        print(id_text_pairs[rand_key][0])
        print(id_text_pairs[rand_key][1])

    with open(dir_of_file + "/output/2_FIGURE_dataset_examples.txt", 'w', encoding="utf-8") as f:
        print("\\begin{figure}[hbtp!]", file=f)
        print("\\begin{quote}", file=f)
        print("\\textbf{Input Claim:} \\textit{" + input_claims["clef_2022_checkthat_2a_english"] + "}\\\\", file=f)
        print("\\textbf{Verified Claim:} \\textit{" + verified_claims["clef_2022_checkthat_2a_english"] + "}\\\\", file=f)
        print("\end{quote}", file=f)
        print("\caption{Example of true pair in \\textit{ct 2022 tweets}. Input claims do not necessarily need to be supported by the verified claim to be a match and the verified claim does not need to be true.", file=f)
        print("\end{figure}", file=f)
        print("", file=f)
        print("\\begin{figure}[hbtp!]", file=f)
        print("\\begin{quote}", file=f)
        print("\\textbf{Input Claim:} \\textit{" + input_claims["clef_2022_checkthat_2b_english"] + "}\\\\", file=f)
        print("\\textbf{Verified Claim:} \\textit{" + verified_claims["clef_2022_checkthat_2b_english"] + "}\\\\", file=f)
        print("\end{quote}", file=f)
        print("\caption{Example of true pair in \\textit{ct 2022 politics}.", file=f)
        print("\end{figure}", file=f)


if __name__ == "__main__":
    run()