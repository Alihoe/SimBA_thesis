import os
import subprocess
from random import random, seed, choice
from os.path import realpath, dirname
import pandas as pd
from src.utils import get_queries, get_targets


def run():

    filepath = realpath(__file__)
    dir_of_file = dirname(filepath)
    parent_dir_of_file = dirname(dir_of_file)
    parents_parent_dir_of_file = dirname(parent_dir_of_file)
    repo_path = parents_parent_dir_of_file
    data_path = repo_path + "/data/"

    similarity_features = [
                           #"infersent",
                           #"https://tfhub.dev/google/universal-sentence-encoder/4",
                           "all-mpnet-base-v2",
                           "multi-qa-mpnet-base-dot-v1",
                           #"all-distilroberta-v1",
                           #"princeton-nlp/unsup-simcse-roberta-large",
                           #"princeton-nlp/sup-simcse-roberta-large",
                            "sentence-transformers/sentence-t5-base",
                            "Muennighoff/SGPT-5.8B-weightedmean-msmarco-specb-bitfit",
                            "similar_words_ratio"
        ]

    similarity_feature_names = [#"\makecell{Infersent\\\GloVe}",
    #"USE",
    "\makecell{all-\\\mpnet-\\\\base-\\\\v2}",
    "\makecell{ multi-\\\qa-\\\mpnet-\\\\base-\\\dot-\\\\v1}",
    #"\makecell{all-\\\distil-\\\\roberta-\\\\v1}",
    #"\makecell{Unsup\\\Sim\\\CSE}",
    #"\makecell{Sup\\\Sim\\\CSE}",
    "ST5",
    "SGPT",
    "Lexical"]

    data_names = ["clef_2020_checkthat_2_english",
                  "clef_2021_checkthat_2a_english",
                  "clef_2022_checkthat_2a_english",
                  "clef_2021_checkthat_2b_english",
                  "clef_2022_checkthat_2b_english"]

    data_names_table = ["ct 2020 2a",
                    "ct 2021 2a",
                      "ct 2022 2a",
                     "ct 2021 2b",
                      "ct 2022 2b"]

    hard_cases_dict = {}

    for data_idx, data_name in enumerate(data_names):

        data_name_table = data_names_table[data_idx]

        dataset_path = data_path + data_name
        data_name_queries = dataset_path + "/queries.tsv"
        data_name_targets = dataset_path + "/corpus"
        data_name_gold = dataset_path + "/gold.tsv"

        gold_df = pd.read_csv(data_name_gold, sep='\t', dtype=str, names=["query_id", "0", "target_id", "score"])
        query_ids = gold_df["query_id"]
        queries = get_queries(data_name_queries)
        targets = get_targets(data_name_targets)

        all_not_predicted = {}
        all_predicted = {}

        for idx, similarity_feature in enumerate(similarity_features):
            print(similarity_feature)

            if similarity_feature == "similar_words_ratio":
                sim_feature_name = similarity_feature_names[idx]
                similarity_feature_path_name = similarity_feature

                data_name_pred = dataset_path + "/" + similarity_feature_path_name + "/pred_qrels.tsv"

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
                                 '-lexical_similarity_measures', similarity_feature])

                pred_df = pd.read_csv(data_name_pred, sep='\t', dtype=str,
                                      names=["query_id", "Q0", "target_id", "1", "score", "name"])
                not_predicted = {}
                predicted = {}
                for query_id in query_ids:
                    predicted_target_ids = list(pred_df.loc[pred_df["query_id"] == query_id]["target_id"])[:5]
                    correct_target_ids = list(gold_df.loc[gold_df["query_id"] == query_id]["target_id"])
                    for target_id in correct_target_ids:
                        if target_id not in predicted_target_ids:
                            not_predicted[query_id] = (target_id, predicted_target_ids)
                        else:
                            predicted[query_id] = (target_id, predicted_target_ids)

                all_not_predicted[sim_feature_name] = not_predicted
                all_predicted[sim_feature_name] = predicted

            else:
                sim_feature_name = similarity_feature_names[idx]

                if "/" or ":" or "." in str(similarity_feature):
                    similarity_feature_path_name = str(similarity_feature).replace("/", "_").replace(":", "_").replace(".", "_")
                else:
                    similarity_feature_path_name = similarity_feature

                data_name_pred = dataset_path + "/" + similarity_feature_path_name + "/pred_qrels.tsv"

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

                pred_df = pd.read_csv(data_name_pred, sep='\t', dtype=str, names=["query_id", "Q0", "target_id", "1", "score", "name"])
                not_predicted = {}
                predicted = {}
                for query_id in query_ids:
                    predicted_target_ids = list(pred_df.loc[pred_df["query_id"] == query_id]["target_id"])[:5]
                    correct_target_ids = list(gold_df.loc[gold_df["query_id"] == query_id]["target_id"])
                    for target_id in correct_target_ids:
                        if target_id not in predicted_target_ids:
                            not_predicted[query_id] = (target_id, predicted_target_ids)
                        else:
                            predicted[query_id] = (target_id, predicted_target_ids)

                all_not_predicted[sim_feature_name] = not_predicted
                all_predicted[sim_feature_name] = predicted

        not_at_all_predicted = {}
        for query_id in query_ids:
            predicted = False
            for idx, similarity_feature in enumerate(similarity_features):
                if similarity_feature == "similar_words_ratio":
                    sim_feature_name = similarity_feature_names[idx]
                    if query_id not in all_not_predicted[sim_feature_name].keys():
                        predicted = True
                if not predicted:
                    correct_target_ids = list(gold_df.loc[gold_df["query_id"] == query_id]["target_id"])
                    correct_target_texts = [targets[target_id] for target_id in correct_target_ids]
                    not_at_all_predicted[queries[query_id]] = correct_target_texts

        hard_cases_dict[data_name_table] = not_at_all_predicted

    not_st5 = all_not_predicted["ST5"]
    not_sgpt = all_not_predicted["SGPT"]
    not_all_mp = all_not_predicted["\makecell{all-\\\mpnet-\\\\base-\\\\v2}"]
    not_multi_qa = all_not_predicted["\makecell{ multi-\\\qa-\\\mpnet-\\\\base-\\\dot-\\\\v1}"]
    not_lexical = all_not_predicted["Lexical"]

    st5 = all_predicted["ST5"]
    sgpt = all_predicted["SGPT"]
    all_mp = all_predicted["\makecell{all-\\\mpnet-\\\\base-\\\\v2}"]
    multi_qa = all_predicted["\makecell{ multi-\\\qa-\\\mpnet-\\\\base-\\\dot-\\\\v1}"]
    lexical = all_predicted["Lexical"]

    idx=0
    with open(dir_of_file + "/output/4_FIGURE_only_semantic.txt", 'w', encoding="utf-8") as f:
        for unpredicted_query_id, unpredicted_targets in not_lexical.items():
            if unpredicted_query_id in st5.keys() and unpredicted_query_id in all_mp.keys() and unpredicted_query_id in multi_qa.keys() and unpredicted_query_id in sgpt.keys():
                    print("\\begin{figure}[hbtp!]", file=f)
                    print("\label{only_semantic"+"_"+str(idx)+"}", file=f)
                    print("\\begin{quote}", file=f)
                    print("\\textbf{Query:} " + queries[unpredicted_query_id], file=f)
                    print("\end{quote}", file=f)
                    print("\\begin{quote}", file=f)
                    print("\\textbf{Target:} " + targets[unpredicted_targets[0]], file=f)
                    print("\end{quote}", file=f)
                    print("Top 5 targets predicted by Lexical Feature:", file=f)
                    for target in unpredicted_targets[1]:
                        print("\\begin{quote}", file=f)
                        print("\\footnotesize", file=f)
                        print(targets[target], file=f)
                        print("\end{quote}", file=f)
                    print("\caption{Example of true pair not retrieved within the top 5 by the lexical feature.}", file=f)
                    print("\end{figure}", file=f)
                    print("", file=f)
                    idx=idx+1

    idx=0
    with open(dir_of_file + "/output/4_FIGURE_only_lexical.txt", 'w', encoding="utf-8") as f:
        for predicted_query_id, predicted_targets in lexical.items():
            if predicted_query_id in not_st5.keys() and predicted_query_id in not_sgpt.keys() and predicted_query_id in not_all_mp.keys() and predicted_query_id in not_multi_qa.keys():
                    print("\\begin{figure}[hbtp!]", file=f)
                    print("\label{only_lexical"+"_"+str(idx)+"}", file=f)
                    print("\\begin{quote}", file=f)
                    print("\\textbf{Query:} " + queries[predicted_query_id], file=f)
                    print("\end{quote}", file=f)
                    print("\\begin{quote}", file=f)
                    print("\\textbf{Target:} " + targets[predicted_targets[0]], file=f)
                    print("\end{quote}", file=f)
                    print("Top target predicted by \\textit{ST5}:", file=f)
                    print("\\begin{quote}", file=f)
                    print("\\footnotesize", file=f)
                    print(targets[not_st5[predicted_query_id][1][0]], file=f)
                    print("\end{quote}", file=f)
                    print("Top target predicted by \\textit{SGPT}:", file=f)
                    print("\\begin{quote}", file=f)
                    print("\\footnotesize", file=f)
                    print(targets[not_sgpt[predicted_query_id][1][0]], file=f)
                    print("\end{quote}", file=f)
                    print("Top target predicted by \\textit{all-mpnet-base-v2}:", file=f)
                    print("\\begin{quote}", file=f)
                    print("\\footnotesize", file=f)
                    print(targets[not_all_mp[predicted_query_id][1][0]], file=f)
                    print("\end{quote}", file=f)
                    print("Top target predicted by \\textit{multi-qa-mpnet-base-dot-v1}:", file=f)
                    print("\\begin{quote}", file=f)
                    print("\\footnotesize", file=f)
                    print(targets[not_multi_qa[predicted_query_id][1][0]], file=f)
                    print("\end{quote}", file=f)
                    print("\caption{Example of true pair only retrieved by the lexical similarity feature and not by one of the listed sentence embeddings within the top 5 matches.}", file=f)
                    print("\end{figure}", file=f)
                    print("", file=f)
                    idx=idx+1
    
    idx=0
    with open(dir_of_file + "/output/4_FIGURE_only_multi_qa.txt", 'w', encoding="utf-8") as f:
        for predicted_query_id, predicted_targets in multi_qa.items():
            if predicted_query_id in not_all_mp.keys():
                    print("\\begin{figure}[hbtp!]", file=f)
                    print("\label{only_multi_qa"+"_"+str(idx)+"}", file=f)
                    print("\\begin{quote}", file=f)
                    print("\\textbf{Query:} " + queries[predicted_query_id], file=f)
                    print("\end{quote}", file=f)
                    print("\\begin{quote}", file=f)
                    print("\\textbf{Target:} " + targets[predicted_targets[0]], file=f)
                    print("\end{quote}", file=f)
                    print("Top 5 targets predicted by \\textit{all-mpnet-base-v2}:", file=f)
                    for target in not_all_mp[predicted_query_id][1]:
                        print("\\begin{quote}", file=f)
                        print("\\footnotesize", file=f)
                        print(targets[target], file=f)
                        print("\end{quote}", file=f)
                    print("\caption{Example of true pair only retrieved within the top 5 by \\textit{multi-qa-mpnet-base-dot-v1} and not by \\textit{all-mpnet-base-v2}.}", file=f)
                    print("\end{figure}", file=f)
                    print("", file=f)
                    idx=idx+1

    idx=0
    with open(dir_of_file + "/output/4_FIGURE_only_all_mp.txt", 'w', encoding="utf-8") as f:
        for predicted_query_id, predicted_targets in all_mp.items():
            if predicted_query_id in not_multi_qa.keys():
                    print("\\begin{figure}[hbtp!]", file=f)
                    print("\label{only_all_mp"+"_"+str(idx)+"}", file=f)
                    print("\\begin{quote}", file=f)
                    print("\\textbf{Query:} " + queries[predicted_query_id], file=f)
                    print("\end{quote}", file=f)
                    print("\\begin{quote}", file=f)
                    print("\\textbf{Target:} " + targets[predicted_targets[0]], file=f)
                    print("\end{quote}", file=f)
                    print("Top 5 targets predicted by \\textit{multi-qa-mpnet-base-dot-v1}:", file=f)
                    for target in not_multi_qa[predicted_query_id][1]:
                        print("\\begin{quote}", file=f)
                        print("\\footnotesize", file=f)
                        print(targets[target], file=f)
                        print("\end{quote}", file=f)
                    print("\caption{Example of true pair only retrieved within the top 5 by \\textit{all-mpnet-base-v2} and not by \\textit{multi-qa-mpnet-base-dot-v1}.}", file=f)
                    print("\end{figure}", file=f)
                    print("", file=f)
                    idx=idx+1

    idx=0
    with open(dir_of_file + "/output/4_FIGURE_only_sgpt.txt", 'w', encoding="utf-8") as f:
        for predicted_query_id, predicted_targets in sgpt.items():
            if predicted_query_id in not_st5.keys() and predicted_query_id in not_all_mp.keys() and predicted_query_id in not_multi_qa.keys() :
                    print("\\begin{figure}[hbtp!]", file=f)
                    print("\label{only_sgpt"+"_"+str(idx)+"}", file=f)
                    print("\\begin{quote}", file=f)
                    print("\\textbf{Query:} " + queries[predicted_query_id], file=f)
                    print("\end{quote}", file=f)
                    print("\\begin{quote}", file=f)
                    print("\\textbf{Target:} " + targets[predicted_targets[0]], file=f)
                    print("\end{quote}", file=f)
                    print("Top 5 targets predicted by \\textit{ST5}:", file=f)
                    for target in not_st5[predicted_query_id][1]:
                        print("\\begin{quote}", file=f)
                        print("\\footnotesize", file=f)
                        print(targets[target], file=f)
                        print("\end{quote}", file=f)
                    print("Top 5 targets predicted by \\textit{all-mpnet-base-v2}:", file=f)
                    for target in not_all_mp[predicted_query_id][1]:
                        print("\\begin{quote}", file=f)
                        print("\\footnotesize", file=f)
                        print(targets[target], file=f)
                        print("\end{quote}", file=f)
                    print("Top 5 targets predicted by \\textit{multi-qa-mpnet-base-dot-v1}:", file=f)
                    for target in not_multi_qa[predicted_query_id][1]:
                        print("\\begin{quote}", file=f)
                        print("\\footnotesize", file=f)
                        print(targets[target], file=f)
                        print("\end{quote}", file=f)
                    print("\caption{Example of true pair only retrieved within the top 5 by \\textit{SGPT}.}", file=f)
                    print("\end{figure}", file=f)
                    print("", file=f)
                    idx=idx+1

    # idx=0
    # with open(dir_of_file + "/output/4_FIGURE_only_all_mpnet.txt", 'w', encoding="utf-8") as f:
    #     for st5_not_predicted_query, st5_not_predicted_targets in not_st5.items():
    #             if st5_not_predicted_query in all_mp.keys() and st5_not_predicted_query in not_sgpt.keys():
    #                 print("\\begin{figure}[hbtp!]", file=f)
    #                 print("\label{only_all_mpnet"+"_"+str(idx)+"}", file=f)
    #                 print("\\begin{quote}", file=f)
    #                 print("\\textbf{Query:} " + queries[st5_not_predicted_query], file=f)
    #                 print("\end{quote}", file=f)
    #                 print("\\begin{quote}", file=f)
    #                 print("\\textbf{Target:} " + targets[all_mp[st5_not_predicted_query][0]], file=f)
    #                 print("\end{quote}", file=f)
    #                 print("Top 5 targets predicted by \\textit{ST5}:", file=f)
    #                 for target in st5_not_predicted_targets[1]:
    #                     print("\\begin{quote}", file=f)
    #                     print("\\footnotesize", file=f)
    #                     print(targets[target], file=f)
    #                     print("\end{quote}", file=f)
    #                 print("Top 5 targets predicted by \\textit{ST5}:", file=f)
    #                 for target in not_sgpt[st5_not_predicted_query][1]:
    #                     print("\\begin{quote}", file=f)
    #                     print("\\footnotesize", file=f)
    #                     print(targets[target], file=f)
    #                     print("\end{quote}", file=f)
    #                 print("\caption{Example of true pair only retrieved within the top 5 by \\textit{all-mpnet-base-v2}.}", file=f)
    #                 print("\end{figure}", file=f)
    #                 idx=idx+1


    with open(dir_of_file + "/output/4_FIGURE_hard_examples.txt", 'w', encoding="utf-8") as f:
        for data_name in data_names_table:
            dataset_dict = hard_cases_dict[data_name]
            if dataset_dict:
                print("\\begin{figure}[hbtp!]", file=f)
                print("\\begin{quote}", file=f)
                for query in list(dataset_dict.keys()):
                    correct_target = " \\textbf{and} ".join(list(dataset_dict.values())[0])
                    n_unpredicted_queries = str(len(dataset_dict))
                    print("\\textbf{Query:} \\textit{" + query + "}\\\\", file=f)
                    print("\\textbf{Correct Target:} \\textit{" + correct_target + "}\\\\", file=f)
                    print("\end{quote}", file=f)
                    print("\caption{Example of true pair in dataset \""+ data_name+"\" that was not retrieved within the five first matchings by all sentence embedding models. This was the case for "+ n_unpredicted_queries + " queries for this dataset.}", file=f)
                    print("\hline", file=f)
                print("\end{figure}", file=f)
                print("", file=f)


if __name__ == "__main__":
    run()