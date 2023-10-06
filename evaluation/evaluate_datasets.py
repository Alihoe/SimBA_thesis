import argparse
import os
import sys

import pandas as pd
import numpy as np


base_path = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(base_path, "../data/")
sys.path.append(os.path.join(base_path, "../src"))
from src.utils import get_queries, get_targets

def run():

    parser = argparse.ArgumentParser()
    #input
    parser.add_argument('data_name', type=str)
    parser.add_argument('data_name_queries', type=str)
    parser.add_argument('data_name_targets', type=str)
    args = parser.parse_args()

    output_path = DATA_PATH + args.data_name + "/dataset_analysis.tsv"

    queries_path = args.data_name_queries
    targets_path = args.data_name_targets

    queries = get_queries(queries_path)
    query_texts = list(queries.values())
    targets = get_targets(targets_path)
    target_texts = list(targets.values())

    query_lengths = [len(text) for text in query_texts]
    avg_query_lengths = int(round(np.mean(query_lengths), 0))
    target_lengths = [len(text) for text in target_texts]
    avg_target_lengths = int(round(np.mean(target_lengths), 0))

    columns = ['dataset', '\# queries', '\# targets', '\makecell{avg.\\\query\\\length}', '\makecell{avg.\\target\\\length}']

    data = {'dataset': [args.data_name], '\# queries': [str(len(query_texts))],
               '\# targets': [str(len(target_texts))], '\makecell{avg.\\\query\\\length}': [str(avg_query_lengths)],
               '\makecell{avg.\\target\\\length}': [str(avg_target_lengths)]}

    analysis_df = pd.DataFrame(data, columns=columns)

    analysis_df.to_csv(output_path, index=False, header=True, sep='\t')


if __name__ == "__main__":
    run()