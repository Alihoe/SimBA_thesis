import argparse
import os
import sys

import pandas as pd
import numpy as np


base_path = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(base_path, "../data/")
sys.path.append(os.path.join(base_path, "../../src"))
from utils import get_queries, get_targets

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
    avg_query_lengths = np.mean(query_lengths)
    target_lengths = [len(text) for text in target_texts]
    avg_target_lengths = np.mean(target_lengths)

    columns = ['Dataset', 'Number of Queries', 'Number of Targets', 'Average Query Length', 'Average Target Length']

    analysis_df = pd.DataFrame(columns=columns)

    analysis_df['Dataset'] = args.data_name
    analysis_df['Number of Queries'] = len(query_texts)
    analysis_df['Number of Targets'] = len(target_texts)
    analysis_df['Average Query Length'] = avg_query_lengths
    analysis_df['Average Target Length'] = avg_target_lengths

    print(analysis_df)

    analysis_df.to_csv(output_path, index=False, header=True, sep='\t')
    #df.to_latex()