import operator
import os
import numpy as np
import torch
import pandas as pd

from scipy.stats import spearmanr

base_path = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(base_path, "../../data/")


def analyse_feature_correlation(all_features, all_sim_scores_df, data_name, k):
    feature_top_ks = {}
    for feature in all_features:
        this_feature_sim_scores = np.stack(all_sim_scores_df[feature])
        sim_scores_top_k_values, sim_scores_top_k_idx = torch.topk(torch.tensor(this_feature_sim_scores), k=k,
                                                                   dim=1, largest=True)
        sim_scores_top_k_values = sim_scores_top_k_values.cpu().tolist()
        feature_top_ks[feature] = sim_scores_top_k_values

    feature_scores = np.array(list(feature_top_ks.values()))
    a = feature_scores.shape[0]
    b = feature_scores.shape[1]
    c = feature_scores.shape[2]
    feature_scores = feature_scores.reshape(a, b * c)

    correlation, p_value = spearmanr(feature_scores, axis=1, nan_policy='propagate')

    pd.DataFrame(correlation, columns=all_features).to_csv(DATA_PATH + data_name + "/" + data_name + "_correlation.tsv",
                                                           index=False, header=True, sep='\t')

    with open(DATA_PATH + data_name + "/" + data_name + "_correlation.txt", 'w') as f:
        print(all_features, file=f)
        print(correlation, file=f)
        for idx, feature in enumerate(all_features):
            correlations = correlation[idx]
            feature_correlation = dict(zip(all_features, correlations))
            feature_correlation_sorted = dict(sorted(feature_correlation.items(), key=operator.itemgetter(1), reverse=True))
            feature_correlation = dict(list(feature_correlation_sorted.items())[1:])
            print('-------------------------------------------', file=f)
            print('The correlation for feature ' + str(feature), file=f)
            for key, value in feature_correlation.items():
                print('with feature ' + str(key) + ' is ' + str(round(value, 3)), file=f)