import json
import os
import shutil
from pathlib import Path
from zipfile import ZipFile
import numpy as np

import requests
import pandas as pd
from evaluation import DATA_PATH


def run():

    data_name_dir = "fever"

    file_path = DATA_PATH + data_name_dir
    Path(file_path).mkdir(parents=True, exist_ok=True)
    corpus_path = file_path + "/corpus"
    corpus_df = pd.read_csv(corpus_path, header=None, sep='\t', dtype=str)

    subdataframes = np.array_split(corpus_df, 20)

    for idx in range(20):
        corpus_path = file_path + "_" + str(idx + 1) + "/corpus"
        subdataframes[idx].to_csv(corpus_path, sep='\t', header=False, index=False)


if __name__ == "__main__":
    run()
