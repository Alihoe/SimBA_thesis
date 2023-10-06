import json
import os
import shutil
from pathlib import Path
from zipfile import ZipFile

import requests
import pandas as pd
from evaluation import DATA_PATH


def run():

    data_name_dir = "cqa_dupstack"
    old_data_name = "cqadupstack"
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip"

    # Get data
    file_path = DATA_PATH + data_name_dir
    Path(file_path).mkdir(parents=True, exist_ok=True)
    file_url = requests.get(url)

    with open(file_path + "/"+data_name_dir+".zip", 'wb') as f:
        f.write(file_url.content)
    with ZipFile(file_path + "/"+data_name_dir+".zip", 'r') as zipObj:
        for file in zipObj.namelist():
            zipObj.extract(file, file_path)

    os.remove(file_path + "/" + data_name_dir + ".zip")

    for sub_dataset_name in os.listdir(file_path + "/" + old_data_name):
        print(sub_dataset_name)

        new_file_path = DATA_PATH + data_name_dir + "_" + sub_dataset_name
        Path(new_file_path).mkdir(parents=True, exist_ok=True)

        queries_path = new_file_path + "/queries.tsv"
        corpus_path = new_file_path + "/corpus"
        qrels_path = new_file_path + "/gold.tsv"

        # Get QRELS
        qrels_df = pd.read_csv(file_path + "/" + old_data_name + "/" + sub_dataset_name + "/qrels/test.tsv", sep='\t', dtype=str)
        qrels_df = qrels_df.loc[qrels_df['score'] != str(0)]
        qrels_df["0"] = str(0)
        qrels_df = qrels_df[["query-id", "0", "corpus-id", "score"]]
        qrels_df.to_csv(qrels_path, sep='\t', header=False, index=False)
        test_ids = set(list(qrels_df["query-id"]))

        # Get Test Queries
        queries_df = pd.DataFrame(columns=['id', 'text'])
        with open(file_path + "/" + old_data_name + "/" + sub_dataset_name + "/queries.jsonl", encoding='utf-8') as f:
            json_list = list(f)
        ids = []
        texts = []
        for json_str in json_list:
            if json.loads(json_str)["_id"] in test_ids:
                ids.append(json.loads(json_str)["_id"])
                texts.append(json.loads(json_str)["text"])
        queries_df['id'] = pd.Series(ids)
        queries_df['text'] = pd.Series(texts)
        queries_df.to_csv(queries_path, sep='\t', header=False, index=False)

        # Get Corpus
        corpus_df = pd.DataFrame(columns=['id', 'text'])
        with open(file_path + "/" + old_data_name + "/" + sub_dataset_name + "/corpus.jsonl") as f:
            json_list = list(f)
        ids = []
        texts = []
        for json_str in json_list:
            ids.append(json.loads(json_str)["_id"])
            texts.append(json.loads(json_str)["text"])
        corpus_df['id'] = pd.Series(ids)
        corpus_df['text'] = pd.Series(texts)
        corpus_df.to_csv(corpus_path, sep='\t', header=False, index=False)

    shutil.rmtree(file_path, ignore_errors=False, onerror=None)


if __name__ == "__main__":
    run()
