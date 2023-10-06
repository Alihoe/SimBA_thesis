import json
import os
import shutil
from pathlib import Path
from zipfile import ZipFile

import requests
import pandas as pd
from evaluation import DATA_PATH


def run():

    datasets = [
                #("ms_marco", ("msmarco", "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip")),
                #("trec_covid", ("trec-covid", "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip")),
                #("nf", ("nfcorpus", "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip")),
                #("nq", ("nq", "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip")),
                #("hotpot_qa", ("hotpotqa", "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip")),
                #("fiqa", ("fiqa", "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip")),
                #("arguana", ("arguana", "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip")),
                #("touche", ("webis-touche2020", "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/webis-touche2020.zip")),
                #("quora", ("quora", "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip")),
                #("dbpedia", ("dbpedia-entity", "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/dbpedia-entity.zip")),
                #("scidocs", ("scidocs", "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip")),
                #("fever", ("fever", "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fever.zip")),
                #("climate-fever", ("climate-fever", "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/climate-fever.zip")),
                #("scifact", ("scifact", "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"))
                ]

    for dataset in datasets:

        data_name = dataset[0]
        print(data_name)
        old_data_name = dataset[1][0]
        url = dataset[1][1]

        # Get data
        file_path = DATA_PATH + data_name
        Path(file_path).mkdir(parents=True, exist_ok=True)
        file_url = requests.get(url)

        with open(file_path + "/"+data_name+".zip", 'wb') as f:
            f.write(file_url.content)
        with ZipFile(file_path + "/"+data_name+".zip", 'r') as zipObj:
            for file in zipObj.namelist():
                zipObj.extract(file, file_path)

        queries_path = file_path + "/queries.tsv"
        corpus_path = file_path + "/corpus"
        qrels_path = file_path + "/gold.tsv"

        # Get QRELS
        qrels_df = pd.read_csv(file_path + "/" + old_data_name + "/qrels/test.tsv", sep='\t', dtype=str)
        qrels_df = qrels_df.loc[qrels_df['score'] != str(0)]
        qrels_df["0"] = str(0)
        qrels_df = qrels_df[["query-id", "0", "corpus-id", "score"]]
        qrels_df.to_csv(qrels_path, sep='\t', header=False, index=False)
        test_ids = set(list(qrels_df["query-id"]))

        # Get Test Queries
        queries_df = pd.DataFrame(columns=['id', 'text'])
        with open(file_path + "/" + old_data_name + "/queries.jsonl", encoding='utf-8') as f:
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
        with open(file_path + "/"+ old_data_name + "/corpus.jsonl") as f:
            json_list = list(f)
        ids = []
        texts = []
        for json_str in json_list:
            ids.append(json.loads(json_str)["_id"])
            texts.append(json.loads(json_str)["text"])
        corpus_df['id'] = pd.Series(ids)
        corpus_df['text'] = pd.Series(texts)
        corpus_df.to_csv(corpus_path, sep='\t', header=False, index=False)

        os.remove(file_path + "/"+data_name+".zip")
        shutil.rmtree(file_path + "/" + old_data_name, ignore_errors=False, onerror=None)


if __name__ == "__main__":
    run()
