import pandas as pd

from src.get_data import DATA_PATH


def run():

        # Get data
        qrels_path = DATA_PATH + "nf/gold.tsv"

        # Get QRELS
        qrels_df = pd.read_csv(qrels_path, names=["query-id", "0", "corpus-id", "score"], sep='\t', dtype=str)
        print(qrels_df)
        qrels_df = qrels_df.loc[qrels_df['score'] != str(1)]
        qrels_df["score"] = str(1)
        qrels_df = qrels_df[["query-id", "0", "corpus-id", "score"]]
        qrels_df.to_csv(qrels_path, sep='\t', header=False, index=False)


if __name__ == "__main__":
    run()