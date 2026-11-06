import sys
import pandas as pd
from model import FIELDS


def evaluate(pred_csv, gold_csv):
    pred = pd.read_csv(pred_csv, keep_default_na=False)
    gold = pd.read_csv(gold_csv, keep_default_na=False)
    pred["File Name"] = pred["File Name"].astype(str).str.strip()
    gold["File Name"] = gold["File Name"].astype(str).str.strip()
    pred = pred.set_index("File Name")
    gold = gold.set_index("File Name")
    gold_cols = list(gold.columns)

    for field, col in zip(FIELDS, gold_cols):
        hits = 0
        for name in gold.index:
            if name in pred.index:
                hits += str(pred.loc[name, field]).strip() == str(gold.loc[name, col]).strip()
        print(f"{field}: recall = {hits / len(gold):.0%}")


if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2])