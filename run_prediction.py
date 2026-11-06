import time
from pathlib import Path
import pandas as pd

from extract_text import process_document
from model import extract_metadata_from_text, set_examples, FIELDS

# Training files shown to the model as examples. They have different notice periods (30 and 90).
# On a GPU (Colab) you can add more, e.g. "47854715-RENTAL-AGREEMENT" and "24158401-Rental-Agreement".
EXAMPLE_NAMES = [
    "50070534-RENTAL-AGREEMENT",
    "36199312-Rental-Agreement",
]


def find_document(base_name, folder):
    for path in Path(folder).iterdir():
        if path.name == base_name or path.stem == base_name or path.stem.startswith(base_name + "."):
            return path
    raise FileNotFoundError(f"No document found for {base_name}")


def load_examples(train_csv, train_folder, names):
    df = pd.read_csv(train_csv, keep_default_na=False)
    df["File Name"] = df["File Name"].astype(str).str.strip()
    csv_fields = [c for c in df.columns if c != "File Name"]
    items = []
    for name in names:
        row = df[df["File Name"] == name].iloc[0]
        gold = {field: str(row[col]).strip() for field, col in zip(FIELDS, csv_fields)}
        text = process_document(find_document(name, train_folder))
        items.append((text, gold))
    return items


def generate_test_predictions(test_csv, test_folder, train_csv, train_folder):
    set_examples(load_examples(train_csv, train_folder, EXAMPLE_NAMES))

    df = pd.read_csv(test_csv, keep_default_na=False)
    results = []
    done = set()
    if Path("predictions_test.csv").is_file():
        old = pd.read_csv("predictions_test.csv", keep_default_na=False)
        results = old.to_dict("records")
        done = set(old["File Name"].astype(str).str.strip())

    for _, row in df.iterrows():
        name = str(row["File Name"]).strip()
        if name in done:
            print("Skipping (already done):", name, flush=True)
            continue
        print("Processing:", name, flush=True)
        start = time.time()
        try:
            text = process_document(find_document(name, test_folder))
            prediction = extract_metadata_from_text(text)
        except Exception as error:
            print("ERROR:", name, error)
            prediction = {f: "N/A" for f in FIELDS}
        prediction["File Name"] = name
        results.append(prediction)
        print(f"Done in {time.time() - start:.0f}s: {prediction}", flush=True)

        pd.DataFrame(results, columns=["File Name"] + FIELDS).to_csv(
            "predictions_test.csv", index=False
        )

    print(pd.DataFrame(results, columns=["File Name"] + FIELDS).to_string(index=False))


if __name__ == "__main__":
    generate_test_predictions(
        "data/test.csv", "data/test", "data/train.csv", "data/train"
    )