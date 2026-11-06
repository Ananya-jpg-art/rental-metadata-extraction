# Metadata Extraction from Rental Agreements

An AI system that extracts six metadata fields from rental agreements, given as
scanned images (`.png`) or Word files (`.docx`), irrespective of the template:

- Agreement Value
- Agreement Start Date
- Agreement End Date
- Renewal Notice (Days)
- Party One
- Party Two

## Approach

No rule-based approach is used: there is no RegEx, keyword search or hand-written
extraction logic anywhere in the code.

1. **Text extraction** (`extract_text.py`)
   - `.docx`: text is read from paragraphs and tables with `python-docx`.
   - `.png` / `.jpg`: text is read with EasyOCR (a neural OCR model).
2. **Field extraction** (`model.py`)
   - A pretrained instruction-tuned LLM (Qwen2.5-Instruct) reads the full document text.
   - Two labelled documents from `data/train` are put in the prompt as in-context
     (few-shot) examples, so the model learns the expected output formats
     (date format, rent as a number, notice period in days, names without titles).
   - The model returns one JSON object with the six fields, which is read with a
     standard JSON decoder.
   - Model choice: `Qwen2.5-1.5B-Instruct` on CPU, `Qwen2.5-3B-Instruct` if a GPU is available.
3. **Prediction and evaluation** (`run_prediction.py`, `evaluate.py`)
   - `run_prediction.py` processes every file listed in `data/test.csv` and writes `predictions_test.csv`.
   - `evaluate.py` computes per-field recall (exact match) against `data/test.csv`.

## Project structure

```
rental_project/
├── data/
│   ├── train/                 # training .docx and .png files
│   ├── test/                  # test .docx and .png files
│   ├── train.csv              # labels for train/
│   └── test.csv               # labels for test/
├── extract_text.py            # docx reading + OCR
├── model.py                   # LLM prompt and field extraction
├── run_prediction.py          # runs the pipeline on the test set
├── evaluate.py                # per-field recall
├── predictions_test.csv       # predictions for the test set
├── prediction_errors.txt      # error log from earlier runs (can be ignored)
├── assignment-details.pdf     # assignment description
└── README.md
```

(`debug_text/`, `__pycache__/` and `venv/` are generated locally and are not part of the submission.)

## Setup

Python 3.10+ is recommended.

```
pip install transformers torch accelerate pandas python-docx easyocr
```

The first run downloads the model (about 3 GB for the 1.5B model) and needs internet access.

## How to run

1. Place the dataset in the `data/` folder as shown above.
2. Generate predictions (this also saves progress after each file, and skips files
   already present in `predictions_test.csv` when re-run; delete that file for a fresh run):

```
python run_prediction.py
```

3. Compute per-field recall:

```
python evaluate.py predictions_test.csv data/test.csv
```

Running on CPU is slow (several minutes per document). On a GPU (for example a free
Google Colab T4) the same code runs in seconds per document.

## Results

Per-field recall (exact match) on the 4 documents in `data/test.csv`:

| Field                 | Recall |
|-----------------------|--------|
| Agreement Value       | 100%   |
| Agreement Start Date  | 100%   |
| Agreement End Date    | 50%    |
| Renewal Notice (Days) | 100%   |
| Party One             | 50%    |
| Party Two             | 50%    |

Predictions are in `predictions_test.csv`. (Re-run `evaluate.py` to reproduce these numbers.)

## Notes and limitations

- Exact-match scoring is strict. The dataset labels contain some inconsistencies
  (for example impossible dates such as `31.02.2011`, and typos in names), so 100% is not
  always reachable.
- The small model sometimes picks the wrong person as a party, such as the father named
  in "son of ..." or the representative of a company instead of the company itself.
  A larger model or more in-context examples (see `EXAMPLE_NAMES` in `run_prediction.py`)
  improves this.
- OCR errors in scanned images can propagate into the extracted values.
- `24158401-Rental-Agreement` appears in both `train.csv` and `test.csv`; it is
  deliberately not used as a prompt example so the test score is not inflated.
- The CSV headers in the dataset spell "Aggrement"; the code maps the metadata columns by
  position, so this does not matter.
- The optional REST API is not included in this submission.