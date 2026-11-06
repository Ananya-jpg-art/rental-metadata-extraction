# rental-metadata-extraction
AI system that extracts metadata (agreement value, start/end dates, renewal notice days, landlord and tenant names) from rental agreements given as scanned images or DOCX files. EasyOCR and python-docx read the text, then a Qwen2.5 LLM with few-shot examples returns JSON. No regex or rule-based extraction. Includes per-field recall evaluation.
