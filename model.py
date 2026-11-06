import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

FIELDS = [
    "Agreement Value",
    "Agreement Start Date",
    "Agreement End Date",
    "Renewal Notice (Days)",
    "Party One",
    "Party Two",
]

USE_GPU = torch.cuda.is_available()
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct" if USE_GPU else "Qwen/Qwen2.5-1.5B-Instruct"
DTYPE = torch.float16 if USE_GPU else torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=DTYPE, low_cpu_mem_usage=True
)
if USE_GPU:
    model = model.to("cuda")
model.eval()

INSTRUCTION = (
    "You extract metadata from rental agreements. Return one JSON object with exactly these keys: "
    + ", ".join(FIELDS)
    + ". Party One is the landlord/owner/lessor and Party Two is the tenant/lessee. "
    "Follow the value formats of the examples exactly. "
    "Answer only from the final agreement; the earlier agreements are just examples."
)

examples = []  # (document_text, gold_values) pairs taken from train data


def set_examples(items):
    global examples
    examples = items


def build_messages(document_text):
    messages = [{"role": "system", "content": INSTRUCTION}]
    for text, gold in examples:
        messages.append({"role": "user", "content": text})
        messages.append({"role": "assistant", "content": json.dumps(gold)})
    messages.append({"role": "user", "content": document_text})
    return messages


def extract_metadata_from_text(document_text):
    prompt = tokenizer.apply_chat_template(
        build_messages(document_text), tokenize=False, add_generation_prompt=True
    )
    prompt += "{"  # start the answer as JSON
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print("Prompt tokens:", inputs["input_ids"].shape[1], flush=True)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    generated = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    print("RAW MODEL OUTPUT:", "{" + generated, flush=True)
    try:
        data, _ = json.JSONDecoder().raw_decode("{" + generated)
    except json.JSONDecodeError:
        data = {}
    return {field: str(data.get(field, "N/A")).strip() for field in FIELDS}