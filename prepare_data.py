import json
from datasets import Dataset
import random

def load_jsonl(path):
    records = []
    decoder = json.JSONDecoder()
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    idx = 0
    while idx < len(content):
        while idx < len(content) and content[idx] in ' \t\n\r':
            idx += 1
        if idx >= len(content):
            break
        try:
            obj, end = decoder.raw_decode(content, idx)
            records.append(obj)
            idx = end
        except json.JSONDecodeError:
            idx += 1
    return records

raw_data = load_jsonl("train.jsonl")

data = [r for r in raw_data if r.get("input", "").strip() and r.get("target", "").strip()]
print(f"Loaded {len(data)} valid sentence pairs (out of {len(raw_data)} total)")

PREFIX = "translate patois to english: "

formatted = [
    {
        "input_text": PREFIX + item["input"].strip(),
        "target_text": item["target"].strip(),
        "domain": item.get("domain", ""),
        "contains_slang": item.get("contains_slang", False),
    }
    for item in data
]   

#splitting training(90%) and validation(10%)
random.seed(42)
random.shuffle(formatted)

split_idx = int(len(formatted) * 0.9)
train_data = formatted[:split_idx]
val_data = formatted[split_idx:]

print(f"Train set: {len(train_data)} examples")
print(f"Validation set: {len(val_data)} examples")

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

train_dataset.save_to_disk("data/train")
val_dataset.save_to_disk("data/val")

print("Datasets saved to ./data/train and ./data/val")
print("\nSample training example: ")
print(train_dataset[0])