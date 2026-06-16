from transformers import T5Tokenizer
from datasets import load_from_disk

MODEL_NAME = "t5-small"
MAX_INPUT_LEN = 128
MAX_TARGET_LEN = 128

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

train_dataset = load_from_disk("data/train")
val_dataset = load_from_disk("data/val")

def tokenize_batch(batch):
    model_inputs = tokenizer(
        batch["input_text"],
        max_length = MAX_INPUT_LEN,
        truncation = True,
        padding = "max_length"
    )


    labels = tokenizer(
        text_target=batch["target_text"],
        max_length = MAX_TARGET_LEN,
        truncation = True,
        padding = "max_length"
    )

    #excluding loss calculation from padding tokens
    labels["input_ids"] = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

#tokenizing training and val splits
tokenized_train = train_dataset.map(tokenize_batch, batched=True, batch_size=16)
tokenized_val = val_dataset.map(tokenize_batch, batched=True, batch_size=16)

#save tokenized datasets
tokenized_train.save_to_disk("data/tokenized_train")
tokenized_val.save_to_disk("data/tokenized_val")

print("Tokenization complete!")
print(f"Train: {len(tokenized_train)} examples")
print(f"Val: {len(tokenized_val)} examples")