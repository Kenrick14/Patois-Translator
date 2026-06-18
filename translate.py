from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

MODEL_PATH = "./patois-translator-model"

tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)

def translate_with_confidence(patois_text):
    input_text = "translate patois to english: " + patois_text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)

    outputs = model.generate(
        inputs["input_ids"],
        max_length=128,
        num_beams=4,
        early_stopping=True,
        output_scores=True,
        return_dict_in_generate=True
    )

    translation = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    if hasattr(outputs, "sequences_scores"):
        overall_confidence = torch.exp(outputs.sequences_scores[0]).item()
    else:
        overall_confidence = None

    return translation, overall_confidence

print("Patois-to-English Translator (type 'quit' to exit)\n")
while True:
    user_input = input("Patois: ")
    translation, overall_confidence = translate_with_confidence(user_input)
    if user_input.lower() in ("quit", "exit"):
        break
    print(f"English: {translation}\n") 
    print(f"Confidence: {overall_confidence:.2%}\n" if overall_confidence else "Confidence: N/A \n") 