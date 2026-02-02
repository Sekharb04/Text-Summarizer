from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model once
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# Abstractive Model


def generate_summary(text, length=3):
    # Map length (1-5) to summary lengths
    # 1: short, 2: medium-short, 3: medium, 4: medium-long, 5: long
    length_configs = {
        1: (60, 20),
        2: (100, 40),
        3: (150, 50),
        4: (200, 70),
        5: (250, 100)
    }

    max_len, min_len = length_configs.get(
        length, (150, 50))  # default to medium

    input_text = "summarize: " + text
    input_ids = tokenizer.encode(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    summary_ids = model.generate(
        input_ids,
        max_length=max_len,
        min_length=min_len,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
