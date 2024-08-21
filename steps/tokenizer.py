from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

checkpoint='Helsinki-NLP/opus-mt-fr-en'

tokenizer=AutoTokenizer.from_pretrained(checkpoint)
model=AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

def tokenize_data(example):
    return tokenizer(
        example['French'],
        text_target=example['English'],
        truncation=True,
        max_length=512
    )