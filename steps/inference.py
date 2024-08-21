from zenml import step
from zenml.client import Client
import pickle
import torch
from .tokenizer import tokenizer
from .artifact_loader import load_artifact_from_pipeline

def load_model():
    
    return load_artifact_from_pipeline("train_evaluate_pipeline", "train_evaluate_model", "output")['model']


def process_input(input_text:str):
    inputs=tokenizer(input_text, padding=True, truncation=True, return_tensors='pt')

    return inputs

@step(enable_cache=False)
def run_inference(input_text:str):
    model=load_model()
    inputs=process_input(input_text)
    with torch.no_grad():
        output=model.generate(**inputs)

    decoded_output=tokenizer.batch_decode(output, skip_special_tokens=True)
    decoded_output=[output.strip() for output in decoded_output]

    return decoded_output