from zenml import step
from zenml.client import Client
import pickle
import torch
from .tokenizer import tokenizer

def load_artifact_from_pipeline(pipeline_name:str, step_name:str, output_name:str):
    pipeline=Client().get_pipeline(pipeline_name)
    last_run=pipeline.last_successful_run if pipeline.last_successful_run else pipeline.last_run
    step=last_run.steps[step_name]
    output=step.outputs
    artifact_id=output[output_name].id
    artifact=Client().get_artifact_version(artifact_id)

    return artifact.load()


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