from zenml import pipeline
from steps.inference import run_inference

@pipeline
def inference_pipeline(input_text:str):
    output=run_inference(input_text)

    return output


if __name__ == "__main__":
    # Execute the pipeline
    pipeline_run = inference_pipeline(input_text="il pleut dehors")

    # Directly use the pipeline_run response
    final_output_artifact = pipeline_run.steps['run_inference'].output.load()

    print(final_output_artifact)