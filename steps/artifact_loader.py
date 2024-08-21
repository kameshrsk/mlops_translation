from zenml.client import Client

def load_artifact_from_pipeline(pipeline_name:str, step_name:str, output_name:str):
    pipeline=Client().get_pipeline(pipeline_name)
    last_run=pipeline.last_successful_run if pipeline.last_successful_run else pipeline.last_run
    step=last_run.steps[step_name]
    output=step.outputs
    artifact_id=output[output_name].id
    artifact=Client().get_artifact_version(artifact_id)

    return artifact.load()