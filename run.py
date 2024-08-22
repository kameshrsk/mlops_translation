import click

from pipelines.data_loader_pipeline import data_pipeline
from pipelines.train_evaluate_pipeline import train_evaluate_pipeline
from pipelines.inference_pipeline import inference_pipeline

import mlflow

@click.command()

@click.option(
    '--load-data',
    is_flag=True,
    default=False,
    help='Load the Dataset'
)

@click.option(
    '--train-evaluate',
    is_flag=True,
    default=False,
    help="Train and Evaluate the Model"
)

@click.option(
    "--translate",
    is_flag=True,
    default=False,
    help="Transflate French sentence to English"
)

@click.option(
    "--path",
    default=None,
    type=click.STRING,
    help="Path to the dataset"
)

@click.option(
    "--learning-rate",
    default=1e-5,
    type=click.FLOAT,
    help="Learning Rate for the model to train on"
)

@click.option(
    "--epochs",
    default=3,
    type=click.INT,
    help="Number of Epochs to Train the model"
)

@click.option(
    "--french",
    type=click.STRING,
    help="French Sentence that is to be translated"
)

def main(
    load_data:bool=False,
    train_evaluate:bool=False,
    translate:bool=False,
    path:str=None,
    epochs:int=3,
    learning_rate:int=1e-5,
    french:str=None
):
    if load_data:
        data_pipeline(path)

    if train_evaluate:
        train_evaluate_pipeline(path, epochs, learning_rate)

    if translate:
        print(inference_pipeline(french).steps['run_inference'].output.load())


if __name__=="__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    main()