import click

from pipelines.data_loader_pipeline import data_pipeline

@click.option(
    '--load-data',
    is_flag=True,
    default=False,
    help='Load the Dataset'
)

@click.option(
    "--path",
    default=None,
    type=click.STRING,
    help="Path to the dataset"
)

def main(
    load_data:bool=False,
    path:str=None
):
    if load_data:
        data_pipeline(path)


if __name__=="__main__":
    main()