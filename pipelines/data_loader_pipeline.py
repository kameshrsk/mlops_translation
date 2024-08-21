from zenml import pipeline

from steps.data_loader import load_data


@pipeline
def data_pipeline(path:str):
    training_batch, testing_batch=load_data(path=path)


if __name__=='__main__':
    data_pipeline('data/translation_data.csv')