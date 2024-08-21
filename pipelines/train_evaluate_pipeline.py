from zenml import pipeline
from steps.train_evaluate import train_evaluate_model
from steps.data_loader import load_data
from torch.utils.data import DataLoader

@pipeline
def train_evaluate_pipeline(path:str, num_epochs:int, learning_rate:float):
    training_batch, testing_batch=load_data(path)

    model_dict=train_evaluate_model(training_batch, testing_batch, num_epochs, learning_rate)

if __name__=="__main__":

    train_evaluate_pipeline("data/translation_data.csv", num_epochs=1, learning_rate=1e-5)