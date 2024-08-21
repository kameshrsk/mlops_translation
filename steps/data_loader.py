from zenml import step
from torch.utils.data import DataLoader
from typing_extensions import Annotated, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from .tokenizer import *
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq

@step(enable_cache=True)
def load_data(path:str)->Tuple[
    Annotated[DataLoader, 'training_batch'],
    Annotated[DataLoader, 'testing_batch']
]:
    data=pd.read_csv('data/translation_data.csv')
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)
    data=data.sample(10000)

    train_data, test_data=train_test_split(data, test_size=0.2, shuffle=True, random_state=101)

    train_data=Dataset.from_pandas(train_data)
    test_data=Dataset.from_pandas(test_data)

    tokenized_train_data=train_data.map(tokenize_data, remove_columns=['English', 'French','__index_level_0__'])
    tokenized_test_data=test_data.map(tokenize_data, remove_columns=['English', 'French','__index_level_0__'])

    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors='pt')

    training_batch=DataLoader(tokenized_train_data, shuffle=True, collate_fn=data_collator)
    testing_batch=DataLoader(tokenized_test_data, shuffle=True, collate_fn=data_collator)

    return training_batch, testing_batch
