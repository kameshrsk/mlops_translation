from zenml import step
import numpy as np
import torch
from torch.utils.data import DataLoader
from .tokenizer import tokenizer
from tqdm.auto import tqdm
from accelerate import Accelerator
import pickle

import evaluate
import mlflow

metrics=evaluate.load('sacrebleu')
mlflow.set_tracking_uri("http://localhost:5000")

def postprocess(predictions, labels):
    predictions=predictions.cpu().numpy()
    labels=labels.cpu().numpy()

    decoded_preds=tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels=np.where(labels!=-100, labels, tokenizer.pad_token_id)

    decoded_labels=tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds=[pred.strip() for pred in decoded_preds]
    decoded_labels=[[label.strip()] for label in decoded_labels]

    return decoded_preds, decoded_labels

@step(enable_cache=False)
def train_evaluate_model(training_batch:DataLoader, testing_batch:DataLoader, num_epochs:int, learning_rate:float):

    from .tokenizer import model

    num_training_steps=num_epochs*len(training_batch)

    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("learning_rate", learning_rate)

    progress=tqdm(range(num_training_steps))

    device=('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

    accelerator=Accelerator()

    model, optimizer, training_batch, testing_batch=accelerator.prepare(
        model, optimizer, training_batch, testing_batch
    )   

    best_score=0.0 
    model_dict={}

    for epoch in range(num_epochs):
        model.train()

        for batch in training_batch:
            batch={k:v.to(device) for k,v in batch.items()}
            output=model(**batch)

            loss=output.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress.update(1)

        model.eval()

        for batch in testing_batch:
            batch={k:v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                output=accelerator.unwrap_model(model).generate(
                    batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=512
                )

            gathered_tokens=accelerator.pad_across_processes(output, dim=1, pad_index=tokenizer.pad_token_id)
            labels=accelerator.pad_across_processes(batch['labels'], dim=1, pad_index=tokenizer.pad_token_id)

            gathered_tokens=accelerator.gather(gathered_tokens)
            labels=accelerator.gather(labels)

            decoded_preds, decoded_labels=postprocess(gathered_tokens, labels)

            metrics.add_batch(predictions=decoded_preds, references=decoded_labels)

        result=metrics.compute()

        if result['score']>best_score:
            best_score=result['score']
    
        print(f"Epoch: {epoch+1} | BLEU Score: {result['score']:.2f}")

    model_dict={"model": model.cpu(), "score": best_score}

    if best_score>40:

        with open('saved_model/model.pkl', 'wb') as f:
            pickle.dump(model_dict, f)

        print(f"The best score is: {best_score} - Model Saved")

    else:
        print(f"The best score is: {best_score} - Model is Not Saved")

    mlflow.log_metric("BLEU Score", best_score)

    components={
        "model":model,
        "tokenizer":tokenizer
    }

    mlflow.transformers.log_model(transformers_model=components, artifact_path='model')

    return model_dict
