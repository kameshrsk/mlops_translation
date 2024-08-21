from steps.artifact_loader import load_artifact_from_pipeline

bleu_score=load_artifact_from_pipeline("train_evaluate_pipeline", "train_evaluate_model", "output")['score']

with open('accuracy.txt', 'w') as file:
    file.write(str(bleu_score))