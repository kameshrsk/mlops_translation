import gradio as gr
import torch
import pickle

with open('saved_model/model.pkl', 'rb') as f:
    model_dict=pickle.load(f)

model=model_dict['model']
tokenizer=model_dict['tokenizer']

def translate(french_text):
    inputs=tokenizer(french_text, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        output=model.generate(**inputs)

    decoded_output=tokenizer.batch_decode(output, skip_special_tokens=True)
    decoded_output=[output.strip() for output in decoded_output]

    return " ".join(decoded_output)

iface=gr.Interface(
    fn=translate,
    inputs=gr.Text(label="French Text"),
    outputs=gr.Text(),
    title="Fr-En Translation",
)

if __name__=="__main__":
    iface.launch(server_name='0.0.0.0', server_port=7860)
