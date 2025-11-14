## Development of a Named Entity Recognition (NER) Prototype Using a Fine-Tuned BART Model and Gradio Framework

### AIM:
To design and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation.

### PROBLEM STATEMENT:
To develop a Named Entity Recognition (NER) web application using the dslim/bert-base-NER model from Hugging Face that identifies and highlights entities such as persons, organizations, locations, and dates in user-provided text through an interactive Gradio interface.
### DESIGN STEPS:

#### STEP 1:
Import Libraries: Import necessary modules — os, requests, json, gradio, and dotenv.
#### STEP 2:
Load API Key: Load the Hugging Face API key using environment variables.

#### STEP 3:

Set API Endpoint: Define the model endpoint (dslim/bert-base-NER).
### PROGRAM:
```
import os
import requests
import json
import gradio as gr
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  
hf_api_key = os.environ.get("HF_API_KEY")
API_URL = os.environ.get(
    "HF_API_NER_BASE",  
    "https://api-inference.huggingface.co/models/dslim/bert-base-NER"
)
def get_completion(inputs, parameters=None, ENDPOINT_URL=API_URL):
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json"
    }
    data = {"inputs": inputs}
    if parameters is not None:
        data["parameters"] = parameters

    response = requests.post(ENDPOINT_URL, headers=headers, data=json.dumps(data))
    
    try:
        return response.json()  
    except json.JSONDecodeError:
        print("Invalid response from Hugging Face API:")
        print(response.text)
        return []

def merge_tokens(tokens):
    merged_tokens = []
    for token in tokens:
        if merged_tokens:
            last_token = merged_tokens[-1]
            # Check if current token continues the last entity
            same_entity_type = last_token['entity'][-3:] == token['entity'][-3:]
            if (token['entity'].startswith('I-') or token['entity'].startswith('B-')) and same_entity_type:
                # Merge token words
                last_token['word'] += token['word'].replace('##', '')
                last_token['end'] = token['end']
                last_token['score'] = (last_token['score'] + token['score']) / 2
                continue
        # Otherwise, add as new entity
        merged_tokens.append(token)
    return merged_tokens


def ner(input):
    output = get_completion(input, parameters=None, ENDPOINT_URL=API_URL)
    merged_tokens = merge_tokens(output)
    return {"text": input, "entities": merged_tokens}

gr.close_all()
demo = gr.Interface(fn=ner,
                    inputs=[gr.Textbox(label="Text to find entities", lines=2)],
                    outputs=[gr.HighlightedText(label="Text with entities")],
                    title="NER with dslim/bert-base-NER",
                    description="Find entities using the `dslim/bert-base-NER` model under the hood!",
                    allow_flagging="never",
                    examples = ["My name is Santhosh and I live in India])

demo.launch(share=True, server_port=int(os.environ['PORT4']))
gr.close_all()

```
### OUTPUT:
<img width="780" height="457" alt="Screenshot 2025-11-14 103343" src="https://github.com/user-attachments/assets/c1459194-a96e-4973-9ca8-67010f4d78a1" />

### RESULT:
Thus the program todesign and develop a prototype application for Named Entity Recognition (NER) by leveraging a fine-tuned BART model and deploying the application using the Gradio framework for user interaction and evaluation was executed successfully.
