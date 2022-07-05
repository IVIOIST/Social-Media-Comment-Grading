import matplotlib.pyplot as plt
import gradio as gr
import numpy as np 
import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization

labels_list = ['Toxic', 'Severe toxic', 'Obscene', 'Threat', 'Insult', 'Identity hate']

class Toxicity:
    def __init__(self):
        self.toxicity = load_model('toxicity.h5')
        self.vect_model = load_model('vectorizer.tf').layers[0]

    def vect_str(self, input_str):
        self.vectorized = self.vect_model(input_str)
        return self.vectorized
    
    def predict(self, vectorized_string):
        results = self.toxicity.predict(vectorized_string)
        return results

toxicity = Toxicity()

def score_comment(comment):
    vectorized_comment = toxicity.vect_str([comment])
    prediction = toxicity.predict(vectorized_comment)

    text = ''
    for idx, col in enumerate(labels_list):
        text += f'{col}:{round(prediction[0][idx]*100, 2)}%\n'
    return text

interface = gr.Interface(fn=score_comment, inputs=gr.inputs.Textbox(lines=2, placeholder='Comment to score'),outputs='text')

interface.launch(share=True)
