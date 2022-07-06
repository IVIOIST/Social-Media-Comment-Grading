from cProfile import label
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np 
import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras.models import load_model


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

labels_list = ['Toxic', 'Severe toxic', 'Obscene', 'Threat', 'Insult', 'Identity hate']

def score_comment(comment):
    comment = comment
    vectorized_comment = toxicity.vect_str([comment])
    prediction = toxicity.predict(vectorized_comment)
    text = ''
    prediction_list = []
    for idx, col in enumerate(labels_list):
        text += f'{col}:{round(prediction[0][idx]*100, 2)}%\n'
        prediction_list.append(float(prediction[0][idx]))
    dictionary = (dict(zip(labels_list, prediction_list)))   
    print(type(dictionary)) 
    return dictionary
  
outputs = [gr.Label(num_top_classes=6, label='Breakdown')]

if __name__ == '__main__':
    gr.Interface(fn=score_comment, inputs=gr.inputs.Textbox(lines=2, placeholder='Comment to score'),outputs=outputs).launch()