import os
import matplotlib.pyplot as plt
import gradio as gr
import numpy as np
import pandas as pd
import seaborn as sbn
import tensorflow as tf
from googleapiclient.discovery import build
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

with open('api_key.txt') as myfile:
    DEVELOPER_KEY = myfile.read()
youtube = build('youtube', 'v3', developerKey=DEVELOPER_KEY)    

def scrape_comment(id, maxnum, order):
    labels_list = ['Negative', 'Toxic', 'Obscene', 'Threat', 'Insult', 'Identity hate']
    data = youtube.commentThreads().list(part='snippet', order=order, videoId=id, maxResults=maxnum).execute()
    comment_list = []
    for i in range(10):
        comment = data['items'][i]['snippet']['topLevelComment']['snippet']['textDisplay']
        comment_list.append(comment)
    
    vectorized_list = toxicity.vect_str(comment_list)
    predicted_list = toxicity.predict(vectorized_list)
    
    toxic_perc = (lambda x: int((sum(x)))/maxnum)(predicted_list[:, 0])
    stoxi_perc = (lambda x: int((sum(x)))/maxnum)(predicted_list[:, 1])
    osbce_perc = (lambda x: int((sum(x)))/maxnum)(predicted_list[:, 2])
    threa_perc = (lambda x: int((sum(x)))/maxnum)(predicted_list[:, 3])
    insul_perc = (lambda x: int((sum(x)))/maxnum)(predicted_list[:, 4])
    idhat_perc = (lambda x: int((sum(x)))/maxnum)(predicted_list[:, 5])
    results_percentage = [toxic_perc, stoxi_perc, osbce_perc, threa_perc, insul_perc, idhat_perc]
    dictionary = (dict(zip(labels_list, results_percentage))) 

    return dictionary

choices = ['time', 'relevance']
inputs = ['text', gr.Slider(0, 100, step=1), gr.Dropdown(choices=choices)]
outputs = [gr.Label(num_top_classes=6)]

gr.Interface(fn=scrape_comment, inputs=inputs, outputs=outputs).launch(share=False)