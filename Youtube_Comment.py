# Importing dependencies

# visualisation 
import matplotlib.pyplot as plt
import gradio as gr
import seaborn as sbn
# math and data
import numpy as np
import pandas as pd
# google data api v3
from googleapiclient.discovery import build
# tensorflow
import tensorflow as tf
from tensorflow.keras.models import load_model
# general
import os


# Defining class for the tensorflow models
class Toxicity:
    def __init__(self):
        # using a keras function to load a saved model
        # prediction model
        self.toxicity = load_model(os.path.join('models', 'toxicity.h5'))
        # text vectorization model
        self.vect_model = load_model(os.path.join('models', 'vectorizer.tf')).layers[0]
    # taking the input and vectorizing it 
    def vect_str(self, input_str):
        self.vectorized = self.vect_model(input_str)
        return self.vectorized
    # takingg the vecotrized input and sending it through the Recursive Nueral Network (RNN)
    def predict(self, vectorized_string):
        results = self.toxicity.predict(vectorized_string)
        return results
# calling the class 
toxicity = Toxicity()

# using a simple method to keep the api key secret (The text file is excluded from to git as specified in .gitignore)
with open('api_key.txt') as myfile:
    DEVELOPER_KEY = myfile.read()
# calling  the youtube data api v3
youtube = build('youtube', 'v3', developerKey=DEVELOPER_KEY)    

# defining the core function for gradio
def scrape_comment(VidUrl, maxnum, order):
    # defining the labels of the prediction outputs for use later in creating a dictionary 
    labels_list = ['Negative', 'Toxic', 'Obscene', 'Threat', 'Insult', 'Identity hate']
    # using google data api v3 to scrape the data including comments off a scpecific video
    # (note the VidUrl is indexed to extract the video id as it is always in the same place for youtube urls)
    data = youtube.commentThreads().list(part='snippet', order=order, videoId=VidUrl[32:43], maxResults=maxnum).execute()
    comment_list = []
    for i in range(maxnum):
        # navigating to the specific location where the comment is stored
        comment = data['items'][i]['snippet']['topLevelComment']['snippet']['textDisplay']
        # appending it to a list consiting purely of comments for easier use 
        comment_list.append(comment)
    
    # vectorizing the comments in the comment list from a saved text vectorizer model
    vectorized_list = toxicity.vect_str(comment_list)
    # using a saved RNN model performing text sentiment analysis on the individual comments
    predicted_list = toxicity.predict(vectorized_list)
    
    # using a lambda function to efficiently and concisely work out the value of each of the labels 
    toxic_perc = (lambda x: int((sum(x)))/maxnum)(predicted_list[:, 0])
    stoxi_perc = (lambda x: int((sum(x)))/maxnum)(predicted_list[:, 1])
    osbce_perc = (lambda x: int((sum(x)))/maxnum)(predicted_list[:, 2])
    threa_perc = (lambda x: int((sum(x)))/maxnum)(predicted_list[:, 3])
    insul_perc = (lambda x: int((sum(x)))/maxnum)(predicted_list[:, 4])
    idhat_perc = (lambda x: int((sum(x)))/maxnum)(predicted_list[:, 5])
    results_percentage = [toxic_perc, stoxi_perc, osbce_perc, threa_perc, insul_perc, idhat_perc]
    # converting the two seperate lists of labels and predicted values into a single dictionary for gradio to read
    dictionary = (dict(zip(labels_list, results_percentage))) 
    # returning the dictionary
    return dictionary

# providing choices in the gradio dropdown 
choices = ['time', 'relevance']
# providing inputs for gradio
inputs = ['text', gr.Slider(0, 100, step=1), gr.Dropdown(choices=choices)]
# specifying the type of output gradio should excpect (note the top classes is 6 and the total classes outputed by the RNN is also 6)
outputs = [gr.Label(num_top_classes=6)]

# calling the interface function to load the gradio interface 
# (The sharing is set to false at the moment, change it to true to share the link across the web to other people)
gr.Interface(fn=scrape_comment, inputs=inputs, outputs=outputs).launch(share=False)