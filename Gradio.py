# Importing dependencies

# general
import os 
# visualisation
import matplotlib.pyplot as plt
import gradio as gr
# loading tensorflow models
from tensorflow.keras.models import load_model

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

# defining the list in which the results are outputed 
labels_list = ['Toxic', 'Severe toxic', 'Obscene', 'Threat', 'Insult', 'Identity hate']

# defining a function in which gradio takes as its main
def score_comment(comment):
    # vectorizing comment
    comment = comment
    vectorized_comment = toxicity.vect_str([comment])
    prediction = toxicity.predict(vectorized_comment)
    text = ''
    prediction_list = []
    # looping through the list of predicted values and changing them into a more readable format using f strings
    for idx, col in enumerate(labels_list):
        text += f'{col}:{round(prediction[0][idx]*100, 2)}%\n'
        prediction_list.append(float(prediction[0][idx]))
    # changing the predictions into a dictionary for gradio to read as labels
    dictionary = (dict(zip(labels_list, prediction_list)))
    # returning both the dictionary and the formatted string
    return dictionary, text
# defining outputs for gradio
outputs = [gr.Label(num_top_classes=6, label='Breakdown'), 'text']

# initialising the gradio interface, defining the function, inputs and outputs
if __name__ == '__main__':
    gr.Interface(fn=score_comment, inputs=gr.inputs.Textbox(lines=2, placeholder='Comment to score'),outputs=outputs).launch()