# Text Sentiment Analysis with a Recursive Neural Network
## Overview
- This project involves the training of a Recurrent Neural Network (RNN) to perform text sentiment analysis.
- The idea of this project is to collect comments off of social media platforms and grade them according to 'toxicity' (levels of negativity).
- The model for the RNN is also included and can be used for any other function.
## Getting Started 
As a prerequisite please make sure that:
- All dependencies outlined in the requirements.txt file are installed either in a global environment of a virtual environment.
- Git the latest version of Git is installed 
- Python 3.9+ is installed and added to the system path
- It is expected that the GPU environment, if available, is correctly installed
### 0. Cloning Repository and Installing Dependencies
#### 1. Clone the repository
``` [Terminal]
C:\> git clone https://github.com/IVIOIST/Social-Media-Comment-Grading
```
#### 2. Installing dependencies via requirements.txt
``` [Terminal]
C:\> pip install -r requirements.txt
```
### 1. Overview of Files
There are three main source files in this repository are outlined below.
#### Training.ipynb
A jupyter notebook containing the code used to train the RNN.
#### Gradio.py
A simple implementation and proof of concept used for interacting with the trained model.
#### Youtube_Comment.py
A real world application of the model using data gathered from the Youtube Data API to outline the negativity and other characteristics of comments on any video on YouTube.
### 2. Using the Files
#### Training.ipynb
Please note that a trained model is already included and extra training does not have to be done.
#### 1. Running the Cells
Once opened click on the play button on each cell to run.\
Run the program from top to bottom.\
![Alt text](/images/train.png?raw=true "Title")
#### Gradio.py
#### 1. Running the Program
Click the play icon on the top right of the interface\
![Alt text](/images/gradio.png?raw=true "Title")
In the terminal (ctrl+shift+`) control click the link\
![Alt text](/images/terminal.png?raw=true "Title")
#### 2. Using Gradio
In the red box fill in the desired comment or sentece and press submit.\
![Alt text](/images/gradioint.png?raw=true "Title")
The returned results will be displayed on the right.\
![Alt text](/images/results.png?raw=true "Title")
#### Training.ipynb
#### 1. Running the program
Click the  play icon on the top right of the interface.\
![Alt text](/images/comment.png?raw=true "Title")
In the terminal (ctrl+shift+`) control click the link.\
![Alt text](/images/comterm.png?raw=true "Title")
#### 2. Using the application
Fill in the full URL for the desired YouTube Video.\
![Alt text](/images/vidurl.png?raw=true "Title")
Select the number of comments to analyse.\
![Alt text](/images/comnum.png?raw=true "Title")
Choose the order in which the comments are requested.\
![Alt text](/images/order.png?raw=true "Title")
Click submit and wait for results.\
![Alt text](/images/comresults.png?raw=true "Title")
## Troubleshooting 
### Module Not Found
```  [Terminal]
C:\> ModuleNotFoundError: No module named 'xxx'
```
Install the specified module 
```[Terminal]
C:\> pip install xxx
```
### GPU Is Not Being Utilised 
Check if tensorflow-gpu is installed
``` [Terminal]
C:\> pip list
```
Check if GPU is being recognised by the system and take note of the CUDA version.
``` [Terminal]
C:\> nvidia-smi
```
In jupyter notebooks or python run the following code to see how many GPUs and CPUs Tensorflow recognises
``` [Python]
Import tensorflow as tf

GPU = tf.config.list_physical_devices('GPU')
CPU = tf.config.list_physical_devices('CPU')
print(f'Available GPU/s:{len(GPU)}')
print(f'Available CPU/s:{len(CPU)}')
```
Once everything is checked go to https://www.tensorflow.org/install/source#tested_build_configurations and check if the versions match up if not, install the correct versions.
### For other errors please contact me, I will try my best to resolve them