# Comment Toxicity Grading
## Overview
This project involves the training of a Recurrent Neural Network (RNN) to perform text sentiment analysis.\
The idea of this project is to collect comments off of social media platforms and grade them accorting to 'toxicity'.\
However, the model for the RNN is also included and can be used for any other function.
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
Check if GPU is being recognised by the system and the  version of CUDA
``` [Terminal]
C:\> nvidia-smi
```
In jupyter notebooks or python run the following command to see how many GPUs and CPUs Tensorflow recognises
``` [Python]
Import tensorflow as tf

GPU = tf.config.list_physical_devices('GPU')
CPU = tf.config.list_physical_devices('CPU')
print(f'Available GPU/s:{len(GPU)}')
print(f'Available CPU/s:{len(CPU)}')
```
Once everything is checked go to https://www.tensorflow.org/install/source#tested_build_configurations and check if the versions match up if not, install the correct versions.