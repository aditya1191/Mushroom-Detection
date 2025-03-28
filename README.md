Details :

Name: Aditya Garimella

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
Project Description:

We are trying to detect if the mushrooms in the dataset are poisonous or edible. 
we are using Artificial neural network for the same.

# Dataset.py
This file is used to split the dataset into three parts: training, validation, and testing. 
We convert the dataset into a one-hot binary format using `pandas.get_dummies()` essentially 
converting the . Then, with the help of NumPy's `split` function which accepts array, 
indices for split. I have divided the dataset into three parts: 60% for training (`training.txt`), 
20% for validation (`val.txt`), and 20% for testing (`testing.txt`).

# formulas.py
In this file, we implement formulas for our model using functions from the NumPy library.

# models.py
The functions  file. The `eval()` function utilizes NumPy's `dot()` function and the 
sigmoid function, both implemented in the `formulas.py` file. Additionally, the `backdrop()` 
function is calculated using the learning rate and other formulas.

# proj_test.py
This file contains the code for training and fitting the model. We call the `eval()` function, 
followed by the `backdrop()` function which implemented backpropagation, and calculate the error 
values for the dataset.

# output.docx
This document contains the output along with the error recevied for each of the datasets training, validation, testing.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Steps to run the code:

1. Install modules using requirements.txt 

2. Command:   run Dataset.py for converting orginal data into required format.  
   command --> python Dataset.py

3. Run proj_text.py for getting the final output
   command --> python proj_test.py

   ->  press enter to start the validation process. This will result in validation error value

   ->  Then press enter to start the testing process. the final test error value will be the output.

FYI: cfile class in models.py is modified according to latest version.
     raw_input() is no longer working in latest edition of python => it is replaced with input().

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
