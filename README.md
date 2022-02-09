# Applied Text Mining 1: Methods
## Vrije Universiteit Amsterdam 
Group 1: Laura Alvarez, Ravi Meijer and Martijn Wesselink

For this course 5 assignments build up to making an automatic negation detector. 

**Assignment 1**
The Github was created. 

**Assignment 2**
Every member annotated ten documents independently on negation cues. 
The “saved” directories from the eHost annotation task were stored for every group member under the name “saved- groupNumber-annotatorName”. This directories are available in the IAA folder.

**Assignment 3** 
During this phase the preprocessing and feature extraction file was performed (run process_extract_final.py file). The code for preprocessing and feature extraction can be found in the preprocessing-feature-extraction folder. In addition, extra code developed for the class, but no included in the final version, can be found in the folder named extra.

**Assignment 4**
For assignment 4 the models were created. We created a baseline model, and experiment with SVMs, Naive Bayes and CRF. The code for this can be found in the folder named models. In additon, a hyperparameter search was performed on the CRF model, this implementation can be found in the hyperparameter-opt folder.

**Assignment 5**
For assigment 5 we performed an error analysis to evaluate the models created during assigment 4. the code for this is available in the folder error analysis.


The data used for this experimentation can be found in the folder data. The code for each of the tasks described above is 


We have also provided a requirements file that can be used to create an enviroment to test the code.

using pip
pip install -r requirements.txt

using Conda
conda create --name <env_name> --file requirements.txt
