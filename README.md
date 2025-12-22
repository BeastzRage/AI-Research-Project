# AI-Research-Project Marouan El Marnissi s0222555

## Overview
The code in this repository was created to answer the research question: **"Does encoding a “negative” interaction as a
negative value in the interaction matrix improve accuracy when using ItemKNN?"** for my final project of the Artificial 
Intelligence Project course at the University of Antwerp. Running `main.py` will generate a CSV in the recommendation_results 
folder containing item recommendations. This CSV was then zipped and uploaded to Codabench to obtain the recommendation metrics.


## Project Structure

### Project root

`main.py:` The file that brings everything together. Running `main.py` loads and processes the training and test data, 
tunes the neighborhood size hyperparameter for ItemKNN, calculates ItemKNN scores, and generates a CSV file with recommendations.
A local accuracy test is also run at the end but this is not required to generate recommendations.

`requirements.txt:` Required libraries to run the code.

`README.md:` The file you are currently reading.

### dataset folder 

`user_reviews.CSV :` CSV file containing the data used for training the ItemKNN recommender algorithm

`train_interactions_in.CSV :` CSV file containing the fold in data used for generating the recommendations with ItemKNN

### recommendation_results folder

After running `main.py`, the generated recommendations are saved as a CSV file in this folder.

`CSV :` contains all CSV files used during the presentation of my research

### src folder

`DataPreprocessor.py:` Class that remaps user and item IDs to row and column indices in dataframes and converts dataframes 
into sparse CSR matrices

`HelperFunctions.py:` File containing utility and helper functions

`ItemKNN.py:` Class that implements the ItemKNN algorithm using cosine similarity. The neighborhood size k can be 
tuned by the class itself or manually assigned.

`metrics.py:` File containing methods to calculate different recommendation metrics.

`NCoreFilter.py:` Class that filters a CSR matrix so that every row and column have at least N non-zero interactions

`StrongGeneralizationSplitter.py:` Class that makes a strong generalization split from a CSR matrix.


### unit_tests folder
`DataPreprocessor_test.py:` Unit tests for the DataPreprocessor class.

`NCoreFilter_test.py:` Unit tests for the NCoreFilter class.

`StrongGeneralizationSplitter_test.py:` Unit tests for the StrongGeneralizationSplitter class.


## setup

Before running the code, install the required libraries by running:
  ```
  pip install -r requirements.txt
  ```
  

