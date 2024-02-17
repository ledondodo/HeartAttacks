# Classification for prediction of heart attacks

## *Class Project 1, Machine Learning CS-433, EPFL*

## :dart: Aim
This project employs various machine learning methods for a classification task aiming to predict who has an enhanced susceptibility to a heart attack. By doing so, we want to enhance our understanding of CVD and contribute to more effective prevention and management strategies.

## :bar_chart: Dataset
The dataset can be downloaded [here](https://github.com/epfml/ML_course/tree/master/projects/project1/data) and need to be located in a folder `dataset` next to the repository folder.
Behavioral Risk Factor Surveillance System (BRFSS) offers us a large health survey dataset to conduct an analysis of CVD risk factors. It includes many features describing U.S. residents' health-related risk behaviors, chronic health conditions, and use of preventive services. 

## :handshake: Contributors
Arthur Chansel, Marianne Scoglio & Gilles de Waha

This is a private school project.

## :microscope: Project description and results
Logistic regression with gradient descent was used to predict heart attack risk (class 1) or no risk (class -1). Various techniques were used to improve and evaluate the model: feature selection and processing, addition of penalty term and class weights, tuning of hyperparameters, K-fold cross validation. All the knowledge comes from the previous classes and from researches on internet.

The best model we obtained reaches a F1 score of 0.426.

## :card_file_box: Files and contents

`run.ipynb`: code for the relevant content and results, from data loading to final predictions.

`implementations.py`: different types of regression functions.

`utility.py`: all the functions related to the model conception.

`feature_engineering.py`: all the functions related to feature selection and processing.

`helpers.py`: all the functions related to data loading and submission.

`cross_validation.py`: all the functions related to cross validation.

`submission.csv`: the final predictions of the best model.

## :thinking: How does it work?
To generate the predictions, open the notebook `run.ipynb` and run all the cells in the order.


## :book: Requirements
Programming language: Python
-  Python >= 3.6
- Libraries and modules: 
  - numpy
  - matplotlib
  - csv
  - os
