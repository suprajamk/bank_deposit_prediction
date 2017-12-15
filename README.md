# bank_deposit_prediction

The data is related with direct marketing campaigns of a Portuguese banking institution. 
The task is to predict whether the product (bank term deposit) would be ('yes') or not ('no') subscribed. 
Namely, it's a binary classification task.

The dataset is from the in-class kaggle, given as train and test

## Data mining algorithms applied:

Linear Discriminants, Logistic Regression, Gaussian Naive Bayes, 
Decision Tree, Random Forest Classifier, KNN,
Gradient Boosting Classifier

Models are evaluated by MCC scores as the dataset is imbalanced.

## Analysis

Got best MCC score for Gradient Boosting Classifier. Hence it was used
for final submission

## Required libraries :

Python Version = 3.6

Pandas = 0.21.0

sklearn


## Files:
data_analysis.py --> Analysis of Data and different Classification Algorithms

submission.py --> Classification file to create 'submission.csv'

## To run the Gradient boost classifier for the data set:

python code/submission.py

## Future Work
Find right parameters for Decision tree, Random Forest Classifier and
Gradient Boost Classifier and use Majority Voting Classifier over them.