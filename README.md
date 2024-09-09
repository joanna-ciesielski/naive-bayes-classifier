# naive-bayes-classifier
Naive Bayes classifier for the Iris dataset using Python and scikit-learn.

# Naive Bayes Classifier for Iris Dataset

This project demonstrates a simple implementation of a Gaussian Naive Bayes classifier using the `scikit-learn` library in Python. The classifier is applied to the Iris dataset, a standard dataset used for classification problems.

## Program Description

The Python script `naive_bayes_classifier.py` uses the Gaussian Naive Bayes model to classify iris flower species based on four features: sepal length, sepal width, petal length, and petal width. The script calculates posterior probabilities for each class and evaluates the model's accuracy.

- **Dataset:** Iris dataset from the `scikit-learn` library, containing 150 samples of iris flowers divided into three classes: setosa, versicolor, and virginica.
- **Model:** Gaussian Naive Bayes, which assumes that the features follow a Gaussian distribution.
- **Output:** Test set predictions, posterior probabilities, and model accuracy.

## Requirements

- Python 3.x
- `scikit-learn` library
- `numpy` library

## Installation

To install the required libraries, run:

```bash
pip install scikit-learn numpy

