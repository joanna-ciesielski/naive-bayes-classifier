from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np

# Load example dataset (Iris dataset)
data = datasets.load_iris()
X = data.data  # Features
y = data.target  # Labels

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Gaussian Naive Bayes classifier
gnb = GaussianNB()

# Fit the classifier to the training data
gnb.fit(X_train, y_train)

# Predict on the test set
y_pred = gnb.predict(X_test)

# Calculate the posterior probability for each class
probabilities = gnb.predict_proba(X_test)

# Display results
print("Test Set Predictions: ", y_pred)
print("Posterior Probabilities: \n", probabilities)

# Accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Instructions for testing
"""
To test the script:
1. Make sure you have Python installed along with scikit-learn and numpy libraries.
2. Save this script as `naive_bayes_classifier.py`.
3. Run the script using the command `python naive_bayes_classifier.py`.
4. The script will display predictions, posterior probabilities, and the accuracy of the model.
"""
