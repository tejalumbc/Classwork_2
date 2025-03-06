# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load the dataset
# Replace 'hamspam.csv' with the actual file path
data = pd.read_csv('hamspam.csv')

# Display the first few rows of the dataset
print(data.head())

# Encode categorical features into numerical values
label_encoder = LabelEncoder()
data['Contains Link'] = label_encoder.fit_transform(data['Contains Link'])
data['Contains Money Words'] = label_encoder.fit_transform(data['Contains Money Words'])
data['Length'] = label_encoder.fit_transform(data['Length'])

# Features (X) and Target (y)
X = data.drop('Class', axis=1)  # Features
y = data['Class']  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Naive Bayes Classifier
print("\nTraining Naive Bayes Classifier...")
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Predict using Naive Bayes
y_pred_nb = nb_classifier.predict(X_test)

# Evaluate Naive Bayes
print("\nNaive Bayes Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print("Classification Report:\n", classification_report(y_test, y_pred_nb))

# K-Nearest Neighbor (KNN) Classifier
print("\nTraining KNN Classifier...")
knn_classifier = KNeighborsClassifier(n_neighbors=2)  # Use k=2
knn_classifier.fit(X_train, y_train)

# Predict using KNN
y_pred_knn = knn_classifier.predict(X_test)

# Evaluate KNN
print("\nKNN Results:")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Classification Report:\n", classification_report(y_test, y_pred_knn))