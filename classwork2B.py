# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('hamspam.csv')

# Display the first few rows of the dataset
print(data.head())

# Encode categorical features into numerical values
label_encoder = LabelEncoder()
data['Contains Link'] = label_encoder.fit_transform(data['Contains Link'])
data['Contains Money Words'] = label_encoder.fit_transform(data['Contains Money Words'])
data['Length'] = label_encoder.fit_transform(data['Length'])

# Encode the target variable
data['Class'] = label_encoder.fit_transform(data['Class'])

# Features (X) and Target (y)
X = data.drop('Class', axis=1)  # Features
y = data['Class']  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Naive Bayes Classifier
print("\nTraining Naive Bayes Classifier...")
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Predict probabilities for the test set
y_pred_proba = nb_classifier.predict_proba(X_test)[:, 1]  # Probabilities for the positive class (Spam)

# Calculate ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Print AUC score
print(f"\nAUC Score: {roc_auc:.2f}")