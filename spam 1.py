import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
# Ensure the dataset has "label" and "message" columns
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['label', 'message']]
data['label'] = data['label'].map({'spam': 1, 'ham': 0})

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.3, random_state=42)

# Vectorize the messages using TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)  # Limiting features can improve both accuracy and speed
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Set up Logistic Regression with hyperparameter tuning
log_reg = LogisticRegression(max_iter=1000)
param_grid = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'solver': ['liblinear', 'lbfgs']  # Solvers to try
}
grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_vectorized, y_train)

# Get the best model from Grid Search
best_model = grid_search.best_estimator_

# Make predictions and evaluate
y_pred = best_model.predict(X_test_vectorized)

# Print model accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimized Model Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
