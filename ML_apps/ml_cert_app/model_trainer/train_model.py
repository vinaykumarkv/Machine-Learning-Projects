import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Generate synthetic data
X, y = make_classification(
    n_samples=1000,     # number of rows
    n_features=11,      # number of input features
    n_informative=8,    # features actually used
    n_redundant=2,      # correlated ones
    n_classes=2,        # binary classification
    random_state=42
)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate (just to check)
accuracy = model.score(X_test, y_test)
print(f"Model trained successfully! Test Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "certificate_model.pkl")
print("Model saved as certificate_model.pkl")
