"""
Iris Flower Classification - A Beginner-Friendly Machine Learning Project
This script demonstrates how to:
1. Load the Iris dataset
2. Analyze the data
3. Split data into training and testing sets
4. Train a classification model
5. Evaluate model accuracy
6. Make predictions on new data
"""

# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("IRIS FLOWER CLASSIFICATION PROJECT")
print("=" * 60)

# Step 1: Load the Iris dataset
print("\n[Step 1] Loading Iris Dataset...")
iris = load_iris()
X = iris.data  # Features (Sepal Length, Sepal Width, Petal Length, Petal Width)
y = iris.target  # Target (Species: 0=Setosa, 1=Versicolor, 2=Virginica)

print(f"Dataset loaded successfully!")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of classes: {len(np.unique(y))}")

# Step 2: Analyze the data
print("\n[Step 2] Data Analysis...")
df = pd.DataFrame(X, columns=iris.feature_names)
df['Species'] = iris.target_names[y]

print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset statistics:")
print(df.describe())

print("\nSpecies distribution:")
print(df['Species'].value_counts())

# Step 3: Split data into training and testing sets
print("\n[Step 3] Splitting Data into Training (80%) and Testing (20%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Step 4: Train the classification model
print("\n[Step 4] Training Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=100,  # Number of decision trees
    random_state=42,
    max_depth=10
)
model.fit(X_train, y_train)
print("Model training completed!")

# Step 5: Evaluate model accuracy
print("\n[Step 5] Model Evaluation...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.2%}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Step 6: Feature importance
print("\n[Step 6] Feature Importance...")
feature_importance = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance Ranking:")
print(feature_importance)

# Step 7: Make sample predictions
print("\n[Step 7] Sample Predictions...")
print("\nSample 1:")
sample1 = np.array([[5.1, 3.5, 1.4, 0.2]])
pred1 = model.predict(sample1)[0]
pred1_proba = model.predict_proba(sample1)[0]
print(f"  Features: Sepal Length=5.1, Sepal Width=3.5, Petal Length=1.4, Petal Width=0.2")
print(f"  Predicted Species: {iris.target_names[pred1]}")
print(f"  Confidence Scores: {dict(zip(iris.target_names, [f'{p:.2%}' for p in pred1_proba]))}")

print("\nSample 2:")
sample2 = np.array([[6.5, 3.0, 5.5, 1.8]])
pred2 = model.predict(sample2)[0]
pred2_proba = model.predict_proba(sample2)[0]
print(f"  Features: Sepal Length=6.5, Sepal Width=3.0, Petal Length=5.5, Petal Width=1.8")
print(f"  Predicted Species: {iris.target_names[pred2]}")
print(f"  Confidence Scores: {dict(zip(iris.target_names, [f'{p:.2%}' for p in pred2_proba]))}")

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)
