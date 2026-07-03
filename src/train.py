"""
Model Training Script
Trains a classification model on the Iris dataset and saves it to disk.
"""

import numpy as np
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os


def train_model():
    """Train a Random Forest classifier on Iris dataset."""
    
    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize and train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10
    )
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Print metrics
    print(f"Model Training Complete")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Save model
    os.makedirs('C:\\Users\\GOPINATH\\CascadeProjects\\unit3-exp1-advanced-ci-pipeline\\models\\models', exist_ok=True)
    joblib.dump(model, 'C:\\Users\\GOPINATH\\CascadeProjects\\unit3-exp1-advanced-ci-pipeline\\models\\model.joblib')
    print("Model saved to\models\\model.joblib")
    
    # Save metrics for validation
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    joblib.dump(metrics, 'C:\\Users\\GOPINATH\\CascadeProjects\\unit3-exp1-advanced-ci-pipeline\\models\\metrics.joblib')
    print("Metrics saved to  \models\\metrics.joblib")
    
    return metrics


if __name__ == '__main__':
    train_model()
