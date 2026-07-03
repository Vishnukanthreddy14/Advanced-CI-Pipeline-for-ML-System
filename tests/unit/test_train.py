 """
Unit tests for model training module.
"""

import pytest
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))


def test_imports():
    """Test that required modules can be imported."""
    import sklearn
    import pandas
    import numpy
    import joblib
    assert sklearn is not None
    assert pandas is not None
    assert numpy is not None
    assert joblib is not None


def test_model_training():
    """Test that model training produces valid output."""
    from train import train_model

    # Train model
    metrics = train_model()

    # Check that metrics are returned
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics

    # Check that metrics are in valid range
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1'] <= 1

    # Check that model file is created
    assert os.path.exists('../models/model.joblib')
    assert os.path.exists('../models/metrics.joblib')


def test_model_file_exists():
    """Test that model file exists after training."""
    assert os.path.exists('../models/model.joblib')


def test_metrics_file_exists():
    """Test that metrics file exists after training."""
    assert os.path.exists('../models/metrics.joblib')


def test_model_loading():
    """Test that trained model can be loaded."""
    import joblib

    model = joblib.load('../models/model.joblib')
    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')
