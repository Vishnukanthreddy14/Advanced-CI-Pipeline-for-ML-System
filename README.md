# Unit 3 - Experiment 1: Advanced CI Pipeline for ML System

## Overview
This experiment implements a production-grade GitHub Actions CI pipeline for ML systems with comprehensive quality gates including code validation, dependency installation, unit testing, API testing, model metric validation, and Docker image building.

## Architecture
The CI pipeline consists of 6 sequential jobs:
1. **Code Validation**: Flake8 linting, Black format checking, MyPy type checking
2. **Dependency Installation**: Automated Python environment setup with caching
3. **Unit Testing**: Pytest execution with coverage reporting
4. **API Testing**: Automated REST API endpoint testing
5. **Model Metric Validation**: Validation against accuracy, precision, recall, F1 thresholds
6. **Docker Build**: Multi-stage Docker image build and validation

## Directory Structure
```
unit3-exp1-advanced-ci-pipeline/
├── .github/
│   └── workflows/
│       └── ci-pipeline.yml          # GitHub Actions CI workflow
├── src/
│   ├── __init__.py
│   ├── app.py                       # FastAPI application
│   ├── train.py                     # Model training script
│   └── validate_model.py            # Model validation script
├── tests/
│   ├── __init__.py
│   ├── unit/
│   │   ├── __init__.py
│   │   └── test_train.py            # Unit tests for training
│   └── api/
│       ├── __init__.py
│       └── test_api.py              # API integration tests
├── models/                          # Trained model storage
├── data/                            # Data storage
├── Dockerfile                       # Multi-stage Docker build
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Prerequisites
- GitHub repository
- Docker installed (for local testing)
- Python 3.10+

## Verification Steps

### 1. Local Testing - Model Training
```bash
cd unit3-exp1-advanced-ci-pipeline
pip install -r requirements.txt
python src/train.py
```
**Expected Output**: Model trained and saved to `models/model.joblib` with metrics printed.

### 2. Local Testing - API Server
```bash
python src/app.py
```
**Expected Output**: Server starts on `http://localhost:8000`. Access `http://localhost:8000/docs` for Swagger UI.

### 3. Local Testing - API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

### 4. Local Testing - Unit Tests
```bash
pytest tests/unit/ -v --cov=src
```
**Expected Output**: All unit tests pass with coverage report.

### 5. Local Testing - Model Validation
```bash
python src/validate_model.py --accuracy 0.85 --precision 0.80 --recall 0.80 --f1 0.80
```
**Expected Output**: All metrics pass validation thresholds.

### 6. Local Testing - Docker Build
```bash
docker build -t ml-pipeline:latest .
docker run --rm ml-pipeline:latest python -c "import sklearn; print('Docker image validated')"
```

### 7. GitHub Actions CI Pipeline
1. Push code to GitHub repository
2. Navigate to **Actions** tab
3. Verify workflow triggers on push/PR
4. Monitor job execution:
   - Code Validation
   - Dependency Installation
   - Unit Testing
   - API Testing
   - Model Metric Validation
   - Docker Build
5. Confirm all jobs pass successfully

## Key Features
- **Automated Code Quality**: Flake8, Black, MyPy integration
- **Dependency Caching**: Faster pipeline execution
- **Comprehensive Testing**: Unit and API test coverage
- **Model Quality Gates**: Metric-based validation
- **Containerization**: Multi-stage Docker build
- **Artifact Management**: Docker image upload and retention

## Pipeline Thresholds
- Accuracy: ≥ 0.85
- Precision: ≥ 0.80
- Recall: ≥ 0.80
- F1 Score: ≥ 0.80

## Next Steps
Proceed to **Experiment 2: Advanced Testing Pipeline with Quality Gates (PiTU)** to implement automated data schema validation and enhanced quality gates.
