"""
Model Validation Script
Validates trained model metrics against defined thresholds.
"""

import argparse
import joblib
import os
import sys


def validate_metrics(metrics_path, accuracy_threshold, precision_threshold, 
                    recall_threshold, f1_threshold):
    """
    Validate model metrics against thresholds.
    
    Args:
        metrics_path: Path to metrics file
        accuracy_threshold: Minimum acceptable accuracy
        precision_threshold: Minimum acceptable precision
        recall_threshold: Minimum acceptable recall
        f1_threshold: Minimum acceptable F1 score
        
    Returns:
        bool: True if all metrics pass thresholds
    """
    # Load metrics
    if not os.path.exists(metrics_path):
        print(f"ERROR: Metrics file not found at {metrics_path}")
        sys.exit(1)
    
    metrics = joblib.load(metrics_path)
    
    print("=" * 60)
    print("MODEL METRIC VALIDATION")
    print("=" * 60)
    
    # Validate each metric
    results = {}
    
    # Accuracy
    accuracy = metrics['accuracy']
    accuracy_pass = accuracy >= accuracy_threshold
    results['accuracy'] = {
        'value': accuracy,
        'threshold': accuracy_threshold,
        'pass': accuracy_pass
    }
    print(f"Accuracy: {accuracy:.4f} (Threshold: {accuracy_threshold:.4f}) - {'PASS' if accuracy_pass else 'FAIL'}")
    
    # Precision
    precision = metrics['precision']
    precision_pass = precision >= precision_threshold
    results['precision'] = {
        'value': precision,
        'threshold': precision_threshold,
        'pass': precision_pass
    }
    print(f"Precision: {precision:.4f} (Threshold: {precision_threshold:.4f}) - {'PASS' if precision_pass else 'FAIL'}")
    
    # Recall
    recall = metrics['recall']
    recall_pass = recall >= recall_threshold
    results['recall'] = {
        'value': recall,
        'threshold': recall_threshold,
        'pass': recall_pass
    }
    print(f"Recall: {recall:.4f} (Threshold: {recall_threshold:.4f}) - {'PASS' if recall_pass else 'FAIL'}")
    
    # F1 Score
    f1 = metrics['f1']
    f1_pass = f1 >= f1_threshold
    results['f1'] = {
        'value': f1,
        'threshold': f1_threshold,
        'pass': f1_pass
    }
    print(f"F1 Score: {f1:.4f} (Threshold: {f1_threshold:.4f}) - {'PASS' if f1_pass else 'FAIL'}")
    
    print("=" * 60)
    
    # Overall result
    all_pass = all(result['pass'] for result in results.values())
    
    if all_pass:
        print("RESULT: ALL METRICS PASSED - Model validation successful")
        return True
    else:
        failed_metrics = [name for name, result in results.items() if not result['pass']]
        print(f"RESULT: VALIDATION FAILED - Failed metrics: {', '.join(failed_metrics)}")
        sys.exit(1)


def main():
    """Main entry point for model validation."""
    parser = argparse.ArgumentParser(description='Validate model metrics against thresholds')
    parser.add_argument('--accuracy', type=float, default=0.85, help='Minimum accuracy threshold')
    parser.add_argument('--precision', type=float, default=0.80, help='Minimum precision threshold')
    parser.add_argument('--recall', type=float, default=0.80, help='Minimum recall threshold')
    parser.add_argument('--f1', type=float, default=0.80, help='Minimum F1 score threshold')
    parser.add_argument('--metrics-path', type=str, default='../models/metrics.joblib', help='Path to metrics file')
    
    args = parser.parse_args()
    
    validate_metrics(
        metrics_path=args.metrics_path,
        accuracy_threshold=args.accuracy,
        precision_threshold=args.precision,
        recall_threshold=args.recall,
        f1_threshold=args.f1
    )


if __name__ == '__main__':
    main()
