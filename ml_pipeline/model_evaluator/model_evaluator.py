"""
Model Evaluator Module

This module provides functionality for evaluating trained ML models,
calculating performance metrics, and generating evaluation reports.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


class ModelEvaluator:
    """Evaluates trained models and generates performance reports"""
    
    def __init__(self, models: Dict[str, Any]):
        """
        Initialize ModelEvaluator with trained models
        
        Args:
            models: Dictionary of trained models with model names as keys
        """
        self.models = models
        self.evaluation_results = {}
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Calculate accuracy, precision, recall, and F1-score for a model
        
        Args:
            model: Trained model with predict() method
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing accuracy, precision, recall, and F1-score
        """
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='binary', zero_division=0)
        }
        
        return metrics

    def generate_confusion_matrix(self, model, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        """
        Generate confusion matrix for model predictions
        
        Args:
            model: Trained model with predict() method
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Confusion matrix as numpy array
        """
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        return cm
    
    def evaluate_all_models(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Evaluate all models and return comparison DataFrame
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with evaluation metrics for all models
        """
        results = []
        
        for model_name, model in self.models.items():
            # Evaluate model
            metrics = self.evaluate_model(model, X_test, y_test)
            
            # Add model name to metrics
            metrics['model_name'] = model_name
            
            # Store results
            results.append(metrics)
            self.evaluation_results[model_name] = metrics
            
            # Check if accuracy meets 71% threshold
            if metrics['accuracy'] >= 0.71:
                print(f"{model_name}: Accuracy {metrics['accuracy']:.2%} meets 71% threshold ✓")
            else:
                print(f"{model_name}: Accuracy {metrics['accuracy']:.2%} below 71% threshold")
        
        # Create DataFrame with model_name as first column
        df = pd.DataFrame(results)
        cols = ['model_name', 'accuracy', 'precision', 'recall', 'f1_score']
        df = df[cols]
        
        return df

    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """
        Extract and rank feature importance scores from model
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            
        Returns:
            DataFrame with features ranked by importance in descending order
        """
        # Check if model has feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            raise AttributeError(
                f"Model does not have feature_importances_ attribute. "
                f"Feature importance is only available for tree-based models."
            )
        
        # Extract feature importances
        importances = model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort by importance in descending order
        importance_df = importance_df.sort_values('importance', ascending=False)
        importance_df = importance_df.reset_index(drop=True)
        
        return importance_df
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 15) -> None:
        """
        Generate horizontal bar chart visualization of feature importance
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            top_n: Number of top features to display (default: 15)
        """
        # Get top N features
        top_features = importance_df.head(top_n)
        
        # Create horizontal bar chart
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Most Important Features')
        plt.gca().invert_yaxis()  # Highest importance at top
        plt.tight_layout()
        plt.show()

    def generate_evaluation_report(self, X_test: np.ndarray, y_test: np.ndarray) -> str:
        """
        Generate formatted text report of all metrics
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Formatted string containing evaluation report
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MODEL EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        for model_name, model in self.models.items():
            report_lines.append(f"Model: {model_name}")
            report_lines.append("-" * 80)
            
            # Get metrics
            metrics = self.evaluate_model(model, X_test, y_test)
            
            # Format metrics
            report_lines.append(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})")
            report_lines.append(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']:.2%})")
            report_lines.append(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']:.2%})")
            report_lines.append(f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']:.2%})")
            report_lines.append("")
            
            # Generate confusion matrix
            cm = self.generate_confusion_matrix(model, X_test, y_test)
            report_lines.append("  Confusion Matrix:")
            report_lines.append(f"                Predicted 0  Predicted 1")
            report_lines.append(f"    Actual 0    {cm[0][0]:>11}  {cm[0][1]:>11}")
            report_lines.append(f"    Actual 1    {cm[1][0]:>11}  {cm[1][1]:>11}")
            report_lines.append("")
            report_lines.append(f"    True Negatives:  {cm[0][0]}")
            report_lines.append(f"    False Positives: {cm[0][1]}")
            report_lines.append(f"    False Negatives: {cm[1][0]}")
            report_lines.append(f"    True Positives:  {cm[1][1]}")
            report_lines.append("")
            report_lines.append("=" * 80)
            report_lines.append("")
        
        return "\n".join(report_lines)
