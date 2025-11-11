"""
Unit tests for Model Evaluator Module
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from ml_pipeline.model_evaluator import ModelEvaluator


class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator class"""
    
    def setUp(self):
        """Set up test data and models"""
        # Create synthetic test data
        np.random.seed(42)
        self.X_train = np.random.rand(100, 10)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.rand(30, 10)
        self.y_test = np.random.randint(0, 2, 30)
        
        # Train simple models for testing
        self.rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.rf_model.fit(self.X_train, self.y_train)
        
        self.lr_model = LogisticRegression(random_state=42, max_iter=1000)
        self.lr_model.fit(self.X_train, self.y_train)
        
        # Create models dictionary
        self.models = {
            'random_forest': self.rf_model,
            'logistic_regression': self.lr_model
        }
        
        self.evaluator = ModelEvaluator(self.models)
        
        # Feature names for testing
        self.feature_names = [f'feature_{i}' for i in range(10)]
    
    def test_init(self):
        """Test ModelEvaluator initialization"""
        evaluator = ModelEvaluator(self.models)
        self.assertEqual(evaluator.models, self.models)
        self.assertEqual(evaluator.evaluation_results, {})
    
    def test_evaluate_model_returns_metrics(self):
        """Test that evaluate_model returns all required metrics"""
        metrics = self.evaluator.evaluate_model(self.rf_model, self.X_test, self.y_test)
        
        # Check that all metrics are present
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1_score', metrics)
        
        # Check that metrics are floats
        self.assertIsInstance(metrics['accuracy'], float)
        self.assertIsInstance(metrics['precision'], float)
        self.assertIsInstance(metrics['recall'], float)
        self.assertIsInstance(metrics['f1_score'], float)
        
        # Check that metrics are in valid range [0, 1]
        for metric_name, metric_value in metrics.items():
            self.assertGreaterEqual(metric_value, 0.0)
            self.assertLessEqual(metric_value, 1.0)
    
    def test_evaluate_model_with_known_predictions(self):
        """Test metric calculations with known predictions"""
        # Create a simple test case with known outcomes
        X_simple = np.array([[1], [2], [3], [4]])
        y_true = np.array([0, 0, 1, 1])
        
        # Create a model that predicts perfectly
        class PerfectModel:
            def predict(self, X):
                return y_true
        
        perfect_model = PerfectModel()
        metrics = self.evaluator.evaluate_model(perfect_model, X_simple, y_true)
        
        # All metrics should be 1.0 for perfect predictions
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
        self.assertEqual(metrics['f1_score'], 1.0)
    
    def test_generate_confusion_matrix(self):
        """Test confusion matrix generation"""
        cm = self.evaluator.generate_confusion_matrix(self.rf_model, self.X_test, self.y_test)
        
        # Check that confusion matrix is 2x2
        self.assertEqual(cm.shape, (2, 2))
        
        # Check that all values are non-negative integers
        self.assertTrue(np.all(cm >= 0))
        
        # Check that sum equals number of test samples
        self.assertEqual(cm.sum(), len(self.y_test))
    
    def test_generate_confusion_matrix_with_known_predictions(self):
        """Test confusion matrix with known predictions"""
        X_simple = np.array([[1], [2], [3], [4]])
        y_true = np.array([0, 0, 1, 1])
        
        # Create a model with known predictions
        class KnownModel:
            def predict(self, X):
                return np.array([0, 1, 1, 1])  # One false positive, one false negative
        
        known_model = KnownModel()
        cm = self.evaluator.generate_confusion_matrix(known_model, X_simple, y_true)
        
        # Expected confusion matrix:
        # [[1, 1],   # TN=1, FP=1
        #  [0, 2]]   # FN=0, TP=2
        self.assertEqual(cm[0, 0], 1)  # True Negatives
        self.assertEqual(cm[0, 1], 1)  # False Positives
        self.assertEqual(cm[1, 0], 0)  # False Negatives
        self.assertEqual(cm[1, 1], 2)  # True Positives
    
    def test_evaluate_all_models(self):
        """Test evaluation of all models"""
        results_df = self.evaluator.evaluate_all_models(self.X_test, self.y_test)
        
        # Check that DataFrame has correct shape
        self.assertEqual(len(results_df), len(self.models))
        
        # Check that all required columns are present
        expected_columns = ['model_name', 'accuracy', 'precision', 'recall', 'f1_score']
        for col in expected_columns:
            self.assertIn(col, results_df.columns)
        
        # Check that model names are correct
        model_names = set(results_df['model_name'].values)
        self.assertEqual(model_names, set(self.models.keys()))
        
        # Check that evaluation results are stored
        self.assertEqual(len(self.evaluator.evaluation_results), len(self.models))
    
    def test_evaluate_all_models_checks_accuracy_threshold(self):
        """Test that evaluate_all_models checks 71% accuracy threshold"""
        # This test just ensures the method runs without error
        # The actual threshold checking is done via print statements
        results_df = self.evaluator.evaluate_all_models(self.X_test, self.y_test)
        
        # Verify that results are returned
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertGreater(len(results_df), 0)
    
    def test_get_feature_importance(self):
        """Test feature importance extraction"""
        importance_df = self.evaluator.get_feature_importance(self.rf_model, self.feature_names)
        
        # Check that DataFrame has correct columns
        self.assertIn('feature', importance_df.columns)
        self.assertIn('importance', importance_df.columns)
        
        # Check that number of features matches
        self.assertEqual(len(importance_df), len(self.feature_names))
        
        # Check that features are sorted by importance (descending)
        importances = importance_df['importance'].values
        self.assertTrue(all(importances[i] >= importances[i+1] for i in range(len(importances)-1)))
        
        # Check that all importances are non-negative
        self.assertTrue(all(imp >= 0 for imp in importances))
    
    def test_get_feature_importance_raises_error_for_unsupported_model(self):
        """Test that get_feature_importance raises error for models without feature_importances_"""
        # Logistic Regression doesn't have feature_importances_ attribute
        with self.assertRaises(AttributeError) as context:
            self.evaluator.get_feature_importance(self.lr_model, self.feature_names)
        self.assertIn("feature_importances_", str(context.exception))
    
    def test_get_feature_importance_ranking(self):
        """Test that features are correctly ranked by importance"""
        importance_df = self.evaluator.get_feature_importance(self.rf_model, self.feature_names)
        
        # Get the most important feature
        most_important = importance_df.iloc[0]
        
        # Verify it has the highest importance
        self.assertEqual(most_important['importance'], importance_df['importance'].max())
    
    def test_generate_evaluation_report(self):
        """Test evaluation report generation"""
        report = self.evaluator.generate_evaluation_report(self.X_test, self.y_test)
        
        # Check that report is a string
        self.assertIsInstance(report, str)
        
        # Check that report contains key information
        self.assertIn('MODEL EVALUATION REPORT', report)
        self.assertIn('random_forest', report)
        self.assertIn('logistic_regression', report)
        self.assertIn('Accuracy', report)
        self.assertIn('Precision', report)
        self.assertIn('Recall', report)
        self.assertIn('F1-Score', report)
        self.assertIn('Confusion Matrix', report)
    
    def test_generate_evaluation_report_includes_confusion_matrix(self):
        """Test that evaluation report includes confusion matrix details"""
        report = self.evaluator.generate_evaluation_report(self.X_test, self.y_test)
        
        # Check that confusion matrix components are mentioned
        self.assertIn('True Negatives', report)
        self.assertIn('False Positives', report)
        self.assertIn('False Negatives', report)
        self.assertIn('True Positives', report)
    
    def test_plot_feature_importance_does_not_raise_error(self):
        """Test that plot_feature_importance runs without error"""
        importance_df = self.evaluator.get_feature_importance(self.rf_model, self.feature_names)
        
        # This test just ensures the method doesn't raise an error
        # We can't easily test the actual plot without a display
        try:
            # Set matplotlib to non-interactive backend to avoid display issues
            import matplotlib
            matplotlib.use('Agg')
            self.evaluator.plot_feature_importance(importance_df, top_n=5)
        except Exception as e:
            self.fail(f"plot_feature_importance raised an exception: {e}")
    
    def test_evaluation_results_stored_after_evaluate_all(self):
        """Test that evaluation results are stored in evaluator"""
        # Initially empty
        self.assertEqual(self.evaluator.evaluation_results, {})
        
        # After evaluation
        self.evaluator.evaluate_all_models(self.X_test, self.y_test)
        
        # Should have results for all models
        self.assertEqual(len(self.evaluator.evaluation_results), len(self.models))
        
        # Each result should have metrics
        for model_name, metrics in self.evaluator.evaluation_results.items():
            self.assertIn('accuracy', metrics)
            self.assertIn('precision', metrics)
            self.assertIn('recall', metrics)
            self.assertIn('f1_score', metrics)


if __name__ == '__main__':
    unittest.main()
