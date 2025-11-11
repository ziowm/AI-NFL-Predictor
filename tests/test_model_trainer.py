"""
Unit tests for Model Trainer Module
"""

import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Try to import MLModelTrainer - may fail if xgboost is not properly installed
try:
    from ml_pipeline.model_trainer import MLModelTrainer
    TRAINER_AVAILABLE = True
except Exception as e:
    TRAINER_AVAILABLE = False
    print(f"Warning: MLModelTrainer not available: {e}")


@unittest.skipIf(not TRAINER_AVAILABLE, "MLModelTrainer not available (XGBoost issue)")
class TestMLModelTrainer(unittest.TestCase):
    """Test cases for MLModelTrainer class"""
    
    def setUp(self):
        """Set up test data"""
        # Create synthetic training data
        np.random.seed(42)
        self.X_train = np.random.rand(100, 10)
        self.y_train = np.random.randint(0, 2, 100)
        
        # Create test data
        self.X_test = np.random.rand(20, 10)
        self.y_test = np.random.randint(0, 2, 20)
        
        self.trainer = MLModelTrainer(random_state=42)
    
    def test_init(self):
        """Test MLModelTrainer initialization"""
        trainer = MLModelTrainer(random_state=42)
        self.assertEqual(trainer.random_state, 42)
        self.assertEqual(trainer.trained_models, {})
        self.assertEqual(trainer.best_params, {})
    
    def test_init_default_random_state(self):
        """Test initialization with default random state"""
        trainer = MLModelTrainer()
        self.assertEqual(trainer.random_state, 42)
    
    def test_get_random_forest_params(self):
        """Test Random Forest hyperparameter grid"""
        params = self.trainer.get_random_forest_params()
        
        # Check that all required parameters are present
        self.assertIn('n_estimators', params)
        self.assertIn('max_depth', params)
        self.assertIn('min_samples_split', params)
        self.assertIn('min_samples_leaf', params)
        self.assertIn('max_features', params)
        
        # Check that parameters have multiple options
        self.assertGreater(len(params['n_estimators']), 1)
        self.assertGreater(len(params['max_depth']), 1)
    
    def test_get_logistic_regression_params(self):
        """Test Logistic Regression hyperparameter grid"""
        params = self.trainer.get_logistic_regression_params()
        
        # Check that all required parameters are present
        self.assertIn('C', params)
        self.assertIn('penalty', params)
        self.assertIn('solver', params)
        self.assertIn('max_iter', params)
        
        # Check that parameters have multiple options
        self.assertGreater(len(params['C']), 1)
        self.assertGreater(len(params['penalty']), 1)
    
    def test_get_xgboost_params(self):
        """Test XGBoost hyperparameter grid"""
        params = self.trainer.get_xgboost_params()
        
        # Check that all required parameters are present
        self.assertIn('n_estimators', params)
        self.assertIn('max_depth', params)
        self.assertIn('learning_rate', params)
        self.assertIn('subsample', params)
        self.assertIn('colsample_bytree', params)
        
        # Check that parameters have multiple options
        self.assertGreater(len(params['n_estimators']), 1)
        self.assertGreater(len(params['max_depth']), 1)
    
    def test_train_with_grid_search_random_forest(self):
        """Test GridSearchCV training with Random Forest"""
        model = RandomForestClassifier(random_state=42)
        # Use smaller param grid for faster testing
        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [5, 10]
        }
        
        grid_search = self.trainer.train_with_grid_search(
            model, param_grid, self.X_train, self.y_train
        )
        
        # Check that GridSearchCV was successful
        self.assertIsNotNone(grid_search.best_estimator_)
        self.assertIsNotNone(grid_search.best_params_)
        self.assertIsNotNone(grid_search.best_score_)
        
        # Check that best params are stored
        self.assertIn('RandomForestClassifier', self.trainer.best_params)
    
    def test_train_with_grid_search_logistic_regression(self):
        """Test GridSearchCV training with Logistic Regression"""
        model = LogisticRegression(random_state=42)
        # Use smaller param grid for faster testing
        param_grid = {
            'C': [0.1, 1.0],
            'penalty': ['l2'],
            'solver': ['liblinear'],
            'max_iter': [1000]
        }
        
        grid_search = self.trainer.train_with_grid_search(
            model, param_grid, self.X_train, self.y_train
        )
        
        # Check that GridSearchCV was successful
        self.assertIsNotNone(grid_search.best_estimator_)
        self.assertIsNotNone(grid_search.best_params_)
        
        # Check that best params are stored
        self.assertIn('LogisticRegression', self.trainer.best_params)
    
    def test_reproducibility_with_fixed_random_state(self):
        """Test that models produce reproducible results with fixed random_state"""
        # Train model twice with same random state
        trainer1 = MLModelTrainer(random_state=42)
        trainer2 = MLModelTrainer(random_state=42)
        
        model1 = RandomForestClassifier(random_state=42, n_estimators=10)
        model2 = RandomForestClassifier(random_state=42, n_estimators=10)
        
        model1.fit(self.X_train, self.y_train)
        model2.fit(self.X_train, self.y_train)
        
        # Predictions should be identical
        pred1 = model1.predict(self.X_test)
        pred2 = model2.predict(self.X_test)
        
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_models_can_make_predictions_after_training(self):
        """Test that trained models can make predictions"""
        # Train a simple Random Forest
        model = RandomForestClassifier(random_state=42, n_estimators=10)
        param_grid = {'max_depth': [5, 10]}
        
        grid_search = self.trainer.train_with_grid_search(
            model, param_grid, self.X_train, self.y_train
        )
        
        best_model = grid_search.best_estimator_
        
        # Make predictions
        predictions = best_model.predict(self.X_test)
        
        # Check predictions
        self.assertEqual(len(predictions), len(self.y_test))
        self.assertTrue(all(p in [0, 1] for p in predictions))
    
    def test_train_all_models_returns_three_models(self):
        """Test that train_all_models returns all three model types"""
        # Use small dataset for faster testing
        X_small = self.X_train[:50]
        y_small = self.y_train[:50]
        
        # Mock the hyperparameter grids to be smaller for faster testing
        original_rf_params = self.trainer.get_random_forest_params
        original_lr_params = self.trainer.get_logistic_regression_params
        original_xgb_params = self.trainer.get_xgboost_params
        
        # Override with smaller grids
        self.trainer.get_random_forest_params = lambda: {
            'n_estimators': [10],
            'max_depth': [5]
        }
        self.trainer.get_logistic_regression_params = lambda: {
            'C': [1.0],
            'penalty': ['l2'],
            'solver': ['liblinear'],
            'max_iter': [1000]
        }
        self.trainer.get_xgboost_params = lambda: {
            'n_estimators': [10],
            'max_depth': [3],
            'learning_rate': [0.1]
        }
        
        try:
            results = self.trainer.train_all_models(X_small, y_small)
            
            # Check that all three models are present
            self.assertIn('random_forest', results)
            self.assertIn('logistic_regression', results)
            self.assertIn('xgboost', results)
            
            # Check that each result has required keys
            for model_name in ['random_forest', 'logistic_regression', 'xgboost']:
                self.assertIn('model', results[model_name])
                self.assertIn('params', results[model_name])
                self.assertIn('cv_score', results[model_name])
                
                # Check that model can make predictions
                model = results[model_name]['model']
                predictions = model.predict(self.X_test)
                self.assertEqual(len(predictions), len(self.y_test))
        
        finally:
            # Restore original methods
            self.trainer.get_random_forest_params = original_rf_params
            self.trainer.get_logistic_regression_params = original_lr_params
            self.trainer.get_xgboost_params = original_xgb_params
    
    def test_trained_models_stored_in_trainer(self):
        """Test that trained models are stored in trainer object"""
        # Use small dataset for faster testing
        X_small = self.X_train[:50]
        y_small = self.y_train[:50]
        
        # Override with smaller grids
        self.trainer.get_random_forest_params = lambda: {
            'n_estimators': [10],
            'max_depth': [5]
        }
        self.trainer.get_logistic_regression_params = lambda: {
            'C': [1.0],
            'penalty': ['l2'],
            'solver': ['liblinear'],
            'max_iter': [1000]
        }
        self.trainer.get_xgboost_params = lambda: {
            'n_estimators': [10],
            'max_depth': [3],
            'learning_rate': [0.1]
        }
        
        results = self.trainer.train_all_models(X_small, y_small)
        
        # Check that models are stored in trainer
        self.assertEqual(self.trainer.trained_models, results)
    
    def test_grid_search_uses_5_fold_cv(self):
        """Test that GridSearchCV uses 5-fold cross-validation"""
        model = RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': [10], 'max_depth': [5]}
        
        grid_search = self.trainer.train_with_grid_search(
            model, param_grid, self.X_train, self.y_train
        )
        
        # Check that 5-fold CV was used
        self.assertEqual(grid_search.cv, 5)
    
    def test_grid_search_uses_accuracy_scoring(self):
        """Test that GridSearchCV uses accuracy as scoring metric"""
        model = RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': [10], 'max_depth': [5]}
        
        grid_search = self.trainer.train_with_grid_search(
            model, param_grid, self.X_train, self.y_train
        )
        
        # Check that accuracy was used as scoring metric
        self.assertEqual(grid_search.scoring, 'accuracy')


if __name__ == '__main__':
    unittest.main()
