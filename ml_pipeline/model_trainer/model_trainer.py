"""
Model Training Module

This module provides the MLModelTrainer class for training multiple ML classifiers
with hyperparameter tuning using GridSearchCV.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import numpy as np
from typing import Dict, Any


class MLModelTrainer:
    """Trains and tunes multiple ML classifiers for NFL game prediction"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize MLModelTrainer with random seed for reproducibility
        
        Args:
            random_state: Random seed for reproducible results
        """
        self.random_state = random_state
        self.trained_models = {}
        self.best_params = {}
    
    def get_random_forest_params(self) -> Dict[str, list]:
        """
        Return hyperparameter grid for Random Forest classifier
        
        Returns:
            Dictionary containing hyperparameter options for GridSearchCV
        """
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
    
    def get_logistic_regression_params(self) -> Dict[str, list]:
        """
        Return hyperparameter grid for Logistic Regression classifier
        
        Returns:
            Dictionary containing hyperparameter options for GridSearchCV
        """
        return {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000]
        }
    
    def get_xgboost_params(self) -> Dict[str, list]:
        """
        Return hyperparameter grid for XGBoost classifier
        
        Returns:
            Dictionary containing hyperparameter options for GridSearchCV
        """
        return {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }

    def train_with_grid_search(self, model, param_grid: Dict[str, list], 
                               X_train: np.ndarray, y_train: np.ndarray) -> GridSearchCV:
        """
        Train model with GridSearchCV hyperparameter tuning
        
        Args:
            model: Scikit-learn compatible model instance
            param_grid: Dictionary of hyperparameters to search
            X_train: Training feature matrix
            y_train: Training labels
            
        Returns:
            Fitted GridSearchCV object with best estimator
        """
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,  # 5-fold cross-validation
            scoring='accuracy',
            n_jobs=-1,  # Use all available CPU cores
            verbose=1,
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store best parameters
        model_name = type(model).__name__
        self.best_params[model_name] = grid_search.best_params_
        
        return grid_search

    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train all three classifiers (Random Forest, Logistic Regression, XGBoost)
        with hyperparameter tuning and return best models
        
        Args:
            X_train: Training feature matrix
            y_train: Training labels
            
        Returns:
            Dictionary containing trained models with their best parameters
            Format: {
                'random_forest': {'model': trained_model, 'params': best_params},
                'logistic_regression': {'model': trained_model, 'params': best_params},
                'xgboost': {'model': trained_model, 'params': best_params}
            }
        """
        results = {}
        
        # Train Random Forest
        print("\n" + "="*60)
        print("Training Random Forest Classifier...")
        print("="*60)
        rf_model = RandomForestClassifier(random_state=self.random_state)
        rf_params = self.get_random_forest_params()
        rf_grid_search = self.train_with_grid_search(rf_model, rf_params, X_train, y_train)
        
        # Retrain with best parameters on full training set
        best_rf = RandomForestClassifier(**rf_grid_search.best_params_, random_state=self.random_state)
        best_rf.fit(X_train, y_train)
        
        results['random_forest'] = {
            'model': best_rf,
            'params': rf_grid_search.best_params_,
            'cv_score': rf_grid_search.best_score_
        }
        print(f"Best Random Forest params: {rf_grid_search.best_params_}")
        print(f"Best CV score: {rf_grid_search.best_score_:.4f}")
        
        # Train Logistic Regression
        print("\n" + "="*60)
        print("Training Logistic Regression Classifier...")
        print("="*60)
        lr_model = LogisticRegression(random_state=self.random_state)
        lr_params = self.get_logistic_regression_params()
        lr_grid_search = self.train_with_grid_search(lr_model, lr_params, X_train, y_train)
        
        # Retrain with best parameters on full training set
        best_lr = LogisticRegression(**lr_grid_search.best_params_, random_state=self.random_state)
        best_lr.fit(X_train, y_train)
        
        results['logistic_regression'] = {
            'model': best_lr,
            'params': lr_grid_search.best_params_,
            'cv_score': lr_grid_search.best_score_
        }
        print(f"Best Logistic Regression params: {lr_grid_search.best_params_}")
        print(f"Best CV score: {lr_grid_search.best_score_:.4f}")
        
        # Train XGBoost
        print("\n" + "="*60)
        print("Training XGBoost Classifier...")
        print("="*60)
        xgb_model = xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss')
        xgb_params = self.get_xgboost_params()
        xgb_grid_search = self.train_with_grid_search(xgb_model, xgb_params, X_train, y_train)
        
        # Retrain with best parameters on full training set
        best_xgb = xgb.XGBClassifier(**xgb_grid_search.best_params_, 
                                      random_state=self.random_state, 
                                      eval_metric='logloss')
        best_xgb.fit(X_train, y_train)
        
        results['xgboost'] = {
            'model': best_xgb,
            'params': xgb_grid_search.best_params_,
            'cv_score': xgb_grid_search.best_score_
        }
        print(f"Best XGBoost params: {xgb_grid_search.best_params_}")
        print(f"Best CV score: {xgb_grid_search.best_score_:.4f}")
        
        # Store trained models
        self.trained_models = results
        
        print("\n" + "="*60)
        print("All models trained successfully!")
        print("="*60)
        
        return results
