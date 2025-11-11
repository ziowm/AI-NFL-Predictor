import unittest
import numpy as np
import os
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from ml_pipeline.model_storage import ModelStorage


class TestModelStorage(unittest.TestCase):
    """Test suite for ModelStorage class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_storage_dir = 'test_models'
        self.storage = ModelStorage(storage_dir=self.test_storage_dir)
        
        # Create sample data for testing
        np.random.seed(42)
        self.X_train = np.random.rand(100, 10)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.rand(20, 10)
        
        # Train a simple model for testing
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        
        # Create a scaler for testing
        self.scaler = StandardScaler()
        self.scaler.fit(self.X_train)
        
        # Feature names for testing
        self.feature_names = [f'feature_{i}' for i in range(10)]
    
    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_storage_dir):
            shutil.rmtree(self.test_storage_dir)
    
    def test_storage_directory_creation(self):
        """Test that storage directory is created"""
        self.assertTrue(os.path.exists(self.test_storage_dir))
    
    def test_save_and_load_model(self):
        """Test saving and loading a model"""
        model_name = 'test_random_forest'
        
        # Save the model
        self.storage.save_model(self.model, model_name)
        
        # Verify file exists
        model_path = os.path.join(self.test_storage_dir, f'{model_name}.joblib')
        self.assertTrue(os.path.exists(model_path))
        
        # Load the model
        loaded_model = self.storage.load_model(model_name)
        
        # Verify loaded model is not None
        self.assertIsNotNone(loaded_model)
    
    def test_save_and_load_scaler(self):
        """Test saving and loading a scaler"""
        scaler_name = 'test_scaler'
        
        # Save the scaler
        self.storage.save_scaler(self.scaler, scaler_name)
        
        # Verify file exists
        scaler_path = os.path.join(self.test_storage_dir, f'{scaler_name}.joblib')
        self.assertTrue(os.path.exists(scaler_path))
        
        # Load the scaler
        loaded_scaler = self.storage.load_scaler(scaler_name)
        
        # Verify loaded scaler is not None
        self.assertIsNotNone(loaded_scaler)
    
    def test_save_and_load_feature_names(self):
        """Test saving and loading feature names"""
        # Save feature names
        self.storage.save_feature_names(self.feature_names)
        
        # Verify file exists
        feature_path = os.path.join(self.test_storage_dir, 'feature_names.json')
        self.assertTrue(os.path.exists(feature_path))
        
        # Load feature names
        loaded_features = self.storage.load_feature_names()
        
        # Verify loaded feature names match original
        self.assertEqual(loaded_features, self.feature_names)
    
    def test_loaded_model_predictions_match_original(self):
        """Test that loaded model produces identical predictions to original"""
        model_name = 'test_model_predictions'
        
        # Get predictions from original model
        original_predictions = self.model.predict(self.X_test)
        
        # Save and load the model
        self.storage.save_model(self.model, model_name)
        loaded_model = self.storage.load_model(model_name)
        
        # Get predictions from loaded model
        loaded_predictions = loaded_model.predict(self.X_test)
        
        # Verify predictions are identical
        np.testing.assert_array_equal(original_predictions, loaded_predictions)
    
    def test_loaded_scaler_transforms_match_original(self):
        """Test that loaded scaler produces identical transforms to original"""
        scaler_name = 'test_scaler_transform'
        
        # Get transform from original scaler
        original_transform = self.scaler.transform(self.X_test)
        
        # Save and load the scaler
        self.storage.save_scaler(self.scaler, scaler_name)
        loaded_scaler = self.storage.load_scaler(scaler_name)
        
        # Get transform from loaded scaler
        loaded_transform = loaded_scaler.transform(self.X_test)
        
        # Verify transforms are identical
        np.testing.assert_array_almost_equal(original_transform, loaded_transform)
    
    def test_load_nonexistent_model_raises_error(self):
        """Test that loading a nonexistent model raises FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            self.storage.load_model('nonexistent_model')
    
    def test_load_nonexistent_scaler_raises_error(self):
        """Test that loading a nonexistent scaler raises FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            self.storage.load_scaler('nonexistent_scaler')
    
    def test_load_nonexistent_feature_names_raises_error(self):
        """Test that loading nonexistent feature names raises FileNotFoundError"""
        with self.assertRaises(FileNotFoundError):
            self.storage.load_feature_names('nonexistent_features.json')


if __name__ == '__main__':
    unittest.main()
