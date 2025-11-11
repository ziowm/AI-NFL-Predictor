import os
import joblib
import json
from typing import Any, List


class ModelStorage:
    """Handles model serialization and deserialization"""
    
    def __init__(self, storage_dir: str = 'models'):
        """
        Initialize with storage directory path
        
        Args:
            storage_dir: Directory path where models will be saved
        """
        self.storage_dir = storage_dir
        # Create storage directory if it doesn't exist
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def save_model(self, model: Any, model_name: str) -> None:
        """
        Serialize model to disk using joblib
        
        Args:
            model: Trained model object to save
            model_name: Name to use for the saved model file
        """
        model_path = os.path.join(self.storage_dir, f"{model_name}.joblib")
        joblib.dump(model, model_path)
    
    def save_scaler(self, scaler: Any, scaler_name: str = 'feature_scaler') -> None:
        """
        Save feature scaler for consistent preprocessing
        
        Args:
            scaler: StandardScaler or other scaler object to save
            scaler_name: Name to use for the saved scaler file
        """
        scaler_path = os.path.join(self.storage_dir, f"{scaler_name}.joblib")
        joblib.dump(scaler, scaler_path)
    
    def save_feature_names(self, feature_names: List[str], filename: str = 'feature_names.json') -> None:
        """
        Save feature names for reference
        
        Args:
            feature_names: List of feature names to save
            filename: Name of the JSON file to save feature names
        """
        feature_path = os.path.join(self.storage_dir, filename)
        with open(feature_path, 'w') as f:
            json.dump(feature_names, f, indent=2)
    
    def load_model(self, model_name: str) -> Any:
        """
        Deserialize model from disk
        
        Args:
            model_name: Name of the model file to load
            
        Returns:
            Loaded model object
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        model_path = os.path.join(self.storage_dir, f"{model_name}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                f"Please train the model first using the --train command."
            )
        return joblib.load(model_path)
    
    def load_scaler(self, scaler_name: str = 'feature_scaler') -> Any:
        """
        Load feature scaler
        
        Args:
            scaler_name: Name of the scaler file to load
            
        Returns:
            Loaded scaler object
            
        Raises:
            FileNotFoundError: If scaler file doesn't exist
        """
        scaler_path = os.path.join(self.storage_dir, f"{scaler_name}.joblib")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Scaler file not found: {scaler_path}. "
                f"Please train the model first using the --train command."
            )
        return joblib.load(scaler_path)
    
    def load_feature_names(self, filename: str = 'feature_names.json') -> List[str]:
        """
        Load feature names
        
        Args:
            filename: Name of the JSON file containing feature names
            
        Returns:
            List of feature names
            
        Raises:
            FileNotFoundError: If feature names file doesn't exist
        """
        feature_path = os.path.join(self.storage_dir, filename)
        if not os.path.exists(feature_path):
            raise FileNotFoundError(
                f"Feature names file not found: {feature_path}. "
                f"Please train the model first using the --train command."
            )
        with open(feature_path, 'r') as f:
            return json.load(f)
