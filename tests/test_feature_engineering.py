"""
Unit tests for Feature Engineering Module
"""

import unittest
import pandas as pd
import numpy as np
from ml_pipeline.feature_engineering import FeatureEngineer


class TestFeatureEngineer(unittest.TestCase):
    """Test cases for FeatureEngineer class"""
    
    def setUp(self):
        """Set up test data"""
        self.engineer = FeatureEngineer()
        
        # Create sample game data with team statistics
        self.sample_data = pd.DataFrame({
            'season': ['2023', '2023', '2023'],
            'week': [1, 2, 3],
            'team1': ['Team A', 'Team A', 'Team B'],
            'team2': ['Team B', 'Team C', 'Team C'],
            'team1_GP': [17, 17, 17],
            'team1_off_PTS': [28.5, 28.5, 26.8],
            'team1_off_All': [82.3, 82.3, 78.9],
            'team1_off_Run': [25.2, 25.2, 23.1],
            'team1_off_Pass': [36.8, 36.8, 35.4],
            'team1_off_QB': [22.5, 22.5, 21.8],
            'team1_def_PA': [21.8, 21.8, 23.4],
            'team1_def_DEF': [8.5, 8.5, 9.1],
            'team1_def_QB': [17.2, 17.2, 18.5],
            'team2_GP': [17, 17, 17],
            'team2_off_PTS': [26.8, 24.3, 24.3],
            'team2_off_All': [78.9, 75.2, 75.2],
            'team2_off_Run': [23.1, 22.4, 22.4],
            'team2_off_Pass': [35.4, 33.6, 33.6],
            'team2_off_QB': [21.8, 20.2, 20.2],
            'team2_def_PA': [23.4, 25.1, 25.1],
            'team2_def_DEF': [9.1, 9.8, 9.8],
            'team2_def_QB': [18.5, 19.2, 19.2]
        })
    
    def test_init(self):
        """Test FeatureEngineer initialization"""
        engineer = FeatureEngineer()
        self.assertIsNotNone(engineer.scaler)
        self.assertEqual(engineer.feature_names, [])
        self.assertFalse(engineer.is_fitted)
    
    def test_create_efficiency_features(self):
        """Test efficiency feature creation"""
        result = self.engineer.create_efficiency_features(self.sample_data)
        
        # Check that efficiency features were created
        self.assertIn('team1_points_per_game', result.columns)
        self.assertIn('team1_yards_per_game', result.columns)
        self.assertIn('team1_pass_yards_per_game', result.columns)
        self.assertIn('team1_rush_yards_per_game', result.columns)
        self.assertIn('team1_offensive_efficiency', result.columns)
        self.assertIn('team1_points_allowed_per_game', result.columns)
        self.assertIn('team1_defensive_efficiency', result.columns)
        
        self.assertIn('team2_points_per_game', result.columns)
        self.assertIn('team2_yards_per_game', result.columns)
        
        # Verify calculations with known values
        expected_ppg = 28.5 / 17
        self.assertAlmostEqual(result.iloc[0]['team1_points_per_game'], expected_ppg, places=5)
        
        expected_ypg = 82.3 / 17
        self.assertAlmostEqual(result.iloc[0]['team1_yards_per_game'], expected_ypg, places=5)
    
    def test_create_turnover_features(self):
        """Test turnover feature creation"""
        result = self.engineer.create_turnover_features(self.sample_data)
        
        # Check that turnover features were created
        self.assertIn('team1_turnovers_gained', result.columns)
        self.assertIn('team1_turnovers_lost', result.columns)
        self.assertIn('team1_turnover_ratio', result.columns)
        
        self.assertIn('team2_turnovers_gained', result.columns)
        self.assertIn('team2_turnovers_lost', result.columns)
        self.assertIn('team2_turnover_ratio', result.columns)
        
        # Verify that turnover ratio is calculated correctly
        for idx in range(len(result)):
            expected_ratio = (
                result.iloc[idx]['team1_turnovers_gained'] - 
                result.iloc[idx]['team1_turnovers_lost']
            )
            self.assertAlmostEqual(
                result.iloc[idx]['team1_turnover_ratio'], 
                expected_ratio, 
                places=5
            )
    
    def test_create_rolling_features(self):
        """Test rolling window feature creation"""
        # First create efficiency features (required for rolling)
        df_with_efficiency = self.engineer.create_efficiency_features(self.sample_data)
        
        result = self.engineer.create_rolling_features(df_with_efficiency, window=3)
        
        # Check that rolling features were created
        self.assertIn('team1_points_per_game_rolling_3games', result.columns)
        self.assertIn('team1_yards_per_game_rolling_3games', result.columns)
        self.assertIn('team1_offensive_efficiency_rolling_3games', result.columns)
        self.assertIn('team1_defensive_efficiency_rolling_3games', result.columns)
        
        # Verify rolling calculations
        # First game should equal the value itself (min_periods=1)
        self.assertAlmostEqual(
            result.iloc[0]['team1_points_per_game_rolling_3games'],
            result.iloc[0]['team1_points_per_game'],
            places=5
        )
    
    def test_create_rolling_features_edge_cases(self):
        """Test rolling features with edge cases"""
        # Create data with single row
        single_row = self.sample_data.iloc[[0]].copy()
        df_with_efficiency = self.engineer.create_efficiency_features(single_row)
        
        result = self.engineer.create_rolling_features(df_with_efficiency, window=3)
        
        # Should not raise error and should have rolling features
        self.assertIn('team1_points_per_game_rolling_3games', result.columns)
        self.assertEqual(len(result), 1)
    
    def test_create_matchup_features(self):
        """Test matchup feature creation"""
        result = self.engineer.create_matchup_features(self.sample_data)
        
        # Check that matchup features were created
        self.assertIn('head_to_head_win_pct', result.columns)
        self.assertIn('head_to_head_avg_point_diff', result.columns)
        
        # Default values should be 0.5 and 0.0
        self.assertEqual(result.iloc[0]['head_to_head_win_pct'], 0.5)
        self.assertEqual(result.iloc[0]['head_to_head_avg_point_diff'], 0.0)
    
    def test_encode_categorical_features(self):
        """Test categorical feature encoding"""
        result = self.engineer.encode_categorical_features(self.sample_data)
        
        # Check that categorical features were created
        self.assertIn('home_field_advantage', result.columns)
        self.assertIn('weather_clear', result.columns)
        self.assertIn('weather_rain', result.columns)
        self.assertIn('weather_snow', result.columns)
        self.assertIn('weather_wind', result.columns)
        self.assertIn('rest_days_difference', result.columns)
        
        # Verify home field advantage is set to 1
        self.assertEqual(result.iloc[0]['home_field_advantage'], 1)
        
        # Verify weather defaults (clear conditions)
        self.assertEqual(result.iloc[0]['weather_clear'], 1)
        self.assertEqual(result.iloc[0]['weather_rain'], 0)
    
    def test_encode_categorical_features_with_weather(self):
        """Test categorical encoding with weather column"""
        data_with_weather = self.sample_data.copy()
        data_with_weather['weather'] = ['Clear', 'Rain', 'Snow']
        
        result = self.engineer.encode_categorical_features(data_with_weather)
        
        # Check that weather was one-hot encoded
        self.assertIn('weather_Clear', result.columns)
        self.assertIn('weather_Rain', result.columns)
        self.assertIn('weather_Snow', result.columns)
    
    def test_create_all_features(self):
        """Test creation of all features"""
        result = self.engineer.create_all_features(self.sample_data, fit_scaler=True)
        
        # Check that all major feature categories exist
        feature_names = self.engineer.get_feature_names()
        
        # Verify minimum feature count (25+)
        self.assertGreaterEqual(len(feature_names), 25, 
                               f"Expected at least 25 features, got {len(feature_names)}")
        
        # Check that scaler was fitted
        self.assertTrue(self.engineer.is_fitted)
        
        # Verify key features exist
        expected_features = [
            'team1_points_per_game',
            'team1_offensive_efficiency',
            'team1_defensive_efficiency',
            'team1_turnover_ratio',
            'team2_points_per_game',
            'team2_offensive_efficiency',
            'head_to_head_win_pct',
            'home_field_advantage',
            'weather_clear'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, feature_names, 
                         f"Expected feature '{feature}' not found in feature list")
    
    def test_create_all_features_normalization(self):
        """Test that features are normalized"""
        result = self.engineer.create_all_features(self.sample_data, fit_scaler=True)
        
        feature_names = self.engineer.get_feature_names()
        
        # Check that normalized features have reasonable ranges
        # After standardization, most values should be within [-3, 3]
        for feature in feature_names:
            if feature in result.columns:
                values = result[feature].values
                # Allow some outliers but most should be in reasonable range
                self.assertTrue(np.abs(values).max() < 10, 
                               f"Feature {feature} has extreme values after normalization")
    
    def test_create_all_features_without_fitting(self):
        """Test that transforming without fitting raises error"""
        with self.assertRaises(ValueError) as context:
            self.engineer.create_all_features(self.sample_data, fit_scaler=False)
        self.assertIn("Scaler must be fitted", str(context.exception))
    
    def test_create_all_features_transform_mode(self):
        """Test feature creation in transform mode"""
        # First fit the scaler
        self.engineer.create_all_features(self.sample_data, fit_scaler=True)
        
        # Create new data for transformation
        new_data = self.sample_data.copy()
        new_data['week'] = [4, 5, 6]
        
        # Transform without fitting
        result = self.engineer.create_all_features(new_data, fit_scaler=False)
        
        # Should work without error
        self.assertEqual(len(result), 3)
        self.assertTrue(self.engineer.is_fitted)
    
    def test_get_feature_names(self):
        """Test getting feature names"""
        # Before creating features
        self.assertEqual(self.engineer.get_feature_names(), [])
        
        # After creating features
        self.engineer.create_all_features(self.sample_data, fit_scaler=True)
        feature_names = self.engineer.get_feature_names()
        
        self.assertIsInstance(feature_names, list)
        self.assertGreater(len(feature_names), 0)
        
        # Verify it returns a copy (not reference)
        feature_names.append('test_feature')
        self.assertNotIn('test_feature', self.engineer.get_feature_names())
    
    def test_feature_count_meets_requirement(self):
        """Verify total feature count is 25+"""
        self.engineer.create_all_features(self.sample_data, fit_scaler=True)
        feature_count = len(self.engineer.get_feature_names())
        
        self.assertGreaterEqual(feature_count, 25,
                               f"Feature count {feature_count} does not meet requirement of 25+")
        
        print(f"\nTotal features created: {feature_count}")
        print("Feature list:")
        for i, feature in enumerate(self.engineer.get_feature_names(), 1):
            print(f"  {i}. {feature}")


if __name__ == '__main__':
    unittest.main()
