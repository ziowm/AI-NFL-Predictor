"""
Unit tests for Data Loader Module
"""

import unittest
import pandas as pd
import numpy as np
import os
import shutil
from ml_pipeline.data_loader import DataLoader


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures directory"""
        cls.fixtures_dir = 'tests/fixtures'
        cls.test_season = '2023'
        
        # Create test CSV files in root directory for testing
        cls.offense_data = pd.DataFrame({
            'Name': ['Team A', 'Team B', 'Team C'],
            'GP': [17, 17, 17],
            'PTS': [28.5, 26.8, 24.3],
            'All': [82.3, 78.9, 75.2],
            'Run': [25.2, 23.1, 22.4],
            'Pass': [36.8, 35.4, 33.6],
            'QB': [22.5, 21.8, 20.2],
            'RB': [12.3, 11.5, 10.8],
            'WR': [18.7, 17.9, 16.5],
            'TE': [8.2, 7.8, 7.2]
        })
        
        cls.defense_data = pd.DataFrame({
            'Name': ['Team A', 'Team B', 'Team C'],
            'GP': [17, 17, 17],
            'PA': [21.8, 23.4, 25.1],
            'DEF': [8.5, 9.1, 9.8],
            'QB': [17.2, 18.5, 19.2],
            'RB': [16.8, 17.2, 17.8],
            'WR': [27.3, 28.1, 29.5],
            'TE': [9.8, 10.2, 10.8]
        })
        
        cls.schedule_data = pd.DataFrame({
            'Week': [1, 2, 3],
            'Team1': ['Team A', 'Team B', 'Team A'],
            'Team2': ['Team B', 'Team C', 'Team C']
        })
        
        # Save test files
        cls.offense_data.to_csv(f'{cls.test_season} Fantasy Offense Stats.csv', index=False)
        cls.defense_data.to_csv(f'{cls.test_season} Fantasy Defense Stats.csv', index=False)
        cls.schedule_data.to_csv(f'{cls.test_season} Schedule.csv', index=False)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files"""
        test_files = [
            f'{cls.test_season} Fantasy Offense Stats.csv',
            f'{cls.test_season} Fantasy Defense Stats.csv',
            f'{cls.test_season} Schedule.csv'
        ]
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)
    
    def test_init(self):
        """Test DataLoader initialization"""
        seasons = ['2023']
        loader = DataLoader(seasons)
        self.assertEqual(loader.seasons, seasons)
    
    def test_validate_data_offense(self):
        """Test validation of offense data"""
        loader = DataLoader(['2023'])
        # Should not raise exception
        self.assertTrue(loader.validate_data(self.offense_data, 'offense'))
    
    def test_validate_data_defense(self):
        """Test validation of defense data"""
        loader = DataLoader(['2023'])
        # Should not raise exception
        self.assertTrue(loader.validate_data(self.defense_data, 'defense'))
    
    def test_validate_data_schedule(self):
        """Test validation of schedule data"""
        loader = DataLoader(['2023'])
        # Should not raise exception
        self.assertTrue(loader.validate_data(self.schedule_data, 'schedule'))
    
    def test_validate_data_empty_dataframe(self):
        """Test that empty DataFrame raises ValueError"""
        loader = DataLoader(['2023'])
        empty_df = pd.DataFrame()
        
        with self.assertRaises(ValueError) as context:
            loader.validate_data(empty_df, 'offense')
        self.assertIn("empty", str(context.exception).lower())
    
    def test_validate_data_missing_columns(self):
        """Test that missing required columns raises ValueError"""
        loader = DataLoader(['2023'])
        
        # Create DataFrame missing required columns
        invalid_df = pd.DataFrame({
            'Name': ['Team A', 'Team B'],
            'GP': [17, 17]
            # Missing PTS column for offense data
        })
        
        with self.assertRaises(ValueError) as context:
            loader.validate_data(invalid_df, 'offense')
        self.assertIn("Missing required columns", str(context.exception))
    
    def test_merge_team_stats(self):
        """Test merging offense and defense statistics"""
        loader = DataLoader(['2023'])
        
        merged_df = loader.merge_team_stats(self.offense_data, self.defense_data)
        
        # Check that merge was successful
        self.assertEqual(len(merged_df), 3)  # 3 teams
        self.assertIn('Name', merged_df.columns)
        self.assertIn('off_PTS', merged_df.columns)
        self.assertIn('def_PA', merged_df.columns)
    
    def test_merge_team_stats_no_common_teams(self):
        """Test that merge with no common teams raises ValueError"""
        loader = DataLoader(['2023'])
        
        # Create offense data with different team names
        different_offense = pd.DataFrame({
            'Name': ['Team X', 'Team Y'],
            'GP': [17, 17],
            'PTS': [28.5, 26.8]
        })
        
        with self.assertRaises(ValueError) as context:
            loader.merge_team_stats(different_offense, self.defense_data)
        self.assertIn("empty DataFrame", str(context.exception))
    
    def test_handle_missing_values_with_low_threshold(self):
        """Test missing value handling with low threshold"""
        loader = DataLoader(['2023'])
        
        # Create DataFrame with some missing values
        df_with_missing = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [10, 20, 30, 40, 50],
            'col3': [np.nan, np.nan, np.nan, np.nan, 100]  # 80% missing
        })
        
        # With 20% threshold, col3 should be dropped
        result = loader.handle_missing_values(df_with_missing, threshold=0.20)
        
        # col3 should be dropped, col1 should be imputed
        self.assertNotIn('col3', result.columns)
        self.assertIn('col1', result.columns)
        self.assertIn('col2', result.columns)
        self.assertEqual(len(result), 5)  # No rows dropped after imputation
    
    def test_handle_missing_values_with_high_threshold(self):
        """Test missing value handling with high threshold"""
        loader = DataLoader(['2023'])
        
        # Create DataFrame with some missing values
        df_with_missing = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [10, 20, 30, 40, 50],
            'col3': [np.nan, np.nan, np.nan, np.nan, 100]  # 80% missing
        })
        
        # With 90% threshold, col3 should be kept and imputed
        result = loader.handle_missing_values(df_with_missing, threshold=0.90)
        
        # All columns should be kept
        self.assertIn('col1', result.columns)
        self.assertIn('col2', result.columns)
        self.assertIn('col3', result.columns)
    
    def test_load_season_data(self):
        """Test loading data for a single season"""
        loader = DataLoader([self.test_season])
        
        games_df = loader.load_season_data(self.test_season)
        
        # Check that games were loaded
        self.assertGreater(len(games_df), 0)
        self.assertIn('season', games_df.columns)
        self.assertIn('week', games_df.columns)
        self.assertIn('team1', games_df.columns)
        self.assertIn('team2', games_df.columns)
        
        # Check that team stats are included
        self.assertIn('team1_off_PTS', games_df.columns)
        self.assertIn('team2_def_PA', games_df.columns)
    
    def test_load_season_data_missing_file(self):
        """Test that loading nonexistent season raises FileNotFoundError"""
        loader = DataLoader(['9999'])
        
        with self.assertRaises(FileNotFoundError) as context:
            loader.load_season_data('9999')
        self.assertIn("not found", str(context.exception))
    
    def test_load_all_seasons(self):
        """Test loading multiple seasons"""
        loader = DataLoader([self.test_season])
        
        all_games_df = loader.load_all_seasons()
        
        # Check that data was loaded
        self.assertGreater(len(all_games_df), 0)
        self.assertIn('season', all_games_df.columns)
        self.assertIn('team1', all_games_df.columns)
        self.assertIn('team2', all_games_df.columns)


if __name__ == '__main__':
    unittest.main()
