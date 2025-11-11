"""
Unit tests for Data Splitter Module
"""

import unittest
import pandas as pd
import numpy as np
from ml_pipeline.data_splitter import TemporalDataSplitter


class TestTemporalDataSplitter(unittest.TestCase):
    """Test cases for TemporalDataSplitter class"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample multi-season data
        self.sample_data = pd.DataFrame({
            'season': ['2019'] * 100 + ['2020'] * 100 + ['2021'] * 100 + ['2022'] * 100 + ['2023'] * 100,
            'week': list(range(1, 101)) * 5,
            'team1': ['Team A'] * 500,
            'team2': ['Team B'] * 500,
            'team1_score': np.random.randint(10, 40, 500),
            'team2_score': np.random.randint(10, 40, 500)
        })
    
    def test_init_valid_seasons(self):
        """Test initialization with valid season assignments"""
        splitter = TemporalDataSplitter(
            train_seasons=['2019', '2020', '2021'],
            val_seasons=['2022'],
            test_seasons=['2023']
        )
        self.assertEqual(splitter.train_seasons, ['2019', '2020', '2021'])
        self.assertEqual(splitter.val_seasons, ['2022'])
        self.assertEqual(splitter.test_seasons, ['2023'])
    
    def test_init_empty_train_seasons(self):
        """Test that empty train_seasons raises ValueError"""
        with self.assertRaises(ValueError) as context:
            TemporalDataSplitter(
                train_seasons=[],
                val_seasons=['2022'],
                test_seasons=['2023']
            )
        self.assertIn("train_seasons cannot be empty", str(context.exception))
    
    def test_init_overlapping_seasons(self):
        """Test that overlapping seasons raise ValueError"""
        with self.assertRaises(ValueError) as context:
            TemporalDataSplitter(
                train_seasons=['2019', '2020', '2021'],
                val_seasons=['2021', '2022'],
                test_seasons=['2023']
            )
        self.assertIn("both train and val sets", str(context.exception))
    
    def test_init_temporal_ordering_violation(self):
        """Test that incorrect temporal ordering raises ValueError"""
        with self.assertRaises(ValueError) as context:
            TemporalDataSplitter(
                train_seasons=['2019', '2020', '2023'],
                val_seasons=['2021'],
                test_seasons=['2022']
            )
        self.assertIn("Temporal ordering violation", str(context.exception))
    
    def test_split_data_valid(self):
        """Test splitting data with valid DataFrame"""
        splitter = TemporalDataSplitter(
            train_seasons=['2019', '2020', '2021'],
            val_seasons=['2022'],
            test_seasons=['2023']
        )
        
        train_df, val_df, test_df = splitter.split_data(self.sample_data)
        
        # Check split sizes
        self.assertEqual(len(train_df), 300)  # 3 seasons * 100 games
        self.assertEqual(len(val_df), 100)    # 1 season * 100 games
        self.assertEqual(len(test_df), 100)   # 1 season * 100 games
        
        # Check that seasons are correctly assigned
        self.assertTrue(all(train_df['season'].isin(['2019', '2020', '2021'])))
        self.assertTrue(all(val_df['season'].isin(['2022'])))
        self.assertTrue(all(test_df['season'].isin(['2023'])))
    
    def test_split_data_empty_dataframe(self):
        """Test that empty DataFrame raises ValueError"""
        splitter = TemporalDataSplitter(
            train_seasons=['2019'],
            val_seasons=['2020'],
            test_seasons=['2021']
        )
        
        empty_df = pd.DataFrame()
        
        with self.assertRaises(ValueError) as context:
            splitter.split_data(empty_df)
        self.assertIn("Cannot split empty DataFrame", str(context.exception))
    
    def test_split_data_missing_season_column(self):
        """Test that DataFrame without 'season' column raises ValueError"""
        splitter = TemporalDataSplitter(
            train_seasons=['2019'],
            val_seasons=['2020'],
            test_seasons=['2021']
        )
        
        df_no_season = pd.DataFrame({
            'week': [1, 2, 3],
            'team1': ['A', 'B', 'C'],
            'team2': ['D', 'E', 'F']
        })
        
        with self.assertRaises(ValueError) as context:
            splitter.split_data(df_no_season)
        self.assertIn("must contain 'season' column", str(context.exception))
    
    def test_split_data_missing_required_seasons(self):
        """Test that missing required seasons raises ValueError"""
        splitter = TemporalDataSplitter(
            train_seasons=['2019', '2020'],
            val_seasons=['2021'],
            test_seasons=['2024']  # 2024 not in sample data
        )
        
        with self.assertRaises(ValueError) as context:
            splitter.split_data(self.sample_data)
        self.assertIn("Required seasons not found", str(context.exception))
    
    def test_split_data_no_leakage(self):
        """Test that there is no data leakage between splits"""
        splitter = TemporalDataSplitter(
            train_seasons=['2019', '2020', '2021'],
            val_seasons=['2022'],
            test_seasons=['2023']
        )
        
        train_df, val_df, test_df = splitter.split_data(self.sample_data)
        
        # Check that no rows overlap
        train_seasons = set(train_df['season'].unique())
        val_seasons = set(val_df['season'].unique())
        test_seasons = set(test_df['season'].unique())
        
        self.assertEqual(len(train_seasons & val_seasons), 0)
        self.assertEqual(len(train_seasons & test_seasons), 0)
        self.assertEqual(len(val_seasons & test_seasons), 0)
    
    def test_get_split_statistics(self):
        """Test get_split_statistics returns correct counts"""
        splitter = TemporalDataSplitter(
            train_seasons=['2019', '2020', '2021'],
            val_seasons=['2022'],
            test_seasons=['2023']
        )
        
        # Split data first
        splitter.split_data(self.sample_data)
        
        # Get statistics
        stats = splitter.get_split_statistics()
        
        self.assertEqual(stats['train'], 300)
        self.assertEqual(stats['val'], 100)
        self.assertEqual(stats['test'], 100)
        self.assertEqual(stats['total'], 500)
    
    def test_get_split_statistics_before_split(self):
        """Test that get_split_statistics raises error before split_data is called"""
        splitter = TemporalDataSplitter(
            train_seasons=['2019'],
            val_seasons=['2020'],
            test_seasons=['2021']
        )
        
        with self.assertRaises(RuntimeError) as context:
            splitter.get_split_statistics()
        self.assertIn("Call split_data() first", str(context.exception))


if __name__ == '__main__':
    unittest.main()
