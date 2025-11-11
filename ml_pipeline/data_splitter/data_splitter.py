"""
Data Splitter Module

Handles temporal splitting of multi-season NFL data to prevent data leakage.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class TemporalDataSplitter:
    """Splits multi-season data maintaining temporal order to prevent data leakage"""
    
    def __init__(self, train_seasons: List[str], val_seasons: List[str], test_seasons: List[str]):
        """
        Initialize TemporalDataSplitter with season assignments for each split.
        
        Args:
            train_seasons: List of season identifiers for training set (e.g., ['2019', '2020', '2021'])
            val_seasons: List of season identifiers for validation set (e.g., ['2022'])
            test_seasons: List of season identifiers for test set (e.g., ['2023'])
            
        Raises:
            ValueError: If season assignments are invalid or overlap
        """
        self.train_seasons = train_seasons
        self.val_seasons = val_seasons
        self.test_seasons = test_seasons
        
        # Validate season assignments
        self._validate_season_assignments()
        
        # Store split statistics (will be populated after split_data is called)
        self._split_stats = None
        
        logger.info(
            f"TemporalDataSplitter initialized - "
            f"Train: {train_seasons}, Val: {val_seasons}, Test: {test_seasons}"
        )
    
    def _validate_season_assignments(self) -> None:
        """
        Validate that season assignments are valid and don't overlap.
        
        Raises:
            ValueError: If validation fails
        """
        # Check that all lists are non-empty
        if not self.train_seasons:
            raise ValueError("train_seasons cannot be empty")
        if not self.val_seasons:
            raise ValueError("val_seasons cannot be empty")
        if not self.test_seasons:
            raise ValueError("test_seasons cannot be empty")
        
        # Check for overlaps between splits
        train_set = set(self.train_seasons)
        val_set = set(self.val_seasons)
        test_set = set(self.test_seasons)
        
        train_val_overlap = train_set & val_set
        train_test_overlap = train_set & test_set
        val_test_overlap = val_set & test_set
        
        if train_val_overlap:
            raise ValueError(f"Seasons appear in both train and val sets: {train_val_overlap}")
        if train_test_overlap:
            raise ValueError(f"Seasons appear in both train and test sets: {train_test_overlap}")
        if val_test_overlap:
            raise ValueError(f"Seasons appear in both val and test sets: {val_test_overlap}")
        
        # Validate temporal ordering (train < val < test)
        all_seasons = self.train_seasons + self.val_seasons + self.test_seasons
        
        # Try to convert to integers for comparison (assumes YYYY format)
        try:
            train_years = [int(s) for s in self.train_seasons]
            val_years = [int(s) for s in self.val_seasons]
            test_years = [int(s) for s in self.test_seasons]
            
            max_train = max(train_years)
            min_val = min(val_years)
            max_val = max(val_years)
            min_test = min(test_years)
            
            if max_train >= min_val:
                raise ValueError(
                    f"Temporal ordering violation: training seasons must be before validation seasons. "
                    f"Max train year: {max_train}, Min val year: {min_val}"
                )
            
            if max_val >= min_test:
                raise ValueError(
                    f"Temporal ordering violation: validation seasons must be before test seasons. "
                    f"Max val year: {max_val}, Min test year: {min_test}"
                )
            
            logger.debug("Temporal ordering validated successfully")
            
        except ValueError as e:
            if "invalid literal" in str(e):
                logger.warning(
                    "Could not validate temporal ordering: season identifiers are not numeric years. "
                    "Proceeding without temporal validation."
                )
            else:
                raise
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame by season maintaining temporal order.
        
        Args:
            df: DataFrame containing multi-season game data with 'season' column
            
        Returns:
            Tuple of (train_df, val_df, test_df)
            
        Raises:
            ValueError: If DataFrame is invalid or missing required columns
        """
        logger.info("Splitting data by season")
        
        # Validate input DataFrame
        if df.empty:
            raise ValueError("Cannot split empty DataFrame")
        
        if 'season' not in df.columns:
            raise ValueError(
                f"DataFrame must contain 'season' column. Available columns: {list(df.columns)}"
            )
        
        # Get unique seasons in the data
        available_seasons = set(df['season'].unique())
        required_seasons = set(self.train_seasons + self.val_seasons + self.test_seasons)
        
        # Check if all required seasons are available
        missing_seasons = required_seasons - available_seasons
        if missing_seasons:
            raise ValueError(
                f"Required seasons not found in DataFrame: {missing_seasons}. "
                f"Available seasons: {available_seasons}"
            )
        
        # Split data by season
        train_df = df[df['season'].isin(self.train_seasons)].copy()
        val_df = df[df['season'].isin(self.val_seasons)].copy()
        test_df = df[df['season'].isin(self.test_seasons)].copy()
        
        # Validate splits are non-empty
        if train_df.empty:
            raise ValueError("Training split is empty")
        if val_df.empty:
            raise ValueError("Validation split is empty")
        if test_df.empty:
            raise ValueError("Test split is empty")
        
        # Verify no data leakage (no overlapping rows)
        train_indices = set(train_df.index)
        val_indices = set(val_df.index)
        test_indices = set(test_df.index)
        
        if train_indices & val_indices:
            raise ValueError("Data leakage detected: overlapping indices between train and val sets")
        if train_indices & test_indices:
            raise ValueError("Data leakage detected: overlapping indices between train and test sets")
        if val_indices & test_indices:
            raise ValueError("Data leakage detected: overlapping indices between val and test sets")
        
        # Store split statistics
        self._split_stats = {
            'train': len(train_df),
            'val': len(val_df),
            'test': len(test_df),
            'total': len(df)
        }
        
        logger.info(
            f"Data split complete - Train: {len(train_df)} games, "
            f"Val: {len(val_df)} games, Test: {len(test_df)} games"
        )
        
        # Reset indices for clean splits
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        return train_df, val_df, test_df
    
    def get_split_statistics(self) -> Dict[str, int]:
        """
        Return game counts for each split.
        
        Returns:
            Dictionary containing game counts for train, val, test, and total
            
        Raises:
            RuntimeError: If split_data() has not been called yet
        """
        if self._split_stats is None:
            raise RuntimeError(
                "No split statistics available. Call split_data() first."
            )
        
        return self._split_stats.copy()
