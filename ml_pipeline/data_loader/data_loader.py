"""
Data Loader Module

Handles loading, validation, and preprocessing of multi-season NFL statistics data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and validation of NFL statistics data"""
    
    def __init__(self, seasons: List[str]):
        """
        Initialize DataLoader with list of season identifiers.
        
        Args:
            seasons: List of season identifiers (e.g., ['2019', '2020', '2021'])
        """
        self.seasons = seasons
        logger.info(f"DataLoader initialized with seasons: {seasons}")
    
    def load_season_data(self, season: str) -> pd.DataFrame:
        """
        Load offense, defense, and schedule data for a single season.
        
        Args:
            season: Season identifier (e.g., '2023')
            
        Returns:
            DataFrame containing merged game data with team statistics
            
        Raises:
            FileNotFoundError: If required CSV files are not found
            ValueError: If data format is invalid
        """
        logger.info(f"Loading data for season {season}")
        
        # Construct file paths
        offense_file = f"{season} Fantasy Offense Stats.csv"
        defense_file = f"{season} Fantasy Defense Stats.csv"
        schedule_file = f"{season} Schedule.csv"
        
        # Check if files exist
        for file_path in [offense_file, defense_file, schedule_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Required file not found: {file_path}. "
                    f"Please ensure all CSV files for season {season} are in the working directory."
                )
        
        # Load CSV files
        try:
            offense_df = pd.read_csv(offense_file)
            defense_df = pd.read_csv(defense_file)
            schedule_df = pd.read_csv(schedule_file)
        except Exception as e:
            raise ValueError(f"Error reading CSV files for season {season}: {str(e)}")
        
        # Validate data
        self.validate_data(offense_df, 'offense')
        self.validate_data(defense_df, 'defense')
        self.validate_data(schedule_df, 'schedule')
        
        # Merge team statistics
        team_stats_df = self.merge_team_stats(offense_df, defense_df)
        
        # Merge with schedule to create game-level data
        games_list = []
        for _, game in schedule_df.iterrows():
            team1 = game['Team1']
            team2 = game['Team2']
            week = game['Week']
            
            # Get stats for both teams
            team1_stats = team_stats_df[team_stats_df['Name'] == team1]
            team2_stats = team_stats_df[team_stats_df['Name'] == team2]
            
            if team1_stats.empty or team2_stats.empty:
                logger.warning(f"Missing stats for game: {team1} vs {team2} in week {week}")
                continue
            
            # Create game record
            game_record = {
                'season': season,
                'week': week,
                'team1': team1,
                'team2': team2
            }
            
            # Add team1 stats with prefix
            for col in team1_stats.columns:
                if col != 'Name':
                    game_record[f'team1_{col}'] = team1_stats.iloc[0][col]
            
            # Add team2 stats with prefix
            for col in team2_stats.columns:
                if col != 'Name':
                    game_record[f'team2_{col}'] = team2_stats.iloc[0][col]
            
            games_list.append(game_record)
        
        games_df = pd.DataFrame(games_list)
        logger.info(f"Loaded {len(games_df)} games for season {season}")
        
        return games_df

    def validate_data(self, df: pd.DataFrame, data_type: str = 'general') -> bool:
        """
        Validate that DataFrame contains required columns and data types.
        
        Args:
            df: DataFrame to validate
            data_type: Type of data ('offense', 'defense', 'schedule', or 'general')
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If required columns are missing or data types are invalid
        """
        if df.empty:
            raise ValueError(f"DataFrame is empty for {data_type} data")
        
        # Define required columns for each data type
        required_columns = {
            'offense': ['Name', 'GP', 'PTS'],
            'defense': ['Name', 'GP', 'PA'],
            'schedule': ['Week', 'Team1', 'Team2'],
            'general': ['Name']
        }
        
        # Get required columns for this data type
        required = required_columns.get(data_type, [])
        
        # Check for missing columns
        missing_columns = [col for col in required if col not in df.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in {data_type} data: {missing_columns}. "
                f"Available columns: {list(df.columns)}"
            )
        
        # Validate data types for schedule data
        if data_type == 'schedule':
            if not pd.api.types.is_numeric_dtype(df['Week']):
                raise ValueError("'Week' column must be numeric")
            if not pd.api.types.is_string_dtype(df['Team1']) and not pd.api.types.is_object_dtype(df['Team1']):
                raise ValueError("'Team1' column must be string type")
            if not pd.api.types.is_string_dtype(df['Team2']) and not pd.api.types.is_object_dtype(df['Team2']):
                raise ValueError("'Team2' column must be string type")
        
        # Validate data types for offense/defense data
        if data_type in ['offense', 'defense']:
            if not pd.api.types.is_string_dtype(df['Name']) and not pd.api.types.is_object_dtype(df['Name']):
                raise ValueError(f"'Name' column must be string type in {data_type} data")
            
            # Check that numeric columns are numeric or can be converted
            numeric_cols = [col for col in df.columns if col != 'Name']
            for col in numeric_cols:
                if col in df.columns:
                    try:
                        pd.to_numeric(df[col], errors='coerce')
                    except Exception as e:
                        raise ValueError(f"Column '{col}' in {data_type} data cannot be converted to numeric: {str(e)}")
        
        logger.debug(f"Validation passed for {data_type} data with {len(df)} rows")
        return True

    def merge_team_stats(self, offense_df: pd.DataFrame, defense_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge offensive and defensive statistics by team name.
        
        Args:
            offense_df: DataFrame containing offensive statistics
            defense_df: DataFrame containing defensive statistics
            
        Returns:
            DataFrame with merged offensive and defensive stats
            
        Raises:
            ValueError: If merge fails or results in unexpected data
        """
        logger.debug(f"Merging {len(offense_df)} offense records with {len(defense_df)} defense records")
        
        # Rename columns to avoid conflicts (except 'Name' and 'GP')
        offense_renamed = offense_df.copy()
        defense_renamed = defense_df.copy()
        
        # Rename offense columns with 'off_' prefix (except Name and GP)
        for col in offense_renamed.columns:
            if col not in ['Name', 'GP']:
                offense_renamed.rename(columns={col: f'off_{col}'}, inplace=True)
        
        # Rename defense columns with 'def_' prefix (except Name and GP)
        for col in defense_renamed.columns:
            if col not in ['Name', 'GP']:
                defense_renamed.rename(columns={col: f'def_{col}'}, inplace=True)
        
        # Merge on team name
        merged_df = pd.merge(
            offense_renamed,
            defense_renamed,
            on='Name',
            how='inner',
            suffixes=('_off', '_def')
        )
        
        if merged_df.empty:
            raise ValueError("Merge resulted in empty DataFrame. Check that team names match between offense and defense files.")
        
        # Use GP from offense if both exist
        if 'GP_off' in merged_df.columns and 'GP_def' in merged_df.columns:
            merged_df['GP'] = merged_df['GP_off']
            merged_df.drop(['GP_off', 'GP_def'], axis=1, inplace=True)
        
        logger.info(f"Successfully merged stats for {len(merged_df)} teams")
        return merged_df
    
    def handle_missing_values(self, df: pd.DataFrame, threshold: float = 0.20) -> pd.DataFrame:
        """
        Handle missing values by imputing or removing based on threshold.
        
        Args:
            df: DataFrame to process
            threshold: Maximum proportion of missing values allowed (default 20%)
            
        Returns:
            DataFrame with missing values handled
        """
        logger.debug(f"Handling missing values with threshold {threshold}")
        
        initial_rows = len(df)
        initial_cols = len(df.columns)
        
        # Calculate missing percentage for each column
        missing_pct = df.isnull().sum() / len(df)
        
        # Remove columns with too many missing values
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        if cols_to_drop:
            logger.warning(f"Dropping columns with >{threshold*100}% missing values: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
        
        # For remaining columns, impute with median for numeric columns
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    median_value = df[col].median()
                    df[col].fillna(median_value, inplace=True)
                    logger.debug(f"Imputed {col} with median value {median_value}")
                else:
                    # For non-numeric, fill with mode or 'Unknown'
                    if not df[col].mode().empty:
                        mode_value = df[col].mode()[0]
                        df[col].fillna(mode_value, inplace=True)
                        logger.debug(f"Imputed {col} with mode value {mode_value}")
                    else:
                        df[col].fillna('Unknown', inplace=True)
        
        # Remove rows that still have missing values
        df = df.dropna()
        
        final_rows = len(df)
        final_cols = len(df.columns)
        
        logger.info(
            f"Missing value handling complete. "
            f"Rows: {initial_rows} -> {final_rows}, "
            f"Columns: {initial_cols} -> {final_cols}"
        )
        
        return df
    
    def load_all_seasons(self) -> pd.DataFrame:
        """
        Load and combine data from all seasons.
        
        Returns:
            DataFrame containing combined multi-season game data
            
        Raises:
            ValueError: If no valid season data could be loaded
        """
        logger.info(f"Loading all seasons: {self.seasons}")
        
        all_games = []
        
        for season in self.seasons:
            try:
                season_df = self.load_season_data(season)
                all_games.append(season_df)
                logger.info(f"Successfully loaded {len(season_df)} games from season {season}")
            except Exception as e:
                logger.error(f"Failed to load season {season}: {str(e)}")
                raise
        
        if not all_games:
            raise ValueError("No valid season data could be loaded")
        
        # Combine all seasons
        combined_df = pd.concat(all_games, ignore_index=True)
        
        # Handle missing values
        combined_df = self.handle_missing_values(combined_df)
        
        logger.info(f"Successfully loaded {len(combined_df)} total games from {len(self.seasons)} seasons")
        
        return combined_df
