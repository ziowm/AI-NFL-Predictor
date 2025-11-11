"""
NFL Predictor - Main Prediction Interface

This module provides the main entry point for the NFL game prediction ML pipeline.
It orchestrates data loading, feature engineering, model training, and predictions.
"""

import argparse
import logging
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

from ml_pipeline.data_loader.data_loader import DataLoader
from ml_pipeline.feature_engineering.feature_engineer import FeatureEngineer
from ml_pipeline.data_splitter.data_splitter import TemporalDataSplitter
from ml_pipeline.model_trainer.model_trainer import MLModelTrainer
from ml_pipeline.model_evaluator.model_evaluator import ModelEvaluator
from ml_pipeline.model_storage.model_storage import ModelStorage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nfl_predictor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class NFLPredictor:
    """Main interface for NFL game prediction system"""
    
    def __init__(self):
        """Initialize predictor with models and feature engineering"""
        self.data_loader = None
        self.feature_engineer = FeatureEngineer()
        self.model_storage = ModelStorage()
        self.trained_models = {}
        self.feature_names = []
        
        logger.info("NFLPredictor initialized")
    
    def train_pipeline(self, seasons: List[str]) -> None:
        """
        Execute full training pipeline
        
        This method orchestrates the complete ML pipeline:
        1. Load data from multiple seasons
        2. Engineer features
        3. Split data temporally
        4. Train models with hyperparameter tuning
        5. Evaluate models
        6. Save trained models and scalers
        
        Args:
            seasons: List of season identifiers (e.g., ['2019', '2020', '2021', '2022', '2023'])
            
        Raises:
            ValueError: If insufficient seasons provided or data loading fails
        """
        logger.info(f"Starting training pipeline with seasons: {seasons}")
        
        # Validate input
        if len(seasons) < 3:
            raise ValueError(
                f"At least 3 seasons required for training. Provided: {len(seasons)}"
            )
        
        # Step 1: Load data
        print("\n" + "="*80)
        print("STEP 1: Loading Data")
        print("="*80)
        self.data_loader = DataLoader(seasons)
        df = self.data_loader.load_all_seasons()
        print(f"Loaded {len(df)} games from {len(seasons)} seasons")
        
        # Step 2: Feature engineering
        print("\n" + "="*80)
        print("STEP 2: Feature Engineering")
        print("="*80)
        df = self.feature_engineer.create_all_features(df, fit_scaler=True)
        self.feature_names = self.feature_engineer.get_feature_names()
        print(f"Created {len(self.feature_names)} features")
        
        # Step 3: Split data temporally
        print("\n" + "="*80)
        print("STEP 3: Splitting Data")
        print("="*80)
        
        # Determine split based on number of seasons
        if len(seasons) == 5:
            # Standard 3-1-1 split
            train_seasons = seasons[:3]
            val_seasons = [seasons[3]]
            test_seasons = [seasons[4]]
        elif len(seasons) == 4:
            # 2-1-1 split
            train_seasons = seasons[:2]
            val_seasons = [seasons[2]]
            test_seasons = [seasons[3]]
        else:
            # For 3 seasons: 1-1-1 split
            train_seasons = [seasons[0]]
            val_seasons = [seasons[1]]
            test_seasons = [seasons[2]]
        
        splitter = TemporalDataSplitter(train_seasons, val_seasons, test_seasons)
        train_df, val_df, test_df = splitter.split_data(df)
        
        split_stats = splitter.get_split_statistics()
        print(f"Train: {split_stats['train']} games ({train_seasons})")
        print(f"Val: {split_stats['val']} games ({val_seasons})")
        print(f"Test: {split_stats['test']} games ({test_seasons})")
        
        # Prepare feature matrices and labels
        X_train = train_df[self.feature_names].values
        X_val = val_df[self.feature_names].values
        X_test = test_df[self.feature_names].values
        
        # Create labels (for now, use a simple heuristic: team1 wins if they have higher points per game)
        # In a real scenario, you would have actual game outcomes
        y_train = (train_df['team1_points_per_game'] > train_df['team2_points_per_game']).astype(int).values
        y_val = (val_df['team1_points_per_game'] > val_df['team2_points_per_game']).astype(int).values
        y_test = (test_df['team1_points_per_game'] > test_df['team2_points_per_game']).astype(int).values
        
        # Step 4: Train models
        print("\n" + "="*80)
        print("STEP 4: Training Models")
        print("="*80)
        trainer = MLModelTrainer(random_state=42)
        trained_models_dict = trainer.train_all_models(X_train, y_train)
        
        # Extract just the models for evaluation
        self.trained_models = {
            name: info['model'] for name, info in trained_models_dict.items()
        }
        
        # Step 5: Evaluate models
        print("\n" + "="*80)
        print("STEP 5: Evaluating Models")
        print("="*80)
        evaluator = ModelEvaluator(self.trained_models)
        
        # Evaluate on validation set
        print("\nValidation Set Performance:")
        print("-" * 80)
        val_results = evaluator.evaluate_all_models(X_val, y_val)
        print(val_results.to_string(index=False))
        
        # Evaluate on test set
        print("\nTest Set Performance:")
        print("-" * 80)
        test_results = evaluator.evaluate_all_models(X_test, y_test)
        print(test_results.to_string(index=False))
        
        # Generate detailed report
        print("\n" + "="*80)
        print("DETAILED EVALUATION REPORT")
        print("="*80)
        report = evaluator.generate_evaluation_report(X_test, y_test)
        print(report)
        
        # Step 6: Save models and scalers
        print("\n" + "="*80)
        print("STEP 6: Saving Models")
        print("="*80)
        
        for model_name, model in self.trained_models.items():
            self.model_storage.save_model(model, model_name)
            print(f"Saved {model_name}")
        
        # Save scaler
        self.model_storage.save_scaler(self.feature_engineer.scaler)
        print("Saved feature scaler")
        
        # Save feature names
        self.model_storage.save_feature_names(self.feature_names)
        print("Saved feature names")
        
        # Save model metadata
        metadata = {
            'seasons': seasons,
            'train_seasons': train_seasons,
            'val_seasons': val_seasons,
            'test_seasons': test_seasons,
            'num_features': len(self.feature_names),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'model_params': {
                name: info['params'] for name, info in trained_models_dict.items()
            },
            'cv_scores': {
                name: info['cv_score'] for name, info in trained_models_dict.items()
            }
        }
        
        import json
        metadata_path = f"{self.model_storage.storage_dir}/metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print("Saved model metadata")
        
        print("\n" + "="*80)
        print("TRAINING PIPELINE COMPLETE!")
        print("="*80)
        print(f"\nModels saved to: {self.model_storage.storage_dir}/")
        print("You can now use --evaluate or --predict commands")
        
        logger.info("Training pipeline completed successfully")
    
    def load_trained_models(self) -> None:
        """
        Load pre-trained models from disk
        
        This method loads:
        - All trained models (Random Forest, Logistic Regression, XGBoost)
        - Feature scaler
        - Feature names
        - Model metadata
        
        Raises:
            FileNotFoundError: If model files are not found
        """
        logger.info("Loading trained models from disk")
        
        try:
            # Load models
            model_names = ['random_forest', 'logistic_regression', 'xgboost']
            for model_name in model_names:
                try:
                    model = self.model_storage.load_model(model_name)
                    self.trained_models[model_name] = model
                    logger.info(f"Loaded {model_name}")
                except FileNotFoundError:
                    logger.warning(f"Model {model_name} not found, skipping")
            
            if not self.trained_models:
                raise FileNotFoundError(
                    "No trained models found. Please train models first using --train command."
                )
            
            # Load scaler
            self.feature_engineer.scaler = self.model_storage.load_scaler()
            self.feature_engineer.is_fitted = True
            logger.info("Loaded feature scaler")
            
            # Load feature names
            self.feature_names = self.model_storage.load_feature_names()
            self.feature_engineer.feature_names = self.feature_names
            logger.info(f"Loaded {len(self.feature_names)} feature names")
            
            print(f"Successfully loaded {len(self.trained_models)} models")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise
    
    def predict_game(self, team1: str, team2: str, 
                    team1_stats: Dict[str, float], 
                    team2_stats: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict outcome of a specific matchup
        
        Args:
            team1: Name of first team
            team2: Name of second team
            team1_stats: Dictionary of team1 statistics
            team2_stats: Dictionary of team2 statistics
            
        Returns:
            Dictionary containing predictions from all models:
            {
                'team1': team1_name,
                'team2': team2_name,
                'predictions': {
                    'random_forest': {'winner': team_name, 'probability': float},
                    'logistic_regression': {'winner': team_name, 'probability': float},
                    'xgboost': {'winner': team_name, 'probability': float}
                },
                'consensus': team_name  # Most common prediction
            }
            
        Raises:
            ValueError: If models not loaded or invalid input
        """
        if not self.trained_models:
            raise ValueError(
                "No models loaded. Call load_trained_models() first or train models with --train."
            )
        
        logger.info(f"Predicting game: {team1} vs {team2}")
        
        # Create a DataFrame with the game data
        game_data = {
            'season': '2024',  # Placeholder
            'week': 1,
            'team1': team1,
            'team2': team2
        }
        
        # Add team1 stats with prefix
        for key, value in team1_stats.items():
            game_data[f'team1_{key}'] = value
        
        # Add team2 stats with prefix
        for key, value in team2_stats.items():
            game_data[f'team2_{key}'] = value
        
        # Create DataFrame
        df = pd.DataFrame([game_data])
        
        # Engineer features (without fitting scaler)
        df = self.feature_engineer.create_all_features(df, fit_scaler=False)
        
        # Extract feature matrix
        X = df[self.feature_names].values
        
        # Get predictions from all models
        predictions = {}
        votes = []
        
        for model_name, model in self.trained_models.items():
            # Get prediction
            pred = model.predict(X)[0]
            
            # Get probability if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                prob_team1_wins = proba[1]  # Probability of class 1 (team1 wins)
            else:
                prob_team1_wins = float(pred)
            
            # Determine winner
            winner = team1 if pred == 1 else team2
            votes.append(winner)
            
            predictions[model_name] = {
                'winner': winner,
                'probability': prob_team1_wins if pred == 1 else (1 - prob_team1_wins),
                'team1_win_probability': prob_team1_wins
            }
        
        # Determine consensus (most common prediction)
        from collections import Counter
        vote_counts = Counter(votes)
        consensus = vote_counts.most_common(1)[0][0]
        
        result = {
            'team1': team1,
            'team2': team2,
            'predictions': predictions,
            'consensus': consensus
        }
        
        logger.info(f"Prediction complete. Consensus: {consensus}")
        
        return result
    
    def predict_season(self, schedule_file: str, 
                      offense_stats_file: str,
                      defense_stats_file: str) -> pd.DataFrame:
        """
        Predict all games in a schedule
        
        Args:
            schedule_file: Path to CSV file containing game schedule
            offense_stats_file: Path to CSV file containing offensive statistics
            defense_stats_file: Path to CSV file containing defensive statistics
            
        Returns:
            DataFrame with predictions for all games in the schedule
            
        Raises:
            FileNotFoundError: If input files not found
            ValueError: If models not loaded
        """
        if not self.trained_models:
            raise ValueError(
                "No models loaded. Call load_trained_models() first or train models with --train."
            )
        
        logger.info(f"Predicting season from schedule: {schedule_file}")
        
        # Load schedule and stats
        import os
        if not os.path.exists(schedule_file):
            raise FileNotFoundError(f"Schedule file not found: {schedule_file}")
        if not os.path.exists(offense_stats_file):
            raise FileNotFoundError(f"Offense stats file not found: {offense_stats_file}")
        if not os.path.exists(defense_stats_file):
            raise FileNotFoundError(f"Defense stats file not found: {defense_stats_file}")
        
        schedule_df = pd.read_csv(schedule_file)
        offense_df = pd.read_csv(offense_stats_file)
        defense_df = pd.read_csv(defense_stats_file)
        
        # Merge team stats
        temp_loader = DataLoader(['temp'])
        team_stats_df = temp_loader.merge_team_stats(offense_df, defense_df)
        
        # Create game data similar to data loader
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
                'season': '2024',
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
        
        # Engineer features
        games_df = self.feature_engineer.create_all_features(games_df, fit_scaler=False)
        
        # Extract feature matrix
        X = games_df[self.feature_names].values
        
        # Get predictions from all models
        predictions_dict = {}
        
        for model_name, model in self.trained_models.items():
            preds = model.predict(X)
            predictions_dict[f'{model_name}_prediction'] = preds
            
            # Add probabilities if available
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(X)
                predictions_dict[f'{model_name}_team1_win_prob'] = probas[:, 1]
        
        # Add predictions to DataFrame
        for key, values in predictions_dict.items():
            games_df[key] = values
        
        # Create consensus prediction (majority vote)
        pred_cols = [col for col in games_df.columns if col.endswith('_prediction')]
        games_df['consensus_prediction'] = games_df[pred_cols].mode(axis=1)[0]
        
        # Add winner names
        games_df['consensus_winner'] = games_df.apply(
            lambda row: row['team1'] if row['consensus_prediction'] == 1 else row['team2'],
            axis=1
        )
        
        logger.info(f"Predicted {len(games_df)} games")
        
        # Return relevant columns
        result_cols = ['week', 'team1', 'team2', 'consensus_winner'] + pred_cols
        result_cols = [col for col in result_cols if col in games_df.columns]
        
        return games_df[result_cols]
    
    def display_feature_importance(self, model_name: str, top_n: int = 15) -> None:
        """
        Display feature importance for specified model
        
        Args:
            model_name: Name of the model ('random_forest', 'logistic_regression', or 'xgboost')
            top_n: Number of top features to display (default: 15)
            
        Raises:
            ValueError: If model not found or doesn't support feature importance
        """
        if not self.trained_models:
            raise ValueError(
                "No models loaded. Call load_trained_models() first or train models with --train."
            )
        
        if model_name not in self.trained_models:
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {list(self.trained_models.keys())}"
            )
        
        model = self.trained_models[model_name]
        
        # Check if model supports feature importance
        if not hasattr(model, 'feature_importances_'):
            raise ValueError(
                f"Model '{model_name}' does not support feature importance. "
                f"Feature importance is only available for tree-based models (random_forest, xgboost)."
            )
        
        logger.info(f"Displaying feature importance for {model_name}")
        
        # Create evaluator and get feature importance
        evaluator = ModelEvaluator({model_name: model})
        importance_df = evaluator.get_feature_importance(model, self.feature_names)
        
        # Display results
        print("\n" + "="*80)
        print(f"FEATURE IMPORTANCE - {model_name.upper()}")
        print("="*80)
        print()
        
        top_features = importance_df.head(top_n)
        
        # Format as table
        print(f"{'Rank':<6} {'Feature':<50} {'Importance':<12}")
        print("-" * 80)
        
        for idx, row in top_features.iterrows():
            rank = idx + 1
            feature = row['feature']
            importance = row['importance']
            
            # Create visual bar
            bar_length = int(importance * 50 / top_features.iloc[0]['importance'])
            bar = '█' * bar_length
            
            print(f"{rank:<6} {feature:<50} {importance:<12.6f}")
            print(f"       {bar}")
            print()
        
        print("="*80)
        
        # Optionally plot if matplotlib is available
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            evaluator.plot_feature_importance(importance_df, top_n)
            plot_file = f"{model_name}_feature_importance.png"
            plt.savefig(plot_file)
            print(f"\nFeature importance plot saved to: {plot_file}")
        except Exception as e:
            logger.warning(f"Could not generate plot: {str(e)}")


def main():
    """
    Main entry point for CLI with comprehensive error handling
    """
    """
    Main entry point for CLI
    """
    parser = argparse.ArgumentParser(
        description='NFL Game Prediction ML Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train models with 5 seasons
  python nfl_predictor.py --train --seasons 2019 2020 2021 2022 2023
  
  # Evaluate trained models on test set
  python nfl_predictor.py --evaluate
  
  # Predict a specific matchup
  python nfl_predictor.py --predict --team1 "Kansas City Chiefs" --team2 "Buffalo Bills"
  
  # Display feature importance for a model
  python nfl_predictor.py --feature-importance --model random_forest
  
  # Predict full season
  python nfl_predictor.py --predict-season --schedule 2024_schedule.csv --offense 2024_offense.csv --defense 2024_defense.csv
        """
    )
    
    # Training arguments
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train all models with hyperparameter tuning'
    )
    parser.add_argument(
        '--seasons',
        nargs='+',
        help='List of season identifiers for training (e.g., 2019 2020 2021 2022 2023)'
    )
    
    # Evaluation arguments
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate trained models on test set'
    )
    
    # Single game prediction arguments
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Predict outcome of a specific matchup'
    )
    parser.add_argument(
        '--team1',
        type=str,
        help='Name of first team for prediction'
    )
    parser.add_argument(
        '--team2',
        type=str,
        help='Name of second team for prediction'
    )
    
    # Feature importance arguments
    parser.add_argument(
        '--feature-importance',
        action='store_true',
        help='Display feature importance for a specified model'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['random_forest', 'logistic_regression', 'xgboost'],
        help='Model name for feature importance display'
    )
    
    # Season prediction arguments
    parser.add_argument(
        '--predict-season',
        action='store_true',
        help='Predict all games in a season schedule'
    )
    parser.add_argument(
        '--schedule',
        type=str,
        help='Path to schedule CSV file'
    )
    parser.add_argument(
        '--offense',
        type=str,
        help='Path to offensive stats CSV file'
    )
    parser.add_argument(
        '--defense',
        type=str,
        help='Path to defensive stats CSV file'
    )
    
    args = parser.parse_args()
    
    # Create predictor instance
    predictor = NFLPredictor()
    
    try:
        # Handle training
        if args.train:
            if not args.seasons:
                print("\n" + "="*80)
                print("ERROR: Missing required argument")
                print("="*80)
                print("\nThe --seasons argument is required for training.")
                print("\nUsage:")
                print("  python nfl_predictor.py --train --seasons <season1> <season2> ...")
                print("\nExample:")
                print("  python nfl_predictor.py --train --seasons 2019 2020 2021 2022 2023")
                print("\nNote: At least 3 seasons are required for proper train/val/test split")
                print("="*80)
                sys.exit(1)
            
            if len(args.seasons) < 3:
                print("\n" + "="*80)
                print("ERROR: Insufficient seasons for training")
                print("="*80)
                print(f"\nAt least 3 seasons are required for training.")
                print(f"You provided: {len(args.seasons)} season(s): {args.seasons}")
                print("\nThe pipeline requires:")
                print("  - Training set: At least 1 season")
                print("  - Validation set: At least 1 season")
                print("  - Test set: At least 1 season")
                print("\nRecommended: Use 5 seasons for optimal results (3 train, 1 val, 1 test)")
                print("\nExample:")
                print("  python nfl_predictor.py --train --seasons 2019 2020 2021 2022 2023")
                print("="*80)
                sys.exit(1)
            
            try:
                predictor.train_pipeline(args.seasons)
            except FileNotFoundError as e:
                print("\n" + "="*80)
                print("ERROR: Data files not found")
                print("="*80)
                print(f"\n{str(e)}")
                print("\nPlease ensure the following CSV files exist for each season:")
                print("  - <YEAR> Fantasy Offense Stats.csv")
                print("  - <YEAR> Fantasy Defense Stats.csv")
                print("  - <YEAR> Schedule.csv")
                print("\nExample for 2023:")
                print("  - 2023 Fantasy Offense Stats.csv")
                print("  - 2023 Fantasy Defense Stats.csv")
                print("  - 2023 Schedule.csv")
                print("="*80)
                sys.exit(1)
            except ValueError as e:
                print("\n" + "="*80)
                print("ERROR: Invalid data")
                print("="*80)
                print(f"\n{str(e)}")
                print("\nPlease check that your CSV files:")
                print("  - Contain all required columns")
                print("  - Have valid data types")
                print("  - Are properly formatted")
                print("="*80)
                sys.exit(1)
            
            return
        
        # Handle evaluation
        if args.evaluate:
            try:
                print("Loading trained models...")
                predictor.load_trained_models()
            except FileNotFoundError as e:
                print("\n" + "="*80)
                print("ERROR: Models not found")
                print("="*80)
                print(f"\n{str(e)}")
                print("\nYou need to train models first before evaluation.")
                print("\nTo train models, run:")
                print("  python nfl_predictor.py --train --seasons 2019 2020 2021 2022 2023")
                print("="*80)
                sys.exit(1)
            
            # Load test data
            print("\nLoading test data...")
            # We need to determine which seasons to use for testing
            # Load metadata to get test seasons
            import json
            import os
            metadata_path = f"{predictor.model_storage.storage_dir}/metadata.json"
            
            if not os.path.exists(metadata_path):
                print("Error: Model metadata not found. Please train models first.")
                sys.exit(1)
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            test_seasons = metadata['test_seasons']
            all_seasons = metadata['seasons']
            
            print(f"Test seasons: {test_seasons}")
            
            # Load and process test data
            data_loader = DataLoader(all_seasons)
            df = data_loader.load_all_seasons()
            
            # Filter to test seasons only
            test_df = df[df['season'].isin(test_seasons)].copy()
            
            # Engineer features
            predictor.feature_engineer.create_all_features(test_df, fit_scaler=False)
            
            # Extract features and labels
            X_test = test_df[predictor.feature_names].values
            y_test = (test_df['team1_points_per_game'] > test_df['team2_points_per_game']).astype(int).values
            
            # Evaluate models
            print("\n" + "="*80)
            print("MODEL EVALUATION ON TEST SET")
            print("="*80)
            
            evaluator = ModelEvaluator(predictor.trained_models)
            test_results = evaluator.evaluate_all_models(X_test, y_test)
            print("\n" + test_results.to_string(index=False))
            
            # Generate detailed report
            print("\n")
            report = evaluator.generate_evaluation_report(X_test, y_test)
            print(report)
            
            return
        
        # Handle single game prediction
        if args.predict:
            if not args.team1 or not args.team2:
                print("\n" + "="*80)
                print("ERROR: Missing required arguments")
                print("="*80)
                print("\nBoth --team1 and --team2 are required for game prediction.")
                print("\nUsage:")
                print("  python nfl_predictor.py --predict --team1 <team_name> --team2 <team_name>")
                print("\nExample:")
                print("  python nfl_predictor.py --predict --team1 'Kansas City Chiefs' --team2 'Buffalo Bills'")
                print("\nNote: Use quotes around team names with spaces")
                print("="*80)
                sys.exit(1)
            
            try:
                print("Loading trained models...")
                predictor.load_trained_models()
            except FileNotFoundError as e:
                print("\n" + "="*80)
                print("ERROR: Models not found")
                print("="*80)
                print(f"\n{str(e)}")
                print("\nYou need to train models first before making predictions.")
                print("\nTo train models, run:")
                print("  python nfl_predictor.py --train --seasons 2019 2020 2021 2022 2023")
                print("="*80)
                sys.exit(1)
            
            # For demonstration, we'll use placeholder stats
            # In a real scenario, you would load actual team statistics
            print(f"\nNote: Using placeholder statistics for demonstration")
            print(f"In production, load actual team statistics from your data source\n")
            
            team1_stats = {
                'GP': 16,
                'off_PTS': 400,
                'off_All': 5500,
                'off_Pass': 3500,
                'off_Run': 2000,
                'off_QB': 300,
                'def_PA': 320,
                'def_DEF': 280,
                'def_QB': 250
            }
            
            team2_stats = {
                'GP': 16,
                'off_PTS': 380,
                'off_All': 5300,
                'off_Pass': 3300,
                'off_Run': 2000,
                'off_QB': 290,
                'def_PA': 340,
                'def_DEF': 300,
                'def_QB': 260
            }
            
            result = predictor.predict_game(args.team1, args.team2, team1_stats, team2_stats)
            
            # Display results
            print("="*80)
            print(f"PREDICTION: {result['team1']} vs {result['team2']}")
            print("="*80)
            print()
            
            for model_name, pred in result['predictions'].items():
                print(f"{model_name}:")
                print(f"  Winner: {pred['winner']}")
                print(f"  Confidence: {pred['probability']:.2%}")
                print(f"  {result['team1']} win probability: {pred['team1_win_probability']:.2%}")
                print()
            
            print("="*80)
            print(f"CONSENSUS PREDICTION: {result['consensus']}")
            print("="*80)
            
            return
        
        # Handle feature importance display
        if args.feature_importance:
            if not args.model:
                print("\n" + "="*80)
                print("ERROR: Missing required argument")
                print("="*80)
                print("\nThe --model argument is required for feature importance display.")
                print("\nUsage:")
                print("  python nfl_predictor.py --feature-importance --model <model_name>")
                print("\nAvailable models:")
                print("  - random_forest")
                print("  - xgboost")
                print("\nNote: logistic_regression does not support feature importance")
                print("\nExample:")
                print("  python nfl_predictor.py --feature-importance --model random_forest")
                print("="*80)
                sys.exit(1)
            
            try:
                print("Loading trained models...")
                predictor.load_trained_models()
                predictor.display_feature_importance(args.model)
            except FileNotFoundError as e:
                print("\n" + "="*80)
                print("ERROR: Models not found")
                print("="*80)
                print(f"\n{str(e)}")
                print("\nYou need to train models first before viewing feature importance.")
                print("\nTo train models, run:")
                print("  python nfl_predictor.py --train --seasons 2019 2020 2021 2022 2023")
                print("="*80)
                sys.exit(1)
            except ValueError as e:
                print("\n" + "="*80)
                print("ERROR: Invalid model or unsupported operation")
                print("="*80)
                print(f"\n{str(e)}")
                print("\nNote: Feature importance is only available for tree-based models:")
                print("  - random_forest")
                print("  - xgboost")
                print("="*80)
                sys.exit(1)
            
            return
        
        # Handle season prediction
        if args.predict_season:
            if not args.schedule or not args.offense or not args.defense:
                print("\n" + "="*80)
                print("ERROR: Missing required arguments")
                print("="*80)
                print("\nAll three arguments are required for season prediction:")
                print("  --schedule: Path to schedule CSV file")
                print("  --offense:  Path to offensive stats CSV file")
                print("  --defense:  Path to defensive stats CSV file")
                print("\nUsage:")
                print("  python nfl_predictor.py --predict-season --schedule <file> --offense <file> --defense <file>")
                print("\nExample:")
                print("  python nfl_predictor.py --predict-season \\")
                print("    --schedule 2024_schedule.csv \\")
                print("    --offense 2024_offense.csv \\")
                print("    --defense 2024_defense.csv")
                print("="*80)
                sys.exit(1)
            
            try:
                print("Loading trained models...")
                predictor.load_trained_models()
            except FileNotFoundError as e:
                print("\n" + "="*80)
                print("ERROR: Models not found")
                print("="*80)
                print(f"\n{str(e)}")
                print("\nYou need to train models first before making predictions.")
                print("\nTo train models, run:")
                print("  python nfl_predictor.py --train --seasons 2019 2020 2021 2022 2023")
                print("="*80)
                sys.exit(1)
            
            try:
                print(f"\nPredicting season from schedule: {args.schedule}")
                predictions_df = predictor.predict_season(args.schedule, args.offense, args.defense)
            except FileNotFoundError as e:
                print("\n" + "="*80)
                print("ERROR: Input files not found")
                print("="*80)
                print(f"\n{str(e)}")
                print("\nPlease ensure all input files exist:")
                print(f"  - Schedule: {args.schedule}")
                print(f"  - Offense:  {args.offense}")
                print(f"  - Defense:  {args.defense}")
                print("="*80)
                sys.exit(1)
            except Exception as e:
                print("\n" + "="*80)
                print("ERROR: Prediction failed")
                print("="*80)
                print(f"\n{str(e)}")
                print("\nPlease check that your input files:")
                print("  - Are properly formatted CSV files")
                print("  - Contain all required columns")
                print("  - Have matching team names across files")
                print("="*80)
                sys.exit(1)
            
            print("\n" + "="*80)
            print("SEASON PREDICTIONS")
            print("="*80)
            print()
            print(predictions_df.to_string(index=False))
            
            # Save predictions to file
            output_file = args.schedule.replace('.csv', '_predictions.csv')
            predictions_df.to_csv(output_file, index=False)
            print(f"\nPredictions saved to: {output_file}")
            
            return
        
        # If no action specified, show help
        if not any([args.train, args.evaluate, args.predict, args.feature_importance, args.predict_season]):
            print("\n" + "="*80)
            print("NFL Game Prediction ML Pipeline")
            print("="*80)
            print("\nNo action specified. Please choose one of the following commands:\n")
            parser.print_help()
            print("\n" + "="*80)
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        print("\n" + "="*80)
        print("UNEXPECTED ERROR")
        print("="*80)
        print(f"\n{str(e)}")
        print("\nFor more details, check the log file: nfl_predictor.log")
        print("\nIf this error persists, please check:")
        print("  - All required dependencies are installed (see requirements.txt)")
        print("  - Input files are properly formatted")
        print("  - You have sufficient disk space and memory")
        print("="*80)
        sys.exit(1)


if __name__ == '__main__':
    main()
