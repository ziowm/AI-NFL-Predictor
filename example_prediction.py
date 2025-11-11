#!/usr/bin/env python3
"""
Example Prediction Script

This script demonstrates how to use trained models to make predictions
for NFL games.

Usage:
    python example_prediction.py
"""

from nfl_predictor import NFLPredictor
import pandas as pd

def example_single_game_prediction():
    """
    Example: Predict the outcome of a single game
    """
    print("="*80)
    print("Example 1: Single Game Prediction")
    print("="*80)
    print()
    
    # Create predictor and load trained models
    predictor = NFLPredictor()
    
    try:
        print("Loading trained models...")
        predictor.load_trained_models()
        print(f"✓ Loaded {len(predictor.trained_models)} models\n")
    except FileNotFoundError:
        print("ERROR: No trained models found.")
        print("Please train models first using:")
        print("  python nfl_predictor.py --train --seasons 2019 2020 2021 2022 2023")
        return
    
    # Define team names
    team1 = "Kansas City Chiefs"
    team2 = "Buffalo Bills"
    
    print(f"Predicting: {team1} vs {team2}\n")
    
    # Define team statistics
    # In a real scenario, you would load these from your data source
    team1_stats = {
        'GP': 16,
        'off_PTS': 450,  # Total points scored
        'off_All': 6200,  # Total offensive yards
        'off_Pass': 4200,  # Passing yards
        'off_Run': 2000,  # Rushing yards
        'off_QB': 320,    # QB fantasy points
        'def_PA': 310,    # Points allowed
        'def_DEF': 270,   # Defensive fantasy points allowed
        'def_QB': 240     # QB fantasy points allowed
    }
    
    team2_stats = {
        'GP': 16,
        'off_PTS': 430,
        'off_All': 6000,
        'off_Pass': 4000,
        'off_Run': 2000,
        'off_QB': 310,
        'def_PA': 330,
        'def_DEF': 290,
        'def_QB': 255
    }
    
    # Make prediction
    result = predictor.predict_game(team1, team2, team1_stats, team2_stats)
    
    # Display results
    print("PREDICTIONS:")
    print("-" * 80)
    
    for model_name, pred in result['predictions'].items():
        print(f"\n{model_name.upper().replace('_', ' ')}:")
        print(f"  Winner: {pred['winner']}")
        print(f"  Confidence: {pred['probability']:.1%}")
        print(f"  {team1} win probability: {pred['team1_win_probability']:.1%}")
        print(f"  {team2} win probability: {(1 - pred['team1_win_probability']):.1%}")
    
    print("\n" + "="*80)
    print(f"CONSENSUS PREDICTION: {result['consensus']}")
    print("="*80)
    print()


def example_season_prediction():
    """
    Example: Predict all games in a season
    """
    print("="*80)
    print("Example 2: Full Season Prediction")
    print("="*80)
    print()
    
    # Create predictor and load trained models
    predictor = NFLPredictor()
    
    try:
        print("Loading trained models...")
        predictor.load_trained_models()
        print(f"✓ Loaded {len(predictor.trained_models)} models\n")
    except FileNotFoundError:
        print("ERROR: No trained models found.")
        print("Please train models first using:")
        print("  python nfl_predictor.py --train --seasons 2019 2020 2021 2022 2023")
        return
    
    # Define file paths
    # These files should contain the schedule and team statistics for the season
    schedule_file = "2024 Schedule.csv"
    offense_file = "2024 Fantasy Offense Stats.csv"
    defense_file = "2024 Fantasy Defense Stats.csv"
    
    print(f"Predicting season from:")
    print(f"  Schedule: {schedule_file}")
    print(f"  Offense:  {offense_file}")
    print(f"  Defense:  {defense_file}")
    print()
    
    try:
        # Make predictions for all games
        predictions_df = predictor.predict_season(
            schedule_file,
            offense_file,
            defense_file
        )
        
        # Display results
        print("SEASON PREDICTIONS:")
        print("="*80)
        print()
        
        # Show first 10 games
        print("First 10 games:")
        print(predictions_df.head(10).to_string(index=False))
        print()
        
        # Summary statistics
        print("SUMMARY:")
        print("-" * 80)
        print(f"Total games predicted: {len(predictions_df)}")
        
        # Count predictions by model
        if 'random_forest_prediction' in predictions_df.columns:
            rf_team1_wins = (predictions_df['random_forest_prediction'] == 1).sum()
            print(f"Random Forest predicts Team1 wins: {rf_team1_wins}/{len(predictions_df)}")
        
        if 'xgboost_prediction' in predictions_df.columns:
            xgb_team1_wins = (predictions_df['xgboost_prediction'] == 1).sum()
            print(f"XGBoost predicts Team1 wins: {xgb_team1_wins}/{len(predictions_df)}")
        
        # Save to file
        output_file = schedule_file.replace('.csv', '_predictions.csv')
        predictions_df.to_csv(output_file, index=False)
        print(f"\n✓ Full predictions saved to: {output_file}")
        print()
        
    except FileNotFoundError as e:
        print(f"ERROR: {str(e)}")
        print("\nPlease ensure the following files exist:")
        print(f"  - {schedule_file}")
        print(f"  - {offense_file}")
        print(f"  - {defense_file}")
        print()


def example_feature_importance():
    """
    Example: View feature importance for a model
    """
    print("="*80)
    print("Example 3: Feature Importance Analysis")
    print("="*80)
    print()
    
    # Create predictor and load trained models
    predictor = NFLPredictor()
    
    try:
        print("Loading trained models...")
        predictor.load_trained_models()
        print(f"✓ Loaded {len(predictor.trained_models)} models\n")
    except FileNotFoundError:
        print("ERROR: No trained models found.")
        print("Please train models first using:")
        print("  python nfl_predictor.py --train --seasons 2019 2020 2021 2022 2023")
        return
    
    # Display feature importance for Random Forest
    model_name = 'random_forest'
    print(f"Analyzing feature importance for: {model_name}\n")
    
    try:
        predictor.display_feature_importance(model_name, top_n=15)
    except ValueError as e:
        print(f"ERROR: {str(e)}")
        print("\nNote: Feature importance is only available for tree-based models:")
        print("  - random_forest")
        print("  - xgboost")
        print()


def main():
    """
    Run all examples
    """
    print("\n" + "="*80)
    print("NFL PREDICTOR - PREDICTION EXAMPLES")
    print("="*80)
    print()
    print("This script demonstrates three prediction scenarios:")
    print("  1. Single game prediction")
    print("  2. Full season prediction")
    print("  3. Feature importance analysis")
    print()
    
    # Run examples
    example_single_game_prediction()
    print("\n")
    
    # Uncomment to run season prediction example
    # Note: Requires 2024 season data files
    # example_season_prediction()
    # print("\n")
    
    example_feature_importance()
    
    print("\n" + "="*80)
    print("EXAMPLES COMPLETE")
    print("="*80)
    print()
    print("For more options, see:")
    print("  python nfl_predictor.py --help")
    print()


if __name__ == '__main__':
    main()
