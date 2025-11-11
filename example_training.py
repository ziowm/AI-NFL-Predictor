#!/usr/bin/env python3
"""
Example Training Script

This script demonstrates how to use the NFL Predictor ML pipeline
to train models on multiple seasons of data.

Usage:
    python example_training.py
"""

from nfl_predictor import NFLPredictor
import logging

# Configure logging to see detailed progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """
    Example: Train models on 5 seasons of NFL data
    """
    print("="*80)
    print("NFL Predictor - Training Example")
    print("="*80)
    print()
    
    # Step 1: Create predictor instance
    print("Step 1: Initializing NFL Predictor...")
    predictor = NFLPredictor()
    print("✓ Predictor initialized\n")
    
    # Step 2: Define seasons for training
    # The pipeline will automatically split these into train/val/test sets
    # For 5 seasons: First 3 for training, 4th for validation, 5th for testing
    seasons = ['2019', '2020', '2021', '2022', '2023']
    
    print("Step 2: Training Configuration")
    print(f"  Seasons: {seasons}")
    print(f"  Total seasons: {len(seasons)}")
    print(f"  Expected split:")
    print(f"    - Training: {seasons[:3]} (3 seasons)")
    print(f"    - Validation: {seasons[3:4]} (1 season)")
    print(f"    - Test: {seasons[4:]} (1 season)")
    print()
    
    # Step 3: Train the pipeline
    print("Step 3: Starting training pipeline...")
    print("This will:")
    print("  1. Load data from CSV files")
    print("  2. Engineer 36 features from raw statistics")
    print("  3. Split data temporally (train/val/test)")
    print("  4. Train 3 models with hyperparameter tuning")
    print("  5. Evaluate models on validation and test sets")
    print("  6. Save trained models to disk")
    print()
    print("Note: This may take 10-20 minutes depending on your hardware")
    print()
    
    try:
        predictor.train_pipeline(seasons)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print()
        print("Next steps:")
        print("  1. Evaluate models: python nfl_predictor.py --evaluate")
        print("  2. Make predictions: python nfl_predictor.py --predict --team1 'Team A' --team2 'Team B'")
        print("  3. View feature importance: python nfl_predictor.py --feature-importance --model random_forest")
        print()
        
    except FileNotFoundError as e:
        print("\n" + "="*80)
        print("ERROR: Missing Data Files")
        print("="*80)
        print(f"\n{str(e)}")
        print("\nPlease ensure you have the following CSV files:")
        for season in seasons:
            print(f"\n{season}:")
            print(f"  - {season} Fantasy Offense Stats.csv")
            print(f"  - {season} Fantasy Defense Stats.csv")
            print(f"  - {season} Schedule.csv")
        print()
        
    except ValueError as e:
        print("\n" + "="*80)
        print("ERROR: Invalid Data")
        print("="*80)
        print(f"\n{str(e)}")
        print("\nPlease check your CSV files for:")
        print("  - Required columns")
        print("  - Valid data types")
        print("  - Consistent team names")
        print()
        
    except Exception as e:
        print("\n" + "="*80)
        print("ERROR: Unexpected Error")
        print("="*80)
        print(f"\n{str(e)}")
        print("\nCheck nfl_predictor.log for detailed error information")
        print()


if __name__ == '__main__':
    main()
