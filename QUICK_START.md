# Quick Start Guide

## You Just Ran Your First NFL Prediction! 🎉

Here's what happened and what you can do next.

## What You Just Did

1. **Installed OpenMP** - Required for XGBoost to work on macOS
2. **Trained 3 ML Models** - Random Forest, Logistic Regression, and XGBoost on 3 seasons of data (60 games)
3. **Made Predictions** - Predicted Kansas City Chiefs vs Buffalo Bills matchup
4. **Analyzed Features** - Saw which statistics are most important for predictions

## Your Trained Models

Your models are saved in the `models/` directory:
- `random_forest.joblib` - 99KB
- `xgboost.joblib` - 148KB  
- `logistic_regression.joblib` - 1KB
- `feature_scaler.joblib` - Normalizes features
- `feature_names.json` - List of all 36 features
- `metadata.json` - Training configuration

## Quick Commands

### 1. Make a Single Game Prediction

```bash
python3 nfl_predictor.py --predict --team1 "Kansas City Chiefs" --team2 "Buffalo Bills"
```

### 2. View Feature Importance

```bash
python3 nfl_predictor.py --feature-importance --model random_forest
```

Or for XGBoost:
```bash
python3 nfl_predictor.py --feature-importance --model xgboost
```

### 3. Evaluate Models

```bash
python3 nfl_predictor.py --evaluate
```

### 4. Run Example Scripts

**Training Example:**
```bash
python3 example_training.py
```

**Prediction Examples:**
```bash
python3 example_prediction.py
```

## Understanding the Results

### Prediction Output

When you predict a game, you get:

```
RANDOM FOREST:
  Winner: Buffalo Bills
  Confidence: 53.0%
  
LOGISTIC REGRESSION:
  Winner: Kansas City Chiefs
  Confidence: 51.1%
  
XGBOOST:
  Winner: Kansas City Chiefs
  Confidence: 74.0%

CONSENSUS PREDICTION: Kansas City Chiefs
```

- **Winner**: Which team the model predicts will win
- **Confidence**: How confident the model is (higher = more confident)
- **Consensus**: The majority vote from all 3 models

### Feature Importance

The top features that drive predictions:

1. **team1_yards_per_game** (11.8%) - Total offensive production
2. **team1_points_per_game_rolling_3games** (9.6%) - Recent scoring trend
3. **team1_points_per_game** (7.2%) - Average scoring
4. **team1_offensive_efficiency_rolling_3games** (7.0%) - Recent efficiency
5. **team1_offensive_efficiency** (6.1%) - Yards per play

## Using Your Own Data

### Data Format Required

You need 3 CSV files per season:

**Offense Stats:** `<YEAR> Fantasy Offense Stats.csv`
```csv
Name,GP,PTS,All,Pass,Run,QB
Kansas City Chiefs,17,485,6200,4200,2000,320
```

**Defense Stats:** `<YEAR> Fantasy Defense Stats.csv`
```csv
Name,GP,PA,DEF,QB
Kansas City Chiefs,17,371,289,255
```

**Schedule:** `<YEAR> Schedule.csv`
```csv
Week,Team1,Team2
1,Kansas City Chiefs,Buffalo Bills
```

### Training with Your Data

1. Place your CSV files in the project directory
2. Run training:
```bash
python3 nfl_predictor.py --train --seasons 2021 2022 2023
```

For more details, see `DATA_PREPARATION_GUIDE.md`

## Current Sample Data

You're currently using sample data with:
- **3 seasons**: 2021, 2022, 2023
- **8 teams**: Chiefs, Bills, Dolphins, Patriots, Bengals, Ravens, Browns, Steelers
- **60 total games**: 20 games per season
- **36 features**: Engineered from raw statistics

## Model Performance (on sample data)

Your models achieved:
- **Random Forest**: 85% cross-validation accuracy
- **Logistic Regression**: 100% cross-validation accuracy (may be overfitting on small dataset)
- **XGBoost**: High accuracy with best hyperparameters

**Note**: These results are on a small sample dataset. With full NFL season data (256+ games per season), you should see more realistic accuracy around 70-72%.

## Next Steps

### Option 1: Get Real NFL Data

1. Download statistics from:
   - https://www.thefantasyfootballers.com/team-stats/
   - https://www.pro-football-reference.com/

2. Format according to `DATA_PREPARATION_GUIDE.md`

3. Train with 5 seasons for best results:
```bash
python3 nfl_predictor.py --train --seasons 2019 2020 2021 2022 2023
```

### Option 2: Experiment with Current Models

1. Try different team matchups:
```bash
python3 nfl_predictor.py --predict --team1 "Cincinnati Bengals" --team2 "Baltimore Ravens"
```

2. Compare feature importance across models:
```bash
python3 nfl_predictor.py --feature-importance --model random_forest
python3 nfl_predictor.py --feature-importance --model xgboost
```

### Option 3: Modify the Pipeline

The code is modular and easy to extend:

- **Add features**: Edit `ml_pipeline/feature_engineering/feature_engineer.py`
- **Try new models**: Edit `ml_pipeline/model_trainer/model_trainer.py`
- **Change hyperparameters**: Modify the parameter grids in model trainer
- **Add new metrics**: Edit `ml_pipeline/model_evaluator/model_evaluator.py`

## Troubleshooting

### "No trained models found"
Run training first: `python3 nfl_predictor.py --train --seasons 2021 2022 2023`

### "Team not found"
Make sure team names match exactly (including spaces and capitalization)

### "Missing required columns"
Check your CSV files match the format in `DATA_PREPARATION_GUIDE.md`

### XGBoost errors on macOS
Install OpenMP: `brew install libomp`

## Getting Help

- **Full documentation**: See `README.md`
- **Data preparation**: See `DATA_PREPARATION_GUIDE.md`
- **Example code**: Check `example_training.py` and `example_prediction.py`
- **Logs**: Check `nfl_predictor.log` for detailed error messages

## Summary

You now have a working ML pipeline that can:
✅ Train multiple models on NFL data
✅ Make game predictions with confidence scores
✅ Analyze which statistics matter most
✅ Handle new data for future predictions

Have fun predicting games! 🏈
