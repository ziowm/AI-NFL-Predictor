# Data Preparation Guide

This guide explains how to prepare your NFL data for use with the ML prediction pipeline.

## Overview

The pipeline requires three CSV files per season:
1. Offensive statistics
2. Defensive statistics  
3. Game schedule

All files must follow specific naming conventions and column requirements.

## File Naming Convention

Files must be named using the following format:

```
<YEAR> Fantasy Offense Stats.csv
<YEAR> Fantasy Defense Stats.csv
<YEAR> Schedule.csv
```

**Examples:**
- `2023 Fantasy Offense Stats.csv`
- `2023 Fantasy Defense Stats.csv`
- `2023 Schedule.csv`

**Important:** The year must match exactly when you specify seasons in the training command.

## Required Columns

### Offensive Statistics File

**Filename:** `<YEAR> Fantasy Offense Stats.csv`

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `Name` | string | Team name | "Kansas City Chiefs" |
| `GP` | integer | Games played | 16 |
| `off_PTS` | float | Total points scored | 450.0 |
| `off_All` | float | Total offensive yards | 6200.0 |
| `off_Pass` | float | Total passing yards | 4200.0 |
| `off_Run` | float | Total rushing yards | 2000.0 |
| `off_QB` | float | QB fantasy points | 320.0 |

**Example CSV:**
```csv
Name,GP,off_PTS,off_All,off_Pass,off_Run,off_QB
Kansas City Chiefs,16,450,6200,4200,2000,320
Buffalo Bills,16,430,6000,4000,2000,310
```

### Defensive Statistics File

**Filename:** `<YEAR> Fantasy Defense Stats.csv`

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `Name` | string | Team name (must match offense file) | "Kansas City Chiefs" |
| `GP` | integer | Games played | 16 |
| `def_PA` | float | Points allowed | 310.0 |
| `def_DEF` | float | Defensive fantasy points allowed | 270.0 |
| `def_QB` | float | QB fantasy points allowed | 240.0 |

**Example CSV:**
```csv
Name,GP,def_PA,def_DEF,def_QB
Kansas City Chiefs,16,310,270,240
Buffalo Bills,16,330,290,255
```

### Schedule File

**Filename:** `<YEAR> Schedule.csv`

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `Week` | integer | Week number (1-18) | 1 |
| `Team1` | string | Home team name | "Kansas City Chiefs" |
| `Team2` | string | Away team name | "Buffalo Bills" |

**Example CSV:**
```csv
Week,Team1,Team2
1,Kansas City Chiefs,Buffalo Bills
1,Green Bay Packers,Chicago Bears
2,Buffalo Bills,Miami Dolphins
```

## Data Quality Requirements

### 1. Team Name Consistency

Team names **must be identical** across all three files for each season.

**✓ Correct:**
```
Offense file: "Kansas City Chiefs"
Defense file: "Kansas City Chiefs"
Schedule file: "Kansas City Chiefs"
```

**✗ Incorrect:**
```
Offense file: "Kansas City Chiefs"
Defense file: "KC Chiefs"
Schedule file: "Kansas City"
```

### 2. Complete Data

- All teams in the schedule must have corresponding entries in both offense and defense files
- No missing values in required columns
- All numeric columns must contain valid numbers (not text or empty cells)

### 3. Data Types

- Team names: Text/string
- Week numbers: Integers (1-18)
- Games played: Integers (typically 16-17)
- All statistics: Floating point numbers (decimals allowed)

### 4. Reasonable Value Ranges

The pipeline will validate that values fall within reasonable ranges:

| Statistic | Typical Range | Notes |
|-----------|---------------|-------|
| GP | 1-17 | Games played in a season |
| off_PTS | 150-600 | Total points scored |
| off_All | 3000-7000 | Total offensive yards |
| off_Pass | 2000-5000 | Passing yards |
| off_Run | 1000-3000 | Rushing yards |
| off_QB | 150-400 | QB fantasy points |
| def_PA | 200-500 | Points allowed |
| def_DEF | 150-400 | Defensive fantasy points |
| def_QB | 150-350 | QB fantasy points allowed |

## Data Sources

### Recommended Sources

1. **Fantasy Football Statistics**
   - https://www.thefantasyfootballers.com/team-stats/
   - Provides comprehensive offensive and defensive statistics

2. **NFL Official Statistics**
   - https://www.nfl.com/stats/
   - Official league statistics

3. **Pro Football Reference**
   - https://www.pro-football-reference.com/
   - Historical data and advanced metrics

### Extracting Data

When extracting data from these sources:

1. **Export to CSV**: Most sites allow CSV export
2. **Rename columns**: Match the required column names exactly
3. **Verify team names**: Ensure consistency across files
4. **Check completeness**: Verify all 32 NFL teams are included

## Preparing New Season Data

### Step-by-Step Process

1. **Collect Raw Data**
   ```bash
   # Download statistics from your preferred source
   # Save as temporary files
   ```

2. **Format Offensive Stats**
   - Open in spreadsheet software (Excel, Google Sheets)
   - Rename columns to match requirements
   - Ensure team names are consistent
   - Save as `<YEAR> Fantasy Offense Stats.csv`

3. **Format Defensive Stats**
   - Repeat process for defensive statistics
   - **Critical:** Use exact same team names as offense file
   - Save as `<YEAR> Fantasy Defense Stats.csv`

4. **Create Schedule**
   - List all games for the season
   - Include week number, home team (Team1), away team (Team2)
   - Use same team names as stats files
   - Save as `<YEAR> Schedule.csv`

5. **Validate Data**
   ```bash
   # Run validation script (if available)
   python validate_data.py --season 2024
   
   # Or attempt to load with the pipeline
   python nfl_predictor.py --train --seasons 2024
   ```

## Common Issues and Solutions

### Issue: "Team not found in statistics"

**Cause:** Team name mismatch between schedule and stats files

**Solution:**
1. Open all three CSV files
2. Compare team names character-by-character
3. Ensure exact matches (including spaces, capitalization)
4. Common mistakes:
   - "Kansas City Chiefs" vs "KC Chiefs"
   - Extra spaces: "Buffalo Bills " vs "Buffalo Bills"
   - Different abbreviations

### Issue: "Missing required column"

**Cause:** CSV file doesn't have all required columns

**Solution:**
1. Check column names in your CSV file
2. Compare against required columns list above
3. Rename columns to match exactly (case-sensitive)
4. Ensure no typos: `off_PTS` not `off_pts` or `off_POINTS`

### Issue: "Invalid data type"

**Cause:** Non-numeric values in numeric columns

**Solution:**
1. Open CSV in spreadsheet software
2. Check for text in numeric columns (e.g., "N/A", "-", "TBD")
3. Replace with valid numbers or remove rows
4. Ensure no commas in numbers (use 1000 not 1,000)

### Issue: "Insufficient data for training"

**Cause:** Not enough games or seasons

**Solution:**
1. Ensure each season has at least 100 games in schedule
2. Provide at least 3 seasons for training
3. Verify all scheduled games have team statistics

### Issue: "High percentage of missing values"

**Cause:** Many empty cells in statistics

**Solution:**
1. Fill in missing values from data source
2. If data unavailable, remove incomplete teams
3. Ensure at least 80% of values are present

## Data Validation Checklist

Before training models, verify:

- [ ] All three CSV files exist for each season
- [ ] Files follow naming convention: `<YEAR> Fantasy <Type> Stats.csv`
- [ ] All required columns are present
- [ ] Column names match exactly (case-sensitive)
- [ ] Team names are identical across all files
- [ ] No missing values in required columns
- [ ] All numeric columns contain valid numbers
- [ ] Schedule includes all teams from stats files
- [ ] At least 3 seasons of data available
- [ ] Each season has 100+ games

## Example: Complete Season Data

Here's a minimal example of complete season data:

**2023 Fantasy Offense Stats.csv:**
```csv
Name,GP,off_PTS,off_All,off_Pass,off_Run,off_QB
Kansas City Chiefs,16,450,6200,4200,2000,320
Buffalo Bills,16,430,6000,4000,2000,310
```

**2023 Fantasy Defense Stats.csv:**
```csv
Name,GP,def_PA,def_DEF,def_QB
Kansas City Chiefs,16,310,270,240
Buffalo Bills,16,330,290,255
```

**2023 Schedule.csv:**
```csv
Week,Team1,Team2
1,Kansas City Chiefs,Buffalo Bills
2,Buffalo Bills,Kansas City Chiefs
```

## Advanced: Custom Data Processing

If your data source uses different formats, you can create a preprocessing script:

```python
import pandas as pd

# Load your custom format
df = pd.read_csv('custom_format.csv')

# Transform to required format
offense_df = pd.DataFrame({
    'Name': df['team_name'],
    'GP': df['games'],
    'off_PTS': df['points_scored'],
    'off_All': df['total_yards'],
    'off_Pass': df['passing_yards'],
    'off_Run': df['rushing_yards'],
    'off_QB': df['qb_points']
})

# Save in required format
offense_df.to_csv('2024 Fantasy Offense Stats.csv', index=False)
```

## Getting Help

If you encounter issues with data preparation:

1. Check the error message in `nfl_predictor.log`
2. Verify your data against this guide
3. Review the troubleshooting section in README.md
4. Ensure all dependencies are installed correctly

## Summary

Proper data preparation is crucial for model performance. Follow these key principles:

1. **Consistency**: Use identical team names across all files
2. **Completeness**: Include all required columns and teams
3. **Quality**: Ensure valid data types and reasonable values
4. **Validation**: Check data before training

With properly formatted data, the pipeline will automatically handle feature engineering and model training to achieve optimal prediction accuracy.
