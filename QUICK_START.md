# Quick Start: Using geo_config.yml

## TL;DR

1. **All column names are now auto-detected** - no more KeyErrors!
2. **Config file centralizes everything** - change data/parameters without code changes
3. **Your existing code still works** - backward compatible!

## 30-Second Fix

If you get `KeyError: 'geo'` or similar:

```python
# Your code probably works now automatically!
# Functions now try: 'Groups', 'geo', 'brand', 'region', 'market'

# Run your code again - it should just work!
```

## Using the Config File (Recommended)

### 1. Edit `geo_config.yml` for your data:

```yaml
# Change the data source
source: &data_source "data/my_new_dataset.csv"

# Update column names if different
data:
  columns:
    date_column: "date"        # ← Your date column name
    target_column: "NetGMV"    # ← Your target/KPI column name
    dimension_column: "Groups"  # ← Your geo/brand column name

    channels:
      - name: "Spend_channel_1"
        spend_column: "Spend_channel_1"
        type: "paid"
```

### 2. Load config in your notebook:

```python
from b2b_src.config_loader import load_geo_config, get_channel_config, get_control_config

# Load config
config = load_geo_config("geo_config.yml")

# Get column names
date_col = config.data.columns.date_column
target_col = config.data.columns.target_column
dimension_col = config.data.columns.dimension_column

# Get channel/control info
channel_info = get_channel_config(config)
control_cols = get_control_config(config)
```

### 3. Use in your code:

```python
# Load data using config
data_df = pd.read_csv(config.data.source, parse_dates=[date_col])

# Functions auto-detect columns (works without changes!)
fig = plot_multidimensional_prior_predictive_by_dim(prior, data_df)

# Or be explicit:
fig = plot_multidimensional_prior_predictive_by_dim(
    prior, data_df,
    dimension_col=dimension_col,
    target_col=target_col
)
```

## Common Patterns

### Pattern 1: Quick notebook with auto-detection
```python
# Load data (column names don't matter much)
data_df = pd.read_csv("data/my_data.csv", parse_dates=["date"])

# Just use the functions - they auto-detect!
fig = plot_multidimensional_prior_predictive_by_dim(prior, data_df)
```

### Pattern 2: Config-driven notebook (recommended)
```python
from b2b_src.config_loader import load_geo_config

config = load_geo_config("geo_config.yml")

# Load data from config
data_df = pd.read_csv(
    config.data.source,
    parse_dates=[config.data.columns.date_column]
)

# Use functions (still auto-detects!)
fig = plot_multidimensional_prior_predictive_by_dim(prior, data_df)
```

### Pattern 3: Explicit control
```python
# When you have unusual column names
fig = plot_multidimensional_prior_predictive_by_dim(
    prior, data_df,
    dimension_col="my_custom_geo_col",
    target_col="my_custom_target"
)
```

## What Changed?

### Before (caused errors):
```python
geo_names = data["geo"].unique()  # ❌ KeyError if column is 'Groups'!
```

### After (auto-detects):
```python
# Functions now automatically find the right column
# Tries: 'Groups', 'geo', 'brand', 'region', 'market'
geo_names = data[dimension_col].unique()  # ✅ Works!
```

## Files Created

1. **`geo_config.yml`** - Main configuration file (edit this!)
2. **`b2b_src/config_loader.py`** - Safe config loading utilities
3. **`b2b_src/column_config.py`** - Column management (optional)
4. **`COLUMN_CONFIG_FIX_SUMMARY.md`** - Detailed documentation
5. **`QUICK_START.md`** - This file!

## Next Steps

1. ✅ Run your notebook - should work with auto-detection
2. ✅ Edit `geo_config.yml` for your specific data
3. ✅ (Optional) Update notebook to use config file
4. ✅ Change datasets easily by updating config only!

## Help!

### Error: "Could not infer dimension column"

Your column has an unusual name. Use explicit parameter:

```python
fig = plot_multidimensional_prior_predictive_by_dim(
    prior, data_df,
    dimension_col="your_column_name"  # ← Add this
)
```

### Error: "KeyError: 'some_column'"

Check which function is failing. Most are fixed, but if you find one:

1. Look for parameter like `dimension_col` or `target_col`
2. Pass your column name explicitly
3. Report it so we can add auto-detection!

## That's It!

🎉 You're ready to go! The system now adapts to your data structure automatically.
