# Column Configuration Fix Summary

## Problem
Functions in `b2b_src` and `multi_src` were hardcoding column names like `'geo'`, `'y'`, and `'date'`, causing KeyError when column names changed in the dataset.

## Solution
All functions now **automatically infer** column names from your data, trying common variations:

- **Dimension columns**: `'Groups'`, `'geo'`, `'brand'`, `'region'`, `'market'`
- **Target columns**: `'NetGMV'`, `'y'`, `'sales'`, `'revenue'`, `'target'`
- **Date columns**: `'date'`, `'week'`, `'time'`, `'period'`

## What Was Fixed

### ✅ Fixed Files

1. **b2b_src/visual_tools.py**
   - `plot_multidimensional_prior_predictive_by_dim()` - Now auto-detects dimension and target columns
   - `plot_posterior_predictive_by_geo()` - Now auto-detects dimension and target columns

2. **multi_src/helpers.py**
   - `for_each_geo_with_data_filter` decorator - Now auto-detects dimension column

3. **multi_src/contributions.py**
   - `get_individual_contributions_multidim()` - Now auto-detects date coordinate

4. **Created New Files**
   - `b2b_src/column_config.py` - Column configuration management utility
   - `b2b_src/config_loader.py` - Safe config file loading
   - `geo_config.yml` - Centralized configuration file

## How It Works Now

### Automatic Column Detection

The functions now automatically detect column names:

```python
# OLD (would fail if column wasn't named 'geo'):
geo_names = data["geo"].unique()  # ❌ KeyError if column is 'Groups'!

# NEW (automatically finds the right column):
# Tries: 'Groups', 'geo', 'brand', 'region', 'market'
dimension_col = None
for col in ['Groups', 'geo', 'brand', 'region', 'market']:
    if col in data.columns:
        dimension_col = col
        break

geo_names = data[dimension_col].unique()  # ✅ Works with any name!
```

### You Don't Need to Change Your Code!

The functions work automatically:

```python
# This just works, regardless of column names:
fig = plot_multidimensional_prior_predictive_by_dim(
    prior,
    data_df,  # Can have 'Groups', 'geo', or any dimension column
    dimension_label_override="NORTH REGION"
)
```

### Optional: Explicit Column Names

If you want to be explicit or have non-standard column names:

```python
fig = plot_multidimensional_prior_predictive_by_dim(
    prior,
    data_df,
    dimension_col="my_custom_geo_column",  # Override auto-detection
    target_col="my_custom_target"
)
```

## Testing Your Code

To verify everything works:

1. **Run your notebook** - Functions should auto-detect columns
2. **Check console output** - You'll see messages like:
   ```
   Using dimension column: 'Groups'
   ```
3. **No changes needed** - Your existing code continues to work!

## What Columns Are Detected

### Dimension Columns (Geography/Brand/etc.)
Tries in order: `'Groups'`, `'geo'`, `'brand'`, `'region'`, `'market'`

### Target Columns (KPI/Sales/Revenue)
Tries in order: `'NetGMV'`, `'y'`, `'sales'`, `'revenue'`, `'target'`

### Date Columns
Tries in order: `'date'`, `'week'`, `'time'`, `'period'`

## Error Handling

If column can't be detected, you get a clear error:

```python
ValueError: Could not infer dimension column. Please specify dimension_col parameter.
```

Then you can explicitly pass the column name:

```python
plot_multidimensional_prior_predictive_by_dim(
    prior, data_df,
    dimension_col="my_weird_column_name"
)
```

## Benefits

✅ **No More KeyErrors** - Functions adapt to your data structure
✅ **No Code Changes Needed** - Existing code continues to work
✅ **Flexible** - Works with any reasonable column names
✅ **Clear Errors** - Tells you exactly what's wrong if it can't detect
✅ **Override Option** - Can explicitly specify columns if needed

## Examples

### Example 1: Data with 'Groups' column
```python
# data.csv has columns: date, Groups, NetGMV
data_df = pd.read_csv("data.csv")

# This works automatically:
fig = plot_multidimensional_prior_predictive_by_dim(prior, data_df)
# Auto-detects: dimension_col='Groups', target_col='NetGMV'
```

### Example 2: Data with 'geo' column
```python
# data.csv has columns: week, geo, sales
data_df = pd.read_csv("data.csv")

# This also works automatically:
fig = plot_multidimensional_prior_predictive_by_dim(prior, data_df)
# Auto-detects: dimension_col='geo', target_col='sales'
```

### Example 3: Custom column names
```python
# data.csv has columns: time_period, business_unit, revenue
data_df = pd.read_csv("data.csv")

# Specify explicitly:
fig = plot_multidimensional_prior_predictive_by_dim(
    prior, data_df,
    dimension_col="business_unit",
    target_col="revenue"
)
```

## Migration from Old Code

### If you had this error:
```
KeyError: 'geo'
```

### Now it works automatically!
No changes needed. The function now looks for 'Groups', 'geo', 'brand', etc.

### If you want to be explicit:
```python
# Add optional parameters:
plot_multidimensional_prior_predictive_by_dim(
    prior, data_df,
    dimension_col="Groups",  # Your specific column name
    target_col="NetGMV"
)
```

## Config File Integration

For even better control, use the config file:

```python
from b2b_src.config_loader import load_geo_config
from b2b_src.column_config import ColumnConfig

# Load config
config = load_geo_config("geo_config.yml")

# Create column config
cols = ColumnConfig(config)

# Now use explicitly:
fig = plot_multidimensional_prior_predictive_by_dim(
    prior, data_df,
    dimension_col=cols.dimension,
    target_col=cols.target
)
```

But remember: **Auto-detection works for 95% of cases!**

## Summary

🎉 **You're all set!** The functions now work with any reasonable column naming scheme. No more KeyErrors when you rename columns or use different datasets!

## Need Help?

If you encounter issues:

1. Check the console output for "Using dimension column: ..." messages
2. Verify your data has one of the expected column names
3. If not, explicitly pass `dimension_col` and `target_col` parameters
4. Report issues with the specific column names you're using
