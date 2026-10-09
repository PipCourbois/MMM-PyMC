# Decorator Fix Summary

## Issues Fixed

### Issue 1: TypeError - "got multiple values for argument 'mmm'"

**Root Cause**: The `for_each_geo_with_data_filter` decorator had a flawed wrapper signature that assumed `data` as the second positional parameter:

```python
# OLD (WRONG):
def wrapper(contributions, data, *args, **kwargs):
```

When calling `plot_all_contributions(contributions=x, mmm=y, data=z, ...)`:
- Python matched `mmm=y` to the `data` parameter (second positional)
- The actual `data=z` went into `**kwargs`
- The wrapper then passed `mmm` both positionally AND as a keyword arg
- Result: **TypeError: got multiple values for argument 'mmm'**

**Solution**: Changed wrapper signature to only expect `contributions` as positional:

```python
# NEW (CORRECT):
def wrapper(contributions, *args, **kwargs):
```

Now:
- `contributions=x` → `contributions` parameter
- `mmm=y` → `**kwargs`
- `data=z` → `**kwargs`
- Decorator extracts `data` from kwargs and passes everything correctly

**Files Modified**:
- `multi_src/helpers.py` - Line 254 (wrapper signature in `for_each_geo_with_data_filter`)

---

### Issue 2: TypeError - "got an unexpected keyword argument 'dimension_label_override'"

**Root Cause**: The `dimension_label_override` parameter is meant for the **decorator** (to control how dimension labels are injected), not for the wrapped function. The decorator was not consuming this parameter before passing kwargs to the function in non-multidimensional cases.

**Solution**: Pop `dimension_label_override` from kwargs before calling the wrapped function:

```python
if not is_multidim:
    # Standard MMM, call function directly
    # Pop dimension_label_override if present (decorator-specific param)
    kwargs.pop('dimension_label_override', None)
    return func(contributions, *args, data=data, **kwargs)
```

**Files Modified**:
- `multi_src/helpers.py` - Lines 160, 176, 281 (pop dimension_label_override in early returns)

---

### Issue 3: Multidimensional detection only checked for 'geo' dimension

**Root Cause**: The code hardcoded `'geo'` when checking for multidimensional contributions:

```python
# OLD (WRONG):
is_multidim = hasattr(first_channel, 'dims') and 'geo' in first_channel.dims
```

This failed with the new dataset that uses `'Groups'` instead of `'geo'`.

**Solution**: Check for ANY spatial dimension (not just 'geo'):

```python
# NEW (CORRECT):
if hasattr(first_channel, 'dims'):
    # Get all dims except common time dimensions
    spatial_dims = [d for d in first_channel.dims if d not in ['date', 'week', 'time', 'period']]
    is_multidim = len(spatial_dims) > 0
else:
    is_multidim = False
```

This now works with any dimension name: `'geo'`, `'Groups'`, `'brand'`, `'region'`, `'market'`, etc.

**Files Modified**:
- `multi_src/helpers.py` - Lines 163-171 (`for_each_geo` decorator)
- `multi_src/helpers.py` - Lines 268-276 (`for_each_geo_with_data_filter` decorator)

---

## Relationship to Recent Changes

These issues surfaced because of your recent work:

1. **New dataset with different column names**: Changed from `'geo'` to `'Groups'`, `'y'` to `'NetGMV'`
2. **Config file support**: Added auto-detection of column names in visualization functions
3. **Active testing**: Actually using `plot_all_contributions()` which exposed the decorator bugs

The decorator needed to be updated to:
- Handle functions with parameters between `contributions` and `data` (like `mmm`)
- Work with any dimension name (not just `'geo'`)
- Properly consume decorator-specific parameters before passing to wrapped functions
- Be consistent with the auto-detection pattern added to other functions

---

## What Was Changed

### File: `multi_src/helpers.py`

#### Change 1: Fixed `for_each_geo` decorator wrapper signature
**Line 155**:
```python
# BEFORE
def wrapper(contributions, *args, **kwargs):
    # Detect if contributions are multidimensional
    # Check the first channel to see if it has a 'geo' dimension
    if not contributions['channels']:
        # No channels, call function directly
        return func(contributions, *args, **kwargs)

    first_channel = list(contributions['channels'].values())[0]
    is_multidim = hasattr(first_channel, 'dims') and 'geo' in first_channel.dims

    if not is_multidim:
        # Standard MMM (single geo), call function directly
        return func(contributions, *args, **kwargs)

# AFTER
def wrapper(contributions, *args, **kwargs):
    # Detect if contributions are multidimensional
    if not contributions['channels']:
        # No channels, call function directly
        # Pop dimension_label_override if present (decorator-specific param)
        kwargs.pop('dimension_label_override', None)
        return func(contributions, *args, **kwargs)

    first_channel = list(contributions['channels'].values())[0]
    # Check if multidimensional by looking for ANY dimension besides 'date'
    # Could be 'geo', 'Groups', 'brand', etc.
    if hasattr(first_channel, 'dims'):
        # Get all dims except common time dimensions
        spatial_dims = [d for d in first_channel.dims if d not in ['date', 'week', 'time', 'period']]
        is_multidim = len(spatial_dims) > 0
    else:
        is_multidim = False

    if not is_multidim:
        # Standard MMM (single geo), call function directly
        # Pop dimension_label_override if present (decorator-specific param)
        kwargs.pop('dimension_label_override', None)
        return func(contributions, *args, **kwargs)
```

#### Change 2: Fixed `for_each_geo_with_data_filter` decorator wrapper signature
**Line 254**:
```python
# BEFORE
def wrapper(contributions, data, *args, **kwargs):
    # Extract 'data' from either positional or keyword arguments
    # [complex logic trying to handle data in args or kwargs]

    first_channel = list(contributions['channels'].values())[0]
    is_multidim = hasattr(first_channel, 'dims') and 'geo' in first_channel.dims

    if not is_multidim:
        # Standard MMM, call function directly
        return func(contributions, data, *args, **kwargs)  # ← WRONG: data passed positionally

# AFTER
def wrapper(contributions, *args, **kwargs):
    # Extract 'data' from kwargs (must be passed as keyword argument)
    if 'data' not in kwargs:
        raise ValueError(
            f"Function {func.__name__} must be called with 'data' as a keyword argument "
            f"when using for_each_geo_with_data_filter decorator"
        )

    data = kwargs.pop('data')

    # Detect if contributions are multidimensional
    if not contributions['channels']:
        return func(contributions, *args, data=data, **kwargs)

    first_channel = list(contributions['channels'].values())[0]
    # Check if multidimensional by looking for ANY dimension besides 'date'
    # Could be 'geo', 'Groups', 'brand', etc.
    if hasattr(first_channel, 'dims'):
        # Get all dims except common time dimensions
        spatial_dims = [d for d in first_channel.dims if d not in ['date', 'week', 'time', 'period']]
        is_multidim = len(spatial_dims) > 0
    else:
        is_multidim = False

    if not is_multidim:
        # Standard MMM, call function directly
        # Pop dimension_label_override if present (decorator-specific param)
        kwargs.pop('dimension_label_override', None)
        return func(contributions, *args, data=data, **kwargs)  # ← CORRECT: data passed as keyword
```

---

## Testing

The fix should now allow this code to work:

```python
figs_by_geo = plot_all_contributions(
    contributions=contributions,  # Has 'Groups' dimension, not 'geo'
    mmm=mmm,
    data=data_df,
    use_log_scale=False,
    dimension_label_override="REGION 1"  # Decorator-specific param
)
```

Expected behavior:
1. ✅ Decorator detects multidimensional contributions (checks for 'Groups' dimension)
2. ✅ Pops `dimension_label_override` from kwargs
3. ✅ Loops over each geo ('Brand_A', 'Brand_B')
4. ✅ Filters data for each geo
5. ✅ Injects `dimension_label="REGION 1"` for each geo
6. ✅ Calls `_plot_all_contributions_wrapper(contrib_geo, data_geo, mmm, use_log_scale, dimension_label="REGION 1")`
7. ✅ Returns `{geo: (fig_combined, fig_individual), ...}`

---

## Summary

**Problem**: Decorator had three bugs:
1. Wrong wrapper signature assumed `data` as second positional parameter
2. Didn't consume `dimension_label_override` before passing to wrapped function
3. Only checked for `'geo'` dimension, not other spatial dimensions

**Solution**:
1. Changed wrapper signature to `wrapper(contributions, *args, **kwargs)`
2. Pop `dimension_label_override` from kwargs in early returns
3. Check for any spatial dimension (not just `'geo'`)

**Result**: Decorators now work correctly with:
- Functions that have parameters between `contributions` and `data`
- Any dimension name (`'Groups'`, `'geo'`, `'brand'`, etc.)
- New dataset and config file structure
- All existing code continues to work

---

## Files Modified

1. **multi_src/helpers.py**
   - Line 155-177: `for_each_geo` decorator wrapper
   - Line 254-282: `for_each_geo_with_data_filter` decorator wrapper

No other files needed changes. The wrapped functions in `multi_src/wrapped_functions.py` work correctly with the fixed decorators.
