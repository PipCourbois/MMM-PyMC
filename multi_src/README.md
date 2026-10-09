# multi_src: Multidimensional MMM Tools

Tools for working with multidimensional MMM models (models with a `geo` dimension).

## Overview

This package implements **Option C** for multidimensional MMM support:

1. **Vectorized computation**: Calculate contributions for ALL geos at once (fast!)
2. **Decorator pattern**: Make existing `b2b_src` functions compatible with multidimensional data
3. **Backward compatible**: Works with both standard and multidimensional MMM models

## Key Benefits

✅ **Fast**: Vectorized operations calculate all geos simultaneously
✅ **Clean**: Minimal code changes to existing workflows
✅ **Valid comparisons**: Same posterior samples across geos
✅ **Flexible**: Choose automatic (decorator) or manual per-geo processing

## Quick Start

### Installation

The package is located in `multi_src/` and can be imported directly:

```python
from multi_src import (
    get_individual_contributions_multidim,
    slice_to_geo,
    for_each_geo,
    for_each_geo_with_data_filter,
)
```

### Basic Usage (Recommended)

**Step 1: Decorate functions once at module level**

```python
from b2b_src.contributions import calculate_roas as _calculate_roas
from multi_src import for_each_geo_with_data_filter

# Decorate once
calculate_roas = for_each_geo_with_data_filter(_calculate_roas)
```

**Step 2: Calculate multidimensional contributions**

```python
from multi_src import get_individual_contributions_multidim

# Fast vectorized calculation for ALL geos
contributions_all = get_individual_contributions_multidim(
    mmm=fitted_mmm,
    data=data,
    channel_cols=['paid_search', 'display'],
    control_cols=['trend', 'seasonality'],
    random_seed=42  # Ensures cross-geo comparison validity
)
```

**Step 3: Use decorated functions normally**

```python
# Automatically processes each geo!
roas_by_geo = calculate_roas(
    contributions=contributions_all,
    data=data,
    channel_cols=['paid_search', 'display'],
    spend_mapping={'paid_search': 'paid_search_spend'}
)

# Results: {'geo1': roas_result1, 'geo2': roas_result2, ...}
```

## Core Functions

### 1. `get_individual_contributions_multidim()`

Calculate contributions for all geos at once (vectorized).

**Signature:**
```python
def get_individual_contributions_multidim(
    mmm,
    data,
    channel_cols: List[str],
    control_cols: List[str],
    random_seed: Optional[Union[int, np.random.Generator]] = None
) -> Dict[str, Any]
```

**Returns:**
```python
{
    'dates': array,              # Date array
    'geos': list,                # List of geo identifiers
    'channels': dict,            # {channel_name: xarray(sample, date, geo)}
    'controls': dict,            # {control_name: xarray(sample, date, geo)}
    'intercept': xarray,         # (sample, date, geo)
    'seasonality': xarray,       # (sample, date, geo)
}
```

**Key features:**
- ✅ Vectorized: Calculates all geos simultaneously (fast!)
- ✅ Consistent sampling: Same posterior samples across geos (valid comparisons)
- ✅ Original scale: Contributions in original units

### 2. `slice_to_geo()`

Extract contributions for a single geo.

**Signature:**
```python
def slice_to_geo(
    contributions_multidim: Dict[str, Any],
    geo: str
) -> Dict[str, Any]
```

**Example:**
```python
# Extract UK contributions
contrib_uk = slice_to_geo(contributions_all, 'uk')

# Now has standard shape: (sample, date)
# Compatible with all b2b_src functions
```

### 3. `@for_each_geo` Decorator

Automatically loop over geos for functions that take contributions.

**Signature:**
```python
def for_each_geo(func: Callable) -> Callable
```

**Usage:**
```python
from multi_src import for_each_geo

# Decorate once
@for_each_geo
def my_analysis(contributions, other_params):
    # Original function expecting (sample, date) shape
    ...

# Use with multidimensional contributions
results_by_geo = my_analysis(contributions_all, other_params)
# Returns: {'geo1': result1, 'geo2': result2, ...}
```

### 4. `@for_each_geo_with_data_filter` Decorator

Like `@for_each_geo`, but also filters the `data` parameter per-geo.

**Signature:**
```python
def for_each_geo_with_data_filter(func: Callable) -> Callable
```

**Usage:**
```python
from multi_src import for_each_geo_with_data_filter

# Decorate once
@for_each_geo_with_data_filter
def calculate_roas(contributions, data, ...):
    # Original function
    # 'data' is automatically filtered to current geo
    ...

# Use with multidimensional contributions
roas_by_geo = calculate_roas(contributions_all, data, ...)
# Each geo gets: contrib_geo + data filtered to that geo
```

## Usage Patterns

### Pattern 1: Pre-Decorated Functions (Recommended ⭐)

**Cleanest and most maintainable approach.**

```python
# In your analysis module, decorate once at top
from b2b_src.contributions import (
    calculate_roas as _calculate_roas,
    calculate_roas_time_series as _calculate_roas_time_series,
)
from multi_src import for_each_geo_with_data_filter

# Apply decorators once
calculate_roas = for_each_geo_with_data_filter(_calculate_roas)
calculate_roas_time_series = for_each_geo_with_data_filter(_calculate_roas_time_series)

# Then use normally throughout your code
contributions_all = get_individual_contributions_multidim(mmm, data, ...)

# Super clean! No wrapper functions needed
roas_by_geo = calculate_roas(contributions_all, data, ...)
roas_ts_by_geo = calculate_roas_time_series(contributions_all, data, ...)
```

### Pattern 2: Functional Style (Call-time decoration)

**More explicit but more verbose.**

```python
from multi_src import for_each_geo_with_data_filter
from b2b_src.contributions import calculate_roas

# Apply decorator at call-time
roas_by_geo = for_each_geo_with_data_filter(calculate_roas)(
    contributions=contributions_all,
    data=data,
    ...
)
```

### Pattern 3: Manual Per-Geo (Maximum control)

**Use when you need fine-grained control.**

```python
from multi_src import slice_to_geo

# Calculate contributions for all geos
contributions_all = get_individual_contributions_multidim(mmm, data, ...)

# Process specific geo manually
contrib_uk = slice_to_geo(contributions_all, 'uk')
data_uk = data[data['geo'] == 'uk']

# Use original b2b_src functions
roas_uk = calculate_roas(contrib_uk, data_uk, ...)
```

## Compatibility Matrix

| Function | Standard MMM | Multidim (decorated) | Notes |
|----------|-------------|---------------------|-------|
| `get_individual_contributions_multidim()` | ❌ | ✅ | Multidim only |
| `slice_to_geo()` | N/A | ✅ | Helper for multidim |
| `@for_each_geo` | ✅ (passthrough) | ✅ | Auto-detects |
| `@for_each_geo_with_data_filter` | ✅ (passthrough) | ✅ | Auto-detects |

## Shape Conventions

| Model Type | Contribution Shape | After `slice_to_geo()` |
|------------|-------------------|----------------------|
| Standard MMM | `(sample, date)` | N/A |
| Multidimensional MMM | `(sample, date, geo)` | `(sample, date)` |

## Examples

See [`example_multidim_usage.py`](../example_multidim_usage.py) for comprehensive examples:

- **Example 1**: Basic decorator pattern workflow (recommended)
- **Example 2**: Manual per-geo extraction
- **Example 3**: Plotting with decorated functions
- **Example 4**: `plot_all_contributions` with forced `use_log_scale=False`
- **Example 5**: Functional style (call-time decoration)

## Special Cases

### `plot_all_contributions` with Log Scale

For multidimensional models, `use_log_scale=True` is not supported because:
- Different geos may have different scales
- Log transformations are applied after mean calculation
- Can cause misleading visualizations across geos

**Solution**: Force `use_log_scale=False` internally.

```python
from multi_src import for_each_geo_with_data_filter
from b2b_src.visual_tools import plot_all_contributions as _plot_all_contributions

def plot_all_contributions_wrapper(contributions, mmm, data, use_log_scale=True, **kwargs):
    # Detect multidimensional
    first_channel = list(contributions['channels'].values())[0]
    is_multidim = hasattr(first_channel, 'dims') and 'geo' in first_channel.dims

    if is_multidim and use_log_scale:
        print("⚠ Warning: Forcing use_log_scale=False for multidimensional model")
        use_log_scale = False

    return _plot_all_contributions(contributions, mmm, data, use_log_scale, **kwargs)

# Decorate the wrapper
plot_all_contributions = for_each_geo_with_data_filter(plot_all_contributions_wrapper)
```

## Performance Considerations

### Memory Usage

Multidimensional contributions can be large:
- **Shape**: `(n_samples, n_dates, n_geos)`
- **Example**: `(4000 samples, 365 dates, 10 geos)` = ~146 MB per channel (float64)

**Tips**:
- Use `random_seed` to enable reproducible sampling (smaller sample sizes if needed)
- Process geos sequentially if memory is constrained
- Delete large objects explicitly: `del contributions_all`

### Computational Speed

**Vectorized (this implementation)**:
- Calculates all geos simultaneously
- ~10-20x faster than looping over geos
- Recommended for production use

**Per-geo loop** (old approach):
- Sequential processing
- Linear scaling: 10 geos = 10x time
- Not recommended

## Implementation Details

### Why Same Posterior Samples Matter

Using the same `random_seed` ensures:
1. Sample[i] is identical across all geos
2. Valid cross-geo comparisons (e.g., "which geo has higher ROAS?")
3. Consistent uncertainty quantification

**Without consistent sampling**:
```python
# BAD: Different samples per geo
for geo in geos:
    contrib_geo = get_contributions(mmm, data_geo)  # Random samples each time
    # Sample[0] for geo_a ≠ Sample[0] for geo_b
    # Cross-geo comparisons are INVALID
```

**With consistent sampling** (this implementation):
```python
# GOOD: Same samples for all geos
contrib_all = get_individual_contributions_multidim(mmm, data, random_seed=42)
# Sample[i] is identical across all geos
# Cross-geo comparisons are VALID ✓
```

### Decorator Detection Logic

Decorators auto-detect multidimensional contributions:

```python
def for_each_geo(func):
    def wrapper(contributions, *args, **kwargs):
        # Check first channel for 'geo' dimension
        first_channel = list(contributions['channels'].values())[0]
        is_multidim = hasattr(first_channel, 'dims') and 'geo' in first_channel.dims

        if not is_multidim:
            return func(contributions, *args, **kwargs)  # Standard MMM
        else:
            # Loop over geos
            results = {}
            for geo in contributions['geos']:
                contrib_geo = slice_to_geo(contributions, geo)
                results[geo] = func(contrib_geo, *args, **kwargs)
            return results
    return wrapper
```

## Troubleshooting

### Error: "Geo 'X' not found in contributions"

**Cause**: Trying to slice a geo that doesn't exist.

**Solution**: Check available geos:
```python
print(contributions_all['geos'])
```

### Error: "'geo' dimension not found"

**Cause**: Passing standard MMM contributions to multidim functions.

**Solution**: Use `get_individual_contributions_multidim()` for multidim models.

### Warning: "use_log_scale=True not supported"

**Cause**: Trying to use log scale with multidimensional `plot_all_contributions`.

**Solution**: Set `use_log_scale=False` or use the wrapper that forces it.

## API Reference

See docstrings in:
- [`contributions.py`](contributions.py) - Contribution calculation
- [`helpers.py`](helpers.py) - Helper functions and decorators

## Related Modules

- `b2b_src/`: Standard MMM functions (single geo)
- `multi_src/`: Multidimensional MMM extensions (this package)

## License

Part of pymc_marketing project.
