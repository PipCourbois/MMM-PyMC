"""
Pre-decorated b2b_src functions for multidimensional MMM.

This module provides pre-decorated versions of all b2b_src functions that
automatically work with both standard and multidimensional MMM models.

Import from this module to get functions that handle geo looping automatically.
No need to create wrapper functions - just use them normally!

Usage
-----
>>> from multi_src.wrapped_functions import calculate_roas, plot_roas_analysis
>>>
>>> # Calculate contributions for all geos (fast, vectorized)
>>> contributions_all = get_individual_contributions_multidim(mmm, data, ...)
>>>
>>> # Use decorated functions normally - decorator handles everything!
>>> roas_by_geo = calculate_roas(contributions_all, data, ...)
>>> # Returns: {'geo1': roas_result1, 'geo2': roas_result2, ...}

Pattern
-------
This implements Pattern 1 from the README: Pre-Decorated Functions
- Decorators applied ONCE at module level
- Import and use decorated versions throughout your code
- Clean, maintainable, no wrapper functions needed

Examples
--------
>>> # Standard MMM (single geo) - works normally
>>> from multi_src.wrapped_functions import calculate_roas
>>> contributions_standard = get_individual_contributions(mmm, data, ...)
>>> roas = calculate_roas(contributions_standard, data, ...)
>>> # Returns single ROAS result
>>>
>>> # Multidimensional MMM - automatic geo looping!
>>> contributions_multi = get_individual_contributions_multidim(mmm, data, ...)
>>> roas_by_geo = calculate_roas(contributions_multi, data, ...)
>>> # Returns: {'geo1': roas_result1, 'geo2': roas_result2, ...}
"""

import sys
sys.path.insert(0, ".")

# Import decorators from helpers
from .helpers import for_each_geo, for_each_geo_with_data_filter

# Import original functions from b2b_src.contributions
from b2b_src.contributions import (
    calculate_roas as _calculate_roas,
    calculate_roas_time_series as _calculate_roas_time_series,
    calculate_marginal_roas as _calculate_marginal_roas,
)

# Import original functions from b2b_src.visual_tools
from b2b_src.visual_tools import (
    plot_roas_analysis as _plot_roas_analysis,
    plot_waterfall_contributions as _plot_waterfall_contributions,
    plot_roas_time_series as _plot_roas_time_series,
    plot_marginal_roas_analysis as _plot_marginal_roas_analysis,
    plot_all_contributions as _plot_all_contributions,
)


# ============================================================================
# DECORATED CALCULATION FUNCTIONS
# ============================================================================
# These functions need both contributions and data, so we use
# for_each_geo_with_data_filter which automatically filters data per-geo

calculate_roas = for_each_geo_with_data_filter(_calculate_roas)
"""
Calculate ROAS (Return on Ad Spend) for each geo.

Decorated version of b2b_src.contributions.calculate_roas that automatically:
- Detects if contributions are multidimensional
- Loops over each geo if multidimensional
- Filters data to match each geo
- Returns results organized by geo

Parameters
----------
contributions : dict
    Contributions from get_individual_contributions() or
    get_individual_contributions_multidim()
data : pd.DataFrame
    Original data (will be filtered per-geo if multidimensional)
channel_cols : list, optional
    Channel names to calculate ROAS for
date_start, date_end : str or datetime, optional
    Date range for calculation
target_column : str, default='y'
    Target variable column name
spend_mapping : dict, optional
    Mapping from channel names to spend columns

Returns
-------
dict or dict of dict
    - Standard MMM: Single ROAS result dict
    - Multidimensional MMM: {'geo1': roas_result1, 'geo2': roas_result2, ...}

Examples
--------
>>> contributions = get_individual_contributions_multidim(mmm, data, ...)
>>> roas_by_geo = calculate_roas(
...     contributions=contributions,
...     data=data,
...     channel_cols=['paid_search', 'display'],
...     spend_mapping={'paid_search': 'paid_search_spend'}
... )
>>> print(roas_by_geo['geo1']['roas'])
"""

calculate_roas_time_series = for_each_geo_with_data_filter(_calculate_roas_time_series)
"""
Calculate per-period ROAS time series for each geo.

Decorated version of b2b_src.contributions.calculate_roas_time_series.

Parameters
----------
contributions : dict
    Contributions dict
data : pd.DataFrame
    Original data
channel_cols : list, optional
    Channels to calculate ROAS for
t_start, t_end : int, optional
    Period range
target_column : str, default='y'
    Target variable
spend_mapping : dict, optional
    Channel to spend column mapping

Returns
-------
dict or dict of dict
    - Standard MMM: Single result
    - Multidimensional MMM: Results per geo
"""

calculate_marginal_roas = for_each_geo_with_data_filter(_calculate_marginal_roas)
"""
Calculate accumulated marginal ROAS over a period for each geo.

Decorated version of b2b_src.contributions.calculate_marginal_roas.

Parameters
----------
mmm : MMM
    Fitted MMM model
data : pd.DataFrame
    Original data
config : dict
    Configuration with channel information
date_start, date_end : str or datetime, optional
    Period dates
target_column : str, default='y'
    Target variable
marginal_increase : float, default=0.01
    Percentage increase (1% default)
prediction_days : int, optional
    Days for prediction range
random_seed : int or np.random.Generator, optional
    Random seed for reproducibility

Returns
-------
dict or dict of dict
    - Standard MMM: Single marginal ROAS result
    - Multidimensional MMM: Results per geo
"""


# ============================================================================
# DECORATED PLOTTING FUNCTIONS
# ============================================================================
# Plotting functions take various parameters, so we choose the appropriate
# decorator based on whether they need data filtering

plot_roas_analysis = for_each_geo_with_data_filter(_plot_roas_analysis)
"""
Visualize ROAS analysis with comprehensive plots for each geo.

Decorated version of b2b_src.visual_tools.plot_roas_analysis.
Creates 7 visualizations per geo.

Parameters
----------
roas_results : dict
    Results from calculate_roas()
contributions : dict
    Contributions dict
data : pd.DataFrame
    Original data
confidence_intervals : tuple, default=(0.85, 0.90)
    Confidence levels for statistics
baseline_components : list, optional
    Components to include in baseline

Returns
-------
dict or dict of dict
    - Standard MMM: Tuple of figures
    - Multidimensional MMM: {geo: tuple_of_figures, ...}
"""

plot_waterfall_contributions = for_each_geo(_plot_waterfall_contributions)
"""
Create horizontal waterfall plot showing aggregated contributions.

Decorated version of b2b_src.visual_tools.plot_waterfall_contributions.

Parameters
----------
roas_results : dict
    Results from calculate_roas()
data : pd.DataFrame, optional
    Original data for validation
channel_cols : list, optional
    Channels to include
control_cols : list, optional
    Controls to include
date_start, date_end : str or datetime, optional
    Date range (deprecated - filter in calculate_roas)
target_column : str, default='y'
    Target variable
figsize : tuple, optional
    Figure size
colors : dict, optional
    Component type colors
show_values : bool, default=True
    Show values on bars
title : str, optional
    Plot title
baseline_components : list, optional
    Components for baseline

Returns
-------
tuple or dict of tuple
    - Standard MMM: (fig, ax, aggregated_contributions)
    - Multidimensional MMM: {geo: (fig, ax, agg_contrib), ...}
"""

plot_roas_time_series = for_each_geo(_plot_roas_time_series)
"""
Visualize ROAS time series with confidence intervals.

Decorated version of b2b_src.visual_tools.plot_roas_time_series.

Parameters
----------
roas_results : dict
    Results from calculate_roas_time_series()
confidence_levels : tuple, default=(0.95, 0.99)
    Confidence levels to display
show_mean_line : bool, default=True
    Show mean ROAS line
date_range : tuple, optional
    (start_date, end_date) filter
figsize : tuple, default=(14, 6)
    Figure size per channel

Returns
-------
dict or dict of dict
    - Standard MMM: {'figures': {channel: fig, ...}}
    - Multidimensional MMM: {geo: {'figures': {...}}, ...}
"""

plot_marginal_roas_analysis = for_each_geo(_plot_marginal_roas_analysis)
"""
Visualize marginal ROAS analysis results.

Decorated version of b2b_src.visual_tools.plot_marginal_roas_analysis.
Creates 3 figures per geo.

Parameters
----------
mroas_results : dict
    Results from calculate_marginal_roas()
data : pd.DataFrame, optional
    Original data
confidence_intervals : tuple, default=(0.85, 0.90)
    Confidence levels

Returns
-------
tuple or dict of tuple
    - Standard MMM: (fig1, fig2, fig3)
    - Multidimensional MMM: {geo: (fig1, fig2, fig3), ...}
"""


# ============================================================================
# SPECIAL: plot_all_contributions with use_log_scale=False enforcement
# ============================================================================

def _plot_all_contributions_wrapper(contributions, mmm, data, use_log_scale=True, **kwargs):
    """
    Wrapper for plot_all_contributions that forces use_log_scale=False for multidimensional models.

    For multidimensional MMM models, using log scale can cause issues because:
    - Different geos may have different scales
    - Log transformations are applied after mean calculation
    - Can create misleading visualizations across geos

    This wrapper automatically detects multidimensional contributions and forces
    use_log_scale=False with a warning message.

    Parameters
    ----------
    contributions : dict
        Contributions from get_individual_contributions() or
        get_individual_contributions_multidim()
    mmm : MMM
        Fitted MMM model
    data : pd.DataFrame
        Original data
    use_log_scale : bool, default=True
        If True, plot in log space (forced to False for multidim)
    **kwargs
        Additional arguments passed to plot_all_contributions

    Returns
    -------
    tuple or dict of tuple
        - Standard MMM: (fig_combined, fig_individual)
        - Multidimensional MMM: {geo: (fig_combined, fig_individual), ...}
    """
    # Detect if contributions are multidimensional
    if contributions['channels']:
        first_channel = list(contributions['channels'].values())[0]
        is_multidim = hasattr(first_channel, 'dims') and 'geo' in first_channel.dims
    else:
        is_multidim = False

    # Force use_log_scale=False for multidimensional models
    if is_multidim and use_log_scale:
        print("⚠ Warning: use_log_scale=True is not supported for multidimensional models")
        print("⚠ Forcing use_log_scale=False to avoid scale mismatches between geos")
        use_log_scale = False

    return _plot_all_contributions(contributions, mmm, data, use_log_scale=use_log_scale, **kwargs)


# Apply decorator to the wrapper
plot_all_contributions = for_each_geo_with_data_filter(_plot_all_contributions_wrapper)
"""
Create contribution plots showing all components.

Decorated version of b2b_src.visual_tools.plot_all_contributions with special
handling for multidimensional models.

IMPORTANT: For multidimensional models, use_log_scale is automatically forced
to False to avoid scale mismatches between geos.

Parameters
----------
contributions : dict
    Contributions dict
mmm : MMM
    Fitted MMM model
data : pd.DataFrame
    Original data
use_log_scale : bool, default=True
    Plot in log space (forced to False for multidim models)

Returns
-------
tuple or dict of tuple
    - Standard MMM: (fig_combined, fig_individual)
    - Multidimensional MMM: {geo: (fig_combined, fig_individual), ...}

Notes
-----
For multidimensional models:
- use_log_scale is automatically set to False
- A warning is printed if you try to use True
- This prevents misleading visualizations across different geo scales
"""


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Calculation functions
    'calculate_roas',
    'calculate_roas_time_series',
    'calculate_marginal_roas',

    # Plotting functions
    'plot_roas_analysis',
    'plot_waterfall_contributions',
    'plot_roas_time_series',
    'plot_marginal_roas_analysis',
    'plot_all_contributions',
]
