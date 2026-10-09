"""
multi_src: Tools for multidimensional MMM models.

This package provides utilities for working with MMM models that have
a geo dimension, including:
- Vectorized contribution calculation for all geos
- Helper functions to transform multidimensional outputs
- Decorators to make b2b_src functions compatible with multidimensional data
- Pre-decorated b2b_src functions ready to use (recommended!)

Quick Start
-----------
Pattern 1: Pre-Decorated Functions (Recommended)
    >>> from multi_src import get_individual_contributions_multidim
    >>> from multi_src.wrapped_functions import calculate_roas, plot_roas_analysis
    >>>
    >>> # Calculate contributions for all geos
    >>> contributions = get_individual_contributions_multidim(mmm, data, ...)
    >>>
    >>> # Use decorated functions - automatic geo handling!
    >>> roas_by_geo = calculate_roas(contributions, data, ...)

Pattern 2: Manual Decoration
    >>> from multi_src import for_each_geo_with_data_filter
    >>> from b2b_src.contributions import calculate_roas
    >>>
    >>> # Decorate at call-time
    >>> roas_by_geo = for_each_geo_with_data_filter(calculate_roas)(contributions, data, ...)

Pattern 3: Manual Per-Geo
    >>> from multi_src import slice_to_geo
    >>>
    >>> contrib_uk = slice_to_geo(contributions, 'uk')
    >>> roas_uk = calculate_roas(contrib_uk, data_uk, ...)
"""

# Core functions
from .contributions import get_individual_contributions_multidim
from .helpers import (
    slice_to_geo,
    for_each_geo,
    for_each_geo_with_data_filter,
)

# Pre-decorated functions (Pattern 1 - Recommended)
from .wrapped_functions import (
    # Calculation functions
    calculate_roas,
    calculate_roas_time_series,
    calculate_marginal_roas,
    # Plotting functions
    plot_roas_analysis,
    plot_waterfall_contributions,
    plot_roas_time_series,
    plot_marginal_roas_analysis,
    plot_all_contributions,
)

__all__ = [
    # Core functions
    'get_individual_contributions_multidim',
    'slice_to_geo',
    'for_each_geo',
    'for_each_geo_with_data_filter',

    # Pre-decorated calculation functions
    'calculate_roas',
    'calculate_roas_time_series',
    'calculate_marginal_roas',

    # Pre-decorated plotting functions
    'plot_roas_analysis',
    'plot_waterfall_contributions',
    'plot_roas_time_series',
    'plot_marginal_roas_analysis',
    'plot_all_contributions',
]
