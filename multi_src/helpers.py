"""
Helper utilities for multidimensional MMM models.

This module provides utilities to transform multidimensional contributions
(sample, date, geo) into per-geo contributions (sample, date) that are compatible
with existing b2b_src functions.
"""

import functools
import inspect
from typing import Dict, Any, Callable
import numpy as np


def slice_to_geo(contributions_multidim: Dict[str, Any], geo: str) -> Dict[str, Any]:
    """
    Transform multidimensional contributions to single-geo contributions.

    Converts contributions from (sample, date, dimension) shape to (sample, date) shape
    for a specific dimension value. This allows multidimensional model outputs to work with
    existing b2b_src functions that expect standard MMM output format.

    Automatically detects the dimension name from the data (e.g., 'geo', 'Groups', 'brand').

    Parameters
    ----------
    contributions_multidim : dict
        Dictionary containing multidimensional contributions with structure:
        - 'dates': array of dates
        - 'geos': list of dimension identifiers
        - 'channels': dict with channel contributions (xarray with spatial dimension)
        - 'controls': dict with control contributions (xarray with spatial dimension)
        - 'intercept': intercept contribution (xarray with spatial dimension)
        - 'seasonality': seasonality contribution (xarray with spatial dimension) or None
    geo : str
        Dimension value to extract (must be in contributions_multidim['geos'])

    Returns
    -------
    dict
        Dictionary with same structure as standard MMM contributions:
        - 'dates': array of dates
        - 'channels': dict with channel contributions for this dimension value (sample, date)
        - 'controls': dict with control contributions for this dimension value (sample, date)
        - 'intercept': intercept contribution for this dimension value (sample, date)
        - 'seasonality': seasonality contribution for this dimension value (sample, date) or None

    Examples
    --------
    >>> # Get multidimensional contributions for all dimension values
    >>> contrib_all = get_individual_contributions_multidim(mmm, data, ...)
    >>>
    >>> # Extract contributions for a specific dimension value
    >>> contrib_uk = slice_to_geo(contrib_all, 'Brand_A')
    >>>
    >>> # Now use with standard b2b_src functions
    >>> roas = calculate_roas(contrib_uk, data, ...)

    Notes
    -----
    - Automatically detects dimension name ('geo', 'Groups', 'brand', etc.)
    - Uses xarray .sel() which creates views, not copies (zero-copy operation)
    - Output has exact same structure as get_individual_contributions() for standard MMM
    - Preserves all posterior samples for uncertainty quantification
    """
    # Validate geo exists in the contributions
    if 'geos' not in contributions_multidim:
        raise ValueError(
            "contributions_multidim must have 'geos' key. "
            "Did you pass multidimensional contributions from get_individual_contributions_multidim()?"
        )

    if geo not in contributions_multidim['geos']:
        raise ValueError(
            f"Geo '{geo}' not found in contributions. "
            f"Available geos: {contributions_multidim['geos']}"
        )

    # Detect the actual dimension name from the data
    # Could be 'geo', 'Groups', 'brand', etc.
    first_channel = list(contributions_multidim['channels'].values())[0]
    if hasattr(first_channel, 'dims'):
        # Get spatial dimensions (exclude time dimensions)
        spatial_dims = [d for d in first_channel.dims if d not in ['date', 'week', 'time', 'period', 'sample']]
        if len(spatial_dims) != 1:
            raise ValueError(
                f"Expected exactly one spatial dimension, found: {spatial_dims}. "
                f"All dimensions: {first_channel.dims}"
            )
        dim_name = spatial_dims[0]
    else:
        # Fallback to 'geo' if no dims attribute
        dim_name = 'geo'

    # Extract contributions for this geo using the detected dimension name
    return {
        'dates': contributions_multidim['dates'],
        'channels': {
            ch: contrib.sel({dim_name: geo})
            for ch, contrib in contributions_multidim['channels'].items()
        },
        'controls': {
            ctrl: contrib.sel({dim_name: geo})
            for ctrl, contrib in contributions_multidim['controls'].items()
        },
        'intercept': contributions_multidim['intercept'].sel({dim_name: geo}),
        'seasonality': (
            contributions_multidim['seasonality'].sel({dim_name: geo})
            if contributions_multidim['seasonality'] is not None
            else None
        ),
    }


def for_each_geo(func: Callable) -> Callable:
    """
    Decorator that automatically loops over geos for b2b_src functions.

    This decorator makes standard b2b_src functions (designed for single-geo MMM)
    compatible with multidimensional MMM outputs. It detects whether the input
    contributions are multidimensional (have 'geo' dimension), and if so,
    automatically loops over each geo, calling the original function with
    per-geo contributions.

    The decorator:
    1. Detects if contributions are multidimensional by checking for 'geo' dimension
    2. If not multidimensional, calls the function normally (backward compatible)
    3. If multidimensional, loops over each geo:
       - Extracts per-geo contributions using slice_to_geo()
       - Calls the original function with per-geo data
       - Collects results in a dictionary keyed by geo

    Parameters
    ----------
    func : callable
        Function that expects standard MMM contributions with (sample, date) shape.
        Typically a b2b_src function like calculate_roas() or plot_roas_analysis().

    Returns
    -------
    callable
        Wrapped function that works with both standard and multidimensional contributions.

    Examples
    --------
    >>> # Apply decorator to existing b2b_src function
    >>> @for_each_geo
    ... def calculate_roas(contributions, data, config, ...):
    ...     # Original implementation expecting (sample, date) shape
    ...     ...
    >>>
    >>> # Now works with both standard and multidimensional contributions
    >>>
    >>> # Standard MMM (single geo):
    >>> contrib_standard = get_individual_contributions(mmm_standard, ...)
    >>> roas = calculate_roas(contrib_standard, data, config)
    >>> # Returns single ROAS result
    >>>
    >>> # Multidimensional MMM (multiple geos):
    >>> contrib_multi = get_individual_contributions_multidim(mmm_multi, ...)
    >>> roas_by_geo = calculate_roas(contrib_multi, data, config)
    >>> # Returns {'geo1': roas_result1, 'geo2': roas_result2, ...}

    Notes
    -----
    - Zero changes required to the original function implementation
    - Backward compatible: works with both standard and multidimensional inputs
    - Preserves function signature and docstring
    - Results are returned as dict {geo: result} for multidimensional inputs
    - The decorator ensures all geos use the SAME posterior samples, enabling
      valid cross-geo comparisons
    """
    @functools.wraps(func)
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

        # Multidimensional MMM: loop over geos
        print(f"\n{'='*70}")
        print(f"Detected multidimensional contributions with {len(contributions['geos'])} geos")
        print(f"Processing each geo separately with {func.__name__}()")
        print(f"{'='*70}")

        # Check if function accepts dimension_label parameter
        sig = inspect.signature(func)
        accepts_dimension_label = 'dimension_label' in sig.parameters

        # Check if user provided a manual override
        dimension_label_override = kwargs.pop('dimension_label_override', None)

        results = {}
        for geo_idx, geo in enumerate(contributions['geos'], 1):
            print(f"\n[{geo_idx}/{len(contributions['geos'])}] Processing geo: {geo}")
            print(f"{'-'*70}")

            # Extract contributions for this geo
            contrib_geo = slice_to_geo(contributions, geo)

            # Inject dimension_label if function accepts it
            if accepts_dimension_label:
                # Use override if provided, otherwise use actual geo name
                if dimension_label_override:
                    kwargs['dimension_label'] = dimension_label_override
                else:
                    kwargs['dimension_label'] = geo

            # Call the original function with per-geo contributions
            result = func(contrib_geo, *args, **kwargs)

            # Store result
            results[geo] = result

        print(f"\n{'='*70}")
        print(f"Completed processing all {len(contributions['geos'])} geos")
        print(f"{'='*70}\n")

        return results

    return wrapper


def for_each_geo_with_data_filter(func: Callable) -> Callable:
    """
    Decorator that loops over geos AND filters data for each geo.

    Similar to for_each_geo(), but also filters the 'data' parameter to only
    include rows for the current geo. This is useful for functions that need
    both contributions and corresponding data for each geo.

    The decorator assumes:
    - First parameter is 'contributions'
    - Function has a 'data' parameter (positional or keyword)

    Parameters
    ----------
    func : callable
        Function that expects (contributions, ..., data, ...) where data is filtered per-geo

    Returns
    -------
    callable
        Wrapped function that filters both contributions and data per-geo

    Examples
    --------
    >>> @for_each_geo_with_data_filter
    ... def calculate_roas(contributions, data, channel_cols, ...):
    ...     # data is automatically filtered to current geo
    ...     ...
    >>>
    >>> # Multidimensional usage:
    >>> contrib_multi = get_individual_contributions_multidim(mmm_multi, data, ...)
    >>> roas_by_geo = calculate_roas(contrib_multi, data, channel_cols)
    >>> # Each geo gets: contrib_geo + data filtered to that geo

    Notes
    -----
    - Requires 'data' parameter to have a dimension column (tries: 'Groups', 'geo', 'brand', 'region', 'market')
    - Automatically filters data to match the current geo being processed
    - Useful for ROAS calculations, plotting, etc. that need aligned data
    """
    @functools.wraps(func)
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
            return func(contributions, *args, data=data, **kwargs)

        # Multidimensional MMM: loop over geos with data filtering
        print(f"\n{'='*70}")
        print(f"Detected multidimensional contributions with {len(contributions['geos'])} geos")
        print(f"Processing each geo separately with filtered data")
        print(f"{'='*70}")

        # Get dimension column from global config
        from b2b_src.column_config import get_default_column_config
        cols = get_default_column_config()
        dimension_col = cols.dimension

        # Validate it exists in data
        if dimension_col not in data.columns:
            raise ValueError(
                f"Dimension column '{dimension_col}' not found in data. "
                f"Available columns: {list(data.columns)}"
            )

        print(f"Using dimension column: '{dimension_col}'")

        # Check if function accepts dimension_label parameter
        sig = inspect.signature(func)
        accepts_dimension_label = 'dimension_label' in sig.parameters

        # Check if user provided a manual override
        dimension_label_override = kwargs.pop('dimension_label_override', None)

        results = {}
        for geo_idx, geo in enumerate(contributions['geos'], 1):
            print(f"\n[{geo_idx}/{len(contributions['geos'])}] Processing geo: {geo}")
            print(f"{'-'*70}")

            # Extract contributions for this geo
            contrib_geo = slice_to_geo(contributions, geo)

            # Filter data for this geo using inferred dimension column
            data_geo = data[data[dimension_col] == geo].copy()

            print(f"  Filtered data: {len(data_geo)} rows for geo '{geo}'")

            # Inject dimension_label if function accepts it
            if accepts_dimension_label:
                # Use override if provided, otherwise use actual geo name
                if dimension_label_override:
                    kwargs['dimension_label'] = dimension_label_override
                else:
                    kwargs['dimension_label'] = geo

            # Call the original function with per-geo contributions and data
            result = func(contrib_geo, *args, data=data_geo, **kwargs)

            # Store result
            results[geo] = result

        print(f"\n{'='*70}")
        print(f"Completed processing all {len(contributions['geos'])} geos")
        print(f"{'='*70}\n")

        return results

    return wrapper
