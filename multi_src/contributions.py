"""
Contribution analysis for multidimensional MMM models.

This module provides functions to calculate individual contributions for
multidimensional MMM models (models with a 'geo' dimension).
"""

import sys
sys.path.insert(0, ".")

import numpy as np
from typing import Dict, Any, List, Optional, Union


def get_individual_contributions_multidim(
    mmm,
    data,
    channel_cols: List[str],
    control_cols: List[str],
    random_seed: Optional[Union[int, np.random.Generator]] = None
) -> Dict[str, Any]:
    """
    Calculate individual contributions for multidimensional MMM models (vectorized).

    This is the multidimensional equivalent of get_individual_contributions() from b2b_src.
    It efficiently calculates contributions for ALL dimensions at once using vectorized operations,
    preserving the dimension (geo/Groups/brand/etc.) in the output.

    The function automatically detects the dimension name from the model coordinates
    (e.g., 'geo', 'Groups', 'brand', 'region', 'market', 'dimension').

    The function:
    1. Uses vectorized shutdown approach for efficiency
    2. Calculates contributions for ALL dimension values simultaneously
    3. Preserves dimension: outputs have shape (sample, date, dimension)
    4. Uses consistent random seeds across all dimension values for valid comparisons

    Parameters
    ----------
    mmm : MMM object
        Fitted multidimensional MMM model instance with a dimension coordinate
        (e.g., 'geo', 'Groups', 'brand', 'region', 'market', 'dimension')
    data : pd.DataFrame
        Original data containing date and dimension columns
    channel_cols : list
        List of channel column names (e.g., ['x1', 'x2'])
    control_cols : list
        List of control column names (e.g., ['t', 'event_1', 'event_2'])
    random_seed : int or np.random.Generator, optional
        Random seed for reproducibility. If None, results may vary between runs.
        IMPORTANT: Using the same seed ensures all geos use the SAME posterior samples,
        which is required for valid cross-geo comparisons.

    Returns
    -------
    dict
        Dictionary containing:
        - 'dates': array of dates
        - 'geos': list of geo identifiers
        - 'channels': dict with channel contributions {channel_name: xarray(sample, date, geo)}
        - 'controls': dict with control contributions {control_name: xarray(sample, date, geo)}
        - 'intercept': xarray with intercept contribution (sample, date, geo)
        - 'seasonality': xarray with seasonality contribution (sample, date, geo) or None

    Examples
    --------
    >>> # Calculate contributions for all geos at once (fast, vectorized)
    >>> contributions_all = get_individual_contributions_multidim(
    ...     mmm=fitted_mmm,
    ...     data=data,
    ...     channel_cols=['paid_search', 'display'],
    ...     control_cols=['trend', 'seasonality'],
    ...     random_seed=42  # Ensures reproducibility and cross-geo comparison validity
    ... )
    >>>
    >>> # Check shape: should have geo dimension
    >>> print(contributions_all['channels']['paid_search'].shape)
    >>> # Output: (n_samples, n_dates, n_geos)
    >>>
    >>> # Option 1: Use with decorator pattern (automatic per-geo processing)
    >>> from multi_src.helpers import for_each_geo
    >>>
    >>> @for_each_geo
    ... def calculate_roas(contributions, data, ...):
    ...     # Standard b2b_src function
    ...     ...
    >>>
    >>> roas_by_geo = calculate_roas(contributions_all, data, ...)
    >>> # Returns: {'geo1': roas_result1, 'geo2': roas_result2, ...}
    >>>
    >>> # Option 2: Manual per-geo extraction
    >>> from multi_src.helpers import slice_to_geo
    >>>
    >>> contrib_uk = slice_to_geo(contributions_all, 'uk')
    >>> roas_uk = calculate_roas(contrib_uk, data_uk, ...)

    Notes
    -----
    - **Computational Efficiency**: This function uses vectorized operations to calculate
      contributions for all geos simultaneously. This is MUCH faster than calling
      get_individual_contributions() in a loop for each geo.

    - **Cross-geo Comparisons**: Using the same random_seed ensures all geos use identical
      posterior samples (sample[i] is the same across all geos). This is critical for
      valid cross-geo comparisons of contributions and ROAS.

    - **Memory Management**: For models with many geos, dates, or posterior samples,
      the output can be large. Consider using context managers or explicit deletion
      if memory is constrained:
      >>> with get_individual_contributions_multidim(mmm, data, ...) as contrib:
      ...     process(contrib)
      >>> # Auto-freed after context

    - **Shape Conventions**:
      - Standard MMM: (sample, date)
      - Multidimensional MMM: (sample, date, geo)

    See Also
    --------
    b2b_src.contributions.get_individual_contributions : Single-geo version
    multi_src.helpers.slice_to_geo : Extract per-geo contributions
    multi_src.helpers.for_each_geo : Decorator for automatic per-geo processing
    """
    import numpy as np

    # Validate that model has dimensional coords
    if not hasattr(mmm, 'model') or not hasattr(mmm.model, 'coords'):
        raise ValueError("MMM model must have coords attribute")

    # Auto-detect the dimension name (could be 'geo', 'Groups', 'brand', etc.)
    # Look for common dimension names used in multidimensional models
    dimension_name = None
    for possible_dim in ['geo', 'Groups', 'brand', 'region', 'market', 'dimension']:
        if possible_dim in mmm.model.coords:
            dimension_name = possible_dim
            break

    if dimension_name is None:
        raise ValueError(
            "This function is for multidimensional models with a dimension coordinate. "
            "Expected one of: 'geo', 'Groups', 'brand', 'region', 'market', 'dimension'. "
            "For standard MMM, use b2b_src.contributions.get_individual_contributions() instead."
        )

    # Get dimension identifiers from model
    # Note: mmm.model.coords[dimension_name] returns a tuple in PyMC, not an xarray DataArray
    geos = list(mmm.model.coords[dimension_name])
    n_geos = len(geos)

    print(f"\n{'='*70}")
    print(f"Calculating contributions for multidimensional MMM")
    print(f"{'='*70}")
    print(f"Dimension: '{dimension_name}'")
    print(f"Dimension values: {geos} (n={n_geos})")
    print(f"Channels: {channel_cols}")
    print(f"Controls: {control_cols}")
    print(f"Random seed: {random_seed}")
    print(f"{'='*70}\n")

    # Prepare random seed sequence for reproducibility
    # We need different seeds for each sampling call (baseline + all components)
    if random_seed is not None:
        if isinstance(random_seed, np.random.Generator):
            ss = np.random.SeedSequence(random_seed.bit_generator.seed_seq.entropy)
        else:
            ss = np.random.SeedSequence(random_seed)

        # Total number of samples: 1 baseline + len(channels) + len(controls) + 2 (intercept, seasonality)
        n_samples = 1 + len(channel_cols) + len(control_cols) + 2
        child_seeds = ss.spawn(n_samples)

        # Create generators for each call
        seed_idx = 0
        baseline_seed = np.random.default_rng(child_seeds[seed_idx])
        seed_idx += 1
        channel_seeds = {ch: np.random.default_rng(child_seeds[seed_idx + i]) for i, ch in enumerate(channel_cols)}
        seed_idx += len(channel_cols)
        control_seeds = {ctrl: np.random.default_rng(child_seeds[seed_idx + i]) for i, ctrl in enumerate(control_cols)}
        seed_idx += len(control_cols)
        intercept_seed = np.random.default_rng(child_seeds[seed_idx])
        seed_idx += 1
        seasonality_seed = np.random.default_rng(child_seeds[seed_idx])
    else:
        baseline_seed = None
        channel_seeds = {ch: None for ch in channel_cols}
        control_seeds = {ctrl: None for ctrl in control_cols}
        intercept_seed = None
        seasonality_seed = None

    # Get baseline predictions (no components shut down) - ALL GEOS AT ONCE
    print(f"[1/{n_samples}] Calculating baseline predictions (all geos)...")
    preds_normal = mmm.sample_posterior_predictive(
        data,
        extend_idata=False,
        combined=True,
        include_last_observations=False,
        original_scale=True,
        random_seed=baseline_seed
    )

    contributions = {
        'dates': None,
        'geos': geos,
        'channels': {},
        'controls': {},
        'intercept': None,
        'seasonality': None
    }

    # Calculate channel contributions - ALL GEOS AT ONCE
    for idx, channel in enumerate(channel_cols, 2):
        print(f"[{idx}/{n_samples}] Calculating contribution for channel: {channel} (all geos)...")
        preds_no_channel = mmm.sample_posterior_predictive(
            data,
            extend_idata=False,
            combined=True,
            include_last_observations=False,
            original_scale=True,
            shutdown_components=[channel],
            random_seed=channel_seeds[channel]
        )
        # Contribution = baseline - shutdown
        # Shape: (sample, date, geo) for multidimensional model
        contributions['channels'][channel] = preds_normal[mmm.output_var] - preds_no_channel[mmm.output_var]

    # Calculate control contributions - ALL GEOS AT ONCE
    control_start_idx = 2 + len(channel_cols)
    for idx, control in enumerate(control_cols, control_start_idx):
        print(f"[{idx}/{n_samples}] Calculating contribution for control: {control} (all geos)...")
        preds_no_control = mmm.sample_posterior_predictive(
            data,
            extend_idata=False,
            combined=True,
            include_last_observations=False,
            original_scale=True,
            shutdown_components=[control],
            random_seed=control_seeds[control]
        )
        contributions['controls'][control] = preds_normal[mmm.output_var] - preds_no_control[mmm.output_var]

    # Calculate intercept contribution - ALL GEOS AT ONCE
    intercept_idx = control_start_idx + len(control_cols)
    print(f"[{intercept_idx}/{n_samples}] Calculating intercept contribution (all geos)...")
    preds_no_intercept = mmm.sample_posterior_predictive(
        data,
        extend_idata=False,
        combined=True,
        include_last_observations=False,
        original_scale=True,
        shutdown_components=['intercept'],
        random_seed=intercept_seed
    )
    contributions['intercept'] = preds_normal[mmm.output_var] - preds_no_intercept[mmm.output_var]

    # Calculate seasonality contribution - ALL GEOS AT ONCE
    seasonality_idx = intercept_idx + 1
    print(f"[{seasonality_idx}/{n_samples}] Calculating seasonality contribution (all geos)...")
    preds_no_seasonality = mmm.sample_posterior_predictive(
        data,
        extend_idata=False,
        combined=True,
        include_last_observations=False,
        original_scale=True,
        shutdown_components=['seasonality'],
        random_seed=seasonality_seed
    )
    contributions['seasonality'] = preds_normal[mmm.output_var] - preds_no_seasonality[mmm.output_var]

    # Extract dates from one of the contribution xarrays
    # Use intercept as it's guaranteed to exist
    dates_obj = contributions['intercept']

    # Try to find date coordinate (could be 'date', 'week', 'time', etc.)
    date_coord = None
    if hasattr(dates_obj, 'coords'):
        for coord_name in ['date', 'week', 'time', 'period']:
            if coord_name in dates_obj.coords:
                date_coord = coord_name
                break

    if date_coord:
        contributions['dates'] = dates_obj.coords[date_coord].values
    else:
        # Fallback: create index array
        contributions['dates'] = np.arange(dates_obj.shape[-2])  # -2 because last dim is geo

    print(f"\n{'='*70}")
    print(f"✓ Contributions calculated successfully for all {n_geos} geos")
    print(f"{'='*70}")

    # Print shape information for verification
    first_channel = list(contributions['channels'].values())[0]
    print(f"\nShape verification:")
    print(f"  First channel shape: {first_channel.shape}")
    print(f"  Expected: (n_samples, n_dates, n_geos)")
    if hasattr(first_channel, 'dims'):
        print(f"  Dimensions: {first_channel.dims}")
    print(f"\n{'='*70}\n")

    return contributions
