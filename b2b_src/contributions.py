# environment setup
import sys
sys.path.insert(0, ".")
import yaml
import logging
from typing import Dict, Any

# Get logger for this module (controlled by centralized logging config)
from pymc_marketing.logging_config import get_logger
logger = get_logger(__name__)


# pymc-labs imports
# pymc_marketing upgrade imports
from pymc_marketing.mmm.mmm import MMM
from pymc_extras.prior import Prior
from pymc_extras.prior import VariableFactory
from pymc_marketing.mmm.components.adstock import GeometricAdstock
from pymc_marketing.mmm.components.saturation import LogisticSaturation
from pymc_marketing.mmm.components.adstock import NoAdstock
from pymc_marketing.mmm.components.saturation import NoSaturation

# helper libraries
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytensor 
import jax 
import pytensor.tensor as pt
import numpyro 
import pandas as pd



def compare_predictions(
    mmm,
    data,
    t_start=None,
    t_end=None,
    shutdown_channels=None,
    marginal_changes=None,
    original_scale=True,
    return_predictions=False,
    random_seed=None
):
    """
    Compare model predictions with channel manipulations.

    Parameters
    ----------
    mmm : MMM
        Fitted MMM model
    data : pd.DataFrame
        Original data with all columns
    t_start, t_end : int, optional
        Period range to select (uses 't' column). If None, uses all data
    shutdown_channels : list, optional
        Channel names to set to zero, e.g., ['x1', 'x2']
    marginal_changes : dict, optional
        Marginal changes as {channel: change}, e.g.:
        - {'x1': 0.2} = increase x1 by 20%
        - {'x2': -0.3} = decrease x2 by 30%
    original_scale : bool, default=True
        If True, reverse log transformation (exp(predictions)) for y-axis
        If False, keep predictions in log scale
    return_predictions : bool, default=False
        If True, return dictionary with prediction arrays instead of plotting
    random_seed : int or np.random.Generator, optional
        Random seed for reproducibility. If None, results may vary between runs.
    
    Returns
    -------
    dict or None
        If return_predictions=True, returns dict with:
        - 'dates': date array
        - 'original_mean': original predictions mean
        - 'original_lower': original predictions 94% HDI lower
        - 'original_upper': original predictions 94% HDI upper
        - 'modified_mean': modified predictions mean
        - 'modified_lower': modified predictions 94% HDI lower
        - 'modified_upper': modified predictions 94% HDI upper
        - 'observed': observed values
        - 'relative_change_pct': percentage change in mean predictions
    """
    # 1. Prepare modified data
    if t_start is not None or t_end is not None:
        mask = pd.Series(True, index=data.index)
        if t_start is not None:
            mask &= (data['t'] >= t_start)
        if t_end is not None:
            mask &= (data['t'] <= t_end)
        X = data[mask].copy()
    else:
        X = data.copy()
    
    if 'y' in X.columns:
        X = X.drop('y', axis=1)
    
    if shutdown_channels:
        for channel in shutdown_channels:
            X[channel] = 0
    
    if marginal_changes:
        for channel, change in marginal_changes.items():
            X[channel] = X[channel] * (1 + change)
    
    # 2. Get original predictions (from idata)
    original_preds = mmm.idata.posterior_predictive[mmm.output_var]
    original_mean_log = original_preds.mean(dim=["chain", "draw"]).values
    original_hdi = az.hdi(original_preds, hdi_prob=0.94)
    original_lower_log = original_hdi.sel(hdi="lower")[mmm.output_var].values
    original_upper_log = original_hdi.sel(hdi="higher")[mmm.output_var].values
    
    # 3. Sample modified predictions
    modified_preds = mmm.sample_posterior_predictive(X=X, extend_idata=False, original_scale=False, random_seed=random_seed)
    modified_preds_var = modified_preds[mmm.output_var]
    modified_mean_log = modified_preds_var.mean(dim="sample").values
    modified_lower_log = modified_preds_var.quantile(0.03, dim="sample").values
    modified_upper_log = modified_preds_var.quantile(0.97, dim="sample").values
    
    # 4. Apply scale transformation if requested
    if original_scale:
        # Reverse log transform: exp(log_value)
        original_mean = np.exp(original_mean_log)
        original_lower = np.exp(original_lower_log)
        original_upper = np.exp(original_upper_log)
        modified_mean = np.exp(modified_mean_log)
        modified_lower = np.exp(modified_lower_log)
        modified_upper = np.exp(modified_upper_log)
        observed = mmm._get_observed_data_for_plot(original_scale=True)
        ylabel = 'Sales (original scale)'
    else:
        original_mean = original_mean_log
        original_lower = original_lower_log
        original_upper = original_upper_log
        modified_mean = modified_mean_log
        modified_lower = modified_lower_log
        modified_upper = modified_upper_log
        observed = mmm._get_observed_data_for_plot(original_scale=False)
        ylabel = 'Sales (log scale)'
    
    # 5. Calculate relative change
    # Relative change = (original - modified) / original * 100
    relative_change_pct = (original_mean.mean() - modified_mean.mean()) / original_mean.mean() * 100
    
    dates = modified_preds.date.values
    
    # 6. Return predictions if requested
    if return_predictions:
        return {
            'dates': dates,
            'original_mean': original_mean,
            'original_lower': original_lower,
            'original_upper': original_upper,
            'modified_mean': modified_mean,
            'modified_lower': modified_lower,
            'modified_upper': modified_upper,
            'observed': observed,
            'relative_change_pct': relative_change_pct
        }
    
    # 7. Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Observed as continuous line
    ax.plot(dates, observed, 'k-', label='Observed', linewidth=1.5, alpha=0.7)
    
    # Original predictions
    ax.plot(dates, original_mean, 'b-', label='Original predictions', linewidth=2)
    ax.fill_between(dates, original_lower, original_upper, alpha=0.2, color='blue')
    
    # Modified predictions
    modification_label = f"Modified ({shutdown_channels[0] if shutdown_channels else 'adjusted'} shutdown)" if shutdown_channels else "Modified predictions"
    ax.plot(dates, modified_mean, 'r-', label=modification_label, linewidth=2)
    ax.fill_between(dates, modified_lower, modified_upper, alpha=0.2, color='red')
    
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)
    
    # Title with explanation
    title = f'Comparison: Original vs Modified Predictions\n'
    title += f'Mean decrease: {relative_change_pct:.1f}% '
    title += f'(calculated as: (mean_original - mean_modified) / mean_original × 100)'
    ax.set_title(title, fontsize=11)
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig



def get_individual_contributions(mmm, data, channel_cols, control_cols, random_seed=None):
    """
    Automatically calculate individual contributions for all model components.

    Parameters
    ----------
    mmm : MMM object
        Fitted MMM model instance
    data : pd.DataFrame
        Original data containing date column
    channel_cols : list
        List of channel column names (e.g., ['x1', 'x2'])
    control_cols : list
        List of control column names (e.g., ['t', 'event_1', 'event_2'])
    random_seed : int or np.random.Generator, optional
        Random seed for reproducibility. If None, results may vary between runs.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'dates': array of dates
        - 'channels': dict with channel contributions {channel_name: xarray}
        - 'controls': dict with control contributions {control_name: xarray}
        - 'intercept': xarray with intercept contribution
        - 'seasonality': xarray with seasonality contribution
    """
    import numpy as np

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

    # Get baseline predictions (no components shut down)
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
        'channels': {},
        'controls': {},
        'intercept': None,
        'seasonality': None
    }

    # Calculate channel contributions
    for channel in channel_cols:
        preds_no_channel = mmm.sample_posterior_predictive(
            data,
            extend_idata=False,
            combined=True,
            include_last_observations=False,
            original_scale=True,
            shutdown_components=[channel],
            random_seed=channel_seeds[channel]
        )
        contributions['channels'][channel] = preds_normal[mmm.output_var] - preds_no_channel[mmm.output_var]

    # Calculate control contributions
    for control in control_cols:
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

    # Calculate intercept contribution
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

    # Calculate seasonality contribution
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
    if hasattr(dates_obj, 'coords') and 'date' in dates_obj.coords:
        contributions['dates'] = dates_obj['date'].values
    else:
        # Fallback: create index array
        contributions['dates'] = np.arange(dates_obj.shape[-1])
    
    return contributions


def calculate_roas(
    contributions,
    data,
    channel_cols=None,
    date_start=None,
    date_end=None,
    target_column=None,
    spend_mapping=None
):
    """
    Calculate ROAS (Return on Ad Spend) for channels based on contributions.

    This method implements the following steps:
    1. Calculate total contribution raw by summing ALL component contributions at each time t
    2. Calculate adjusted contributions by dividing each component's raw contribution by total
       contribution raw and multiplying by the actual target value (yt) at that time
    3. Calculate ROAS by dividing the sum of adjusted contributions by the sum of spend
    4. Calculate ROAS distribution across posterior samples for boxplot analysis

    Parameters
    ----------
    contributions : dict
        Dictionary returned by get_individual_contributions() containing:
        - 'dates': array of dates
        - 'channels': dict with channel contributions (xarray)
        - 'controls': dict with control contributions (xarray)
        - 'intercept': intercept contribution (xarray)
        - 'seasonality': seasonality contribution (xarray)
    data : pd.DataFrame
        Original data containing spend data and target column
    channel_cols : list, optional
        List of channel names to calculate ROAS for. If None, uses all channels in contributions.
    date_start, date_end : str or datetime, optional
        Start and end dates for the cumulative period. If None, uses all periods.
        Dates should match the format in contributions['dates'] or data date column.
    target_column : str, optional
        Name of the target variable column in data. If None, uses value from global config.
    spend_mapping : dict, optional
        Dictionary mapping channel names to spend column names.
        Example: {'MV_Paid_1_visits': 'MV_Paid_1_Spend', 'MV_Paid_2_visits': 'MV_Paid_2_Spend'}
        If None, assumes channel name is the spend column (backwards compatible).

    Returns
    -------
    dict
        Dictionary containing:
        - 'roas': dict with mean ROAS values for each channel {channel_name: roas_value}
        - 'adjusted_contributions_channels': dict with adjusted contribution arrays for channels {channel_name: array}
        - 'adjusted_contributions_all': dict with adjusted contributions for ALL components
        - 'total_contribution_raw': array of total raw contributions over time
        - 'spend': dict with total spend for each channel {channel_name: spend_value}
        - 'period': dict with start and end indices {'start': int, 'end': int}
        - 'cumulative_period': dict with date range info {'start_date': date, 'end_date': date, 'days': int}
        - 'roas_samples': dict with ROAS posterior samples for boxplot {channel_name: array(n_samples)}

    Notes
    -----
    ALL components (channels, controls, intercept, seasonality) are ALWAYS included in the
    total_contribution_raw calculation to ensure mathematical correctness. The channel_cols
    parameter only controls which channels you want ROAS calculated for (reporting purposes).
    """
    import numpy as np
    import pandas as pd
    from b2b_src.column_config import get_default_column_config
    
    # Get column configuration
    cols = get_default_column_config()
    
    # Use target_column from config if not provided
    if target_column is None:
        target_column = cols.target

    # 1. Convert dates to period indices if provided
    dates_array = contributions['dates']

    if date_start is not None or date_end is not None:
        # Convert dates to pandas datetime for comparison
        dates_pd = pd.to_datetime(dates_array)
        mask = pd.Series(True, index=range(len(dates_pd)))

        if date_start is not None:
            date_start_pd = pd.to_datetime(date_start)
            mask &= (dates_pd >= date_start_pd)
        else:
            date_start_pd = dates_pd[0]

        if date_end is not None:
            date_end_pd = pd.to_datetime(date_end)
            mask &= (dates_pd <= date_end_pd)
        else:
            date_end_pd = dates_pd[-1]

        period_indices = np.where(mask)[0]

        if len(period_indices) == 0:
            raise ValueError(f"No data found for date range {date_start} to {date_end}")

        # Filter data by the same indices
        data_period = data.iloc[period_indices].copy()

        # Calculate cumulative period info
        cumulative_period = {
            'start_date': dates_pd[period_indices[0]],
            'end_date': dates_pd[period_indices[-1]],
            'days': len(period_indices)
        }
    else:
        data_period = data.copy()
        period_indices = np.arange(len(data))
        dates_pd = pd.to_datetime(dates_array)
        cumulative_period = {
            'start_date': dates_pd[0],
            'end_date': dates_pd[-1],
            'days': len(data)
        }

    # 2. Determine which channels to calculate ROAS for
    if channel_cols is None:
        channel_cols = list(contributions['channels'].keys())

    # 3. Calculate total contribution raw at each time t, preserving samples
    # IMPORTANT: ALWAYS include ALL components for mathematical correctness
    total_contribution_raw_samples = None

    # Store raw contributions for all components with samples preserved
    raw_contributions_all_samples = {}

    # Add ALL channel contributions (preserving samples)
    for channel_name, channel_contrib in contributions['channels'].items():
        # Extract the data for the selected period
        channel_contrib_period = channel_contrib[..., period_indices]

        # Stack chain and draw dimensions to get (n_samples, n_dates)
        dims = getattr(channel_contrib_period, 'dims', None)
        if dims and 'chain' in dims and 'draw' in dims:
            # Stack to (n_samples, n_dates)
            channel_samples = channel_contrib_period.stack(sample=('chain', 'draw'))
            channel_samples_array = np.asarray(channel_samples)
        else:
            # If no chain/draw dims, just convert to array
            channel_samples_array = np.asarray(channel_contrib_period)
            if channel_samples_array.ndim == 1:
                channel_samples_array = channel_samples_array[np.newaxis, :]

        # Store for later
        raw_contributions_all_samples[f'channel_{channel_name}'] = channel_samples_array

        if total_contribution_raw_samples is None:
            total_contribution_raw_samples = channel_samples_array.copy()
        else:
            total_contribution_raw_samples += channel_samples_array

    # Add ALL control contributions
    for control_name, control_contrib in contributions['controls'].items():
        control_contrib_period = control_contrib[..., period_indices]
        dims = getattr(control_contrib_period, 'dims', None)
        if dims and 'chain' in dims and 'draw' in dims:
            control_samples = control_contrib_period.stack(sample=('chain', 'draw'))
            control_samples_array = np.asarray(control_samples)
        else:
            control_samples_array = np.asarray(control_contrib_period)
            if control_samples_array.ndim == 1:
                control_samples_array = control_samples_array[np.newaxis, :]

        raw_contributions_all_samples[f'control_{control_name}'] = control_samples_array

        if total_contribution_raw_samples is None:
            total_contribution_raw_samples = control_samples_array.copy()
        else:
            total_contribution_raw_samples += control_samples_array

    # Add intercept contribution
    intercept_contrib = contributions['intercept']
    intercept_contrib_period = intercept_contrib[..., period_indices]
    dims = getattr(intercept_contrib_period, 'dims', None)
    if dims and 'chain' in dims and 'draw' in dims:
        intercept_samples = intercept_contrib_period.stack(sample=('chain', 'draw'))
        intercept_samples_array = np.asarray(intercept_samples)
    else:
        intercept_samples_array = np.asarray(intercept_contrib_period)
        if intercept_samples_array.ndim == 1:
            intercept_samples_array = intercept_samples_array[np.newaxis, :]

    raw_contributions_all_samples['intercept'] = intercept_samples_array

    if total_contribution_raw_samples is None:
        total_contribution_raw_samples = intercept_samples_array.copy()
    else:
        total_contribution_raw_samples += intercept_samples_array

    # Add seasonality contribution
    seasonality_contrib = contributions['seasonality']
    seasonality_contrib_period = seasonality_contrib[..., period_indices]
    dims = getattr(seasonality_contrib_period, 'dims', None)
    if dims and 'chain' in dims and 'draw' in dims:
        seasonality_samples = seasonality_contrib_period.stack(sample=('chain', 'draw'))
        seasonality_samples_array = np.asarray(seasonality_samples)
    else:
        seasonality_samples_array = np.asarray(seasonality_contrib_period)
        if seasonality_samples_array.ndim == 1:
            seasonality_samples_array = seasonality_samples_array[np.newaxis, :]

    raw_contributions_all_samples['seasonality'] = seasonality_samples_array

    if total_contribution_raw_samples is None:
        total_contribution_raw_samples = seasonality_samples_array.copy()
    else:
        total_contribution_raw_samples += seasonality_samples_array

    # 4. Get actual target values for the period
    y_actual = data_period[target_column].values

    # 5. Calculate adjusted contributions for ALL components (mean across samples for plotting)
    # Calculate mean total contribution for adjustment
    total_contribution_raw = total_contribution_raw_samples.mean(axis=0)

    adjusted_contributions_all = {}

    for component_name, raw_contrib_samples in raw_contributions_all_samples.items():
        # Use mean across samples for the plotting contributions
        raw_contrib = raw_contrib_samples.mean(axis=0)

        # Calculate adjusted contribution:
        # adjusted_contrib_t = (raw_contrib_t / total_contrib_raw_t) * y_actual_t
        # Handle division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            adjustment_factor = np.where(
                total_contribution_raw != 0,
                raw_contrib / total_contribution_raw,
                0
            )

        adjusted_contrib = adjustment_factor * y_actual
        adjusted_contributions_all[component_name] = adjusted_contrib

    # 6. Extract adjusted contributions for channels only (for reporting)
    adjusted_contributions_channels = {}
    for channel in channel_cols:
        if channel not in contributions['channels']:
            raise ValueError(f"Channel '{channel}' not found in contributions")

        adjusted_contributions_channels[channel] = adjusted_contributions_all[f'channel_{channel}']

    # 7. Calculate ROAS for each channel (mean value for reporting)
    roas_values = {}
    spend_values = {}

    for channel in channel_cols:
        # Sum of adjusted contributions over the period
        total_contribution = np.sum(adjusted_contributions_channels[channel])

        # Get the spend column name from mapping, or use channel name if no mapping
        spend_column = spend_mapping.get(channel, channel) if spend_mapping else channel
        
        # Verify spend column exists in data
        if spend_column not in data_period.columns:
            raise ValueError(f"Spend column '{spend_column}' for channel '{channel}' not found in data. "
                           f"Available columns: {list(data_period.columns)}")
        
        # Sum of spend over the period
        total_spend = data_period[spend_column].sum()
        spend_values[channel] = total_spend

        # ROAS = total contribution / total spend
        if total_spend > 0:
            roas = total_contribution / total_spend
        else:
            roas = np.nan

        roas_values[channel] = roas

    # 8. Calculate ROAS samples for boxplot (across all posterior samples)
    roas_samples = {}

    for channel in channel_cols:
        # Get channel contribution samples (n_samples, n_dates)
        channel_contrib_samples = raw_contributions_all_samples[f'channel_{channel}']

        # Calculate adjusted contributions per sample
        # adjusted_contrib[s,t] = (raw_contrib[s,t] / total_contrib_raw[s,t]) * y_actual[t]
        with np.errstate(divide='ignore', invalid='ignore'):
            adjustment_factor_samples = np.where(
                total_contribution_raw_samples != 0,
                channel_contrib_samples / total_contribution_raw_samples,
                0
            )

        adjusted_contrib_samples = adjustment_factor_samples * y_actual[np.newaxis, :]  # broadcast y_actual

        # Sum over time for each sample
        total_contribution_samples = np.sum(adjusted_contrib_samples, axis=1)

        # Get the spend column name from mapping
        spend_column = spend_mapping.get(channel, channel) if spend_mapping else channel
        total_spend = data_period[spend_column].sum()

        if total_spend > 0:
            roas_sample_values = total_contribution_samples / total_spend
        else:
            roas_sample_values = np.full(len(total_contribution_samples), np.nan)

        roas_samples[channel] = roas_sample_values

    # 9. Return results
    return {
        'roas': roas_values,
        'adjusted_contributions_channels': adjusted_contributions_channels,
        'adjusted_contributions_all': adjusted_contributions_all,
        'total_contribution_raw': total_contribution_raw,
        'spend': spend_values,
        'period': {
            'start': period_indices[0] if len(period_indices) > 0 else None,
            'end': period_indices[-1] if len(period_indices) > 0 else None
        },
        'cumulative_period': cumulative_period,
        'roas_samples': roas_samples
    }


def calculate_marginal_roas(
    mmm,
    data,
    config,
    date_start=None,
    date_end=None,
    target_column=None,
    marginal_increase=0.01,
    prediction_days=None,
    random_seed=None
):
    """
    Calculate accumulated marginal ROAS over a cumulative period for PAID channels only.

    Marginal ROAS measures the incremental return from a small increase in spend
    (default 1%). This is calculated by comparing predictions with baseline spend
    vs. predictions with slightly increased spend.

    **IMPORTANT**: Only channels with type='paid' in the config will be analyzed.
    Unpaid channels are automatically excluded as they don't have spend associated with them.

    Formula for channel m over period [t₀, t₁]:

        mROAS_m = Σ[t₀ to t₁] (Ŷ_t^m(x̃) - Ŷ_t^m(x)) / (0.01 × Σ[t₀ to t₁] x_t,m)

    Where:
        - x_t,m: Baseline spend for channel m at time t
        - x̃_t,m: Modified spend = (1 + marginal_increase) × x_t,m
        - Ŷ_t^m(x̃): Predicted sales with increased spend
        - Ŷ_t^m(x): Predicted sales with baseline spend

    Parameters
    ----------
    mmm : MMM object
        Fitted MMM model instance (required for generating predictions)
    data : pd.DataFrame
        Original data containing spend data and target column
    config : dict
        Configuration dictionary containing channel information.
        Expected structure: config['data']['columns']['channels'] should be a list of dicts
        with 'name', 'spend_column', and 'type' keys.
        **Only channels with type='paid' will be included in the analysis.**
        Example:
            config = {
                'data': {
                    'columns': {
                        'channels': [
                            {'name': 'MV_Paid_1_visits', 'spend_column': 'MV_Paid_1_Spend', 'type': 'paid'},
                            {'name': 'MV_Paid_2_visits', 'spend_column': 'MV_Paid_2_Spend', 'type': 'paid'},
                            {'name': 'MV_Unpaid_1_visits', 'spend_column': 'MV_Unpaid_1_visits', 'type': 'unpaid'}
                        ]
                    }
                }
            }
        In this example, only MV_Paid_1_visits and MV_Paid_2_visits will be analyzed.
    date_start, date_end : str or datetime, optional
        Start and end dates for the cumulative period. If None, uses all periods.
        Dates should match the format in data's date column.
    target_column : str, optional
        Name of the target variable column in data. If None, uses value from global config.
    marginal_increase : float, default=0.01
        Percentage increase to apply (default 0.01 = 1% increase)
    prediction_days : int, optional
        Number of days to use for prediction range. If None, uses the entire period
        defined by date_start/date_end (or full dataset if those are None).
        This allows focusing on recent periods, e.g., prediction_days=30 for last 30 days,
        prediction_days=60 for last 60 days. The spend increase is applied during this period,
        and predictions capture carryover effects beyond it.
    random_seed : int or np.random.Generator, optional
        Random seed for reproducibility. If None, results may vary between runs.
        Pass the same seed to get consistent results.

    Returns
    -------
    dict
        Dictionary containing:
        - 'mroas': dict with mean mROAS values for each channel {channel_name: mroas_value}
        - 'mroas_samples': dict with mROAS posterior samples for boxplot {channel_name: array(n_samples)}
        - 'incremental_sales': dict with total incremental sales {channel_name: float}
        - 'incremental_sales_samples': dict with incremental sales samples {channel_name: array(n_samples)}
        - 'incremental_spend': dict with total incremental spend {channel_name: float}
        - 'period': dict with start and end indices {'start': int, 'end': int}
        - 'cumulative_period': dict with date range info {'start_date': date, 'end_date': date, 'days': int}
        - 'marginal_increase': float (the percentage increase used)

    Notes
    -----
    - Marginal ROAS is typically lower than regular ROAS due to diminishing returns (saturation)
    - Uses full posterior distribution for uncertainty quantification
    - All predictions are in original scale (not log-transformed)
    - Carryover effects are naturally incorporated through the model's adstock transformations

    Examples
    --------
    >>> # Calculate marginal ROAS for a specific period
    >>> mroas_results = calculate_marginal_roas(
    ...     mmm=fitted_mmm,
    ...     data=data,
    ...     config=config,
    ...     date_start='2023-01-01',
    ...     date_end='2023-12-31'
    ... )
    >>>
    >>> # Access results for a specific channel
    >>> channel_name = config['data']['columns']['channels'][0]['name']
    >>> print(f"{channel_name} mROAS: {mroas_results['mroas'][channel_name]:.2f}")
    >>> print(f"Incremental Sales: ${mroas_results['incremental_sales'][channel_name]:,.0f}")

    See Also
    --------
    calculate_roas : Calculate regular (non-marginal) ROAS
    calculate_marginal_roas_time_series : Calculate per-period marginal ROAS
    """
    import numpy as np
    import pandas as pd
    from b2b_src.column_config import get_default_column_config
    
    # Get column configuration
    cols = get_default_column_config()
    
    # Use target_column from config if not provided
    if target_column is None:
        target_column = cols.target

    # 1. Extract channel information from config
    channels_config = config['data']['columns']['channels']

    # Filter for PAID channels only and create mapping from channel name to spend column
    channel_spend_mapping = {
        channel_info['name']: channel_info['spend_column']
        for channel_info in channels_config
        if channel_info.get('type', '').lower() == 'paid'  # Only include paid channels
    }

    channel_cols = list(channel_spend_mapping.keys())

    # Validate that we have at least one paid channel
    if len(channel_cols) == 0:
        raise ValueError("No paid channels found in config. Marginal ROAS can only be calculated for paid channels.")

    # Log information about which channels will be analyzed
    logger.info(f"Analyzing marginal ROAS for {len(channel_cols)} paid channel(s): {', '.join(channel_cols)}")

    # 2. Convert dates to period indices if provided
    if 'date' in data.columns:
        dates_array = data['date'].values
    else:
        # Fallback to index if no date column
        dates_array = data.index.values

    dates_pd = pd.to_datetime(dates_array)

    # Handle prediction_days parameter (takes precedence over date_start/date_end)
    if prediction_days is not None:
        # Use the last N days from the data
        if prediction_days > len(data):
            raise ValueError(f"prediction_days ({prediction_days}) exceeds available data length ({len(data)})")

        period_indices = np.arange(len(data) - prediction_days, len(data))

        cumulative_period = {
            'start_date': dates_pd[period_indices[0]],
            'end_date': dates_pd[period_indices[-1]],
            'days': len(period_indices)
        }
    elif date_start is not None or date_end is not None:
        # Convert dates to pandas datetime for comparison
        mask = pd.Series(True, index=range(len(dates_pd)))

        if date_start is not None:
            date_start_pd = pd.to_datetime(date_start)
            mask &= (dates_pd >= date_start_pd)
        else:
            date_start_pd = dates_pd[0]

        if date_end is not None:
            date_end_pd = pd.to_datetime(date_end)
            mask &= (dates_pd <= date_end_pd)
        else:
            date_end_pd = dates_pd[-1]

        period_indices = np.where(mask)[0]

        if len(period_indices) == 0:
            raise ValueError(f"No data found for date range {date_start} to {date_end}")

        # Calculate cumulative period info
        cumulative_period = {
            'start_date': dates_pd[period_indices[0]],
            'end_date': dates_pd[period_indices[-1]],
            'days': len(period_indices)
        }
    else:
        # Use entire dataset
        period_indices = np.arange(len(data))
        cumulative_period = {
            'start_date': dates_pd[0],
            'end_date': dates_pd[-1],
            'days': len(data)
        }

    # 3. Prepare random seed sequence for reproducibility
    # We need different seeds for each sampling call, but deterministically derived from the input seed
    if random_seed is not None:
        # Create a seed sequence that generates child seeds
        if isinstance(random_seed, np.random.Generator):
            # If it's a Generator, extract the seed
            ss = np.random.SeedSequence(random_seed.bit_generator.seed_seq.entropy)
        else:
            # If it's an int, use it directly
            ss = np.random.SeedSequence(random_seed)

        # Generate child seeds: one for baseline + one for each channel
        child_seeds = ss.spawn(1 + len(channel_cols))
        baseline_seed = np.random.default_rng(child_seeds[0])
        channel_seeds = [np.random.default_rng(child_seeds[i+1]) for i in range(len(channel_cols))]
    else:
        baseline_seed = None
        channel_seeds = [None] * len(channel_cols)

    # 4. Get baseline predictions (with original spend)
    logger.info(f"Generating baseline predictions...")
    preds_baseline = mmm.sample_posterior_predictive(
        data,
        extend_idata=False,
        combined=True,
        include_last_observations=False,
        original_scale=False,  # ✅ FIXED: Keep in log/scaled space
        random_seed=baseline_seed
    )

    # Extract baseline samples
    output_var = mmm.output_var
    baseline_preds = preds_baseline[output_var]

    logger.debug(f"Baseline predictions dims: {baseline_preds.dims}")
    logger.debug(f"Baseline predictions shape: {baseline_preds.shape}")

    # ✅ FIXED: Calculate mean in log/scaled space FIRST (per time period)
    # Handle both combined ('sample') and uncombined ('chain', 'draw') dimensions
    if 'sample' in baseline_preds.dims:
        baseline_mean_transformed = baseline_preds.mean(dim="sample").values
    elif 'chain' in baseline_preds.dims and 'draw' in baseline_preds.dims:
        baseline_mean_transformed = baseline_preds.mean(dim=["chain", "draw"]).values
    else:
        # Already averaged (shouldn't happen with combined=True)
        baseline_mean_transformed = baseline_preds.values

    # ✅ FIXED: Apply transformations in CORRECT order to each time period
    # Step 1: Revert log transformation if applicable
    if hasattr(mmm, 'log_transform_vars') and 'target' in mmm.log_transform_vars:
        logger.info("✓ Detected log-transformed target - applying exp() transformation")
        baseline_mean_original = np.exp(baseline_mean_transformed)
    else:
        baseline_mean_original = baseline_mean_transformed

    # Step 2: Revert scaling
    if hasattr(mmm, 'target_scale'):
        logger.info(f"✓ Applying target_scale: {mmm.target_scale}")
        baseline_mean_original = baseline_mean_original * mmm.target_scale

    # baseline_mean_original shape: (n_dates,) - one value per time period
    # We'll use this for calculating incremental sales
    baseline_array = baseline_mean_original  # shape: (n_dates,)

    # Initialize result dictionaries
    mroas_values = {}
    mroas_samples_dict = {}
    incremental_sales_dict = {}
    incremental_sales_samples_dict = {}
    incremental_spend_dict = {}

    # 5. Calculate mROAS for EACH channel independently
    for i, channel in enumerate(channel_cols, 1):
        logger.info(f"Calculating mROAS for channel {i}/{len(channel_cols)}: {channel}...")

        # Get the spend column name from mapping (for calculating incremental spend)
        spend_column = channel_spend_mapping[channel]

        # Verify spend column exists in data
        if spend_column not in data.columns:
            raise ValueError(f"Spend column '{spend_column}' for channel '{channel}' not found in data. "
                           f"Available columns: {list(data.columns)}")

        # Create modified data with ONLY this channel's input increased
        data_modified_channel = data.copy()
        # Modify the CHANNEL column (visits), not the spend column
        # The model was fitted with channel_columns (visits), so we modify that
        data_modified_channel.loc[period_indices, channel] *= (1 + marginal_increase)

        # Get modified predictions for THIS channel only (using channel-specific seed)
        preds_modified_channel = mmm.sample_posterior_predictive(
            data_modified_channel,
            extend_idata=False,
            combined=True,
            include_last_observations=False,
            original_scale=False,  # ✅ FIXED: Keep in log/scaled space
            random_seed=channel_seeds[i-1]
        )

        # ✅ FIXED: Extract modified predictions with proper transformation
        modified_preds = preds_modified_channel[output_var]

        # ✅ FIXED: Calculate mean in log/scaled space FIRST (per time period)
        # Handle both combined ('sample') and uncombined ('chain', 'draw') dimensions
        if 'sample' in modified_preds.dims:
            modified_mean_transformed = modified_preds.mean(dim="sample").values
        elif 'chain' in modified_preds.dims and 'draw' in modified_preds.dims:
            modified_mean_transformed = modified_preds.mean(dim=["chain", "draw"]).values
        else:
            # Already averaged (shouldn't happen with combined=True)
            modified_mean_transformed = modified_preds.values

        # ✅ FIXED: Apply transformations in CORRECT order to each time period
        # Step 1: Revert log transformation if applicable
        if hasattr(mmm, 'log_transform_vars') and 'target' in mmm.log_transform_vars:
            modified_mean_original = np.exp(modified_mean_transformed)
        else:
            modified_mean_original = modified_mean_transformed

        # Step 2: Revert scaling
        if hasattr(mmm, 'target_scale'):
            modified_mean_original = modified_mean_original * mmm.target_scale

        # modified_mean_original shape: (n_dates,)
        modified_array_channel = modified_mean_original  # shape: (n_dates,)

        # Calculate incremental sales for this channel
        # Sum over ALL time periods (not just the spend increase period) to capture carryover effects
        incremental_sales_per_period = (
            modified_array_channel - baseline_array
        )  # shape: (n_dates,)

        # Sum over time to get total incremental sales
        total_incremental_sales = incremental_sales_per_period.sum()  # scalar

        # Calculate incremental spend (only during the spend increase period)
        total_spend_baseline = data.iloc[period_indices][spend_column].sum()
        incremental_spend = marginal_increase * total_spend_baseline

        # Calculate mROAS (now working with mean values, not sample distributions)
        if incremental_spend > 0:
            mroas_value = total_incremental_sales / incremental_spend
        else:
            mroas_value = np.nan

        # Store results
        # Note: With the corrected approach, we're using mean predictions rather than
        # preserving the full posterior distribution. This is necessary because we need
        # to apply exp() transformation which should be done on the mean in log space,
        # not on individual samples (due to Jensen's inequality).
        mroas_values[channel] = mroas_value
        mroas_samples_dict[channel] = np.array([mroas_value])  # Single value wrapped in array for compatibility
        incremental_sales_dict[channel] = total_incremental_sales
        incremental_sales_samples_dict[channel] = np.array([total_incremental_sales])  # Single value wrapped in array
        incremental_spend_dict[channel] = incremental_spend

    # 8. Return results
    return {
        'mroas': mroas_values,
        'mroas_samples': mroas_samples_dict,
        'incremental_sales': incremental_sales_dict,
        'incremental_sales_samples': incremental_sales_samples_dict,
        'incremental_spend': incremental_spend_dict,
        'period': {
            'start': period_indices[0] if len(period_indices) > 0 else None,
            'end': period_indices[-1] if len(period_indices) > 0 else None
        },
        'cumulative_period': cumulative_period,
        'marginal_increase': marginal_increase
    }

def calculate_roas_time_series(
    contributions,
    data,
    channel_cols=None,
    t_start=None,
    t_end=None,
    target_column=None,
    spend_mapping=None
):
    """
    Calculate per-period ROAS preserving all posterior samples for uncertainty quantification.
    
    This function calculates ROAS (Return on Ad Spend) for each time period while preserving
    the full posterior distribution from the Bayesian model. This allows us to visualize
    ROAS over time with confidence intervals.
    
    Parameters
    ----------
    contributions : dict
        Dictionary returned by get_individual_contributions() containing:
        - 'dates': array of dates
        - 'channels': dict with channel contributions (xarray with sample dimension)
        - 'controls': dict with control contributions (xarray with sample dimension)
        - 'intercept': intercept contribution (xarray with sample dimension)
        - 'seasonality': seasonality contribution (xarray with sample dimension)
    data : pd.DataFrame
        Original data containing spend data and target column
    channel_cols : list, optional
        List of channel names to calculate ROAS for. If None, uses all channels.
    t_start, t_end : int, optional
        Period range to select. If None, uses all periods.
    target_column : str, optional
        Name of the target variable column in data. If None, uses value from global config.
    spend_mapping : dict, optional
        Dictionary mapping channel names to spend column names.
        Example: {'MV_Paid_1_visits': 'MV_Paid_1_Spend', 'MV_Paid_2_visits': 'MV_Paid_2_Spend'}
        If None, assumes channel name is the spend column (backwards compatible).
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'roas_time_series': dict with ROAS arrays {channel_name: array(n_samples, n_dates)}
        - 'dates': array of dates for the period
        - 'spend': dict with spend arrays {channel_name: array(n_dates)}
        - 'channel_cols': list of channels processed
    
    Notes
    -----
    - Preserves ALL posterior samples (chain, draw) for uncertainty quantification
    - Zero spend periods result in NaN ROAS values
    - All components are included in total contribution calculation
    """
    import numpy as np
    import pandas as pd
    import xarray as xr
    from b2b_src.column_config import get_default_column_config
    
    # Get column configuration
    cols = get_default_column_config()
    
    # Use target_column from config if not provided
    if target_column is None:
        target_column = cols.target
    
    # 1. Filter data by period if specified
    if t_start is not None or t_end is not None:
        mask = pd.Series(True, index=data.index)
        if t_start is not None:
            mask &= (data['t'] >= t_start)
        if t_end is not None:
            mask &= (data['t'] <= t_end)
        data_period = data[mask].copy()
        period_indices = np.where(mask)[0]
    else:
        data_period = data.copy()
        period_indices = np.arange(len(data))
    
    # 2. Determine which channels to calculate ROAS for
    if channel_cols is None:
        channel_cols = list(contributions['channels'].keys())
    
    # 3. Get dates for the period
    dates = contributions['dates'][period_indices]
    
    # 4. Get actual target values for the period
    y_actual = data_period[target_column].values  # shape: (n_dates,)
    
    # 5. Calculate total contribution raw per sample, per time
    # We need to preserve the sample dimension (chain, draw)
    
    total_contribution_raw = None
    raw_contributions_channels = {}  # Store raw contributions for channels
    
    # Helper function to extract and reshape contributions
    def extract_samples(contrib_xr, period_indices):
        """Extract contribution data preserving sample dimension."""
        # Select the time period
        contrib_period = contrib_xr[..., period_indices]
        
        # Check if it's an xarray with dims
        if hasattr(contrib_period, 'dims'):
            # Stack chain and draw into a single 'sample' dimension
            if 'chain' in contrib_period.dims and 'draw' in contrib_period.dims:
                contrib_stacked = contrib_period.stack(sample=('chain', 'draw'))
                # Result should be (sample, date)
                if 'date' in contrib_stacked.dims:
                    contrib_array = contrib_stacked.transpose('sample', 'date').values
                else:
                    contrib_array = contrib_stacked.values
            else:
                # If no chain/draw dims, just convert to array
                contrib_array = np.asarray(contrib_period)
                # Ensure it's 2D: (samples, dates)
                if contrib_array.ndim == 1:
                    contrib_array = contrib_array[np.newaxis, :]
        else:
            # If not xarray, convert to numpy array
            contrib_array = np.asarray(contrib_period)
            if contrib_array.ndim == 1:
                contrib_array = contrib_array[np.newaxis, :]
        
        return contrib_array
    
    # Add ALL channel contributions
    for channel_name, channel_contrib in contributions['channels'].items():
        channel_array = extract_samples(channel_contrib, period_indices)
        raw_contributions_channels[channel_name] = channel_array
        
        if total_contribution_raw is None:
            total_contribution_raw = channel_array.copy()
        else:
            total_contribution_raw += channel_array
    
    # Add ALL control contributions
    for control_name, control_contrib in contributions['controls'].items():
        control_array = extract_samples(control_contrib, period_indices)
        
        if total_contribution_raw is None:
            total_contribution_raw = control_array.copy()
        else:
            total_contribution_raw += control_array
    
    # Add intercept contribution
    intercept_array = extract_samples(contributions['intercept'], period_indices)
    if total_contribution_raw is None:
        total_contribution_raw = intercept_array.copy()
    else:
        total_contribution_raw += intercept_array
    
    # Add seasonality contribution
    seasonality_array = extract_samples(contributions['seasonality'], period_indices)
    if total_contribution_raw is None:
        total_contribution_raw = seasonality_array.copy()
    else:
        total_contribution_raw += seasonality_array
    
    # Now total_contribution_raw has shape: (n_samples, n_dates)
    
    # 6. Calculate adjusted contributions per sample, per time for each channel
    adjusted_contributions_channels = {}
    
    for channel in channel_cols:
        if channel not in contributions['channels']:
            raise ValueError(f"Channel '{channel}' not found in contributions")
        
        raw_contrib = raw_contributions_channels[channel]  # shape: (n_samples, n_dates)
        
        # Calculate adjusted contribution:
        # adjusted_contrib[s, t] = (raw_contrib[s, t] / total_contrib_raw[s, t]) * y_actual[t]
        with np.errstate(divide='ignore', invalid='ignore'):
            adjustment_factor = np.where(
                total_contribution_raw != 0,
                raw_contrib / total_contribution_raw,
                0
            )
        
        # Broadcast y_actual across samples: y_actual has shape (n_dates,)
        # adjustment_factor has shape (n_samples, n_dates)
        adjusted_contrib = adjustment_factor * y_actual[np.newaxis, :]  # Broadcasting
        
        adjusted_contributions_channels[channel] = adjusted_contrib
    
    # 7. Calculate ROAS per sample, per time for each channel
    roas_time_series = {}
    spend_by_channel = {}
    
    for channel in channel_cols:
        # Get the spend column name from mapping
        spend_column = spend_mapping.get(channel, channel) if spend_mapping else channel
        
        # Verify spend column exists in data
        if spend_column not in data_period.columns:
            raise ValueError(f"Spend column '{spend_column}' for channel '{channel}' not found in data. "
                           f"Available columns: {list(data_period.columns)}")
        
        # Get spend for this channel over the period
        spend = data_period[spend_column].values  # shape: (n_dates,)
        spend_by_channel[channel] = spend
        
        # Get adjusted contributions: shape (n_samples, n_dates)
        adjusted_contrib = adjusted_contributions_channels[channel]
        
        # Calculate ROAS: roas[s, t] = adjusted_contrib[s, t] / spend[t]
        with np.errstate(divide='ignore', invalid='ignore'):
            roas = np.where(
                spend[np.newaxis, :] != 0,  # Broadcasting spend to (1, n_dates)
                adjusted_contrib / spend[np.newaxis, :],
                np.nan
            )
        
        roas_time_series[channel] = roas
    
    # 8. Return results
    return {
        'roas_time_series': roas_time_series,
        'dates': dates,
        'spend': spend_by_channel,
        'channel_cols': channel_cols
    }


def analyze_revenue_vs_spend_levels(
    mmm,
    data,
    config,
    multipliers=None,
    date_start=None,
    date_end=None,
    random_seed=None
):
    """
    Analyze revenue vs spend level by simulating different spending multipliers for paid channels.
    
    For each paid channel, this function simulates predictions at different spend levels
    (e.g., 0.2x, 0.4x, 0.6x, 0.8x, 1x, 2x, 3x, 4x, 5x current spend) while keeping all
    other components constant. The results show how revenue and ROAS change at different
    spend levels, which helps identify optimal budget allocations.
    
    Parameters
    ----------
    mmm : MMM object
        Fitted MMM model instance
    data : pd.DataFrame
        Original data containing spend columns and date information
    config : dict
        Configuration dictionary containing channel information.
        Expected structure: config['data']['columns']['channels'] should be a list of dicts
        with 'name', 'spend_column', and 'type' keys.
        Only channels with type='paid' will be analyzed.
    multipliers : list of float, optional
        List of multipliers to apply to current spend levels.
        Default: [0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0]
    date_start, date_end : str or datetime, optional
        Start and end dates for the analysis period. If None, uses all periods.
    random_seed : int or np.random.Generator, optional
        Random seed for reproducibility. If None, results may vary between runs.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'channels': dict with results for each paid channel
        - 'baseline': dict with baseline metrics
        - 'period': dict with analysis period info
        - 'multiplier_list': list of multipliers used
    """
    import numpy as np
    import pandas as pd
    
    # 1. Set default multipliers if not provided
    if multipliers is None:
        multipliers = [0.2,0.4, 0.6, 0.8, 1, 2.0, 3.0, 4.0, 5.0]
    
    # 2. Extract channel information from config (PAID channels only)
    channels_config = config['data']['columns']['channels']
    paid_channels = [ch for ch in channels_config if ch.get('type') == 'paid']
    paid_channel_names = [ch['name'] for ch in paid_channels]
    channel_spend_mapping = {ch['name']: ch['spend_column'] for ch in paid_channels}

    logger.info(f"Found {len(paid_channel_names)} paid channels to analyze")
    logger.info(f"Paid channels: {paid_channel_names}")
    
    # 3. Filter data by date range if specified
    data_period = data.copy()
    period_info = {}
    
    if date_start is not None or date_end is not None:
        date_col = config['data']['columns']['date']
        data_period[date_col] = pd.to_datetime(data_period[date_col])
        
        if date_start is not None:
            date_start = pd.to_datetime(date_start)
            data_period = data_period[data_period[date_col] >= date_start]
            period_info['start'] = str(date_start)
        
        if date_end is not None:
            date_end = pd.to_datetime(date_end)
            data_period = data_period[data_period[date_col] <= date_end]
            period_info['end'] = str(date_end)

    logger.info(f"\nAnalysis period: {len(data_period)} observations")
    if period_info:
        logger.info(f"Period: {period_info.get('start', 'beginning')} to {period_info.get('end', 'end')}")
    
    # 4. Setup random seed handling
    if random_seed is not None:
        if isinstance(random_seed, np.random.Generator):
            ss = np.random.SeedSequence(random_seed.bit_generator.seed_seq.entropy)
        else:
            # If it's an int, use it directly
            ss = np.random.SeedSequence(random_seed)
        
        # We need: 1 baseline + (n_channels * n_multipliers) seeds
        n_total_seeds = 1 + len(paid_channel_names) * len(multipliers)
        child_seeds = ss.spawn(n_total_seeds)
        
        baseline_seed = np.random.default_rng(child_seeds[0])
        seed_idx = 1
    else:
        baseline_seed = None
        seed_idx = None
    
    # 5. Get baseline predictions (1.0x spend for all channels)
    logger.info(f"\nGenerating baseline predictions (1.0x spend)...")
    preds_baseline = mmm.sample_posterior_predictive(
        data_period,
        extend_idata=False,
        combined=True,
        include_last_observations=False,
        original_scale=False,  # ✅ Keep in log/scaled space
        random_seed=baseline_seed
    )
    
    # 6. Extract baseline revenue with CORRECT transformation order
    output_var = mmm.output_var
    baseline_preds = preds_baseline[output_var]
    
    logger.debug(f"Baseline predictions dims: {baseline_preds.dims}")
    logger.debug(f"Baseline predictions shape: {baseline_preds.shape}")
    
    # ✅ CRITICAL: Calculate mean in log/scaled space FIRST (per time period)
    # Handle both combined ('sample') and uncombined ('chain', 'draw') dimensions
    if 'sample' in baseline_preds.dims:
        baseline_mean_transformed = baseline_preds.mean(dim="sample").values
    elif 'chain' in baseline_preds.dims and 'draw' in baseline_preds.dims:
        baseline_mean_transformed = baseline_preds.mean(dim=["chain", "draw"]).values
    else:
        # Already averaged (shouldn't happen with combined=True)
        baseline_mean_transformed = baseline_preds.values
    
    # ✅ Apply transformations in CORRECT order to each time period
    # Step 1: Revert log transformation if applicable
    if hasattr(mmm, 'log_transform_vars') and 'target' in mmm.log_transform_vars:
        logger.info("✓ Detected log-transformed target - applying exp() transformation")
        baseline_mean_original = np.exp(baseline_mean_transformed)
    else:
        baseline_mean_original = baseline_mean_transformed
    
    # Step 2: Revert scaling
    if hasattr(mmm, 'target_scale'):
        logger.info(f"✓ Applying target_scale: {mmm.target_scale}")
        baseline_mean_original = baseline_mean_original * mmm.target_scale
    
    # Step 3: Sum across time periods to get total revenue
    baseline_total_revenue = baseline_mean_original.sum()

    logger.info(f"✓ Baseline total revenue: ${baseline_total_revenue:,.2f}")
    
    # Calculate baseline spend and ROAS for each channel
    baseline_spend_by_channel = {}
    baseline_roas_by_channel = {}
    
    for channel in paid_channel_names:
        spend_col = channel_spend_mapping[channel]
        baseline_spend_by_channel[channel] = data_period[spend_col].sum()
        
        # Calculate ROAS (this is placeholder - ideally use marginal ROAS)
        baseline_roas_by_channel[channel] = baseline_total_revenue / baseline_spend_by_channel[channel] if baseline_spend_by_channel[channel] > 0 else 0
    
    # 7. Initialize results structure
    results = {
        'channels': {},
        'baseline': {
            'total_revenue': baseline_total_revenue,
            'spend_by_channel': baseline_spend_by_channel,
            'roas_by_channel': baseline_roas_by_channel
        },
        'period': period_info,
        'multiplier_list': multipliers
    }
    
    # 8. Simulate each multiplier for each paid channel
    for channel_idx, channel in enumerate(paid_channel_names):
        logger.info(f"\n{'='*70}")
        logger.info(f"Analyzing channel {channel_idx+1}/{len(paid_channel_names)}: {channel}")
        logger.info(f"{'='*70}")
        
        spend_column = channel_spend_mapping[channel]
        baseline_spend = baseline_spend_by_channel[channel]
        baseline_revenue = baseline_total_revenue
        baseline_roas = baseline_roas_by_channel[channel]
        
        channel_results = {'multipliers': {}}
        
        for mult_idx, multiplier in enumerate(multipliers):
            logger.info(f"  Simulating multiplier {mult_idx+1}/{len(multipliers)}: {multiplier}x...")
            
            # Create modified data with this channel's spend multiplied
            data_modified = data_period.copy()
            # Modify the CHANNEL column (visits), not the spend column
            # The model was fitted with channel_columns (visits), so we modify that
            data_modified[channel] = data_period[channel] * multiplier
            
            # Get seed for this specific simulation
            if seed_idx is not None:
                current_seed = np.random.default_rng(child_seeds[seed_idx])
                seed_idx += 1
            else:
                current_seed = None
            
            # Generate predictions with modified spend
            preds_modified = mmm.sample_posterior_predictive(
                data_modified,
                extend_idata=False,
                combined=True,
                include_last_observations=False,
                original_scale=False,  # ✅ Keep in log/scaled space
                random_seed=current_seed
            )
            
            # ✅ Extract modified revenue with CORRECT transformation order
            modified_preds = preds_modified[output_var]
            
            # ✅ CRITICAL: Calculate mean in log/scaled space FIRST (per time period)
            # Handle both combined ('sample') and uncombined ('chain', 'draw') dimensions
            if 'sample' in modified_preds.dims:
                modified_mean_transformed = modified_preds.mean(dim="sample").values
            elif 'chain' in modified_preds.dims and 'draw' in modified_preds.dims:
                modified_mean_transformed = modified_preds.mean(dim=["chain", "draw"]).values
            else:
                # Already averaged (shouldn't happen with combined=True)
                modified_mean_transformed = modified_preds.values
            
            # ✅ Apply transformations in CORRECT order to each time period
            # Step 1: Revert log transformation if applicable
            if hasattr(mmm, 'log_transform_vars') and 'target' in mmm.log_transform_vars:
                modified_mean_original = np.exp(modified_mean_transformed)
            else:
                modified_mean_original = modified_mean_transformed
            
            # Step 2: Revert scaling
            if hasattr(mmm, 'target_scale'):
                modified_mean_original = modified_mean_original * mmm.target_scale
            
            # Step 3: Sum across time periods to get total revenue
            new_revenue = modified_mean_original.sum()
            
            # Calculate spend at this multiplier level
            new_spend = baseline_spend * multiplier
            
            # Calculate new ROAS at this spend level
            new_roas = new_revenue / new_spend if new_spend > 0 else 0
            
            # Calculate changes
            spend_change = new_spend - baseline_spend
            revenue_change = new_revenue - baseline_revenue
            roas_change = new_roas - baseline_roas
            
            # Store results for this multiplier
            channel_results['multipliers'][str(multiplier)] = {
                'previous_spend': baseline_spend,
                'new_spend': new_spend,
                'spend_change': spend_change,
                'previous_revenue': baseline_revenue,
                'new_revenue': new_revenue,
                'revenue_change': revenue_change,
                'previous_roas': baseline_roas,
                'new_roas': new_roas,
                'roas_change': roas_change
            }

            logger.debug(f"    ✓ New revenue: ${new_revenue:,.2f} (Δ: ${revenue_change:,.2f})")
        
        # Store channel results
        results['channels'][channel] = channel_results

    logger.info(f"\n{'='*70}")
    logger.info("Analysis complete!")
    logger.info(f"{'='*70}\n")
    
    return results