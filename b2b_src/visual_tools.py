# environment setup
import sys
sys.path.insert(0, ".")
import yaml
import logging
from typing import Dict, Any


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


def plot_sales_and_channels(data, config, figsize=(12, 9), channel_colors=None):
    """
    Plot sales and channel spending over time in paired subplots with dual y-axes.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the sales and channel data
    config : dict
        Configuration dictionary with data column specifications
    figsize : tuple, optional
        Figure size (width, height), default is (12, 9)
    channel_colors : list, optional
        List of colors for channels. If None, uses matplotlib defaults

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : np.ndarray
        Array of axes objects (primary axes for sales)
    """
    from b2b_src.column_config import get_default_column_config
    
    # Get column configuration
    cols = get_default_column_config()
    
    # Get column names from config
    date_col = config['data']['columns']['date_column']
    # Extract channel information from the new config structure
    channels = config['data']['columns']['channels']

    # Create subplots - one for each channel
    n_channels = len(channels)
    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)

    # Handle case where there's only one channel (axes would not be array)
    if n_channels == 1:
        axes = [axes]

    # Plot each channel paired with sales on the same subplot
    for i, channel_info in enumerate(channels):
        channel_name = channel_info['name']
        spend_column = channel_info['spend_column']

        if channel_colors and i < len(channel_colors):
            color = channel_colors[i]
        else:
            color = f'C{i}'

        # Primary axis: Sales
        ax1 = axes[i]
        ax1.plot(data[date_col], data[cols.target], color="black", linewidth=2.5, label='Sales', alpha=0.8)
        ax1.set_ylabel("Sales", fontsize=11, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.grid(alpha=0.3)

        # Secondary axis: Channel spend/visits
        ax2 = ax1.twinx()
        ax2.plot(data[date_col], data[spend_column], color=color, linewidth=2, label=channel_name, alpha=0.7)
        ax2.set_ylabel("Spend/Visits", fontsize=11, fontweight='bold', color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        # Title for this subplot
        ax1.set_title(f"Sales vs Channel: {channel_name}", fontsize=12, fontweight='bold')

        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    # Set xlabel on the bottom plot
    axes[-1].set_xlabel("Date", fontsize=11, fontweight='bold')

    plt.tight_layout()
    return fig, axes


def plot_all_contributions(contributions, mmm, data, use_log_scale=True, dimension_label=None, custom_titles=None):
    """
    Create two plots: one with all contributions together, and subplots with individual contributions.
    Always includes the actual target data and full model predictions in both plots.

    Parameters
    ----------
    contributions : dict
        Dictionary returned by get_individual_contributions() containing:
        - 'dates': array of dates
        - 'channels': dict with channel contributions
        - 'controls': dict with control contributions
        - 'intercept': intercept contribution
        - 'seasonality': seasonality contribution
    mmm : MMM object
        Fitted MMM model instance (required)
    data : pd.DataFrame
        Original data containing the target column 'y'
    use_log_scale : bool, default=True
        If True, plot everything in log space. Requires that the model was trained
        with log transformation (log_transform_vars containing 'target').
        If False, plot everything in normal space with proper handling of log
        transformation and scaling reversion.
    dimension_label : str, optional
        Label for the dimension (e.g., 'geo_a', 'geo_b'). When provided, this label
        is added to plot titles to differentiate between different dimensions.
        Automatically injected by decorators for multidimensional models.
    custom_titles : dict, optional
        Dictionary to customize plot titles. Keys:
        - 'combined': Custom title for the combined contributions plot
        - 'individual': Custom title for the individual subplots figure
        If provided, these override the default titles (dimension_label still appended).

    Returns
    -------
    tuple
        (fig_combined, fig_individual) - Two matplotlib figure objects

    Raises
    ------
    ValueError
        If use_log_scale=True but the model was not trained with log transformation.

    Notes
    -----
    - Automatically detects and handles log transformation via mmm.log_transform_vars
    - Automatically detects and handles scaling via mmm.target_scale
    - Works correctly whether log transformation is ON or OFF in the model
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import arviz as az
    from b2b_src.column_config import get_default_column_config
    
    # Get column configuration
    cols = get_default_column_config()

    dates = contributions['dates']

    # Collect all components and their mean contributions
    all_components = {}

    if use_log_scale:
        # Check if model was trained with log transformation
        if not (hasattr(mmm, 'log_transform_vars') and 'target' in mmm.log_transform_vars):
            raise ValueError(
                "Cannot use use_log_scale=True when model was not trained with log transformation. "
                "Either set use_log_scale=False or train the model with log_transform_vars=['target', ...]"
            )

        # Convert contributions from normal space to log space
        # Add channels
        for channel_name, channel_data in contributions['channels'].items():
            mean_dims = [d for d in getattr(channel_data, 'dims', []) if d != 'date']
            mean_contrib = channel_data.mean(dim=mean_dims) if mean_dims else channel_data
            all_components[f'Channel: {channel_name}'] = np.asarray(mean_contrib)

        # Add controls
        for control_name, control_data in contributions['controls'].items():
            mean_dims = [d for d in getattr(control_data, 'dims', []) if d != 'date']
            mean_contrib = control_data.mean(dim=mean_dims) if mean_dims else control_data
            all_components[f'Control: {control_name}'] = np.asarray(mean_contrib)

        # Add intercept
        intercept_data = contributions['intercept']
        mean_dims = [d for d in getattr(intercept_data, 'dims', []) if d != 'date']
        mean_contrib = intercept_data.mean(dim=mean_dims) if mean_dims else intercept_data
        all_components['Intercept'] = np.asarray(mean_contrib)

        # Add seasonality
        seasonality_data = contributions['seasonality']
        mean_dims = [d for d in getattr(seasonality_data, 'dims', []) if d != 'date']
        mean_contrib = seasonality_data.mean(dim=mean_dims) if mean_dims else seasonality_data
        all_components['Seasonality'] = np.asarray(mean_contrib)

        # Get full model predictions (already in log space)
        preds = mmm.idata.posterior_predictive[mmm.output_var]
        predictions_mean = preds.mean(dim=["chain", "draw"]).values

        # Get HDI for uncertainty bands (already in log space)
        preds_hdi = az.hdi(preds, hdi_prob=0.94)
        predictions_hdi_lower = preds_hdi.sel(hdi="lower")[mmm.output_var].values
        predictions_hdi_upper = preds_hdi.sel(hdi="higher")[mmm.output_var].values

        # Get actual observed target data and convert to log space
        y_observed = np.log(data[cols.target].values)

        ylabel = 'Value (Log Scale)'
        title_suffix = '(Log Scale)'
        reference_line = 0

    else:
        # Keep everything in normal space
        # Add channels (already in normal space from get_individual_contributions)
        for channel_name, channel_data in contributions['channels'].items():
            mean_dims = [d for d in getattr(channel_data, 'dims', []) if d != 'date']
            mean_contrib = channel_data.mean(dim=mean_dims) if mean_dims else channel_data
            all_components[f'Channel: {channel_name}'] = np.asarray(mean_contrib)

        # Add controls (already in normal space)
        for control_name, control_data in contributions['controls'].items():
            mean_dims = [d for d in getattr(control_data, 'dims', []) if d != 'date']
            mean_contrib = control_data.mean(dim=mean_dims) if mean_dims else control_data
            all_components[f'Control: {control_name}'] = np.asarray(mean_contrib)

        # Add intercept (already in normal space)
        intercept_data = contributions['intercept']
        mean_dims = [d for d in getattr(intercept_data, 'dims', []) if d != 'date']
        mean_contrib = intercept_data.mean(dim=mean_dims) if mean_dims else intercept_data
        all_components['Intercept'] = np.asarray(mean_contrib)

        # Add seasonality (already in normal space)
        seasonality_data = contributions['seasonality']
        mean_dims = [d for d in getattr(seasonality_data, 'dims', []) if d != 'date']
        mean_contrib = seasonality_data.mean(dim=mean_dims) if mean_dims else seasonality_data
        all_components['Seasonality'] = np.asarray(mean_contrib)

        # Get full model predictions from posterior_predictive
        # Check if we have original_scale version or need to transform
        if 'y_original_scale' in mmm.idata.posterior_predictive:
            # Already in original scale, use directly
            preds = mmm.idata.posterior_predictive['y_original_scale']

            # Filter to specific geo if this is multidimensional data
            if cols.dimension in preds.dims and cols.dimension in data.columns:
                # Get the geo value from the filtered data (all rows should have same geo)
                dimension_value = data[cols.dimension].iloc[0]
                preds = preds.sel({cols.dimension: dimension_value})

            predictions_mean = np.asarray(preds.mean(dim=["chain", "draw"])).flatten()

            # Get HDI for uncertainty bands (already in original scale)
            preds_hdi = az.hdi(preds, hdi_prob=0.94)
            # az.hdi returns a Dataset, convert to DataArray
            if hasattr(preds_hdi, 'to_dataarray'):
                preds_hdi = preds_hdi.to_dataarray().squeeze()
            # Ensure we extract as numpy arrays with proper flattening
            predictions_hdi_lower = np.asarray(preds_hdi.sel(hdi="lower")).flatten()
            predictions_hdi_upper = np.asarray(preds_hdi.sel(hdi="higher")).flatten()
        else:
            # Need to transform from scaled space
            preds = mmm.idata.posterior_predictive[mmm.output_var]

            # Calculate mean in log/scaled space FIRST
            predictions_mean = preds.mean(dim=["chain", "draw"]).values

            # Apply transformations in CORRECT order
            # Step 1: Revert log transformation if applicable
            if hasattr(mmm, 'log_transform_vars') and 'target' in mmm.log_transform_vars:
                predictions_mean = np.exp(predictions_mean)

            # Step 2: Revert scaling if applicable
            if hasattr(mmm, 'target_scale'):
                predictions_mean = predictions_mean * mmm.target_scale

            # Get HDI for uncertainty bands with proper transformation
            preds_hdi = az.hdi(preds, hdi_prob=0.94)
            predictions_hdi_lower = preds_hdi.sel(hdi="lower")[mmm.output_var].values
            predictions_hdi_upper = preds_hdi.sel(hdi="higher")[mmm.output_var].values

            # Apply same transformations to HDI bounds
            if hasattr(mmm, 'log_transform_vars') and 'target' in mmm.log_transform_vars:
                predictions_hdi_lower = np.exp(predictions_hdi_lower)
                predictions_hdi_upper = np.exp(predictions_hdi_upper)

            if hasattr(mmm, 'target_scale'):
                predictions_hdi_lower = predictions_hdi_lower * mmm.target_scale
            predictions_hdi_upper = predictions_hdi_upper * mmm.target_scale

        # Get actual observed target data (already in normal space)
        y_observed = data[cols.target].values

        ylabel = 'Value (Normal Scale)'
        title_suffix = '(Normal Scale)'
        reference_line = 0

    # === PLOT 1: All contributions together ===
    fig_combined, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_components)))

    for idx, (component_name, component_values) in enumerate(all_components.items()):
        ax.plot(dates, component_values, label=component_name, linewidth=2, color=colors[idx])

    # Add actual observed target data
    ax.plot(dates, y_observed, label='Actual Target (Y)', linewidth=2.5,
            color='red', linestyle='--', alpha=0.9, zorder=11)

    # Add full model predictions
    ax.plot(dates, predictions_mean, label='Full Model Prediction', linewidth=2.5,
            color='black', linestyle='-', alpha=0.8, zorder=10)
    ax.fill_between(dates, predictions_hdi_lower, predictions_hdi_upper,
                    alpha=0.2, color='gray', label='94% HDI', zorder=5)

    ax.axhline(reference_line, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

    # Build title: use custom if provided, otherwise use default
    if custom_titles and 'combined' in custom_titles:
        title = custom_titles['combined']
    else:
        title = f'All Component Contributions Over Time {title_suffix}'

    # Add dimension label if provided
    if dimension_label:
        title = f'{title} - {dimension_label.upper()}'

    ax.set_title(title, fontsize=14, fontweight='bold')

    ax.legend(loc='best', framealpha=0.9)
    ax.grid(alpha=0.3)

    try:
        fig_combined.autofmt_xdate()
    except Exception:
        pass

    plt.tight_layout()

    # === PLOT 2: Individual subplots ===
    # Add one extra subplot for predictions
    n_base_components = len(all_components)
    n_total_components = n_base_components + 1
    n_cols = 2
    n_rows = int(np.ceil(n_total_components / n_cols))

    fig_individual, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))

    # Flatten axes array for easier iteration
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()

    # Plot all component contributions
    for idx, (component_name, component_values) in enumerate(all_components.items()):
        ax = axes_flat[idx]
        ax.plot(dates, component_values, linewidth=2, color=colors[idx])
        ax.axhline(reference_line, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(component_name, fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)

        try:
            ax.tick_params(axis='x', rotation=45)
        except Exception:
            pass

    # Add full model predictions subplot
    ax = axes_flat[n_base_components]
    ax.plot(dates, predictions_mean, linewidth=2.5, color='black', label='Mean Prediction')
    ax.fill_between(dates, predictions_hdi_lower, predictions_hdi_upper,
                    alpha=0.3, color='gray', label='94% HDI')
    ax.set_xlabel('Date', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title('Full Model Prediction', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

    try:
        ax.tick_params(axis='x', rotation=45)
    except Exception:
        pass

    # Hide unused subplots
    for idx in range(n_total_components, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Build suptitle: use custom if provided, otherwise use default
    if custom_titles and 'individual' in custom_titles:
        suptitle = custom_titles['individual']
    else:
        suptitle = f'Individual Component Contributions {title_suffix}'

    # Add dimension label if provided
    if dimension_label:
        suptitle = f'{suptitle} - {dimension_label.upper()}'

    plt.suptitle(suptitle, fontsize=16, fontweight='bold', y=1.00)

    plt.tight_layout()

    return fig_combined, fig_individual



def plot_roas_analysis(roas_results, contributions, data, confidence_intervals=(0.85, 0.90), baseline_components=None, dimension_label=None, custom_titles=None):
    """
    Visualize ROAS analysis results with comprehensive visualizations.

    Creates a complete suite of 7 visualizations showing adjusted contributions, ROAS metrics,
    distribution analysis, waterfall chart, pie chart, time series, and ranking analysis.

    Parameters
    ----------
    roas_results : dict
        Results from calculate_roas()
    contributions : dict
        Original contributions dictionary
    data : pd.DataFrame
        Original data with dates
    confidence_intervals : tuple of float, default=(0.85, 0.90)
        Two confidence levels to display in statistics printout.
        For example, (0.85, 0.90) will show 85% CI (7.5%-92.5%) and 90% CI (5%-95%).
        Default shows both 85% and 90% confidence intervals.
    baseline_components : list, optional
        List of component names to include in the baseline.
        If None, defaults to ['intercept', 'seasonality', 'trend'].
        You can customize this to include other components (e.g., specific control variables).
    dimension_label : str, optional
        Label for the dimension (e.g., 'geo_a', 'geo_b'). When provided, this label
        is added to plot titles to differentiate between different dimensions.
        Automatically injected by decorators for multidimensional models.
    custom_titles : dict, optional
        Dictionary of custom titles for the 7 figures. Keys can be:
        - 'fig1': Adjusted contributions time series + ROAS bar chart
        - 'fig2': Spend vs contribution comparison bar chart
        - 'fig3': ROAS boxplot distribution
        - 'fig4': Waterfall contributions plot
        - 'fig5': Contribution pie chart
        - 'fig6': Contribution over time (stacked area)
        - 'fig7': Contribution rank over time
        If a key is not provided, the default title is used.
        Dimension label will still be appended if provided.

    Returns
    -------
    tuple
        (fig1, fig2, fig3, fig4, fig5, fig6, fig7, aggregated_contributions) - Seven matplotlib figure objects and aggregated contributions dict:
        - fig1: Adjusted contributions time series + ROAS bar chart
        - fig2: Spend vs contribution comparison bar chart
        - fig3: ROAS boxplot distribution with confidence intervals
        - fig4: Waterfall contributions plot (horizontal bars)
        - fig5: Contribution pie chart (Baseline vs Channels vs Controls)
        - fig6: Contribution over time (stacked area chart)
        - fig7: Contribution rank over time (bump chart)
        - aggregated_contributions: Dictionary with aggregated contribution values

    Notes
    -----
    - Baseline includes intercept + seasonality (+ trend if available)
    - All visualizations use adjusted contributions that sum to actual sales
    - Time series plots preserve temporal relationships
    - Rank chart shows relative importance of each component over time
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from b2b_src.column_config import get_default_column_config

    # Get column configuration
    cols = get_default_column_config()

    # Helper function to build titles with custom title and dimension label support
    def _build_title(default_title: str, custom_key: str = None) -> str:
        """
        Build title with optional custom title and dimension label.

        Parameters
        ----------
        default_title : str
            Default title to use if no custom title provided
        custom_key : str, optional
            Key to look up in custom_titles dict (e.g., 'fig1', 'fig2')

        Returns
        -------
        str
            Final title with custom title (if provided) and dimension label (if provided)
        """
        # Use custom title if provided, otherwise use default
        if custom_titles and custom_key and custom_key in custom_titles:
            base_title = custom_titles[custom_key]
        else:
            base_title = default_title

        # Add dimension label if provided
        if dimension_label:
            return f'{base_title} - {dimension_label.upper()}'
        return base_title

    # Validate confidence_intervals
    if not isinstance(confidence_intervals, tuple) or len(confidence_intervals) != 2:
        raise ValueError("confidence_intervals must be a tuple of exactly 2 values")

    for ci in confidence_intervals:
        if not 0 < ci < 1:
            raise ValueError(f"Confidence interval {ci} must be between 0 and 1")

    # Sort to ensure smaller CI is first
    ci1, ci2 = sorted(confidence_intervals)

    # Get dates for the period
    period_start = roas_results['period']['start']
    period_end = roas_results['period']['end']
    dates = contributions['dates'][period_start:period_end+1]

    # Get actual target values using configured target column
    y_actual = data.iloc[period_start:period_end+1][cols.target].values

    # Get cumulative period info
    cumulative_period = roas_results.get('cumulative_period', {})
    start_date = cumulative_period.get('start_date', 'N/A')
    end_date = cumulative_period.get('end_date', 'N/A')
    days = cumulative_period.get('days', 'N/A')

    # Format dates for display
    if hasattr(start_date, 'strftime'):
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
    else:
        start_date_str = str(start_date)
        end_date_str = str(end_date)

    # ============================================================================
    # Print cumulative period and ROAS statistics
    # ============================================================================
    print("=" * 80)
    print(f"ROAS Analysis - Cumulative Period: {start_date_str} to {end_date_str} ({days} days)")
    print("=" * 80)
    print()

    # Calculate and print ROAS statistics from samples
    if 'roas_samples' in roas_results:
        # Calculate percentiles for both confidence intervals
        lower1 = (1 - ci1) / 2 * 100  # e.g., 7.5% for 85% CI
        upper1 = (1 + ci1) / 2 * 100  # e.g., 92.5% for 85% CI
        lower2 = (1 - ci2) / 2 * 100  # e.g., 5% for 90% CI
        upper2 = (1 + ci2) / 2 * 100  # e.g., 95% for 90% CI

        print("Channel ROAS Statistics:")
        print("-" * 80)

        for channel, roas_samples in roas_results['roas_samples'].items():
            # Calculate percentiles for both CIs
            ci1_lower = np.percentile(roas_samples, lower1)
            ci1_upper = np.percentile(roas_samples, upper1)
            ci2_lower = np.percentile(roas_samples, lower2)
            ci2_upper = np.percentile(roas_samples, upper2)
            median = np.percentile(roas_samples, 50)
            mean = np.mean(roas_samples)

            print(f"  {channel}:")
            print(f"    - Lower bound {int(ci2*100)}% CI ({lower2:.1f}%): {ci2_lower:.2f}")
            print(f"    - Lower bound {int(ci1*100)}% CI ({lower1:.1f}%): {ci1_lower:.2f}")
            print(f"    - Median (50%): {median:.2f}")
            print(f"    - Mean: {mean:.2f}")
            print(f"    - Upper bound {int(ci1*100)}% CI ({upper1:.1f}%): {ci1_upper:.2f}")
            print(f"    - Upper bound {int(ci2*100)}% CI ({upper2:.1f}%): {ci2_upper:.2f}")
            print()

    print("=" * 80)
    print()

    # ============================================================================
    # FIGURE 1: Adjusted Contributions + ROAS
    # ============================================================================
    fig1, axes = plt.subplots(2, 1, figsize=(14, 10))

    # ============================================================================
    # Plot 1: ADJUSTED Contributions Over Time - ALL COMPONENTS (with actual Y)
    # ============================================================================
    ax = axes[0]

    # Define colors for different component types
    channel_colors = plt.cm.Set1(np.arange(10))

    # Plot adjusted contributions for ALL components
    color_idx = 0
    for component_name, adj_contrib in roas_results['adjusted_contributions_all'].items():
        # Parse component name for better labeling
        if component_name.startswith('channel_'):
            label = component_name.replace('channel_', '') + ' (channel)'
            color = channel_colors[color_idx % len(channel_colors)]
            color_idx += 1
            linewidth = 2
            alpha = 0.8
        elif component_name.startswith('control_'):
            label = component_name.replace('control_', '') + ' (control)'
            color = 'gray'
            linewidth = 1.5
            alpha = 0.5
        elif component_name == 'intercept':
            label = 'Intercept'
            color = 'purple'
            linewidth = 1.5
            alpha = 0.6
        elif component_name == 'seasonality':
            label = 'Seasonality'
            color = 'orange'
            linewidth = 1.5
            alpha = 0.6
        else:
            label = component_name
            color = 'black'
            linewidth = 1
            alpha = 0.5

        ax.plot(dates, adj_contrib, label=label, linewidth=linewidth, alpha=alpha, color=color)

    # Calculate sum of ALL adjusted contributions
    sum_adjusted_all = np.zeros_like(y_actual)
    for adj_contrib in roas_results['adjusted_contributions_all'].values():
        sum_adjusted_all += adj_contrib

    # Plot sum of ALL adjusted contributions
    ax.plot(dates, sum_adjusted_all, label='Sum of ALL Adjusted',
            linewidth=3, color='green', linestyle='-', alpha=0.9, zorder=10)

    # Plot actual target Y
    ax.plot(dates, y_actual, label='Actual Y (target)', linewidth=3,
            color='black', linestyle='--', alpha=0.9, zorder=10)

    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Contribution', fontsize=11)
    ax.set_title(_build_title('ADJUSTED Contributions (ALL COMPONENTS) vs Actual Target (Y)\n(Notice: Green line (sum of adjusted) = Black dashed line (Actual Y))', 'fig1_plot1'),
                 fontsize=13, fontweight='bold')
    ax.legend(loc='best', ncol=2, fontsize=9)
    ax.grid(alpha=0.3)

    # ============================================================================
    # Plot 2: ROAS Bar Chart
    # ============================================================================
    ax = axes[1]
    channels = list(roas_results['roas'].keys())
    roas_values = [roas_results['roas'][ch] for ch in channels]
    colors = plt.cm.Set2(np.arange(len(channels)))
    bars = ax.bar(channels, roas_values, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, value in zip(bars, roas_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Channel', fontsize=11)
    ax.set_ylabel('ROAS ($ revenue per $ spent)', fontsize=11)
    ax.set_title(_build_title('Return on Ad Spend (ROAS) by Channel', 'fig1_plot2'), fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # Add figure-level suptitle if dimension_label is provided
    if dimension_label:
        fig1.suptitle(f'ROAS Analysis - {dimension_label.upper()}', fontsize=16, fontweight='bold', y=1.00)

    plt.tight_layout()

    # ============================================================================
    # FIGURE 2: Spend vs Contribution Comparison
    # ============================================================================
    fig2, ax = plt.subplots(figsize=(10, 6))

    channels = list(roas_results['roas'].keys())
    x = np.arange(len(channels))
    width = 0.35

    spend_values = [roas_results['spend'][ch] for ch in channels]
    contrib_values = [np.sum(roas_results['adjusted_contributions_channels'][ch]) for ch in channels]

    bars1 = ax.bar(x - width/2, spend_values, width, label='Total Spend',
                   color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, contrib_values, width, label='Total Adjusted Contribution',
                   color='coral', edgecolor='black')

    ax.set_xlabel('Channel', fontsize=12)
    ax.set_ylabel('Amount ($)', fontsize=12)
    ax.set_title(_build_title('Total Spend vs Total Adjusted Contribution by Channel', 'fig2'),
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # ============================================================================
    # FIGURE 3: ROAS Boxplot (NEW)
    # ============================================================================
    if 'roas_samples' in roas_results:
        fig3, ax = plt.subplots(figsize=(12, 7))

        channels = list(roas_results['roas_samples'].keys())
        roas_samples_list = [roas_results['roas_samples'][ch] for ch in channels]

        # Create boxplot
        bp = ax.boxplot(roas_samples_list, labels=channels, patch_artist=True,
                        showmeans=True, meanline=False,
                        medianprops=dict(color='red', linewidth=2),
                        meanprops=dict(marker='D', markerfacecolor='green',
                                      markersize=8, markeredgecolor='darkgreen'),
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        flierprops=dict(marker='o', markerfacecolor='gray',
                                       markersize=5, alpha=0.5))

        ax.set_xlabel('Channel', fontsize=12, fontweight='bold')
        ax.set_ylabel('ROAS ($ revenue per $ spent)', fontsize=12, fontweight='bold')
        ax.set_title(_build_title(f'ROAS Distribution by Channel\nCumulative Period: {start_date_str} to {end_date_str} ({days} days)', 'fig3'),
                     fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')

        # Add legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, label='Median'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='green',
                   markersize=8, markeredgecolor='darkgreen', label='Mean'),
            Patch(facecolor='lightblue', alpha=0.7, edgecolor='black', label='IQR (25%-75%)'),
            Line2D([0], [0], color='black', linewidth=1.5, label=f'Whiskers ({int(ci2*100)}% CI: {lower2:.1f}%-{upper2:.1f}%)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.tight_layout()
    else:
        # If no samples available, create empty figure
        fig3 = None

    # ============================================================================
    # FIGURE 4: Waterfall Contributions Plot (INTEGRATED)
    # ============================================================================
    waterfall_title = _build_title(f"Component Contributions Waterfall\nCumulative Period: {start_date_str} to {end_date_str} ({days} days)", 'fig4')
    fig4, ax4, aggregated_contributions = plot_waterfall_contributions(
        roas_results=roas_results,
        data=data,
        title=waterfall_title,
        baseline_components=baseline_components,
        dimension_label=dimension_label
    )

    # ============================================================================
    # FIGURE 5: Contribution Pie Chart
    # ============================================================================
    fig5, ax5 = plt.subplots(figsize=(10, 8))

    # Calculate total baseline (intercept + seasonality + trend if available)
    baseline_total = aggregated_contributions['baseline']

    # Calculate total channels contribution
    channels_total = sum(aggregated_contributions['channels'].values())

    # Calculate total controls contribution
    controls_total = sum(aggregated_contributions['controls'].values())

    # Prepare data for pie chart: Baseline vs All Channels vs All Controls
    pie_labels = []
    pie_values = []
    pie_colors = []

    # Add baseline
    if abs(baseline_total) > 0:
        pie_labels.append('Baseline\n(Intercept + Seasonality)')
        pie_values.append(abs(baseline_total))
        pie_colors.append('#FDD835')  # Yellow

    # Add channels
    if abs(channels_total) > 0:
        pie_labels.append('All Channels')
        pie_values.append(abs(channels_total))
        pie_colors.append('#1976D2')  # Blue

    # Add controls
    if abs(controls_total) > 0:
        pie_labels.append('All Controls')
        pie_values.append(abs(controls_total))
        pie_colors.append('#FF7043')  # Orange

    # Create pie chart
    wedges, texts, autotexts = ax5.pie(
        pie_values,
        labels=pie_labels,
        colors=pie_colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )

    # Make percentage text white
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)

    ax5.set_title(_build_title(f'Total Contribution Breakdown\nCumulative Period: {start_date_str} to {end_date_str}', 'fig5'),
                  fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    # ============================================================================
    # FIGURE 6: Contribution Over Time (Stacked Area Chart)
    # ============================================================================
    fig6, ax6 = plt.subplots(figsize=(14, 8))

    # Get time series data from adjusted contributions
    # Stack order: Baseline at bottom, then controls, then channels
    components_to_plot = []
    component_labels = []
    component_colors = []

    # Add baseline first (bottom of stack)
    # Use default baseline components if not specified
    if baseline_components is None:
        baseline_components_list = ['intercept', 'seasonality', 'trend']
    else:
        baseline_components_list = baseline_components

    baseline_ts = np.zeros(len(dates))
    for comp in baseline_components_list:
        if comp in roas_results['adjusted_contributions_all']:
            baseline_ts += roas_results['adjusted_contributions_all'][comp]

    if np.any(baseline_ts != 0):
        components_to_plot.append(baseline_ts)
        component_labels.append('Baseline')
        component_colors.append('#FDD835')  # Yellow

    # Add controls
    for control_name in sorted(roas_results['adjusted_contributions_all'].keys()):
        if control_name.startswith('control_'):
            control_label = control_name.replace('control_', '')
            components_to_plot.append(roas_results['adjusted_contributions_all'][control_name])
            component_labels.append(f'{control_label} (control)')
            component_colors.append('#FF7043')  # Orange shades

    # Add channels - use Set1 for better contrast
    channel_colors_map = plt.cm.Set1(np.arange(9))  # Set1 has more vivid, contrasting colors
    for idx, channel_name in enumerate(sorted(roas_results['adjusted_contributions_all'].keys())):
        if channel_name.startswith('channel_'):
            channel_label = channel_name.replace('channel_', '')
            components_to_plot.append(roas_results['adjusted_contributions_all'][channel_name])
            component_labels.append(channel_label)
            component_colors.append(channel_colors_map[idx % len(channel_colors_map)])

    # Create stacked area chart
    ax6.stackplot(dates, *components_to_plot, labels=component_labels,
                  colors=component_colors, alpha=0.8, edgecolor='white', linewidth=0.5)

    ax6.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Contribution', fontsize=12, fontweight='bold')
    ax6.set_title(_build_title(f'Component Contributions Over Time\nCumulative Period: {start_date_str} to {end_date_str}', 'fig6'),
                  fontsize=14, fontweight='bold', pad=20)
    ax6.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, framealpha=0.9)
    ax6.grid(alpha=0.3, axis='y')

    plt.tight_layout()

    # ============================================================================
    # FIGURE 7: Contribution Rank Over Time (Bump Chart) - Quarterly Aggregation
    # ============================================================================
    fig7, ax7 = plt.subplots(figsize=(14, 8))

    # Calculate ranks for each time period
    # Combine all components into a dataframe-like structure
    all_components_ts = {}

    # Add baseline
    if np.any(baseline_ts != 0):
        all_components_ts['Baseline'] = baseline_ts

    # Add controls
    for control_name in roas_results['adjusted_contributions_all'].keys():
        if control_name.startswith('control_'):
            control_label = control_name.replace('control_', '')
            all_components_ts[f'{control_label} (control)'] = roas_results['adjusted_contributions_all'][control_name]

    # Add channels
    for channel_name in roas_results['adjusted_contributions_all'].keys():
        if channel_name.startswith('channel_'):
            channel_label = channel_name.replace('channel_', '')
            all_components_ts[channel_label] = roas_results['adjusted_contributions_all'][channel_name]

    # Aggregate contributions by quarter
    # Create a pandas DataFrame to facilitate quarterly aggregation
    df_contrib = pd.DataFrame(all_components_ts, index=dates)

    # Resample to quarterly (Q - quarter end frequency)
    df_contrib_quarterly = df_contrib.resample('Q').sum()
    quarterly_dates = df_contrib_quarterly.index

    # Calculate ranks at each quarterly time point
    ranks_over_time = {}
    for component_name in all_components_ts.keys():
        ranks_over_time[component_name] = []

    for t in range(len(quarterly_dates)):
        # Get aggregated values for this quarter
        values_at_t = {name: df_contrib_quarterly.iloc[t][name] for name in all_components_ts.keys()}

        # Rank them (higher value = lower rank number, i.e., rank 1 is best)
        sorted_components = sorted(values_at_t.items(), key=lambda x: x[1], reverse=True)

        # Assign ranks
        for rank, (component_name, _) in enumerate(sorted_components, start=1):
            ranks_over_time[component_name].append(rank)

    # Plot bump chart - use Set1 colormap for better contrast
    channel_colors_map_bump = plt.cm.Set1(np.arange(9))  # Set1 has more vivid, contrasting colors
    colors_bump = {}
    colors_bump['Baseline'] = '#FDD835'

    # Assign colors to controls
    for component_name in ranks_over_time.keys():
        if '(control)' in component_name:
            colors_bump[component_name] = '#FF7043'

    # Assign colors to channels
    channel_idx = 0
    for component_name in ranks_over_time.keys():
        if component_name not in colors_bump:
            colors_bump[component_name] = channel_colors_map_bump[channel_idx % len(channel_colors_map_bump)]
            channel_idx += 1

    # Plot lines
    for component_name, ranks in ranks_over_time.items():
        ax7.plot(quarterly_dates, ranks, marker='o', linewidth=2, markersize=8,
                label=component_name, color=colors_bump.get(component_name, 'gray'))

    ax7.set_xlabel('Date (Quarterly)', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Contribution Rank', fontsize=12, fontweight='bold')
    ax7.set_title(_build_title(f'Component Contribution Rank Over Time (Quarterly Aggregation)\nCumulative Period: {start_date_str} to {end_date_str}', 'fig7'),
                  fontsize=14, fontweight='bold', pad=20)
    ax7.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, framealpha=0.9)
    ax7.grid(alpha=0.3, axis='both')
    ax7.invert_yaxis()  # Rank 1 should be at the top

    # Format y-axis to show integer ranks
    from matplotlib.ticker import MaxNLocator
    ax7.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7, aggregated_contributions


def plot_roas_time_series(
    roas_results,
    confidence_levels=(0.95, 0.99),
    show_mean_line=True,
    date_range=None,
    figsize=(14, 6),
    dimension_label=None,
    custom_title=None
):
    """
    Visualize ROAS time series with confidence intervals.

    Creates individual plots for each channel showing ROAS over time with multiple
    confidence intervals and the central tendency (mean).

    Parameters
    ----------
    roas_results : dict
        Dictionary returned by calculate_roas_time_series() containing:
        - 'roas_time_series': dict with ROAS arrays {channel: array(n_samples, n_dates)}
        - 'dates': array of dates
        - 'spend': dict with spend arrays
        - 'channel_cols': list of channel names
    confidence_levels : tuple, default=(0.95, 0.99)
        Tuple of two confidence levels to display (e.g., (0.90, 0.95) or (0.85, 0.95))
    show_mean_line : bool, default=True
        If True, show the mean ROAS line
    date_range : tuple, optional
        Tuple of (start_date, end_date) to filter the visualization.
        Dates should be in a format comparable to the dates in roas_results.
        If None, shows all dates.
    figsize : tuple, default=(14, 6)
        Figure size (width, height) in inches for each individual plot
    dimension_label : str, optional
        Label for the dimension (e.g., 'geo_a', 'geo_b'). When provided, this label
        is added to plot titles to differentiate between different dimensions.
        Automatically injected by decorators for multidimensional models.
    custom_title : str, optional
        Custom title template. If provided, replaces "ROAS Time Series - Channel: {channel}".
        The {channel} placeholder will be replaced with the actual channel name.
        Dimension label will still be appended if provided.

    Returns
    -------
    dict
        Dictionary containing:
        - 'figures': dict with matplotlib Figures {channel_name: Figure}

    Notes
    -----
    - Confidence intervals are calculated using percentiles from posterior samples
    - NaN values (from zero-spend periods) are automatically handled by matplotlib
    - Each channel gets its own independent figure
    - Confidence levels should be provided as a tuple of exactly 2 values
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Validate confidence_levels
    if not isinstance(confidence_levels, tuple) or len(confidence_levels) != 2:
        raise ValueError("confidence_levels must be a tuple of exactly 2 values (e.g., (0.90, 0.95))")

    # Extract data from results
    roas_time_series = roas_results['roas_time_series']
    dates = roas_results['dates']
    channel_cols = roas_results['channel_cols']

    # Convert dates to matplotlib format if needed
    if hasattr(dates[0], 'to_pydatetime'):
        dates_plot = np.array([d.to_pydatetime() for d in dates])
    else:
        dates_plot = np.array(dates)

    # Filter by date range if specified
    if date_range is not None:
        start_date, end_date = date_range
        mask = (dates_plot >= start_date) & (dates_plot <= end_date)
        dates_plot = dates_plot[mask]
        date_indices = np.where(mask)[0]
    else:
        date_indices = np.arange(len(dates_plot))

    # Sort confidence levels (highest to lowest for proper layering)
    confidence_levels_sorted = sorted(confidence_levels, reverse=True)

    # Color palette for channels
    colors = plt.cm.tab10(np.linspace(0, 1, len(channel_cols)))

    # Dictionary to store figures
    figures = {}

    # ===== CREATE INDIVIDUAL PLOT FOR EACH CHANNEL =====
    for idx, channel in enumerate(channel_cols):
        # Create new figure for this channel
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        # Get ROAS samples for this channel
        roas_samples = roas_time_series[channel]  # shape: (n_samples, n_dates)

        # Filter by date range if specified
        roas_samples_filtered = roas_samples[:, date_indices]

        color = colors[idx]

        # Calculate statistics for each time period
        mean_roas = np.nanmean(roas_samples_filtered, axis=0)

        # Plot confidence intervals (largest to smallest)
        for ci_idx, ci_level in enumerate(confidence_levels_sorted):
            alpha_val = (1 - ci_level) / 2 * 100  # Convert to percentile
            lower = np.nanpercentile(roas_samples_filtered, alpha_val, axis=0)
            upper = np.nanpercentile(roas_samples_filtered, 100 - alpha_val, axis=0)

            # Alpha for fill: lighter for wider intervals
            fill_alpha = 0.2 if ci_idx == 0 else 0.3

            ax.fill_between(
                dates_plot,
                lower,
                upper,
                color=color,
                alpha=fill_alpha,
                label=f'{int(ci_level*100)}% CI'
            )

        # Plot mean line
        if show_mean_line:
            ax.plot(
                dates_plot,
                mean_roas,
                color=color,
                linewidth=2,
                label='Mean ROAS'
            )

        # Formatting
        ax.set_xlabel('Date', fontsize=11, fontweight='bold')
        ax.set_ylabel('ROAS', fontsize=11, fontweight='bold')

        # Build title with optional custom title and dimension label
        if custom_title:
            # Replace {channel} placeholder in custom title
            base_title = custom_title.replace('{channel}', channel)
        else:
            base_title = f'ROAS Time Series - Channel: {channel}'

        # Add dimension label if provided
        if dimension_label:
            title = f'{base_title} - {dimension_label.upper()}'
        else:
            title = base_title
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Store figure
        figures[channel] = fig

    return {
        'figures': figures
    }



import numpy as np
import matplotlib.pyplot as plt
import arviz as az

def plot_prior_predictive_with_log_reversion(
    mmm,
    original_scale: bool = True,
    revert_log: bool = True,
    figsize: tuple = (12, 6)
):
    """
    Plot PRIOR predictive with proper log transformation handling.

    Must be called AFTER mmm.sample_prior_predictive(X, y, samples=N)

    Parameters
    ----------
    mmm : MMM object
        The MMM model instance (after sample_prior_predictive has been called)
    original_scale : bool
        If True, multiply by target_scale to revert scaling
    revert_log : bool
        If True, apply exp(x) to revert log transformation
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """

    # Get the prior predictive data from idata
    if not hasattr(mmm, 'idata') or 'prior_predictive' not in mmm.idata:
        raise ValueError("Must call mmm.sample_prior_predictive(X, y, samples=N) first!")

    # Get predictions from idata (in log space if log transformation was applied)
    preds = mmm.idata.prior_predictive[mmm.output_var]

    print(f"\n{'='*70}")
    print(f"PRIOR PREDICTIVE TRANSFORMATION DEBUG")
    print(f"{'='*70}")
    print(f"Predictions shape: {preds.shape}")
    print(f"Predictions sample (first 5): {preds.values.flat[:5]}")
    print(f"Predictions mean: {preds.mean().values:.2f}")

    # Step 1: Revert log transformation FIRST (if requested)
    if revert_log:
        if hasattr(mmm, 'log_transform_vars') and 'target' in mmm.log_transform_vars:
            print(f"\n✓ Reverting log transformation: exp(x)")
            print(f"  Model has log_transform_vars: {mmm.log_transform_vars}")
            # Calculate mean in log space, then exp
            predictions_mean = np.exp(preds.mean(dim=["chain", "draw"]).values)
            # Calculate HDI in log space, then exp
            preds_hdi = az.hdi(preds, hdi_prob=0.94)
            hdi_94_lower = np.exp(preds_hdi.sel(hdi="lower")[mmm.output_var].values)
            hdi_94_upper = np.exp(preds_hdi.sel(hdi="higher")[mmm.output_var].values)

            preds_hdi_50 = az.hdi(preds, hdi_prob=0.50)
            hdi_50_lower = np.exp(preds_hdi_50.sel(hdi="lower")[mmm.output_var].values)
            hdi_50_upper = np.exp(preds_hdi_50.sel(hdi="higher")[mmm.output_var].values)

            print(f"  After log reversion - mean: {predictions_mean.mean():.2f}")
        else:
            print(f"\n✗ Log transformation NOT applied")
            # No log transformation, just use values directly
            predictions_mean = preds.mean(dim=["chain", "draw"]).values
            preds_hdi = az.hdi(preds, hdi_prob=0.94)
            hdi_94_lower = preds_hdi.sel(hdi="lower")[mmm.output_var].values
            hdi_94_upper = preds_hdi.sel(hdi="higher")[mmm.output_var].values

            preds_hdi_50 = az.hdi(preds, hdi_prob=0.50)
            hdi_50_lower = preds_hdi_50.sel(hdi="lower")[mmm.output_var].values
            hdi_50_upper = preds_hdi_50.sel(hdi="higher")[mmm.output_var].values
    else:
        print(f"\n○ Log reversion SKIPPED (revert_log=False)")
        # Keep in log space
        predictions_mean = preds.mean(dim=["chain", "draw"]).values
        preds_hdi = az.hdi(preds, hdi_prob=0.94)
        hdi_94_lower = preds_hdi.sel(hdi="lower")[mmm.output_var].values
        hdi_94_upper = preds_hdi.sel(hdi="higher")[mmm.output_var].values

        preds_hdi_50 = az.hdi(preds, hdi_prob=0.50)
        hdi_50_lower = preds_hdi_50.sel(hdi="lower")[mmm.output_var].values
        hdi_50_upper = preds_hdi_50.sel(hdi="higher")[mmm.output_var].values

    # Step 2: Revert scaling SECOND (if requested)
    if original_scale:
        if hasattr(mmm, 'target_scale'):
            print(f"\n✓ Reverting scaling: multiply by target_scale={mmm.target_scale}")
            predictions_mean = predictions_mean * mmm.target_scale
            hdi_94_lower = hdi_94_lower * mmm.target_scale
            hdi_94_upper = hdi_94_upper * mmm.target_scale
            hdi_50_lower = hdi_50_lower * mmm.target_scale
            hdi_50_upper = hdi_50_upper * mmm.target_scale
            print(f"  After scale reversion - mean: {predictions_mean.mean():.2f}")
        else:
            print(f"\n✗ Scaling NOT applied (target_scale attribute not found)")
    else:
        print(f"\n○ Scale reversion SKIPPED (original_scale=False)")

    print(f"{'='*70}\n")

    # Get dates
    dates = mmm.idata.prior_predictive['date'].values

    # Get observed data
    y_observed = np.asarray(mmm.y)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot 94% HDI (lighter)
    ax.fill_between(
        dates,
        hdi_94_lower,
        hdi_94_upper,
        alpha=0.2,
        color='C0',
        label='94% HDI'
    )

    # Plot 50% HDI (darker)
    ax.fill_between(
        dates,
        hdi_50_lower,
        hdi_50_upper,
        alpha=0.4,
        color='C0',
        label='50% HDI'
    )

    # Plot mean prediction
    ax.plot(dates, predictions_mean, color='blue', linewidth=2, label='Mean Prediction')

    # Plot observed data
    ax.plot(dates, y_observed, color='black', linewidth=2, label='Observed', linestyle='--')

    # Labels and formatting
    ax.set_xlabel('Date')
    ax.set_ylabel('y')
    title = 'Prior Predictive Check'
    if revert_log and original_scale:
        title += ' (Original Scale)'
    elif original_scale:
        title += ' (Scaled Only)'
    elif revert_log:
        title += ' (Log Reverted Only)'
    else:
        title += ' (Model Space)'
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Print summary statistics
    print(f"PLOT STATISTICS:")
    print(f"  Observed Y - mean: {y_observed.mean():.2f}, range: [{y_observed.min():.2f}, {y_observed.max():.2f}]")
    print(f"  Predicted Y - mean: {predictions_mean.mean():.2f}, range: [{predictions_mean.min():.2f}, {predictions_mean.max():.2f}]")
    print(f"  94% HDI - lower mean: {hdi_94_lower.mean():.2f}, upper mean: {hdi_94_upper.mean():.2f}")
    print(f"  50% HDI - lower mean: {hdi_50_lower.mean():.2f}, upper mean: {hdi_50_upper.mean():.2f}")
    print(f"\n  Scale ratio (observed/predicted): {y_observed.mean() / predictions_mean.mean():.4f}")

    return fig, ax


import numpy as np
import matplotlib.pyplot as plt
import arviz as az

def plot_posterior_predictive_with_log_reversion(
    mmm,
    original_scale: bool = True,
    revert_log: bool = True,
    figsize: tuple = (12, 6)
):
    """
    Plot POSTERIOR predictive with proper log transformation handling.

    Must be called AFTER mmm.sample_posterior_predictive(X, extend_idata=True, combined=True)

    Parameters
    ----------
    mmm : MMM object
        The MMM model instance (after sample_posterior_predictive has been called)
    original_scale : bool
        If True, multiply by target_scale to revert scaling
    revert_log : bool
        If True, apply exp(x) to revert log transformation
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """

    # Get the posterior predictive data from idata
    if not hasattr(mmm, 'idata') or 'posterior_predictive' not in mmm.idata:
        raise ValueError("Must call mmm.sample_posterior_predictive(X, extend_idata=True, combined=True) first!")

    # Get predictions from idata (in log space if log transformation was applied)
    preds = mmm.idata.posterior_predictive[mmm.output_var]

    print(f"\n{'='*70}")
    print(f"POSTERIOR PREDICTIVE TRANSFORMATION DEBUG")
    print(f"{'='*70}")
    print(f"Predictions shape: {preds.shape}")
    print(f"Predictions sample (first 5): {preds.values.flat[:5]}")
    print(f"Predictions mean: {preds.mean().values:.2f}")

    # Step 1: Revert log transformation FIRST (if requested)
    if revert_log:
        if hasattr(mmm, 'log_transform_vars') and 'target' in mmm.log_transform_vars:
            print(f"\n✓ Reverting log transformation: exp(x)")
            print(f"  Model has log_transform_vars: {mmm.log_transform_vars}")
            # Calculate mean in log space, then exp
            predictions_mean = np.exp(preds.mean(dim=["chain", "draw"]).values)
            # Calculate HDI in log space, then exp
            preds_hdi = az.hdi(preds, hdi_prob=0.94)
            hdi_94_lower = np.exp(preds_hdi.sel(hdi="lower")[mmm.output_var].values)
            hdi_94_upper = np.exp(preds_hdi.sel(hdi="higher")[mmm.output_var].values)

            preds_hdi_50 = az.hdi(preds, hdi_prob=0.50)
            hdi_50_lower = np.exp(preds_hdi_50.sel(hdi="lower")[mmm.output_var].values)
            hdi_50_upper = np.exp(preds_hdi_50.sel(hdi="higher")[mmm.output_var].values)

            print(f"  After log reversion - mean: {predictions_mean.mean():.2f}")
        else:
            print(f"\n✗ Log transformation NOT applied")
            # No log transformation, just use values directly
            predictions_mean = preds.mean(dim=["chain", "draw"]).values
            preds_hdi = az.hdi(preds, hdi_prob=0.94)
            hdi_94_lower = preds_hdi.sel(hdi="lower")[mmm.output_var].values
            hdi_94_upper = preds_hdi.sel(hdi="higher")[mmm.output_var].values

            preds_hdi_50 = az.hdi(preds, hdi_prob=0.50)
            hdi_50_lower = preds_hdi_50.sel(hdi="lower")[mmm.output_var].values
            hdi_50_upper = preds_hdi_50.sel(hdi="higher")[mmm.output_var].values
    else:
        print(f"\n○ Log reversion SKIPPED (revert_log=False)")
        # Keep in log space
        predictions_mean = preds.mean(dim=["chain", "draw"]).values
        preds_hdi = az.hdi(preds, hdi_prob=0.94)
        hdi_94_lower = preds_hdi.sel(hdi="lower")[mmm.output_var].values
        hdi_94_upper = preds_hdi.sel(hdi="higher")[mmm.output_var].values

        preds_hdi_50 = az.hdi(preds, hdi_prob=0.50)
        hdi_50_lower = preds_hdi_50.sel(hdi="lower")[mmm.output_var].values
        hdi_50_upper = preds_hdi_50.sel(hdi="higher")[mmm.output_var].values

    # Step 2: Revert scaling SECOND (if requested)
    if original_scale:
        if hasattr(mmm, 'target_scale'):
            print(f"\n✓ Reverting scaling: multiply by target_scale={mmm.target_scale}")
            predictions_mean = predictions_mean * mmm.target_scale
            hdi_94_lower = hdi_94_lower * mmm.target_scale
            hdi_94_upper = hdi_94_upper * mmm.target_scale
            hdi_50_lower = hdi_50_lower * mmm.target_scale
            hdi_50_upper = hdi_50_upper * mmm.target_scale
            print(f"  After scale reversion - mean: {predictions_mean.mean():.2f}")
        else:
            print(f"\n✗ Scaling NOT applied (target_scale attribute not found)")
    else:
        print(f"\n○ Scale reversion SKIPPED (original_scale=False)")

    print(f"{'='*70}\n")

    # Get dates
    dates = mmm.idata.posterior_predictive['date'].values

    # Get observed data
    y_observed = np.asarray(mmm.y)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot 94% HDI (lighter)
    ax.fill_between(
        dates,
        hdi_94_lower,
        hdi_94_upper,
        alpha=0.2,
        color='C0',
        label='94% HDI'
    )

    # Plot 50% HDI (darker)
    ax.fill_between(
        dates,
        hdi_50_lower,
        hdi_50_upper,
        alpha=0.4,
        color='C0',
        label='50% HDI'
    )

    # Plot mean prediction
    ax.plot(dates, predictions_mean, color='blue', linewidth=2, label='Mean Prediction')

    # Plot observed data
    ax.plot(dates, y_observed, color='black', linewidth=2, label='Observed', linestyle='--')

    # Labels and formatting
    ax.set_xlabel('Date')
    ax.set_ylabel('y')
    title = 'Posterior Predictive Check'
    if revert_log and original_scale:
        title += ' (Original Scale)'
    elif original_scale:
        title += ' (Scaled Only)'
    elif revert_log:
        title += ' (Log Reverted Only)'
    else:
        title += ' (Model Space)'
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Print summary statistics
    print(f"PLOT STATISTICS:")
    print(f"  Observed Y - mean: {y_observed.mean():.2f}, range: [{y_observed.min():.2f}, {y_observed.max():.2f}]")
    print(f"  Predicted Y - mean: {predictions_mean.mean():.2f}, range: [{predictions_mean.min():.2f}, {predictions_mean.max():.2f}]")
    print(f"  94% HDI - lower mean: {hdi_94_lower.mean():.2f}, upper mean: {hdi_94_upper.mean():.2f}")
    print(f"  50% HDI - lower mean: {hdi_50_lower.mean():.2f}, upper mean: {hdi_50_upper.mean():.2f}")
    print(f"\n  Scale ratio (observed/predicted): {y_observed.mean() / predictions_mean.mean():.4f}")

    return fig, ax


def plot_waterfall_contributions(
    roas_results,
    data=None,
    channel_cols=None,
    control_cols=None,
    date_start=None,
    date_end=None,
    target_column='y',
    figsize=(12, 8),
    colors=None,
    show_values=True,
    title="Component Contributions Waterfall",
    baseline_components=None,
    dimension_label=None
):
    """
    Create a horizontal waterfall plot showing aggregated ADJUSTED contributions from each model component.

    IMPORTANT: This function uses ADJUSTED contributions from calculate_roas() which are scaled
    to match actual sales values. This ensures the waterfall sums to actual sales.

    This function:
    1. Aggregates ADJUSTED contributions over the specified period for each component
    2. Groups intercept + seasonality (+ trend if available) into a 'Baseline' category
    3. Shows channels and controls separately
    4. Creates a horizontal bar waterfall plot

    Parameters
    ----------
    roas_results : dict
        Dictionary returned by calculate_roas() containing:
        - 'adjusted_contributions_all': dict with adjusted contributions for ALL components
          Keys: 'channel_{name}', 'control_{name}', 'intercept', 'seasonality', 'trend' (if available)
        - 'period': dict with start and end indices
    data : pd.DataFrame, optional
        Original data containing target column. Used for validation.
    channel_cols : list, optional
        List of channel names to include. If None, uses all channels in roas_results.
    control_cols : list, optional
        List of control names to include. If None, uses all controls in roas_results.
    date_start, date_end : str or datetime, optional
        NOTE: Date filtering should be done when calling calculate_roas(). This parameter
        is kept for API compatibility but will raise a warning if used.
    target_column : str, default='y'
        Name of the target variable column in data (used for labeling)
    figsize : tuple, optional
        Figure size (width, height), default is (12, 8)
    colors : dict, optional
        Dictionary mapping component types to colors:
        {'baseline': color, 'channels': color, 'controls': color}
        If None, uses default color scheme.
    show_values : bool, default=True
        Whether to show contribution values on the bars
    title : str, optional
        Plot title
    baseline_components : list, optional
        List of component names to include in the baseline.
        If None, defaults to ['intercept', 'seasonality', 'trend'].
        You can customize this to include other components (e.g., specific control variables).
    dimension_label : str, optional
        Label for the dimension (e.g., 'geo_a', 'geo_b'). When provided, this label
        can be used to differentiate between different dimensions. Note: This function
        is typically called from plot_roas_analysis which handles dimension labeling
        in the title parameter.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    aggregated_contributions : dict
        Dictionary containing aggregated values:
        - 'baseline': float (sum of intercept + seasonality + trend)
        - 'channels': dict {channel_name: float}
        - 'controls': dict {control_name: float}
        - 'total': float (sum of all components)

    Notes
    -----
    - Uses ADJUSTED contributions that sum to actual sales
    - Adjusted contributions are calculated using: adjusted_t = (raw_t / total_raw_t) × actual_y_t
    - The waterfall will correctly sum to actual sales for the period

    Examples
    --------
    >>> from b2b_src.contributions import get_individual_contributions, calculate_roas
    >>> from b2b_src.visual_tools import plot_waterfall_contributions
    >>>
    >>> # Get raw contributions from fitted model
    >>> contributions = get_individual_contributions(mmm, data, channel_cols, control_cols)
    >>>
    >>> # Calculate ROAS with adjusted contributions
    >>> roas_results = calculate_roas(
    ...     contributions=contributions,
    ...     data=data,
    ...     channel_cols=paid_channels,
    ...     date_start='2023-01-01',
    ...     date_end='2023-12-31',
    ...     spend_mapping=spend_mapping
    ... )
    >>>
    >>> # Create waterfall plot using adjusted contributions
    >>> fig, ax, agg_contribs = plot_waterfall_contributions(
    ...     roas_results,
    ...     data=data,
    ...     title="MMM Component Contributions (2023)"
    ... )
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    import warnings

    # Default colors
    if colors is None:
        colors = {
            'baseline': '#7f7f7f',    # Gray
            'channels': '#1f77b4',     # Blue
            'controls': '#ff7f0e'      # Orange
        }

    # Warn if date filtering is attempted (should be done in calculate_roas instead)
    if date_start is not None or date_end is not None:
        warnings.warn(
            "Date filtering in plot_waterfall_contributions is deprecated. "
            "Please filter dates when calling calculate_roas() instead.",
            DeprecationWarning,
            stacklevel=2
        )

    # 1. Extract adjusted contributions from roas_results
    adjusted_contributions_all = roas_results['adjusted_contributions_all']

    # 2. Aggregate baseline components (intercept + seasonality + trend if available)
    # Use default baseline components if not specified
    if baseline_components is None:
        baseline_components = ['intercept', 'seasonality', 'trend']

    baseline_value = 0.0

    for component in baseline_components:
        if component in adjusted_contributions_all:
            baseline_value += float(np.sum(adjusted_contributions_all[component]))

    # 3. Aggregate channel contributions
    # Get all channel names from adjusted_contributions_all
    all_channel_keys = [k for k in adjusted_contributions_all.keys() if k.startswith('channel_')]

    if channel_cols is None:
        # Use all channels found in adjusted contributions
        channel_cols = [k.replace('channel_', '') for k in all_channel_keys]

    channel_contributions = {}
    for channel_name in channel_cols:
        channel_key = f'channel_{channel_name}'
        if channel_key in adjusted_contributions_all:
            channel_contributions[channel_name] = float(np.sum(adjusted_contributions_all[channel_key]))

    # 4. Aggregate control contributions
    # Get all control names from adjusted_contributions_all
    all_control_keys = [k for k in adjusted_contributions_all.keys() if k.startswith('control_')]

    if control_cols is None:
        # Use all controls found in adjusted contributions
        control_cols = [k.replace('control_', '') for k in all_control_keys]

    control_contributions = {}
    for control_name in control_cols:
        control_key = f'control_{control_name}'
        if control_key in adjusted_contributions_all:
            control_contributions[control_name] = float(np.sum(adjusted_contributions_all[control_key]))

    # 5. Build the waterfall data structure
    # Order: All components sorted by absolute value in decreasing order

    waterfall_data = []

    # Combine all components into a single list with their types
    all_components = []

    # Add baseline
    all_components.append({
        'label': 'Baseline',
        'value': baseline_value,
        'type': 'baseline'
    })

    # Add all channels
    for channel_name, value in channel_contributions.items():
        all_components.append({
            'label': channel_name,
            'value': value,
            'type': 'channels'
        })

    # Add all controls
    for control_name, value in control_contributions.items():
        all_components.append({
            'label': control_name,
            'value': value,
            'type': 'controls'
        })

    # Sort all components by absolute value (largest first)
    waterfall_data = sorted(all_components, key=lambda x: abs(x['value']), reverse=True)

    # Calculate total
    total_value = baseline_value + sum(channel_contributions.values()) + sum(control_contributions.values())

    # 6. Create the waterfall plot (horizontal bars)
    fig, ax = plt.subplots(figsize=figsize)

    # Extract data for plotting
    labels = [item['label'] for item in waterfall_data]
    values = [item['value'] for item in waterfall_data]
    types = [item['type'] for item in waterfall_data]

    # Calculate cumulative positions for waterfall effect
    cumulative = [0]
    for i in range(len(values)):
        cumulative.append(cumulative[-1] + values[i])

    # Create horizontal bars
    y_positions = np.arange(len(labels))[::-1]  # Reverse so first item is at top

    for i, (label, value, comp_type) in enumerate(zip(labels, values, types)):
        # Get the starting position (left edge of bar)
        left = cumulative[i]

        # Color based on component type
        color = colors.get(comp_type, '#2ca02c')

        # Create bar
        ax.barh(y_positions[i], value, left=left, height=0.6,
               color=color, alpha=0.7, edgecolor='black', linewidth=1)

        # Add value label if requested
        if show_values:
            # Position text in the middle of the bar
            text_x = left + value / 2
            text_y = y_positions[i]

            # Format value with appropriate precision
            if abs(value) >= 1000:
                value_text = f'{value:,.0f}'
            elif abs(value) >= 100:
                value_text = f'{value:,.1f}'
            else:
                value_text = f'{value:,.2f}'

            ax.text(text_x, text_y, value_text,
                   ha='center', va='center', fontsize=9, fontweight='bold')

    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Contribution', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add legend for component types (positioned outside the plot area on the right)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['baseline'], alpha=0.7, edgecolor='black', label='Baseline'),
        Patch(facecolor=colors['channels'], alpha=0.7, edgecolor='black', label='Channels'),
        Patch(facecolor=colors['controls'], alpha=0.7, edgecolor='black', label='Controls')
    ]
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    plt.tight_layout()

    # 7. Return aggregated contributions dictionary
    aggregated_contributions = {
        'baseline': baseline_value,
        'channels': channel_contributions,
        'controls': control_contributions,
        'total': total_value
    }

    return fig, ax, aggregated_contributions


def plot_marginal_roas_analysis(
    mroas_results,
    data=None,
    confidence_intervals=(0.85, 0.90)
):
    """
    Visualize accumulated marginal ROAS analysis results.

    Creates three publication-quality figures showing marginal ROAS statistics:
    1. Bar chart of mean mROAS by channel
    2. Comparison of incremental spend vs incremental sales
    3. Boxplot showing full posterior distribution of mROAS

    Parameters
    ----------
    mroas_results : dict
        Results dictionary returned by calculate_marginal_roas() containing:
        - 'mroas': dict with mean mROAS values
        - 'mroas_samples': dict with posterior samples
        - 'incremental_sales': dict with incremental sales
        - 'incremental_spend': dict with incremental spend
        - 'cumulative_period': dict with date range info
        - 'marginal_increase': float
    data : pd.DataFrame, optional
        Original data (used for additional context if needed)
    confidence_intervals : tuple of float, default=(0.85, 0.90)
        Two confidence levels to display in statistics printout.
        For example, (0.85, 0.90) will show 85% CI (7.5%-92.5%) and 90% CI (5%-95%).

    Returns
    -------
    tuple
        (fig1, fig2, fig3) - Three matplotlib figure objects:
        - fig1: Marginal ROAS bar chart
        - fig2: Incremental spend vs incremental sales comparison
        - fig3: Marginal ROAS boxplot with full distribution

    Examples
    --------
    >>> from b2b_src.contributions import calculate_marginal_roas
    >>> from b2b_src.visual_tools import plot_marginal_roas_analysis
    >>>
    >>> # Calculate marginal ROAS
    >>> mroas_results = calculate_marginal_roas(
    ...     mmm=fitted_mmm,
    ...     data=data,
    ...     channel_cols=['paid_search', 'display'],
    ...     date_start='2023-01-01',
    ...     date_end='2023-12-31'
    ... )
    >>>
    >>> # Visualize results
    >>> fig1, fig2, fig3 = plot_marginal_roas_analysis(mroas_results, data=data)

    See Also
    --------
    plot_roas_analysis : Visualize regular (non-marginal) ROAS
    calculate_marginal_roas : Calculate accumulated marginal ROAS
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Validate confidence_intervals
    if not isinstance(confidence_intervals, tuple) or len(confidence_intervals) != 2:
        raise ValueError("confidence_intervals must be a tuple of exactly 2 values")

    for ci in confidence_intervals:
        if not 0 < ci < 1:
            raise ValueError(f"Confidence interval {ci} must be between 0 and 1")

    # Sort to ensure smaller CI is first
    ci1, ci2 = sorted(confidence_intervals)

    # Get cumulative period info
    cumulative_period = mroas_results.get('cumulative_period', {})
    start_date = cumulative_period.get('start_date', 'N/A')
    end_date = cumulative_period.get('end_date', 'N/A')
    days = cumulative_period.get('days', 'N/A')
    marginal_increase = mroas_results.get('marginal_increase', 0.01)

    # Format dates for display
    if hasattr(start_date, 'strftime'):
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
    else:
        start_date_str = str(start_date)
        end_date_str = str(end_date)

    # ============================================================================
    # Print cumulative period and marginal ROAS statistics
    # ============================================================================
    print("=" * 80)
    print(f"Marginal ROAS Analysis - Cumulative Period: {start_date_str} to {end_date_str} ({days} days)")
    print("=" * 80)
    print(f"Marginal Increase Applied: {marginal_increase*100:.1f}%")
    print()

    # Calculate and print marginal ROAS statistics from samples
    if 'mroas_samples' in mroas_results:
        # Calculate percentiles for both confidence intervals
        lower1 = (1 - ci1) / 2 * 100  # e.g., 7.5% for 85% CI
        upper1 = (1 + ci1) / 2 * 100  # e.g., 92.5% for 85% CI
        lower2 = (1 - ci2) / 2 * 100  # e.g., 5% for 90% CI
        upper2 = (1 + ci2) / 2 * 100  # e.g., 95% for 90% CI

        print("Channel Marginal ROAS Statistics:")
        print("-" * 80)

        for channel, mroas_samples in mroas_results['mroas_samples'].items():
            # Calculate percentiles for both CIs
            ci1_lower = np.percentile(mroas_samples, lower1)
            ci1_upper = np.percentile(mroas_samples, upper1)
            ci2_lower = np.percentile(mroas_samples, lower2)
            ci2_upper = np.percentile(mroas_samples, upper2)
            median = np.percentile(mroas_samples, 50)
            mean = np.mean(mroas_samples)

            print(f"  {channel}:")
            print(f"    - Lower bound {int(ci2*100)}% CI ({lower2:.1f}%): {ci2_lower:.2f}")
            print(f"    - Lower bound {int(ci1*100)}% CI ({lower1:.1f}%): {ci1_lower:.2f}")
            print(f"    - Median (50%): {median:.2f}")
            print(f"    - Mean: {mean:.2f}")
            print(f"    - Upper bound {int(ci1*100)}% CI ({upper1:.1f}%): {ci1_upper:.2f}")
            print(f"    - Upper bound {int(ci2*100)}% CI ({upper2:.1f}%): {ci2_upper:.2f}")
            print()

    print("=" * 80)
    print()

    # ============================================================================
    # FIGURE 1: Marginal ROAS Bar Chart
    # ============================================================================
    fig1, ax = plt.subplots(figsize=(12, 6))

    channels = list(mroas_results['mroas'].keys())
    mroas_values = [mroas_results['mroas'][ch] for ch in channels]
    colors = plt.cm.Set2(np.arange(len(channels)))

    bars = ax.bar(channels, mroas_values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

    # Add value labels on bars
    for bar, value in zip(bars, mroas_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Channel', fontsize=12, fontweight='bold')
    ax.set_ylabel('Marginal ROAS ($ revenue per $ spent)', fontsize=12, fontweight='bold')
    ax.set_title(f'Marginal ROAS by Channel ({marginal_increase*100:.1f}% Spend Increase)\n'
                 f'Period: {start_date_str} to {end_date_str}',
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(alpha=0.3, axis='y', linestyle='--')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    plt.tight_layout()

    # ============================================================================
    # FIGURE 2: Incremental Spend vs Incremental Sales
    # ============================================================================
    fig2, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(channels))
    width = 0.35

    incremental_spend_values = [mroas_results['incremental_spend'][ch] for ch in channels]
    incremental_sales_values = [mroas_results['incremental_sales'][ch] for ch in channels]

    bars1 = ax.bar(x - width/2, incremental_spend_values, width, label='Incremental Spend',
                   color='steelblue', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, incremental_sales_values, width, label='Incremental Sales',
                   color='coral', edgecolor='black', alpha=0.8)

    ax.set_xlabel('Channel', fontsize=12, fontweight='bold')
    ax.set_ylabel('Amount ($)', fontsize=12, fontweight='bold')
    ax.set_title(f'Incremental Spend vs Incremental Sales by Channel\n'
                 f'{marginal_increase*100:.1f}% Spend Increase',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(channels)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y', linestyle='--')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    # ============================================================================
    # FIGURE 3: Marginal ROAS Boxplot
    # ============================================================================
    if 'mroas_samples' in mroas_results:
        fig3, ax = plt.subplots(figsize=(12, 7))

        channels = list(mroas_results['mroas_samples'].keys())
        mroas_samples_list = [mroas_results['mroas_samples'][ch] for ch in channels]

        # Create boxplot
        bp = ax.boxplot(mroas_samples_list, labels=channels, patch_artist=True,
                        showmeans=True, meanline=False,
                        medianprops=dict(color='red', linewidth=2),
                        meanprops=dict(marker='D', markerfacecolor='green',
                                      markersize=8, markeredgecolor='darkgreen'),
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        flierprops=dict(marker='o', markerfacecolor='gray',
                                       markersize=5, alpha=0.5))

        ax.set_xlabel('Channel', fontsize=12, fontweight='bold')
        ax.set_ylabel('Marginal ROAS ($ revenue per $ spent)', fontsize=12, fontweight='bold')
        ax.set_title(f'Marginal ROAS Distribution by Channel\n'
                     f'Cumulative Period: {start_date_str} to {end_date_str} ({days} days)\n'
                     f'{marginal_increase*100:.1f}% Spend Increase',
                     fontsize=14, fontweight='bold', pad=20)
        ax.grid(alpha=0.3, axis='y', linestyle='--')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

        # Add legend
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=2, label='Median'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='green',
                   markersize=8, markeredgecolor='darkgreen', label='Mean'),
            Patch(facecolor='lightblue', alpha=0.7, edgecolor='black', label='IQR (25%-75%)'),
            Line2D([0], [0], color='black', linewidth=1.5, label=f'Whiskers ({int(ci2*100)}% CI: {lower2:.1f}%-{upper2:.1f}%)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.tight_layout()
    else:
        # If no samples available, create empty figure
        fig3 = None

    return fig1, fig2, fig3




def plot_response_curves(analysis_results, dimension_label=None, custom_title=None):
    """
    Plot response curves showing revenue vs spend for each channel.

    Parameters
    ----------
    analysis_results : dict
        Output from analyze_revenue_vs_spend_levels function
    dimension_label : str, optional
        Label for the dimension (e.g., 'geo_a', 'geo_b'). When provided, this label
        is added to plot title to differentiate between different dimensions.
    custom_title : str, optional
        Custom title for the plot. If provided, replaces "Response Curves: Revenue vs Spend by Channel".
        Dimension label will still be appended if provided.
    """
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(10, 6))

    channels_data = analysis_results['channels']

    for channel_name, channel_data in channels_data.items():
        spend_levels = []
        revenue_levels = []

        for multiplier, results in channel_data['multipliers'].items():
            spend_levels.append(results['new_spend'])
            revenue_levels.append(results['new_revenue'])

        ax.plot(spend_levels, revenue_levels, marker='o', label=channel_name, linewidth=2, markersize=6)

    # Format axes to show readable numbers instead of scientific notation
    def millions_formatter(x, pos):
        """Format large numbers as millions with 'M' suffix"""
        if x >= 1_000_000:
            return f'${x/1_000_000:.1f}M'
        elif x >= 1_000:
            return f'${x/1_000:.0f}K'
        else:
            return f'${x:.0f}'

    ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))

    ax.set_xlabel('Spend', fontsize=12)
    ax.set_ylabel('Revenue', fontsize=12)

    # Build title with optional custom title and dimension label
    if custom_title:
        base_title = custom_title
    else:
        base_title = 'Response Curves: Revenue vs Spend by Channel'

    # Add dimension label if provided
    if dimension_label:
        title = f'{base_title} - {dimension_label.upper()}'
    else:
        title = base_title

    ax.set_title(title, fontsize=14, pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


import seaborn as sns


def plot_multidimensional_prior_predictive_by_dim(prior, data, figsize=(12, 6), dimension_label_override=None, custom_title=None, dimension_col=None, target_col=None):
    """
    Plot prior predictive samples by dimension.

    Parameters
    ----------
    prior : InferenceData
        Prior predictive samples from pm.sample_prior_predictive().
        Must contain 'prior_predictive' group with 'y_original_scale' or 'y'.
    data : pd.DataFrame
        Original data containing the actual target values
    figsize : tuple
        Figure size (width, height) for each subplot
    dimension_label_override : str, optional
        Custom label to use instead of the actual geo name (e.g., "REGION" instead of "GEO_A").
        If provided, all subplots will use this same label.
    custom_title : str, optional
        Custom title template. If provided, replaces "Prior Predictive Check".
        Dimension label will still be appended unless dimension_label_override is empty string.
    dimension_col : str, optional
        Name of the dimension column in data (e.g., 'Groups', 'geo', 'brand').
        If None, will try to infer from data (checks 'Groups', 'geo', 'brand', 'region').
    target_col : str, optional
        Name of the target column in data (e.g., 'NetGMV', 'y', 'sales').
        If None, will try to infer from data (checks 'NetGMV', 'y', 'sales', 'revenue').

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing all subplots

    Notes
    -----
    This function accesses y_original_scale from prior.prior group (where deterministics
    are stored), or falls back to prior.prior_predictive for the likelihood sample.
    In PyMC, deterministics computed from prior samples are stored in the 'prior' group,
    while likelihood samples are stored in 'prior_predictive'.
    """
    from b2b_src.column_config import get_default_column_config
    
    # Get column configuration
    cols = get_default_column_config()
    
    # Infer dimension column if not provided
    if dimension_col is None:
        # Try config first, then fallback to auto-detection
        dimension_col = cols.dimension
        if dimension_col not in data.columns:
            # Fallback to old behavior
            for col in ['Groups', 'geo', 'brand', 'region', 'market']:
                if col in data.columns:
                    dimension_col = col
                    break
        if dimension_col not in data.columns:
            raise ValueError("Could not infer dimension column. Please specify dimension_col parameter.")

    # Infer target column if not provided
    if target_col is None:
        # Try config first, then fallback to auto-detection
        target_col = cols.target
        if target_col not in data.columns:
            # Fallback to old behavior
            for col in ['NetGMV', 'y', 'sales', 'revenue', 'target']:
                if col in data.columns:
                    target_col = col
                    break
        if target_col not in data.columns:
            raise ValueError("Could not infer target column. Please specify target_col parameter.")

    # Use y_original_scale if available (deterministic in prior.prior),
    # or fall back to y from prior_predictive (likelihood sample)
    if "y_original_scale" in prior.prior:
        y_prior = prior.prior["y_original_scale"]
    elif "y_original_scale" in prior.prior_predictive:
        y_prior = prior.prior_predictive["y_original_scale"]
    else:
        y_prior = prior.prior_predictive["y"]

    n_geos = y_prior.shape[-1]

    fig, axes = plt.subplots(n_geos, 1, figsize=(figsize[0], figsize[1] * n_geos))
    if n_geos == 1:
        axes = [axes]

    geo_names = data[dimension_col].unique()

    for geo_idx in range(n_geos):
        ax = axes[geo_idx]

        # Get dimension name from xarray coordinates (could be 'geo', 'Groups', etc.)
        dim_name = [d for d in y_prior.dims if d not in ['chain', 'draw', 'date']][0]
        samples = y_prior.isel({dim_name: geo_idx}).values

        mean_samples = samples.mean(axis=(0, 1))
        hdi_94 = np.percentile(samples, [3, 97], axis=(0, 1))
        hdi_50 = np.percentile(samples, [25, 75], axis=(0, 1))

        dates = np.arange(len(mean_samples))

        geo_data = data[data[dimension_col] == geo_names[geo_idx]]
        actual_y = geo_data[target_col].values

        ax.plot(dates, actual_y, color='black', linewidth=2, label='Actual')
        ax.plot(dates, mean_samples, color='blue', linewidth=2, label='Mean')
        ax.fill_between(dates, hdi_94[0], hdi_94[1], alpha=0.2, color='blue', label='94% HDI')
        ax.fill_between(dates, hdi_50[0], hdi_50[1], alpha=0.4, color='blue', label='50% HDI')

        # Build title with optional custom title and dimension label override
        base_title = custom_title if custom_title else 'Prior Predictive Check'

        # Determine dimension label
        if dimension_label_override is not None:
            # User provided override (could be empty string to disable label)
            dim_label = dimension_label_override.upper() if dimension_label_override else ''
        else:
            # Use actual geo name
            dim_label = geo_names[geo_idx].upper()

        # Combine title and dimension label
        if dim_label:
            title = f'{base_title} - {dim_label}'
        else:
            title = base_title

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_posterior_predictive_by_geo(mmm, data, dimension_label_override=None, custom_title=None, dimension_col=None, target_col=None):
    """
    Plot posterior predictive for each geo dimension.

    Parameters
    ----------
    mmm : MMM object
        Fitted MMM model with posterior predictive samples.
        Must have called mmm.sample_posterior_predictive() with
        var_names=['y_original_scale'] before calling this function.
    data : pd.DataFrame
        Original data containing the target column
    dimension_label_override : str, optional
        Custom label to use instead of the actual geo name (e.g., "REGION" instead of "GEO_A").
        If provided, all subplots will use this same label.
    custom_title : str, optional
        Custom title template. If provided, replaces "Posterior Predictive Check".
        Dimension label will still be appended unless dimension_label_override is empty string.
    dimension_col : str, optional
        Name of the dimension column in data (e.g., 'Groups', 'geo', 'brand').
        If None, will try to infer from model coords.
    target_col : str, optional
        Name of the target column in data (e.g., 'NetGMV', 'y', 'sales').
        If None, will try to infer from data.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object with subplots for each geo

    Raises
    ------
    ValueError
        If 'y_original_scale' not found in posterior_predictive group

    Notes
    -----
    This function uses posterior_predictive group (includes observation noise)
    rather than posterior group (deterministic only), which is correct for
    posterior predictive checks per Bayesian best practices.
    """
    from b2b_src.column_config import get_default_column_config
    
    # Get column configuration
    cols = get_default_column_config()
    
    # Infer dimension name from model coords
    if dimension_col is None:
        # Try config first
        dimension_col = cols.dimension
        # Verify it exists in model coords
        if dimension_col not in mmm.model.coords:
            # Try to find dimension in coords (exclude 'date', 'chain', 'draw', etc.)
            for dim in mmm.model.coords:
                if dim not in ['date', 'chain', 'draw', 'channel', 'control', 'fourier_mode']:
                    dimension_col = dim
                    break
        if dimension_col not in mmm.model.coords:
            raise ValueError("Could not infer dimension column from model coords. Please specify dimension_col parameter.")

    # Infer target column if not provided
    if target_col is None:
        # Try config first, then fallback to auto-detection
        target_col = cols.target
        if target_col not in data.columns:
            # Fallback to old behavior
            for col in ['NetGMV', 'y', 'sales', 'revenue', 'target']:
                if col in data.columns:
                    target_col = col
                    break
        if target_col not in data.columns:
            raise ValueError("Could not infer target column. Please specify target_col parameter.")

    geos = mmm.model.coords[dimension_col]
    n_geos = len(geos)

    fig, axes = plt.subplots(n_geos, 1, figsize=(12, 6 * n_geos))
    if n_geos == 1:
        axes = [axes]

    for i, geo in enumerate(geos):
        ax = axes[i]

        # Check if y_original_scale is available in posterior_predictive
        if "y_original_scale" not in mmm.idata["posterior_predictive"]:
            raise ValueError(
                f"y_original_scale not found in posterior_predictive for geo '{geo}'. "
                f"Available variables: {list(mmm.idata['posterior_predictive'].data_vars)}. "
                f"Ensure you called mmm.sample_posterior_predictive(X, var_names=['y_original_scale'])"
            )

        y_pred = mmm.idata["posterior_predictive"].y_original_scale.sel({dimension_col: geo})

        mean_pred = y_pred.mean(dim=["chain", "draw"]).values
        hdi_94 = np.percentile(y_pred.values, [3, 97], axis=(0, 1))
        hdi_50 = np.percentile(y_pred.values, [25, 75], axis=(0, 1))

        geo_data = data[data[dimension_col] == geo]
        actual_y = geo_data[target_col].values

        dates = mmm.model.coords["date"]

        ax.plot(dates, actual_y, color='black', linewidth=2, label='Actual')
        ax.plot(dates, mean_pred, color='blue', linewidth=2, label='Mean Prediction')
        ax.fill_between(dates, hdi_94[0], hdi_94[1], alpha=0.2, color='blue', label='94% HDI')
        ax.fill_between(dates, hdi_50[0], hdi_50[1], alpha=0.4, color='blue', label='50% HDI')

        # Build title with optional custom title and dimension label override
        base_title = custom_title if custom_title else 'Posterior Predictive Check'

        # Determine dimension label
        if dimension_label_override is not None:
            # User provided override (could be empty string to disable label)
            dim_label = dimension_label_override.upper() if dimension_label_override else ''
        else:
            # Use actual geo name
            dim_label = geo.upper()

        # Combine title and dimension label
        if dim_label:
            title = f'{base_title} - {dim_label}'
        else:
            title = base_title

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_channel_spends_by_geo(
    data: pd.DataFrame,
    channel_cols: list[str] = None,
    target_col: str = None,
    dimension_col: str = None,
    date_col: str = None,
    figsize: tuple = (12, 8),
    bar_width: int = 7,
    bar_colors: list = None,
    dimension_label_override: str = None,
    custom_title: str = None
):
    """
    Plot stacked channel spends and target values over time by geo in separate subplots.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data with channels, target, and dimension columns
    channel_cols : list[str], optional
        List of channel column names. If None, uses config default.
    target_col : str, optional
        Target column name. If None, uses config default.
    dimension_col : str, optional
        Dimension column name (e.g., 'Groups'). If None, uses config default.
    date_col : str, optional
        Date column name. If None, uses config default.
    figsize : tuple, optional
        Figure size (width, height), default is (12, 8)
    bar_width : int, optional
        Width of bars in days, default is 7
    bar_colors : list, optional
        List of colors for channels. If None, uses blue palette.
    dimension_label_override : str, optional
        Override dimension label in title. If None, uses actual geo names.
        Set to "" (empty string) to hide dimension label completely.
    custom_title : str, optional
        Override the main figure title. If None, uses "Channel Spends Over Time"

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object

    Examples
    --------
    # Basic usage with automatic column detection
    >>> fig = plot_channel_spends_by_geo(data_df)

    # With custom title and dimension labels
    >>> fig = plot_channel_spends_by_geo(
    ...     data_df,
    ...     custom_title="Marketing Investment Analysis",
    ...     dimension_label_override="REGION"
    ... )

    # With explicit column names
    >>> fig = plot_channel_spends_by_geo(
    ...     data_df,
    ...     channel_cols=['Spend_channel_1', 'Spend_channel_2'],
    ...     target_col='NetGMV',
    ...     dimension_col='Groups',
    ...     date_col='date'
    ... )
    """
    from b2b_src.column_config import get_default_column_config

    # Get column configuration
    cols = get_default_column_config()

    # Use provided columns or fall back to config
    channel_cols = channel_cols or cols.channels
    target_col = target_col or cols.target
    dimension_col = dimension_col or cols.dimension
    date_col = date_col or cols.date

    # Default bar colors (blue palette)
    if bar_colors is None:
        bar_colors = ["#1f77b4", "#7aa6c2", "#aec7e8", "#c5d9ed"]

    # Get unique groups from the data
    groups = data[dimension_col].unique()
    n_groups = len(groups)

    # Create subplots
    fig, axes = plt.subplots(n_groups, 1, figsize=figsize, layout="constrained")

    # Handle single group case
    if n_groups == 1:
        axes = [axes]

    # Set main figure title
    main_title = custom_title if custom_title else "Channel Spends Over Time"
    fig.suptitle(main_title, fontsize=16, fontweight="bold")

    # Plot each group
    for idx, (ax, group) in enumerate(zip(axes, groups)):
        group_data = data[data[dimension_col] == group]

        # Plot stacked bars for channels
        bottom = np.zeros(len(group_data))
        for channel_idx, channel in enumerate(channel_cols):
            color = bar_colors[channel_idx % len(bar_colors)]
            ax.bar(
                group_data[date_col],
                group_data[channel],
                bottom=bottom,
                label=channel,
                width=bar_width,
                color=color
            )
            bottom += group_data[channel].values

        # Plot target line
        ax.plot(
            group_data[date_col],
            group_data[target_col],
            "--",
            label=target_col,
            color="black",
            linewidth=2
        )

        # Build subplot title with optional dimension label
        if dimension_label_override is not None:
            # User provided override (could be empty string to disable label)
            dim_label = dimension_label_override.upper() if dimension_label_override else ''
        else:
            # Use actual group name
            dim_label = str(group).upper()

        ax.set_title(dim_label, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')

    return fig
