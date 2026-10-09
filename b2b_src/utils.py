from pymc_marketing.mmm.components.adstock import GeometricAdstock
from pymc_marketing.mmm.components.saturation import LogisticSaturation
from pymc_marketing.mmm.components.adstock import NoAdstock
from pymc_marketing.mmm.components.saturation import NoSaturation

# Helper function to create adstock object from config
def create_adstock_from_config(adstock_config):
    """Create adstock object based on config type."""
    adstock_type = adstock_config.get('type', 'GeometricAdstock')
    l_max = adstock_config.get('l_max', 8)
    
    if adstock_type == 'GeometricAdstock':
        return GeometricAdstock(l_max=l_max)
    elif adstock_type == 'NoAdstock':
        return NoAdstock()
    else:
        raise ValueError(f"Unknown adstock type: {adstock_type}")

# Helper function to create saturation object from config
def create_saturation_from_config(saturation_config):
    """Create saturation object based on config type."""
    saturation_type = saturation_config.get('type', 'NoSaturation')
    
    if saturation_type == 'NoSaturation':
        return NoSaturation()
    elif saturation_type == 'LogisticSaturation':
        return LogisticSaturation()
    else:
        raise ValueError(f"Unknown saturation type: {saturation_type}")



def plot_trace_with_channel_names(
    mmm,
    config,
    include_params=None,
    figsize=None,
    **kwargs
):
    """
    Plot trace plots with readable channel names from config.
    
    Parameters
    ----------
    mmm : MMM
        Fitted MMM model instance
    config : dict
        Configuration dictionary with channel definitions
    include_params : list of str, optional
        Which parameter types to include. Options:
        - 'adstock_alpha': Adstock decay parameters
        - 'saturation_beta': Saturation curve parameters
        - 'gamma_control': Control variable coefficients
        - 'gamma_fourier': Seasonality (Fourier) coefficients
        - 'intercept': Model intercept
        - 'y_sigma': Likelihood standard deviation
        Default: ['adstock_alpha', 'intercept']
    figsize : tuple, optional
        Figure size (width, height)
    **kwargs
        Additional arguments passed to az.plot_trace()
    
    Returns
    -------
    numpy.ndarray
        Array of matplotlib axes with readable labels
    
    Examples
    --------
    >>> # Default: adstock + intercept
    >>> axes = plot_trace_with_channel_names(mmm, config)
    >>> 
    >>> # Custom parameters
    >>> axes = plot_trace_with_channel_names(
    ...     mmm, config,
    ...     include_params=['adstock_alpha', 'saturation_beta', 'gamma_control']
    ... )
    """
    import arviz as az
    import matplotlib.pyplot as plt
    
    if include_params is None:
        include_params = ['adstock_alpha', 'intercept']
    
    # Extract channel and control names from config
    channels = config['data']['columns']['channels']
    channel_names = [ch['name'] for ch in channels]
    controls = config['data']['columns'].get('controls', [])
    control_names = [ctrl['name'] for ctrl in controls]
    
    # Build technical parameter names and readable name mapping
    var_names = []
    readable_mapping = {}
    
    if 'adstock_alpha' in include_params:
        for i, ch_name in enumerate(channel_names):
            tech_name = f"adstock_alpha_ch{i}"
            var_names.append(tech_name)
            readable_mapping[tech_name] = ch_name
    
    if 'saturation_beta' in include_params:
        for i, ch_name in enumerate(channel_names):
            tech_name = f"saturation_beta_ch{i}"
            var_names.append(tech_name)
            readable_mapping[tech_name] = ch_name
    
    if 'gamma_control' in include_params:
        for i, ctrl_name in enumerate(control_names):
            tech_name = f"gamma_control_ch{i}"  # Note: uses 'ch', not 'ctrl'
            var_names.append(tech_name)
            readable_mapping[tech_name] = ctrl_name
    
    if 'gamma_fourier' in include_params:
        var_names.append("gamma_fourier")
        readable_mapping["gamma_fourier"] = "gamma_fourier"
    
    if 'intercept' in include_params:
        var_names.append("intercept")
        readable_mapping["intercept"] = "intercept"
    
    if 'y_sigma' in include_params:
        var_names.append("y_sigma")
        readable_mapping["y_sigma"] = "y_sigma"
    
    # Print parameter mapping
    print("=" * 70)
    print("PARAMETER MAPPING:")
    print("=" * 70)
    for tech_name, readable_name in readable_mapping.items():
        print(f"  {tech_name:30s} → {readable_name}")
    print("=" * 70)
    
    # Create trace plot - returns array of axes
    axes = az.plot_trace(
        data=mmm.fit_result,
        var_names=var_names,
        figsize=figsize,
        **kwargs
    )
    
    # Flatten axes array for easier iteration
    if hasattr(axes, 'flatten'):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]
    
    # Update subplot labels with readable names
    # Each parameter has 2 subplots (distribution on left, trace on right)
    param_idx = 0
    
    for tech_name in var_names:
        if tech_name == "gamma_fourier":
            # gamma_fourier has multiple fourier modes (sin_1, sin_2, cos_1, cos_2)
            # So it takes up 4 rows (8 subplots total: 4 distributions + 4 traces)
            fourier_modes = ['sin_1', 'sin_2', 'cos_1', 'cos_2']
            for mode_idx, mode in enumerate(fourier_modes):
                row_idx = param_idx * 2
                if row_idx < len(axes_flat) - 1:
                    # Left subplot (distribution)
                    axes_flat[row_idx].set_ylabel(
                        f"gamma_fourier\n[{mode}]",
                        fontsize=9,
                        rotation=0,
                        ha='right',
                        va='center'
                    )
                    # Right subplot (trace)
                    axes_flat[row_idx + 1].set_ylabel(
                        f"gamma_fourier\n[{mode}]",
                        fontsize=9,
                        rotation=0,
                        ha='right',
                        va='center'
                    )
                param_idx += 1
        else:
            # Single parameter
            readable_name = readable_mapping.get(tech_name, tech_name)
            row_idx = param_idx * 2
            
            if row_idx < len(axes_flat) - 1:
                # Left subplot (distribution)
                axes_flat[row_idx].set_ylabel(
                    readable_name,
                    fontsize=10,
                    rotation=0,
                    ha='right',
                    va='center'
                )
                # Right subplot (trace)
                axes_flat[row_idx + 1].set_ylabel(
                    readable_name,
                    fontsize=10,
                    rotation=0,
                    ha='right',
                    va='center'
                )
            param_idx += 1
    
    plt.tight_layout()
    return axes


def summary_with_channel_names(
    mmm,
    config,
    include_params=None,
    **kwargs
):
    """
    Generate parameter summary with readable channel names from config.
    
    Parameters
    ----------
    mmm : MMM
        Fitted MMM model instance
    config : dict
        Configuration dictionary with channel definitions
    include_params : list of str, optional
        Which parameter types to include. Options:
        - 'adstock_alpha': Adstock decay parameters
        - 'saturation_beta': Saturation curve parameters
        - 'gamma_control': Control variable coefficients
        - 'gamma_fourier': Seasonality (Fourier) coefficients
        - 'intercept': Model intercept
        - 'y_sigma': Likelihood standard deviation
        Default: ['adstock_alpha', 'intercept']
    **kwargs
        Additional arguments passed to az.summary()
    
    Returns
    -------
    pandas.DataFrame
        Summary statistics with readable parameter names as index
    
    Examples
    --------
    >>> # Default summary
    >>> summary_df = summary_with_channel_names(mmm, config)
    >>> print(summary_df)
    >>> 
    >>> # Custom parameters with specific stats
    >>> summary_df = summary_with_channel_names(
    ...     mmm, config,
    ...     include_params=['adstock_alpha', 'saturation_beta'],
    ...     kind='stats'
    ... )
    """
    import arviz as az
    import pandas as pd
    
    if include_params is None:
        include_params = ['adstock_alpha', 'intercept']
    
    # Extract channel and control names from config
    channels = config['data']['columns']['channels']
    channel_names = [ch['name'] for ch in channels]
    controls = config['data']['columns'].get('controls', [])
    control_names = [ctrl['name'] for ctrl in controls]
    
    # Build technical parameter names and readable name mapping
    var_names = []
    readable_mapping = {}
    
    if 'adstock_alpha' in include_params:
        for i, ch_name in enumerate(channel_names):
            tech_name = f"adstock_alpha_ch{i}"
            var_names.append(tech_name)
            readable_mapping[tech_name] = f"adstock_alpha[{ch_name}]"
    
    if 'saturation_beta' in include_params:
        for i, ch_name in enumerate(channel_names):
            tech_name = f"saturation_beta_ch{i}"
            var_names.append(tech_name)
            readable_mapping[tech_name] = f"saturation_beta[{ch_name}]"
    
    if 'gamma_control' in include_params:
        for i, ctrl_name in enumerate(control_names):
            tech_name = f"gamma_control_ch{i}"
            var_names.append(tech_name)
            readable_mapping[tech_name] = f"gamma_control[{ctrl_name}]"
    
    if 'gamma_fourier' in include_params:
        var_names.append("gamma_fourier")
        # gamma_fourier will expand to multiple rows in summary
        # We'll handle the renaming after getting the summary
    
    if 'intercept' in include_params:
        var_names.append("intercept")
        readable_mapping["intercept"] = "intercept"
    
    if 'y_sigma' in include_params:
        var_names.append("y_sigma")
        readable_mapping["y_sigma"] = "y_sigma"
    
    # Get summary with technical names
    summary_df = az.summary(
        data=mmm.fit_result,
        var_names=var_names,
        **kwargs
    )
    
    # Rename index to readable names
    def rename_index(idx_name):
        # Handle gamma_fourier with fourier modes
        if idx_name.startswith("gamma_fourier["):
            return idx_name  # Keep as is: gamma_fourier[sin_1], etc.
        return readable_mapping.get(idx_name, idx_name)
    
    summary_df.index = summary_df.index.map(rename_index)
    
    return summary_df