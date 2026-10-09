"""
Configuration Loader for GEO Model
===================================

Safe utility functions to load and access YAML configuration files
without KeyErrors or AttributeErrors.

Usage:
    from b2b_src.config_loader import load_geo_config, get_config_value

    config = load_geo_config("geo_config.yml")
    date_col = get_config_value(config, "data.columns.date_column", default="date")
"""

import yaml
from pathlib import Path
from typing import Any, Optional, Dict, List, Union


class ConfigDict(dict):
    """
    Dictionary subclass that allows dot notation access and returns None
    instead of raising KeyError for missing keys.

    Examples:
        config = ConfigDict({'a': {'b': 1}})
        config.a.b  # Returns 1
        config.a.c  # Returns None instead of KeyError
        config['a']['b']  # Also works
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Convert nested dicts to ConfigDict
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = ConfigDict(value)
            elif isinstance(value, list):
                self[key] = [ConfigDict(item) if isinstance(item, dict) else item
                            for item in value]

    def __getattr__(self, key):
        """Allow attribute-style access: config.model.adstock"""
        try:
            return self[key]
        except KeyError:
            return None

    def __setattr__(self, key, value):
        """Allow attribute-style setting: config.model = {...}"""
        self[key] = value

    def __getitem__(self, key):
        """Return None instead of KeyError for missing keys"""
        try:
            return super().__getitem__(key)
        except KeyError:
            return None

    def get_nested(self, path: str, default: Any = None, separator: str = ".") -> Any:
        """
        Safely get a nested value using dot notation path.

        Args:
            path: Dot-separated path to the value (e.g., "data.columns.date_column")
            default: Value to return if path doesn't exist
            separator: Character used to separate path components (default: ".")

        Returns:
            The value at the path, or default if not found

        Examples:
            config.get_nested("data.columns.date_column", default="date")
            config.get_nested("model/adstock/type", separator="/")
        """
        keys = path.split(separator)
        value = self

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    def require(self, path: str, separator: str = ".") -> Any:
        """
        Get a required nested value, raising a clear error if not found.

        Args:
            path: Dot-separated path to the value
            separator: Character used to separate path components

        Returns:
            The value at the path

        Raises:
            ValueError: If the path doesn't exist in the config

        Examples:
            config.require("data.source")  # Raises error if not found
        """
        value = self.get_nested(path, default=None, separator=separator)
        if value is None:
            raise ValueError(
                f"Required configuration value not found: '{path}'\n"
                f"Please check your configuration file."
            )
        return value


def load_geo_config(config_path: Union[str, Path]) -> ConfigDict:
    """
    Load geo model configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        ConfigDict object with safe attribute/key access

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails

    Examples:
        config = load_geo_config("geo_config.yml")
        config = load_geo_config(Path("configs/geo_config.yml"))
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Current working directory: {Path.cwd()}"
        )

    with open(config_path, 'r') as f:
        try:
            config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(
                f"Error parsing YAML configuration file {config_path}:\n{e}"
            )

    return ConfigDict(config_data)


def get_config_value(
    config: Union[Dict, ConfigDict],
    path: str,
    default: Any = None,
    required: bool = False
) -> Any:
    """
    Safely get a value from config using dot notation.

    Args:
        config: Configuration dictionary
        path: Dot-separated path to the value
        default: Default value if not found (only used if required=False)
        required: If True, raises error when value not found

    Returns:
        The configuration value or default

    Raises:
        ValueError: If required=True and value not found

    Examples:
        date_col = get_config_value(config, "data.columns.date_column", default="date")
        source = get_config_value(config, "data.source", required=True)
    """
    if isinstance(config, ConfigDict):
        if required:
            return config.require(path)
        else:
            return config.get_nested(path, default=default)
    else:
        # Handle regular dict
        keys = path.split(".")
        value = config

        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    if required:
                        raise ValueError(
                            f"Required configuration value not found: '{path}'"
                        )
                    return default
            else:
                if required:
                    raise ValueError(
                        f"Required configuration value not found: '{path}'"
                    )
                return default

        return value if value is not None else default


def validate_geo_config(config: ConfigDict) -> List[str]:
    """
    Validate that all required configuration values are present.

    Args:
        config: Configuration dictionary to validate

    Returns:
        List of error messages (empty if valid)

    Examples:
        errors = validate_geo_config(config)
        if errors:
            print("Configuration errors:")
            for error in errors:
                print(f"  - {error}")
    """
    errors = []

    # Required fields
    required_fields = [
        "data.source",
        "data.columns.date_column",
        "data.columns.target_column",
        "data.columns.dimension_column",
    ]

    for field in required_fields:
        if config.get_nested(field) is None:
            errors.append(f"Missing required field: {field}")

    # Validate channels exist
    channels = config.get_nested("data.columns.channels", default=[])
    if not channels:
        errors.append("No channels defined in data.columns.channels")
    else:
        for i, channel in enumerate(channels):
            if not isinstance(channel, dict):
                errors.append(f"Channel {i} is not a dictionary")
            elif 'name' not in channel:
                errors.append(f"Channel {i} missing 'name' field")
            elif 'spend_column' not in channel:
                errors.append(f"Channel {i} missing 'spend_column' field")

    return errors


def get_channel_config(config: ConfigDict) -> Dict[str, Any]:
    """
    Extract channel configuration in a format ready for model building.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary with channel names, columns, types, etc.

    Examples:
        channel_info = get_channel_config(config)
        channel_cols = channel_info['channel_columns']
        paid_channels = channel_info['paid_channels']
    """
    channels = config.get_nested("data.columns.channels", default=[])

    channel_columns = []
    paid_channels = []
    unpaid_channels = []
    spend_mapping = {}

    for channel in channels:
        name = channel.get('name')
        spend_col = channel.get('spend_column')
        channel_type = channel.get('type', 'paid')

        if name:
            channel_columns.append(name)

            if channel_type == 'paid':
                paid_channels.append(name)
            else:
                unpaid_channels.append(name)

            if spend_col:
                spend_mapping[name] = spend_col

    return {
        'channel_columns': channel_columns,
        'paid_channels': paid_channels,
        'unpaid_channels': unpaid_channels,
        'spend_mapping': spend_mapping,
    }


def get_control_config(config: ConfigDict) -> List[str]:
    """
    Extract control variable names from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        List of control variable names

    Examples:
        controls = get_control_config(config)
    """
    controls = config.get_nested("data.columns.controls", default=[])
    return [ctrl.get('name') for ctrl in controls if ctrl.get('name')]


def get_baseline_components(config: ConfigDict) -> List[str]:
    """
    Get baseline components for contribution analysis.

    Args:
        config: Configuration dictionary

    Returns:
        List of baseline component names

    Examples:
        baseline = get_baseline_components(config)
    """
    return config.get_nested(
        "analysis.baseline_components",
        default=["intercept", "seasonality"]
    )


def print_config_summary(config: ConfigDict) -> None:
    """
    Print a human-readable summary of the configuration.

    Args:
        config: Configuration dictionary

    Examples:
        print_config_summary(config)
    """
    print("=" * 80)
    print("GEO MODEL CONFIGURATION SUMMARY")
    print("=" * 80)

    print(f"\nData Source: {config.get_nested('data.source')}")
    print(f"Date Column: {config.get_nested('data.columns.date_column')}")
    print(f"Target Column: {config.get_nested('data.columns.target_column')}")
    print(f"Dimension Column: {config.get_nested('data.columns.dimension_column')}")

    channels = config.get_nested("data.columns.channels", default=[])
    print(f"\nChannels ({len(channels)}):")
    for channel in channels:
        print(f"  - {channel.get('name')} ({channel.get('type', 'paid')})")

    controls = config.get_nested("data.columns.controls", default=[])
    print(f"\nControls ({len(controls)}):")
    for ctrl in controls:
        print(f"  - {ctrl.get('name')}")

    print(f"\nModel Settings:")
    print(f"  Adstock: {config.get_nested('model.adstock.type')}")
    print(f"  Saturation: {config.get_nested('model.saturation.type')}")
    print(f"  Scaling: {config.get_nested('model.apply_scaling')}")
    print(f"  Log Transform: {config.get_nested('model.log_transformation')}")

    print(f"\nSampling:")
    print(f"  Chains: {config.get_nested('sampling.mcmc.chains')}")
    print(f"  Target Accept: {config.get_nested('sampling.mcmc.target_accept')}")
    print(f"  Sampler: {config.get_nested('sampling.mcmc.nuts_sampler')}")
    print(f"  Random Seed: {config.get_nested('sampling.random_seed')}")

    print("=" * 80)


# Convenience function for common pattern
def load_and_validate_config(config_path: Union[str, Path]) -> ConfigDict:
    """
    Load and validate configuration, raising errors if invalid.

    Args:
        config_path: Path to configuration file

    Returns:
        Validated ConfigDict

    Raises:
        ValueError: If configuration is invalid

    Examples:
        config = load_and_validate_config("geo_config.yml")
    """
    config = load_geo_config(config_path)
    errors = validate_geo_config(config)

    if errors:
        raise ValueError(
            "Configuration validation failed:\n  - " + "\n  - ".join(errors)
        )

    return config
