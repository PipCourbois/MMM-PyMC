"""
Column Configuration Manager
=============================

Centralized column name management to avoid hardcoding and KeyErrors.
All functions should use this to get column names.

Usage:
    from b2b_src.column_config import ColumnConfig

    # Initialize with config or defaults
    cols = ColumnConfig(config)

    # Use in functions
    data[cols.dimension]  # Instead of data["geo"]
    data[cols.date]       # Instead of data["date"]
    data[cols.target]     # Instead of data["y"]
"""

from typing import Optional, Dict, Any, Union
from pathlib import Path


class ColumnConfig:
    """
    Manages column names for MMM models.

    Provides a single source of truth for column names, avoiding hardcoding
    and enabling easy switching between datasets with different column names.

    Attributes:
        dimension: Name of the dimension column (geography, brand, etc.)
        date: Name of the date column
        target: Name of the target/KPI column
        channels: List of channel column names
        controls: List of control variable column names

    Examples:
        # From config file
        from b2b_src.config_loader import load_geo_config
        config = load_geo_config("geo_config.yml")
        cols = ColumnConfig(config)

        # Direct initialization
        cols = ColumnConfig(
            dimension="Groups",
            date="date",
            target="NetGMV"
        )

        # Use in code
        geo_names = data[cols.dimension].unique()
        dates = data[cols.date]
    """

    def __init__(
        self,
        config: Optional[Union[Dict, Any]] = None,
        dimension: Optional[str] = None,
        date: Optional[str] = None,
        target: Optional[str] = None,
        channels: Optional[list] = None,
        controls: Optional[list] = None,
    ):
        """
        Initialize column configuration.

        Args:
            config: Configuration dict or ConfigDict object (optional)
            dimension: Override dimension column name
            date: Override date column name
            target: Override target column name
            channels: Override channel list
            controls: Override control list

        Note:
            If config is provided, values are extracted from it.
            Direct parameters override config values.
        """
        # Default values
        self._dimension = "geo"  # Default for backward compatibility
        self._date = "date"
        self._target = "y"
        self._channels = []
        self._controls = []

        # Load from config if provided
        if config is not None:
            self._load_from_config(config)

        # Override with direct parameters if provided
        if dimension is not None:
            self._dimension = dimension
        if date is not None:
            self._date = date
        if target is not None:
            self._target = target
        if channels is not None:
            self._channels = channels
        if controls is not None:
            self._controls = controls

    def _load_from_config(self, config):
        """Load column names from config object."""
        # Try to access config values safely
        try:
            # Handle ConfigDict objects
            if hasattr(config, 'get_nested'):
                self._dimension = config.get_nested(
                    "data.columns.dimension_column",
                    default="geo"
                )
                self._date = config.get_nested(
                    "data.columns.date_column",
                    default="date"
                )
                self._target = config.get_nested(
                    "data.columns.target_column",
                    default="y"
                )

                # Get channels
                channels_cfg = config.get_nested("data.columns.channels", default=[])
                self._channels = [ch.get('name') for ch in channels_cfg if ch.get('name')]

                # Get controls
                controls_cfg = config.get_nested("data.columns.controls", default=[])
                self._controls = [ctrl.get('name') for ctrl in controls_cfg if ctrl.get('name')]

            # Handle regular dicts
            elif isinstance(config, dict):
                data_cols = config.get('data', {}).get('columns', {})
                self._dimension = data_cols.get('dimension_column', 'geo')
                self._date = data_cols.get('date_column', 'date')
                self._target = data_cols.get('target_column', 'y')

                # Get channels
                channels_cfg = data_cols.get('channels', [])
                self._channels = [ch.get('name') for ch in channels_cfg if ch.get('name')]

                # Get controls
                controls_cfg = data_cols.get('controls', [])
                self._controls = [ctrl.get('name') for ctrl in controls_cfg if ctrl.get('name')]

        except Exception as e:
            # If anything fails, keep defaults
            print(f"Warning: Could not load column config: {e}")
            print("Using default column names")

    @property
    def dimension(self) -> str:
        """Get dimension column name (e.g., 'geo', 'Groups', 'brand')"""
        return self._dimension

    @property
    def date(self) -> str:
        """Get date column name"""
        return self._date

    @property
    def target(self) -> str:
        """Get target/KPI column name"""
        return self._target

    @property
    def channels(self) -> list:
        """Get list of channel column names"""
        return self._channels

    @property
    def controls(self) -> list:
        """Get list of control variable column names"""
        return self._controls

    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration as dictionary.

        Returns:
            Dictionary with all column configurations
        """
        return {
            'dimension': self._dimension,
            'date': self._date,
            'target': self._target,
            'channels': self._channels,
            'controls': self._controls,
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"ColumnConfig(\n"
            f"  dimension='{self._dimension}',\n"
            f"  date='{self._date}',\n"
            f"  target='{self._target}',\n"
            f"  channels={self._channels},\n"
            f"  controls={self._controls}\n"
            f")"
        )


# Convenience function for quick access
def get_column_config(
    config: Optional[Union[Dict, Any]] = None,
    **kwargs
) -> ColumnConfig:
    """
    Create ColumnConfig from config file or parameters.

    Args:
        config: Configuration dict or ConfigDict object
        **kwargs: Override parameters (dimension, date, target, etc.)

    Returns:
        ColumnConfig instance

    Examples:
        # From config
        cols = get_column_config(config)

        # With overrides
        cols = get_column_config(config, dimension="Groups")

        # Direct
        cols = get_column_config(dimension="Groups", date="week", target="revenue")
    """
    return ColumnConfig(config=config, **kwargs)


# Global default instance (can be overridden)
_default_column_config: Optional[ColumnConfig] = None


def set_default_column_config(config: ColumnConfig):
    """
    Set global default column configuration.

    This allows functions to use a default config without passing it every time.

    Args:
        config: ColumnConfig instance to use as default

    Examples:
        from b2b_src.config_loader import load_geo_config
        from b2b_src.column_config import set_default_column_config, ColumnConfig

        config = load_geo_config("geo_config.yml")
        cols = ColumnConfig(config)
        set_default_column_config(cols)

        # Now functions can use the default
        # (if they support it)
    """
    global _default_column_config
    _default_column_config = config


def get_default_column_config() -> ColumnConfig:
    """
    Get global default column configuration.

    Returns:
        Default ColumnConfig instance, or a new one with defaults if not set

    Examples:
        cols = get_default_column_config()
        dimension_col = cols.dimension
    """
    global _default_column_config
    if _default_column_config is None:
        _default_column_config = ColumnConfig()
    return _default_column_config


# Backward compatibility helper
def infer_dimension_column(data, mmm=None, default="geo") -> str:
    """
    Infer dimension column name from data or model.

    This is a backward-compatibility helper for old code.
    New code should use ColumnConfig instead.

    Args:
        data: DataFrame to inspect
        mmm: MMM model object (optional)
        default: Default column name if can't infer

    Returns:
        Inferred dimension column name

    Examples:
        dim_col = infer_dimension_column(data, mmm)
        geo_names = data[dim_col].unique()
    """
    # Try to get from model coords
    if mmm is not None:
        try:
            coords = mmm.model.coords
            # Check common dimension names
            for dim_name in ['Groups', 'geo', 'brand', 'region', 'market']:
                if dim_name in coords:
                    return dim_name
        except Exception:
            pass

    # Try to infer from data columns
    if data is not None:
        for col in ['Groups', 'geo', 'brand', 'region', 'market']:
            if col in data.columns:
                return col

    # Return default
    return default
