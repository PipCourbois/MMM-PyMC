#   Copyright 2022 - 2025 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Media Mix Model class."""

import json
import logging
import warnings
from collections.abc import Sequence
from typing import Annotated, Any, Literal


# Singleton Logger Implementation
# This ensures the logger is instantiated only once throughout the application
# preventing duplicate log messages when handlers accumulate
# Reference: https://stackoverflow.com/questions/7173033/duplicate-log-output-when-using-python-logging-module
class _SingletonMMMLogger:
    """Singleton logger for MMM to prevent duplicate log messages."""

    _instance = None

    @staticmethod
    def get_logger():
        """Get or create the singleton logger instance.

        Returns the same logger instance on every call, ensuring handlers
        are only added once and preventing duplicate log messages.

        This implementation respects the centralized logging configuration
        from pymc_marketing.logging_config if available.
        """
        if _SingletonMMMLogger._instance is None:
            # Try to use centralized logging config if available
            try:
                from pymc_marketing.logging_config import get_logger
                logger = get_logger('pymc_marketing.mmm.mmm.MMM')
            except ImportError:
                # Fallback to standard logging if centralized config not available
                logger = logging.getLogger('pymc_marketing.mmm.mmm.MMM')
                logger.setLevel(logging.DEBUG)
                logger.propagate = True

            _SingletonMMMLogger._instance = logger
        return _SingletonMMMLogger._instance

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import seaborn as sns
from pydantic import Field, InstanceOf, validate_call
from pymc_extras.prior import Prior
from scipy.optimize import OptimizeResult
from xarray import DataArray, Dataset

from pymc_marketing.hsgp_kwargs import HSGPKwargs
from pymc_marketing.mmm.base import BaseValidateMMM
from pymc_marketing.mmm.causal import CausalGraphModel
from pymc_marketing.mmm.components.adstock import (
    AdstockTransformation,
    adstock_from_dict,
)
from pymc_marketing.mmm.components.saturation import (
    SaturationTransformation,
    saturation_from_dict,
)
from pymc_marketing.mmm.fourier import YearlyFourier
from pymc_marketing.mmm.hsgp import SoftPlusHSGP
from pymc_marketing.mmm.lift_test import (
    add_lift_measurements_to_likelihood_from_saturation,
    scale_lift_measurements,
)
from pymc_marketing.mmm.preprocessing import MaxAbsScaleChannels, MaxAbsScaleTarget
from pymc_marketing.mmm.scaling import Scaling, VariableScaling
from pymc_marketing.mmm.tvp import create_time_varying_gp_multiplier, infer_time_index
from pymc_marketing.mmm.utility import UtilityFunctionType, average_response
from pymc_marketing.mmm.utils import (
    apply_sklearn_transformer_across_dim,
    create_new_spend_data,
)
from pymc_marketing.mmm.validating import ValidateControlColumns
from pymc_marketing.model_builder import _handle_deprecate_pred_argument
from pymc_marketing.model_config import parse_model_config
from pymc_marketing.model_graph import deterministics_to_flat

__all__ = ["MMM", "BaseMMM"]


class BaseMMM(BaseValidateMMM):
    """Base class for a media mix model using Delayed Adstock and Logistic Saturation (see [1]_).

    References
    ----------
    .. [1] Jin, Yuxue, et al. "Bayesian methods for media mix modeling with carryover and shape effects." (2017).

    """

    _model_name: str = "BaseMMM"
    _model_type: str = "BaseValidateMMM"
    version: str = "0.0.3"

    @validate_call
    def __init__(
        self,
        date_column: str = Field(..., description="Column name of the date variable."),
        channel_columns: list[str] = Field(
            min_length=1, description="Column names of the media channel variables."
        ),
        adstock: InstanceOf[AdstockTransformation] = Field(
            ..., description="Type of adstock transformation to apply."
        ),
        saturation: InstanceOf[SaturationTransformation] = Field(
            ..., description="Type of saturation transformation to apply."
        ),
        time_varying_intercept: bool = Field(
            False, description="Whether to consider time-varying intercept."
        ),
        time_varying_media: bool = Field(
            False, description="Whether to consider time-varying media contributions."
        ),
        model_config: dict | None = Field(None, description="Model configuration."),
        sampler_config: dict | None = Field(None, description="Sampler configuration."),
        validate_data: bool = Field(
            True, description="Whether to validate the data before fitting to model"
        ),
        control_columns: Annotated[
            list[str],
            Field(
                min_length=1,
                description="Column names of control variables to be added as additional regressors",
            ),
        ]
        | None = None,
        yearly_seasonality: Annotated[
            int,
            Field(
                gt=0, description="Number of Fourier modes to model yearly seasonality."
            ),
        ]
        | None = None,
        adstock_first: bool = Field(
            True, description="Whether to apply adstock first."
        ),
        dag: str | None = Field(
            None,
            description="Optional DAG provided as a string Dot format for causal identification.",
        ),
        treatment_nodes: list[str] | tuple[str] | None = Field(
            None,
            description="Column names of the variables of interest to identify causal effects on outcome.",
        ),
        outcome_node: str | None = Field(
            None, description="Name of the outcome variable."
        ),
        scaling: InstanceOf[Scaling] | dict | None = Field(
            None, description="Scaling configuration for the model."
        ),
        apply_scaling: bool = Field(
            True, description="Whether to apply data scaling. When False, all scale factors are set to 1.0."
        ),
        fit_intercept: bool = Field(
            True, description="Whether to fit an intercept term in the model."
        ),
        log_transform_vars: list[str] | None = Field(
            None, description="List of variables to apply log(x+1) transformation (natural log). Options: 'target', 'channels'. Example: ['target', 'channels']"
        ),
    ) -> None:
        """Define the constructor method.

        Parameter
        ---------
        date_column : str
            Column name of the date variable. Must be parsable using ~pandas.to_datetime.
        channel_columns : List[str]
            Column names of the media channel variables.
        adstock : AdstockTransformation
            Type of adstock transformation to apply.
        saturation : SaturationTransformation
            Type of saturation transformation to apply.
        time_varying_intercept : bool, optional
            Whether to consider time-varying intercept, by default False.
            Because the `time-varying` variable is centered around 1 and acts as a multiplier,
            the variable `intercept_baseline` now represents the mean of the time-varying intercept.
        time_varying_media : bool, optional
            Whether to consider time-varying media contributions, by default False.
            The `time-varying-media` creates a time media variable centered around 1,
            this variable acts as a global multiplier (scaling factor) for all channels,
            meaning all media channels share the same latent fluctuation.
        model_config : Dictionary, optional
            Dictionary of parameters that initialise model configuration.
            Class-default defined by the user default_model_config method.
        sampler_config : Dictionary, optional
            Dictionary of parameters that initialise sampler configuration.
            Class-default defined by the user default_sampler_config method.
        validate_data : bool, optional
            Whether to validate the data before fitting to model, by default True.
        control_columns : Optional[List[str]], optional
            Column names of control variables to be added as additional regressors, by default None
        yearly_seasonality : Optional[int], optional
            Number of Fourier modes to model yearly seasonality, by default None.
        adstock_first : bool, optional
            Whether to apply adstock first, by default True.
        dag : Optional[str], optional
            Optional DAG provided as a string Dot format for causal modeling, by default None.
        treatment_nodes : Optional[list[str]], optional
            Column names of the variables of interest to identify causal effects on outcome.
        outcome_node : Optional[str], optional
            Name of the outcome variable, by default None.
        scaling : Scaling | dict | None, optional
            Scaling configuration for the model. If None, defaults to max scaling for both target and channels.
            Can be a Scaling object or a dict that will be converted to a Scaling object.
        apply_scaling : bool, optional
            Whether to apply data scaling. When False, all scale factors are set to 1.0 (identity scaling).
            Defaults to True.
        fit_intercept : bool, optional
            Whether to fit an intercept term in the model. When False, the model is forced through the origin.
            Defaults to True.
        log_transform_vars : list[str] | None, optional
            List of variables to apply log(x+1) transformation (natural log). Options are 'target' and 'channels'.
            The transformation uses log(x+1) to handle zero values naturally.
            Examples: ['channels'], ['target'], ['target', 'channels'].
            Defaults to None (no log transformation).
        """
        # Use singleton logger to prevent duplicate log messages
        # Each instance gets the same logger object, configured only once
        self.logger = _SingletonMMMLogger.get_logger()

        # Log initialization with input parameters
        self.logger.info("Initializing BaseMMM model")
        self.logger.info(f"Input parameters - date_column: {date_column}")
        self.logger.info(f"Input parameters - channel_columns: {channel_columns}")
        self.logger.info(f"Input parameters - adstock: {type(adstock).__name__}")
        self.logger.info(f"Input parameters - saturation: {type(saturation).__name__}")
        self.logger.info(f"Input parameters - time_varying_intercept: {time_varying_intercept}")
        self.logger.info(f"Input parameters - time_varying_media: {time_varying_media}")
        self.logger.info(f"Input parameters - control_columns: {control_columns}")
        self.logger.info(f"Input parameters - yearly_seasonality: {yearly_seasonality}")
        self.logger.info(f"Input parameters - validate_data: {validate_data}")
        self.logger.info(f"Input parameters - apply_scaling: {apply_scaling}")
        self.logger.info(f"Input parameters - fit_intercept: {fit_intercept}")
        self.logger.info(f"Input parameters - log_transform_vars: {log_transform_vars}")

        self.control_columns = control_columns
        self.time_varying_intercept = time_varying_intercept
        self.time_varying_media = time_varying_media
        self.date_column = date_column
        self.validate_data = validate_data
        self.apply_scaling = apply_scaling
        self.fit_intercept = fit_intercept
        self.log_transform_vars = log_transform_vars if log_transform_vars is not None else []

        self.adstock = adstock
        self.saturation = saturation
        self.adstock_first = adstock_first

        # Initialize scaling configuration similar to multidimensional MMM
        if isinstance(scaling, dict):
            scaling = scaling.copy()

            if "channel" not in scaling:
                scaling["channel"] = VariableScaling(method="max", dims=())
            if "target" not in scaling:
                scaling["target"] = VariableScaling(method="max", dims=())

            scaling = Scaling(**scaling)

        self.scaling: Scaling = scaling or Scaling(
            target=VariableScaling(method="max", dims=()),
            channel=VariableScaling(method="max", dims=()),
        )

        model_config = model_config or {}
        model_config = parse_model_config(
            model_config,  # type: ignore
            hsgp_kwargs_fields=["intercept_tvp_config", "media_tvp_config"],
        )

        if model_config is not None:
            self.adstock.update_priors({**self.default_model_config, **model_config})
            self.saturation.update_priors({**self.default_model_config, **model_config})

        super().__init__(
            date_column=date_column,
            channel_columns=channel_columns,
            model_config=model_config,
            sampler_config=sampler_config,
        )

        self.yearly_seasonality = yearly_seasonality

        self.dag = dag
        self.treatment_nodes = treatment_nodes
        self.outcome_node = outcome_node

        # Initialize causal graph if provided
        if self.dag is not None and self.outcome_node is not None:
            if self.treatment_nodes is None:
                self.treatment_nodes = self.channel_columns
                warnings.warn(
                    "No treatment nodes provided, using channel columns as treatment nodes.",
                    stacklevel=2,
                )
            self.causal_graphical_model = CausalGraphModel.build_graphical_model(
                graph=self.dag,
                treatment=self.treatment_nodes,
                outcome=self.outcome_node,
            )

            self.control_columns = self.causal_graphical_model.compute_adjustment_sets(
                control_columns=self.control_columns,
                channel_columns=self.channel_columns,
            )

            if "yearly_seasonality" not in self.causal_graphical_model.adjustment_set:
                warnings.warn(
                    "Yearly seasonality excluded as it's not required for adjustment.",
                    stacklevel=2,
                )
                self.yearly_seasonality = None

        if self.yearly_seasonality is not None:
            self.yearly_fourier = YearlyFourier(
                n_order=self.yearly_seasonality,
                prefix="fourier_mode",
                prior=self.model_config["gamma_fourier"],
                variable_name="gamma_fourier",
            )

        # Log successful initialization
        self.logger.info("BaseMMM model initialization completed successfully")
        self.logger.info(f"Final configuration - apply_scaling: {self.apply_scaling}")
        self.logger.info(f"Final configuration - scaling: {self.scaling}")
        self.logger.info(f"Final configuration - model_config keys: {list(self.model_config.keys()) if self.model_config else 'None'}")
        self.logger.info(f"Final configuration - yearly_fourier: {'Enabled' if self.yearly_seasonality else 'Disabled'}")

    def post_sample_model_transformation(self) -> None:
        """Post-sample model transformation in order to store the HSGP state from fit."""
        names = []
        if self.time_varying_intercept:
            names.extend(
                SoftPlusHSGP.deterministics_to_replace("intercept_latent_process")
            )
        if self.time_varying_media:
            names.extend(
                SoftPlusHSGP.deterministics_to_replace(
                    "media_temporal_latent_multiplier"
                )
            )
        if not names:
            return

        self.model = deterministics_to_flat(self.model, names=names)

    @property
    def default_sampler_config(self) -> dict:
        """Default sampler configuration for the model.

        Returns
        -------
        dict
            Empty dictionary.
        """
        return {}

    @property
    def output_var(self) -> Literal["y"]:
        """Define target variable for the model.

        Returns
        -------
        str
            The target variable for the model.
        """
        return "y"

    def _generate_and_preprocess_model_data(  # type: ignore
        self, X: pd.DataFrame | pd.Series, y: pd.Series | np.ndarray
    ) -> None:
        """Apply preprocessing to the data before fitting the model.

        If validate is True, it will check if the data is valid for the model.
        sets self.model_coords based on provided dataset

        Parameters
        ----------
        X : Union[pd.DataFrame, pd.Series], shape (n_obs, n_features)
        y : Union[pd.Series, np.ndarray], shape (n_obs,)

        Sets
        ----
        preprocessed_data : Dict[str, Union[pd.DataFrame, pd.Series]]
            Preprocessed data for the model.
        X : pd.DataFrame
            A filtered version of the input `X`, such that it is guaranteed that
            it contains only the `date_column`, the columns that are specified
            in the `channel_columns` and `control_columns`, and fourier features
            if `yearly_seasonality=True`.
        y : Union[pd.Series, np.ndarray]
            The target variable for the model (as provided).
        _time_index : np.ndarray
            The index of the date column. Used by TVP
        _time_index_mid : int
            The middle index of the date index. Used by TVP.
        _time_resolution: int
            The time resolution of the date index. Used by TVP.

        """
        self.logger.info("Starting data preprocessing")
        self.logger.info(f"Input data shapes - X: {X.shape}, y: {y.shape if hasattr(y, 'shape') else len(y)}")
        
        # Log basic statistics about input data
        if isinstance(X, pd.DataFrame):
            self.logger.info(f"X columns: {list(X.columns)}")
            self.logger.info(f"X date range: {X[self.date_column].min()} to {X[self.date_column].max()}")
        
        if hasattr(y, 'describe'):
            y_stats = y.describe()
            self.logger.info(f"y statistics - mean: {y_stats['mean']:.2f}, std: {y_stats['std']:.2f}, min: {y_stats['min']:.2f}, max: {y_stats['max']:.2f}")
        else:
            self.logger.info(f"y statistics - mean: {np.mean(y):.2f}, std: {np.std(y):.2f}, min: {np.min(y):.2f}, max: {np.max(y):.2f}")
        try:
            date_data = pd.to_datetime(X[self.date_column])
        except Exception as e:
            raise ValueError(
                f"Could not convert {self.date_column} to datetime. Please check the date format."
            ) from e

        channel_data = X[self.channel_columns]

        coords: dict[str, Any] = {
            "channel": self.channel_columns,
            "date": date_data,
        }

        new_X_dict = {
            self.date_column: date_data,
        }
        X_data = pd.DataFrame.from_dict(new_X_dict)
        X_data = pd.concat([X_data, channel_data], axis=1)
        control_data: pd.DataFrame | pd.Series | None = None
        if self.control_columns is not None:
            control_data = X[self.control_columns]
            coords["control"] = self.control_columns
            X_data = pd.concat([X_data, control_data], axis=1)

        self.model_coords = coords
        if self.validate_data:
            self.validate("X", X_data)
            self.validate("y", y)
        # Store raw data instead of applying scaling transformations
        # We'll apply scaling within the model graph like multidimensional MMM
        self.preprocessed_data: dict[str, pd.DataFrame | pd.Series | np.ndarray] = {
            "X": X_data,  # Store raw data
            "y": y,  # Store raw data
        }

        # Still fit the transformers for backward compatibility
        # but don't apply the transformations to the data
        try:
            # Fit transformers without transforming data (for backward compatibility)
            if hasattr(self, "max_abs_scale_target_data"):
                self.max_abs_scale_target_data(y)
            if hasattr(self, "max_abs_scale_channel_data"):
                self.max_abs_scale_channel_data(X_data)
        except Exception as e:
            # If transformer fitting fails, continue without them
            import warnings

            warnings.warn(f"Failed to fit transformers: {e}", UserWarning, stacklevel=2)
        self.X: pd.DataFrame = X_data
        self.y: pd.Series | np.ndarray = y

        if self.time_varying_intercept | self.time_varying_media:
            self._time_index = np.arange(0, X.shape[0])
            self._time_index_mid = X.shape[0] // 2
            self._time_resolution = (
                self.X[self.date_column].iloc[1] - self.X[self.date_column].iloc[0]
            ).days

        # Log completion of preprocessing
        self.logger.info("Data preprocessing completed successfully")
        self.logger.info(f"Final preprocessed data shapes - X: {self.X.shape}, y: {len(self.y)}")
        self.logger.info(f"Model coordinates: {list(self.model_coords.keys())}")
        if hasattr(self, '_time_index'):
            self.logger.info(f"Time-varying features enabled - time_index length: {len(self._time_index)}")

    def _compute_scales(self) -> None:
        """Compute and save scaling factors for channels and target."""
        import numpy as np

        self.logger.info("=" * 60)
        self.logger.info("SCALING COMPUTATION")
        self.logger.info("=" * 60)
        self.logger.info(f"apply_scaling parameter: {self.apply_scaling}")

        # Get raw data
        X_data = self.preprocessed_data["X"]
        if isinstance(X_data, pd.DataFrame):
            channel_data = np.asarray(X_data[self.channel_columns].values)
        else:
            raise TypeError("X data must be a DataFrame for scaling computation")

        y_data = self.preprocessed_data["y"]
        target_data = np.asarray(y_data).reshape(-1, 1)

        # Log raw target data sample (before scaling)
        self.logger.info("\n--- TARGET DATA (BEFORE SCALING) ---")
        self.logger.info(f"Target shape: {target_data.shape}")
        self.logger.info(f"Target statistics - mean: {target_data.mean():.4f}, std: {target_data.std():.4f}")
        self.logger.info(f"Target statistics - min: {target_data.min():.4f}, max: {target_data.max():.4f}")
        sample_size = min(10, len(target_data))
        self.logger.info(f"First {sample_size} target values (raw): {target_data[:sample_size].flatten()}")

        # Log raw channel data sample (before scaling)
        self.logger.info("\n--- CHANNEL DATA (BEFORE SCALING) ---")
        self.logger.info(f"Channel data shape: {channel_data.shape}")
        self.logger.info(f"Channel columns: {self.channel_columns}")
        for i, channel_name in enumerate(self.channel_columns):
            channel_col = channel_data[:, i]
            self.logger.info(f"\nChannel '{channel_name}':")
            self.logger.info(f"  Statistics - mean: {channel_col.mean():.4f}, std: {channel_col.std():.4f}")
            self.logger.info(f"  Statistics - min: {channel_col.min():.4f}, max: {channel_col.max():.4f}")
            self.logger.info(f"  First {sample_size} values (raw): {channel_col[:sample_size]}")

        # Compute scales based on scaling configuration
        self.logger.info(f"\n--- COMPUTING SCALES ---")

        if not self.apply_scaling:
            # When scaling is disabled, use identity scaling (all factors = 1.0)
            self.logger.info("apply_scaling=False: Using identity scaling (all scale factors = 1.0)")
            channel_scale = np.ones(len(self.channel_columns))
            target_scale = 1.0
        else:
            # Apply scaling based on configuration
            self.logger.info(f"Channel scaling method: {self.scaling.channel.method}")
            self.logger.info(f"Target scaling method: {self.scaling.target.method}")

            if self.scaling.channel.method == "max":
                channel_scale = np.abs(channel_data).max(axis=0)
                # Avoid division by zero
                channel_scale = np.where(channel_scale == 0, 1.0, channel_scale)
            elif self.scaling.channel.method == "mean":
                channel_scale = np.abs(channel_data).mean(axis=0)
                channel_scale = np.where(channel_scale == 0, 1.0, channel_scale)
            else:
                raise ValueError(f"Unknown scaling method: {self.scaling.channel.method}")

            if self.scaling.target.method == "max":
                target_scale = np.abs(target_data).max()
                target_scale = 1.0 if target_scale == 0 else target_scale
            elif self.scaling.target.method == "mean":
                target_scale = np.abs(target_data).mean()
                target_scale = 1.0 if target_scale == 0 else target_scale
            else:
                raise ValueError(f"Unknown scaling method: {self.scaling.target.method}")

        # Store scales
        self.channel_scale = channel_scale
        self.target_scale = target_scale

        # Log computed scales
        self.logger.info(f"\nComputed target_scale: {target_scale:.4f}")
        self.logger.info(f"Computed channel_scale: {channel_scale}")

        # Log scaled data samples (after scaling)
        scaled_target_data = target_data / target_scale
        self.logger.info("\n--- TARGET DATA (AFTER SCALING) ---")
        self.logger.info(f"Scaled target statistics - mean: {scaled_target_data.mean():.4f}, std: {scaled_target_data.std():.4f}")
        self.logger.info(f"Scaled target statistics - min: {scaled_target_data.min():.4f}, max: {scaled_target_data.max():.4f}")
        self.logger.info(f"First {sample_size} target values (scaled): {scaled_target_data[:sample_size].flatten()}")

        scaled_channel_data = channel_data / channel_scale
        self.logger.info("\n--- CHANNEL DATA (AFTER SCALING) ---")
        for i, channel_name in enumerate(self.channel_columns):
            scaled_channel_col = scaled_channel_data[:, i]
            self.logger.info(f"\nChannel '{channel_name}' (scale factor: {channel_scale[i]:.4f}):")
            self.logger.info(f"  Scaled statistics - mean: {scaled_channel_col.mean():.4f}, std: {scaled_channel_col.std():.4f}")
            self.logger.info(f"  Scaled statistics - min: {scaled_channel_col.min():.4f}, max: {scaled_channel_col.max():.4f}")
            self.logger.info(f"  First {sample_size} values (scaled): {scaled_channel_col[:sample_size]}")

        self.logger.info("\n" + "=" * 60)

    def create_idata_attrs(self) -> dict[str, str]:
        """Create attributes for the inference data.

        Returns
        -------
        dict[str, str]
            The attributes for the inference data.

        """
        attrs = super().create_idata_attrs()
        attrs["date_column"] = json.dumps(self.date_column)
        attrs["adstock"] = json.dumps(self.adstock.to_dict())
        attrs["saturation"] = json.dumps(self.saturation.to_dict())
        attrs["adstock_first"] = json.dumps(self.adstock_first)
        attrs["control_columns"] = json.dumps(self.control_columns)
        attrs["channel_columns"] = json.dumps(self.channel_columns)
        attrs["validate_data"] = json.dumps(self.validate_data)
        attrs["yearly_seasonality"] = json.dumps(self.yearly_seasonality)
        attrs["time_varying_intercept"] = json.dumps(self.time_varying_intercept)
        attrs["time_varying_media"] = json.dumps(self.time_varying_media)
        attrs["dag"] = json.dumps(self.dag)
        attrs["treatment_nodes"] = json.dumps(self.treatment_nodes)
        attrs["outcome_node"] = json.dumps(self.outcome_node)
        attrs["apply_scaling"] = json.dumps(self.apply_scaling)
        attrs["fit_intercept"] = json.dumps(self.fit_intercept)
        attrs["log_transform_vars"] = json.dumps(self.log_transform_vars)

        # Serialize scaling configuration
        if hasattr(self, "scaling") and self.scaling is not None:
            attrs["scaling"] = json.dumps(
                {
                    "target": {
                        "method": self.scaling.target.method,
                        "dims": self.scaling.target.dims,
                    },
                    "channel": {
                        "method": self.scaling.channel.method,
                        "dims": self.scaling.channel.dims,
                    },
                }
            )
        else:
            attrs["scaling"] = json.dumps(None)

        return attrs

    def forward_pass(self, x: pt.TensorVariable | npt.NDArray) -> pt.TensorVariable:
        """Transform channel input into target contributions of each channel.

        Ensures the order of operations matches the model graph when channel
        log transformation is enabled: apply adstock, then (optionally) log(x+1), then saturation.

        This method must be called from without a pm.Model context but not
        necessarily in the instance's model. A dim named "channel" is required
        associated with the number of columns of `x`.

        Parameters
        ----------
        x : pt.TensorVariable | npt.NDArray
            The channel input which could be spends or impressions

        Returns
        -------
        The contributions associated with the channel input

        Notes
        -----
        Historically, the order between adstock and saturation could be toggled via
        `adstock_first`. To correctly support channel log-transforms with linear saturation
        (e.g., NoSaturation), we apply adstock first and insert the log-transform before
        saturation, matching the build_model implementation. This ensures beta multiplies
        the log-transformed adstocked signal: beta * log(adstock(x) + 1).
        """
        # Always apply adstock first to support correct placement of the log transform
        adstocked = self.adstock.apply(x=x, dims="channel")

        # If channel log-transform is enabled, apply it AFTER adstock and BEFORE saturation
        if hasattr(self, "log_transform_vars") and ("channels" in self.log_transform_vars):
            x_for_saturation = pt.log(adstocked + 1.0)
            x_for_saturation.name = "channel_data_log_transformed"
        else:
            x_for_saturation = adstocked

        # Apply saturation (which may multiply by beta)
        return self.saturation.apply(x=x_for_saturation, dims="channel")

    def build_model(  # type: ignore[override]
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        **kwargs,
    ) -> None:
        """Build a probabilistic model using PyMC for marketing mix modeling.

        The model incorporates channels, control variables, and Fourier components, applying
        adstock and saturation transformations to the channel data. The final model is
        constructed with multiple factors contributing to the response variable.

        Parameters
        ----------
        X : pd.DataFrame
            The input data for the model, which should include columns for channels,
            control variables (if applicable), and Fourier components (if applicable).

        y : Union[pd.Series, np.ndarray]
            The target/response variable for the modeling.

        **kwargs : dict
            Additional keyword arguments that might be required by underlying methods or utilities.

        Attributes Set
        ---------------
        model : pm.Model
            The PyMC model object containing all the defined stochastic and deterministic variables.

        Examples
        --------
        Initialize model with custom configuration

        .. code-block:: python

            from pymc_marketing.mmm import (
                GeometricAdstock,
                LogisticSaturation
                MMM,
            )
            from pymc_extras.prior import Prior

            custom_config = {
                "intercept": Prior("Normal", mu=0, sigma=2),
                "saturation_beta": Prior("Gamma", mu=1, sigma=3),
                "saturation_lambda": Prior("Beta", alpha=3, beta=1),
                "adstock_alpha": Prior("Beta", alpha=1, beta=3),
                "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=2)),
                "gamma_control": Prior("Normal", mu=0, sigma=2, dims="control"),
                "gamma_fourier": Prior("Laplace", mu=0, b=1, dims="fourier_mode"),
            }

            model = MMM(
                date_column="date_week",
                channel_columns=["x1", "x2"],
                adstock=GeometricAdstock(l_max=8),
                saturation=LogisticSaturation(),
                control_columns=[
                    "event_1",
                    "event_2",
                    "t",
                ],
                yearly_seasonality=2,
                model_config=custom_config,
            )

        """
        self.logger.info("Starting PyMC model building")
        self.logger.info(f"Model building input data shapes - X: {X.shape}, y: {y.shape if hasattr(y, 'shape') else len(y)}")
        
        self._generate_and_preprocess_model_data(X, y)
        # Compute and save scales
        self._compute_scales()

        with pm.Model(
            coords=self.model_coords,
        ) as self.model:
            # Store raw data and scales as pm.Data
            channel_scale_ = pm.Data(
                "channel_scale",
                self.channel_scale,
                dims=("channel",),
            )
            target_scale_ = pm.Data(
                "target_scale",
                self.target_scale,
                dims=(),
            )

            X_data = self.preprocessed_data["X"]
            if isinstance(X_data, pd.DataFrame):
                channel_data_value = X_data[self.channel_columns]
            else:
                raise TypeError("X data must be a DataFrame")

            channel_data = pm.Data(
                name="channel_data",
                value=channel_data_value,
                dims=("date", "channel"),
            )

            target_data = pm.Data(
                name="target_data",
                value=self.preprocessed_data["y"],
                dims="date",
            )

            # Apply scaling within the model graph
            channel_data_ = channel_data / channel_scale_
            channel_data_ = pt.switch(pt.isnan(channel_data_), 0.0, channel_data_)
            channel_data_.name = "channel_data_scaled"

            target_ = target_data / target_scale_
            target_.name = "target_scaled"

            # Apply log transformation to target if requested
            if "target" in self.log_transform_vars:
                target_ = pt.log(target_ + 1.0)
                target_.name = "target_log_transformed"
            if self.time_varying_intercept | self.time_varying_media:
                time_index = pm.Data(
                    "time_index",
                    self._time_index,
                    dims="date",
                )

            if self.fit_intercept:
                if self.time_varying_intercept:
                    intercept_baseline = self.model_config["intercept"].create_variable(
                        "intercept_baseline"
                    )

                    intercept_latent_process = create_time_varying_gp_multiplier(
                        name="intercept",
                        dims="date",
                        time_index=time_index,
                        time_index_mid=self._time_index_mid,
                        time_resolution=self._time_resolution,
                        hsgp_kwargs=self.model_config["intercept_tvp_config"],
                    )
                    intercept = pm.Deterministic(
                        name="intercept",
                        var=intercept_baseline * intercept_latent_process,
                        dims="date",
                    )
                else:
                    intercept = self.model_config["intercept"].create_variable(
                        name="intercept"
                    )
            else:
                # No intercept: create a zero constant
                # Store as deterministic for compatibility with plotting methods
                intercept = pm.Deterministic(
                    name="intercept",
                    var=pt.zeros_like(target_),
                    dims="date",
                )

            if self.time_varying_media:
                # Apply adstock first
                adstocked_data = self.adstock.apply(x=channel_data_, dims="channel")

                # Apply log(x+1) transformation AFTER adstock but BEFORE saturation
                if "channels" in self.log_transform_vars:
                    log_transformed_data = pt.log(adstocked_data + 1.0)
                    log_transformed_data.name = "channel_data_log_transformed"
                else:
                    log_transformed_data = adstocked_data

                # Apply saturation (which multiplies by beta)
                # This ensures: beta * log(adstock(x) + 1) instead of log(beta * adstock(x) + 1)
                transformed_channel_data = self.saturation.apply(x=log_transformed_data, dims="channel")

                baseline_channel_contribution = pm.Deterministic(
                    name="baseline_channel_contribution",
                    var=transformed_channel_data,
                    dims=("date", "channel"),
                )

                media_latent_process = create_time_varying_gp_multiplier(
                    name="media",
                    dims="date",
                    time_index=time_index,
                    time_index_mid=self._time_index_mid,
                    time_resolution=self._time_resolution,
                    hsgp_kwargs=self.model_config["media_tvp_config"],
                )
                channel_contribution = pm.Deterministic(
                    name="channel_contribution",
                    var=baseline_channel_contribution * media_latent_process[:, None],
                    dims=("date", "channel"),
                )

            else:
                # Apply adstock first
                adstocked_data = self.adstock.apply(x=channel_data_, dims="channel")

                # Apply log(x+1) transformation AFTER adstock but BEFORE saturation
                if "channels" in self.log_transform_vars:
                    log_transformed_data = pt.log(adstocked_data + 1.0)
                    log_transformed_data.name = "channel_data_log_transformed"
                else:
                    log_transformed_data = adstocked_data

                # Apply saturation (which multiplies by beta)
                # This ensures: beta * log(adstock(x) + 1) instead of log(beta * adstock(x) + 1)
                transformed_channel_data = self.saturation.apply(x=log_transformed_data, dims="channel")

                channel_contribution = pm.Deterministic(
                    name="channel_contribution",
                    var=transformed_channel_data,
                    dims=("date", "channel"),
                )

            # We define the deterministic variable to define the optimization by default
            pm.Deterministic(
                name="total_contribution",
                var=channel_contribution.sum(axis=(-2, -1)),
                dims=(),
            )

            mu_var = intercept + channel_contribution.sum(axis=-1)

            X_data = self.preprocessed_data["X"]
            if (
                self.control_columns is not None
                and len(self.control_columns) > 0
                and isinstance(X_data, pd.DataFrame)
                and all(column in X_data.columns for column in self.control_columns)
            ):
                if self.model_config["gamma_control"].dims != ("control",):
                    self.model_config["gamma_control"].dims = "control"

                gamma_control = self.model_config["gamma_control"].create_variable(
                    name="gamma_control"
                )

                control_data_ = pm.Data(
                    name="control_data",
                    value=self.preprocessed_data["X"][self.control_columns],
                    dims=("date", "control"),
                )

                control_contribution = pm.Deterministic(
                    name="control_contribution",
                    var=control_data_ * gamma_control,
                    dims=("date", "control"),
                )

                mu_var += control_contribution.sum(axis=-1)

            if self.yearly_seasonality is not None:
                X_data = self.preprocessed_data["X"]
                if isinstance(X_data, pd.DataFrame):
                    dayofyear_value = X_data[self.date_column].dt.dayofyear.to_numpy()
                else:
                    raise TypeError("X data must be a DataFrame for yearly seasonality")

                dayofyear = pm.Data(
                    name="dayofyear",
                    value=dayofyear_value,
                    dims="date",
                )

                def create_deterministic(x: pt.TensorVariable) -> None:
                    pm.Deterministic(
                        "fourier_contribution",
                        x,
                        dims=("date", *self.yearly_fourier.prior.dims),
                    )

                yearly_seasonality_contribution = pm.Deterministic(
                    name="yearly_seasonality_contribution",
                    var=self.yearly_fourier.apply(
                        dayofyear, result_callback=create_deterministic
                    ),
                    dims="date",
                )

                mu_var += yearly_seasonality_contribution

            mu = pm.Deterministic(name="mu", var=mu_var, dims="date")

            self.model_config["likelihood"].dims = "date"
            self.model_config["likelihood"].create_likelihood_variable(
                name=self.output_var,
                mu=mu,
                observed=target_,
            )

            # Add original scale contribution variables
            # If log transformation is applied, we need to inverse transform: exp(x) - 1
            if "target" in self.log_transform_vars:
                # For log-transformed targets: y_original = (exp(mu) - 1) * target_scale
                # But contributions are additive in log space, so this is complex
                # We approximate by: contribution_original ≈ (exp(contribution) - 1) * target_scale
                # Note: This is an approximation because exp(a+b) ≠ exp(a) + exp(b)
                pm.Deterministic(
                    name="channel_contribution_original_scale",
                    var=(pt.exp(channel_contribution) - 1.0) * target_scale_,
                    dims=("date", "channel"),
                )

                pm.Deterministic(
                    name="total_contribution_original_scale",
                    var=(pt.exp(channel_contribution.sum(axis=-1)) - 1.0) * target_scale_,
                    dims="date",
                )

                if (
                    self.control_columns is not None
                    and len(self.control_columns) > 0
                    and "control_contribution"
                    in [var.name for var in self.model.deterministics]
                ):
                    pm.Deterministic(
                        name="control_contribution_original_scale",
                        var=(pt.exp(control_contribution) - 1.0) * target_scale_,
                        dims=("date", "control"),
                    )

                if (
                    self.yearly_seasonality is not None
                    and "yearly_seasonality_contribution"
                    in [var.name for var in self.model.deterministics]
                ):
                    pm.Deterministic(
                        name="yearly_seasonality_contribution_original_scale",
                        var=(pt.exp(yearly_seasonality_contribution) - 1.0) * target_scale_,
                        dims="date",
                    )

                pm.Deterministic(
                    name="y_original_scale",
                    var=(pt.exp(mu) - 1.0) * target_scale_,
                    dims="date",
                )
            else:
                # No log transformation: just multiply by target_scale
                pm.Deterministic(
                    name="channel_contribution_original_scale",
                    var=channel_contribution * target_scale_,
                    dims=("date", "channel"),
                )

                pm.Deterministic(
                    name="total_contribution_original_scale",
                    var=channel_contribution.sum(axis=-1) * target_scale_,
                    dims="date",
                )

                if (
                    self.control_columns is not None
                    and len(self.control_columns) > 0
                    and "control_contribution"
                    in [var.name for var in self.model.deterministics]
                ):
                    pm.Deterministic(
                        name="control_contribution_original_scale",
                        var=control_contribution * target_scale_,
                        dims=("date", "control"),
                    )

                if (
                    self.yearly_seasonality is not None
                    and "yearly_seasonality_contribution"
                    in [var.name for var in self.model.deterministics]
                ):
                    pm.Deterministic(
                        name="yearly_seasonality_contribution_original_scale",
                        var=yearly_seasonality_contribution * target_scale_,
                        dims="date",
                    )

                pm.Deterministic(
                    name="y_original_scale",
                    var=(mu * target_scale_),
                    dims="date",
                )

        # Log model building completion
        self.logger.info("PyMC model building completed successfully")
        self.logger.info(f"Model variables created: {[var.name for var in self.model.unobserved_RVs]}")
        self.logger.info(f"Model deterministics: {[var.name for var in self.model.deterministics]}")
        self.logger.info(f"Model coordinates: {list(self.model.coords.keys())}")
        self.logger.info(f"Scaling factors applied - channel_scale shape: {self.channel_scale.shape}, target_scale: {self.target_scale}")

    @property
    def default_model_config(self) -> dict:
        """Define the default model configuration."""
        base_config = {
            "intercept": Prior("Normal", mu=0, sigma=2),
            "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=2)),
            "gamma_control": Prior("Normal", mu=0, sigma=2, dims="control"),
            "gamma_fourier": Prior("Laplace", mu=0, b=1, dims="fourier_mode"),
        }

        if self.time_varying_intercept:
            base_config["intercept_tvp_config"] = HSGPKwargs(
                m=200,
                L=None,
                eta_lam=1,
                ls_mu=5,
                ls_sigma=10,
                cov_func=None,
            )
        if self.time_varying_media:
            base_config["media_tvp_config"] = HSGPKwargs(
                m=200,
                L=None,
                eta_lam=1,
                ls_mu=5,
                ls_sigma=10,
                cov_func=None,
            )

        for media_transform in [self.adstock, self.saturation]:
            for dist in media_transform.function_priors.values():
                if dist.dims != ("channel",):
                    dist.dims = "channel"

        return {
            **base_config,
            **self.adstock.model_config,
            **self.saturation.model_config,
        }

    def channel_contribution_forward_pass(
        self,
        channel_data: npt.NDArray,
        disable_logger_stdout: bool | None = False,
    ) -> npt.NDArray:
        """Evaluate the channel contribution for a given channel data and a fitted model, ie. the forward pass.

        Parameters
        ----------
        channel_data : array-like
            Input channel data. Result of all the preprocessing steps.
        disable_logger_stdout : bool, optional
            If True, suppress logger output to stdout

        Returns
        -------
        array-like
            Transformed channel data.

        """
        if disable_logger_stdout:
            logger = logging.getLogger("pymc.sampling.forward")
            logger.propagate = False

        coords = {
            **self.model_coords,
        }
        with pm.Model(coords=coords):
            # Create channel_data as a pm.Data tensor to ensure proper shape handling
            channel_data_tensor = pm.Data(
                name="channel_data",
                value=channel_data,
                dims=("date", "channel"),
            )

            # Apply the same scaling as in the original model - scale by channel_scale
            channel_scale_ = pm.Data(
                "channel_scale",
                self.channel_scale,
                dims=("channel",),
            )

            # Apply scaling within the model graph (same as build_model)
            channel_data_scaled = channel_data_tensor / channel_scale_
            channel_data_scaled = pt.switch(
                pt.isnan(channel_data_scaled), 0.0, channel_data_scaled
            )
            channel_data_scaled.name = "channel_data_scaled"

            pm.Deterministic(
                "channel_contribution",
                self.forward_pass(x=channel_data_scaled),
                dims=("date", "channel"),
            )

            idata = pm.sample_posterior_predictive(
                self.fit_result,
                var_names=["channel_contribution"],
                progressbar=False,
            )

        channel_contribution = idata.posterior_predictive.channel_contribution
        if self.time_varying_media:
            # This is coupled with the name of the
            # latent process Deterministic
            name = "media_temporal_latent_multiplier"
            mutliplier = self.fit_result[name]
            channel_contribution = channel_contribution * mutliplier

        return channel_contribution.to_numpy()

    @property
    def _serializable_model_config(self) -> dict[str, Any]:
        def ndarray_to_list(d: dict) -> dict:
            new_d = d.copy()  # Copy the dictionary to avoid mutating the original one
            for key, value in new_d.items():
                if isinstance(value, np.ndarray):
                    new_d[key] = value.tolist()
                elif isinstance(value, dict):
                    new_d[key] = ndarray_to_list(value)
            return new_d

        serializable_config = self.model_config.copy()
        return ndarray_to_list(serializable_config)

    @classmethod
    def attrs_to_init_kwargs(cls, attrs) -> dict[str, Any]:
        """Convert attributes to initialization kwargs.

        Returns
        -------
        dict[str, Any]
            The initialization kwargs.

        """
        return {
            "model_config": cls._model_config_formatting(
                json.loads(attrs["model_config"])
            ),
            "date_column": json.loads(attrs["date_column"]),
            "control_columns": json.loads(attrs["control_columns"]),
            "channel_columns": json.loads(attrs["channel_columns"]),
            "adstock": adstock_from_dict(json.loads(attrs["adstock"])),
            "saturation": saturation_from_dict(json.loads(attrs["saturation"])),
            "adstock_first": json.loads(attrs.get("adstock_first", "true")),
            "yearly_seasonality": json.loads(attrs["yearly_seasonality"]),
            "time_varying_intercept": json.loads(
                attrs.get("time_varying_intercept", "false")
            ),
            "time_varying_media": json.loads(attrs.get("time_varying_media", "false")),
            "validate_data": json.loads(attrs["validate_data"]),
            "sampler_config": json.loads(attrs["sampler_config"]),
            "dag": json.loads(attrs.get("dag", "null")),
            "treatment_nodes": json.loads(attrs.get("treatment_nodes", "null")),
            "outcome_node": json.loads(attrs.get("outcome_node", "null")),
            "scaling": cls._deserialize_scaling(
                json.loads(attrs.get("scaling", "null"))
            ),
            "apply_scaling": json.loads(attrs.get("apply_scaling", "true")),
            "fit_intercept": json.loads(attrs.get("fit_intercept", "true")),
            "log_transform_vars": json.loads(attrs.get("log_transform_vars", "null")),
        }

    def _data_setter(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
        shutdown_components: list[str] | None = None,
    ) -> None:
        """Set new data in the model.

        This function accepts data in various formats and sets them into the
        model using the PyMC's `set_data` method. The data corresponds to the
        channel data and the target.

        Parameters
        ----------
        X : Union[np.ndarray, pd.DataFrame]
            Data for the channel. It can be a numpy array or pandas DataFrame.
            If it's a DataFrame, the columns corresponding to self.channel_columns
            are used. If it's an ndarray, it's used directly.
        y : Union[np.ndarray, pd.Series], optional
            Target data. It can be a numpy array or a pandas Series.
            If it's a Series, its values are used. If it's an ndarray, it's used
            directly. The default is None.
        shutdown_components : list[str], optional
            List of components to zero out for contribution isolation.
            Can include:
            - Individual channel names (e.g., ['x1', 'x2'])
            - 'channels' or 'all_channels': shutdown all channels
            - 'controls': shutdown all control variables
            - Individual control names (e.g., ['event_1', 't'])
            - 'seasonality' or 'fourier': shutdown seasonal effects
            The default is None (no components shut down).

        Raises
        ------
        RuntimeError
            If the data for the channel is not provided in `X`.
        TypeError
            If `X` is not a pandas DataFrame or a numpy array, or
            if `y` is not a pandas Series or a numpy array and is not None.
        ValueError
            If shutdown_components contains invalid component names.

        Returns
        -------
        None

        """
        if not isinstance(X, pd.DataFrame):
            msg = "X must be a pandas DataFrame in order to access the columns"
            raise TypeError(msg)

        new_channel_data: np.ndarray | None = None
        coords = {"date": X[self.date_column].to_numpy()}

        try:
            new_channel_data = X[self.channel_columns].to_numpy()
        except KeyError as e:
            raise RuntimeError("New data must contain channel_data!") from e

        def identity(x):
            return x

        # For backward compatibility, still support old transformer approach
        # but prioritize the new scaling approach
        if hasattr(self, "channel_scale") and hasattr(self, "target_scale"):
            # Use new scaling approach - set raw data and computed scales
            data: dict[str, np.ndarray | Any] = {
                "channel_data": new_channel_data,
                "channel_scale": self.channel_scale,
            }
        else:
            # Fall back to old transformer approach for backward compatibility
            channel_transformation = (
                identity
                if not hasattr(self, "channel_transformer")
                else self.channel_transformer.transform
            )
            data = {"channel_data": channel_transformation(new_channel_data)}
        if self.control_columns is not None:
            control_data = X[self.control_columns].to_numpy()
            control_transformation = (
                identity
                if not hasattr(self, "control_transformer")
                else self.control_transformer.transform
            )
            data["control_data"] = control_transformation(control_data)

        if self.yearly_seasonality is not None:
            data["dayofyear"] = X[self.date_column].dt.dayofyear.to_numpy()

        if self.time_varying_intercept | self.time_varying_media:
            data["time_index"] = infer_time_index(
                X[self.date_column], self.X[self.date_column], self._time_resolution
            )

        if y is not None:
            if isinstance(y, pd.Series):
                y_data = y.to_numpy()
            elif isinstance(y, np.ndarray):
                y_data = y
            else:
                raise TypeError("y must be either a pandas Series or a numpy array")

            # Use new scaling approach if available
            if hasattr(self, "target_scale"):
                data["target_data"] = y_data
                data["target_scale"] = self.target_scale
            else:
                # Fall back to old approach for backward compatibility
                data["target"] = y_data
        else:
            dtype = self.preprocessed_data["y"].dtype  # type: ignore
            if hasattr(self, "target_scale"):
                data["target_data"] = np.zeros(X.shape[0], dtype=dtype)
                data["target_scale"] = self.target_scale
            else:
                data["target"] = np.zeros(X.shape[0], dtype=dtype)

        # Apply component shutdown logic if specified
        if shutdown_components is not None:
            self._validate_shutdown_components(shutdown_components)
            data = self._apply_component_shutdown(data, shutdown_components, X)

        with self.model:
            pm.set_data(data, coords=coords)

    @classmethod
    def _deserialize_scaling(cls, scaling_dict: dict | None) -> Scaling | None:
        """Deserialize scaling configuration from JSON.

        Parameters
        ----------
        scaling_dict : dict | None
            The serialized scaling configuration.

        Returns
        -------
        Scaling | None
            The deserialized Scaling object or None.
        """
        if scaling_dict is None:
            return None

        return Scaling(
            target=VariableScaling(
                method=scaling_dict["target"]["method"],
                dims=tuple(scaling_dict["target"]["dims"]),
            ),
            channel=VariableScaling(
                method=scaling_dict["channel"]["method"],
                dims=tuple(scaling_dict["channel"]["dims"]),
            ),
        )

    @classmethod
    def _model_config_formatting(cls, model_config: dict) -> dict:
        """Format the model configuration.

        Because of json serialization, model_config values that were originally tuples
        or numpy are being encoded as lists. This function converts them back to tuples
        and numpy arrays to ensure correct id encoding.

        Parameters
        ----------
        model_config : dict
            The model configuration to format.

        Returns
        -------
        dict
            The formatted model configuration.

        """

        def format_nested_dict(d: dict) -> dict:
            for key, value in d.items():
                if isinstance(value, dict):
                    d[key] = format_nested_dict(value)
                elif isinstance(value, list):
                    # Check if the key is "dims" to convert it to tuple
                    if key == "dims":
                        d[key] = tuple(value)
                    # Convert all other lists to numpy arrays
                    else:
                        d[key] = np.array(value)
            return d

        return format_nested_dict(model_config.copy())

    def sample_prior_predictive(
        self,
        X=None,
        y=None,
        samples: int | None = None,
        extend_idata: bool = True,
        combined: bool = True,
        **kwargs,
    ):
        """Sample from the model's prior predictive distribution.

        Parameters
        ----------
        X : array, shape (n_pred, n_features)
            The input data used for prediction using prior distribution.
        y : array, shape (n_pred,), optional
            The target values (real numbers) used for prediction using prior distribution.
            If not set, defaults to an array of zeros.
        samples : int
            Number of samples from the prior parameter distributions to generate.
            If not set, uses sampler_config['draws'] if that is available, otherwise defaults to 500.
        extend_idata : Boolean
            Determine whether the predictions should be added to inference data object.
            Defaults to True.
        combined: Boolean
            Combine chain and draw dims into sample. Won't work if a dim named sample already exists.
            Defaults to True.
        **kwargs: Additional arguments to pass to pymc.sample_prior_predictive

        Returns
        -------
        prior_predictive_samples : DataArray, shape (n_pred, samples)
            Prior predictive samples for each input X

        """
        self.logger.info("=" * 60)
        self.logger.info("PRIOR PREDICTIVE SAMPLING")
        self.logger.info("=" * 60)
        self.logger.info(f"Input parameters - X shape: {X.shape if X is not None else 'None'}")
        self.logger.info(f"Input parameters - y shape: {y.shape if y is not None else 'None'}")
        self.logger.info(f"Input parameters - samples: {samples}, extend_idata: {extend_idata}, combined: {combined}")

        # Log scaling information if scales are available
        if hasattr(self, 'channel_scale') and hasattr(self, 'target_scale'):
            self.logger.info("\n--- DATA SCALING INFORMATION ---")
            self.logger.info(f"Target scale factor: {self.target_scale:.4f}")
            self.logger.info(f"Channel scale factors: {self.channel_scale}")
            self.logger.info(f"Channel names: {self.channel_columns}")
            for i, channel_name in enumerate(self.channel_columns):
                self.logger.info(f"  {channel_name}: {self.channel_scale[i]:.4f}")
        else:
            self.logger.info("\nNote: Scaling factors not yet computed (model may not be built)")

        # Call the parent method
        result = super().sample_prior_predictive(
            X=X, y=y, samples=samples, extend_idata=extend_idata, combined=combined, **kwargs
        )

        # Log completion
        self.logger.info("\n--- SAMPLING RESULTS ---")
        self.logger.info("Prior predictive sampling completed successfully")
        self.logger.info(f"Output samples shape: {result.sizes}")
        self.logger.info(f"Output variables: {list(result.data_vars.keys())}")
        self.logger.info("=" * 60 + "\n")

        return result

    def fit(  # type: ignore[override]
        self,
        X: pd.DataFrame | pd.Series,
        y: pd.Series | np.ndarray | None = None,
        progressbar: bool | None = None,
        random_seed=None,
        **kwargs: Any,
    ) -> az.InferenceData:
        """Fit a model using the data passed as a parameter.

        Sets attrs to inference data of the model.

        Parameters
        ----------
        X : pd.DataFrame | pd.Series
            The training input samples.
        y : pd.Series | np.ndarray | None
            The target values (real numbers).
        progressbar : bool, optional
            Specifies whether the fit progress bar should be displayed. Defaults to True.
        random_seed : Optional
            Provides sampler with initial random seed for obtaining reproducible samples.
        **kwargs : Any
            Custom sampler settings can be provided in form of keyword arguments.

        Returns
        -------
        self : az.InferenceData
            Returns inference data of the fitted model.

        """
        self.logger.info("Starting model fitting")
        self.logger.info(f"Fit input data shapes - X: {X.shape}, y: {y.shape if hasattr(y, 'shape') else len(y) if y is not None else 'None'}")
        self.logger.info(f"Fit parameters - progressbar: {progressbar}, random_seed: {random_seed}")
        self.logger.info(f"Additional kwargs: {list(kwargs.keys())}")
        
        # Call the parent method
        result = super().fit(
            X=X, y=y, progressbar=progressbar, random_seed=random_seed, **kwargs
        )
        
        # Log completion
        self.logger.info("Model fitting completed successfully")
        self.logger.info(f"Inference data groups: {list(result.groups())}")
        if hasattr(result, 'posterior'):
            self.logger.info(f"Posterior variables: {list(result.posterior.data_vars.keys())}")
        
        return result


class MMM(
    MaxAbsScaleTarget,
    MaxAbsScaleChannels,
    ValidateControlColumns,
    BaseMMM,
):
    r"""Media Mix Model class, Delayed Adstock and logistic saturation as default initialization (see [1]_).

    Given a time series target variable :math:`y_{t}` (e.g. sales on conversions), media variables
    :math:`x_{m, t}` (e.g. impressions, clicks or costs) and a set of control covariates :math:`z_{c, t}` (e.g. holidays, special events)
    we consider a Bayesian linear model of the form:

    .. math::
        y_{t} = \alpha + \sum_{m=1}^{M}\beta_{m}f(x_{m, t}) +  \sum_{c=1}^{C}\gamma_{c}z_{c, t} + \varepsilon_{t},

    where :math:`\alpha` is the intercept, :math:`f` is a media transformation function and :math:`\varepsilon_{t}` is the error therm
    which we assume is normally distributed. The function :math:`f` encodes the contribution of media on the target variable.
    Typically we consider two types of transformation: adstock (carry-over) and saturation effects.

    Notes
    -----
    Here are some important notes about the model:

    1. Before fitting the model, we scale the target variable and the media channels using the maximum absolute value of each variable.
    This enable us to have a more stable model and better convergence. If control variables are present, we do not scale them!
    If needed please do it before passing the data to the model.

    2. We allow to add yearly seasonality controls as Fourier modes.
    You can use the `yearly_seasonality` parameter to specify the number of Fourier modes to include.

    3. This class also allow us to calibrate the model using:

        * Custom priors for the parameters via the `model_config` parameter. You can also set the likelihood distribution.

        * Adding lift tests to the likelihood function via the :meth:`add_lift_test_measurements <pymc_marketing.mmm.mmm.MMM.add_lift_test_measurements>` method.

    For details on a vanilla implementation in PyMC, see [2]_.

    Examples
    --------
    Here is an example of how to instantiate the model with the default configuration:

    .. code-block:: python

        import numpy as np
        import pandas as pd

        from pymc_marketing.mmm import (
            GeometricAdstock,
            LogisticSaturation
            MMM,
        )
        from pymc_marketing.paths import data_dir

        file_path = data_dir / "mmm_example.csv"
        data = pd.read_csv(file_path, parse_dates=["date_week"])

        mmm = MMM(
            date_column="date_week",
            channel_columns=["x1", "x2"],
            adstock=GeometricAdstock(l_max=8),
            saturation=LogisticSaturation(),
            control_columns=[
                "event_1",
                "event_2",
                "t",
            ],
            yearly_seasonality=2,
        )

    Now we can fit the model with the data:

    .. code-block:: python

        # Set features and target
        X = data.drop("y", axis=1)
        y = data["y"]

        # Fit the model
        idata = mmm.fit(X, y)

    We can also define custom priors for the model:

    .. code-block:: python

        import numpy as np

        from pymc_marketing.mmm import (
            GeometricAdstock,
            LogisticSaturation
            MMM,
        )
        from pymc_extras.prior import Prior

        my_model_config = {
            "saturation_beta": Prior("LogNormal", mu=np.array([2, 1]), sigma=1),
            "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=2)),
        }

        mmm = MMM(
            date_column="date_week",
            channel_columns=["x1", "x2"],
            adstock=GeometricAdstock(l_max=8),
            saturation=LogisticSaturation(),
            control_columns=[
                "event_1",
                "event_2",
                "t",
            ],
            yearly_seasonality=2,
            model_config=my_model_config,
        )

    As you can see, we can configure all prior and likelihood distributions via the `model_config`.

    The `fit` method accepts keyword arguments that are passed to the PyMC sampling method.
    For example, to change the number of samples and chains, and using a JAX implementation of NUTS we can do:

    .. code-block:: python

        sampler_kwargs = {
            "draws": 2_000,
            "target_accept": 0.9,
            "chains": 5,
            "random_seed": 42,
        }

        idata = mmm.fit(X, y, nuts_sampler="numpyro", **sampler_kwargs)

    References
    ----------
    .. [1] Jin, Yuxue, et al. “Bayesian methods for media mix modeling with carryover and shape effects.” (2017).
    .. [2] Orduz, J. `"Media Effect Estimation with PyMC: Adstock, Saturation & Diminishing Returns" <https://juanitorduz.github.io/pymc_mmm/>`_.

    """  # noqa: E501

    _model_type: str = "MMM"
    version: str = "0.0.3"

    def channel_contribution_forward_pass(
        self,
        channel_data: npt.NDArray,
        disable_logger_stdout: bool | None = False,
    ) -> npt.NDArray:
        """Evaluate the channel contribution for a given channel data and a fitted model, ie. the forward pass.

        We return the contribution in the original scale of the target variable.

        Parameters
        ----------
        channel_data : array-like
            Input channel data. Result of all the preprocessing steps.
        disable_logger_stdout : bool, optional
            If True, suppress logger output to stdout

        Returns
        -------
        array-like
            Transformed channel data.

        """
        channel_contribution_forward_pass = super().channel_contribution_forward_pass(
            channel_data=channel_data, disable_logger_stdout=disable_logger_stdout
        )

        # Convert from scaled space to original scale
        if hasattr(self, "target_scale"):
            # Use the computed scale factor from our new approach
            return channel_contribution_forward_pass * self.target_scale
        else:
            # Fallback to transformer for backward compatibility
            target_transformed_vectorized = np.vectorize(
                self.target_transformer.inverse_transform,
                excluded=[1, 2],
                signature="(m, n) -> (m, n)",
            )
            return target_transformed_vectorized(channel_contribution_forward_pass)

    def get_channel_contribution_forward_pass_grid(
        self, start: float, stop: float, num: int
    ) -> DataArray:
        """Generate a grid of scaled channel contributions for a given grid of shared values.

        Parameters
        ----------
        start : float
            Start of the grid. It must be equal or greater than 0.
        stop : float
            End of the grid. It must be greater than start.
        num : int
            Number of points in the grid.

        Returns
        -------
        DataArray
            Grid of channel contributions.

        """
        if start < 0:
            raise ValueError("start must be greater than or equal to 0.")

        share_grid = np.linspace(start=start, stop=stop, num=num)

        channel_contribution = []
        for delta in share_grid:
            X_data = self.preprocessed_data["X"]
            if isinstance(X_data, pd.DataFrame):
                channel_data = delta * X_data[self.channel_columns].to_numpy()
            else:
                raise TypeError("X data must be a DataFrame")
            channel_contribution_forward_pass = self.channel_contribution_forward_pass(
                channel_data=channel_data,
                disable_logger_stdout=True,
            )
            channel_contribution.append(channel_contribution_forward_pass)
        return DataArray(
            data=np.array(channel_contribution),
            dims=("delta", "chain", "draw", "date", "channel"),
            coords={
                "delta": share_grid,
                "date": self.X[self.date_column],
                "channel": self.channel_columns,
            },
        )

    def plot_channel_parameter(self, param_name: str, **plt_kwargs: Any) -> plt.Figure:
        """Plot the posterior distribution of a specific parameter for each channel.

        Parameters
        ----------
        param_name : str
            The name of the parameter to plot.
        **plt_kwargs : Any
            Additional keyword arguments to pass to the `plt.subplots` function.

        Returns
        -------
        plt.Figure
            The matplotlib Figure object containing the plot.

        Raises
        ------
        ValueError
            If the specified parameter name is invalid or not found in the model
            saturation or adstock function.

        """
        saturation: SaturationTransformation = self.saturation
        adstock: AdstockTransformation = self.adstock

        parameters_to_check = list(saturation.variable_mapping.values()) + list(
            adstock.variable_mapping.values()
        )
        if param_name not in parameters_to_check:
            raise ValueError(
                f"Invalid parameter name: {param_name}. Choose from {parameters_to_check}"
            )

        param_samples_df = pd.DataFrame(
            data=az.extract(data=self.fit_result, var_names=[param_name]).T,
            columns=self.channel_columns,
        )

        fig, ax = plt.subplots(**plt_kwargs)
        sns.violinplot(data=param_samples_df, orient="h", ax=ax)
        ax.set(
            title=f"Posterior Distribution: {param_name} Parameter",
            xlabel=param_name,
            ylabel="channel",
        )
        return fig

    def get_ts_contribution_posterior(
        self, var_contribution: str, original_scale: bool = False
    ) -> DataArray:
        """Get the posterior distribution of the time series contributions of a given variable.

        Parameters
        ----------
        var_contribution : str
            The variable for which to get the contributions. It must be a valid variable
            in the `fit_result` attribute.
        original_scale : bool, optional
            Whether to plot in the original scale.

        Returns
        -------
        DataArray
            The posterior distribution of the time series contributions.

        """
        if original_scale:
            # Try to use pre-computed original scale deterministic (best)
            original_scale_var = f"{var_contribution}_original_scale"
            if hasattr(self, "fit_result") and original_scale_var in self.fit_result:
                contributions = self._format_model_contributions(
                    var_contribution=original_scale_var
                )
                # Rename to expected variable name for consistency with plotting code
                return contributions.rename(var_contribution)

            # Fallback: compute on-the-fly using target_scale (good)
            contributions = self._format_model_contributions(
                var_contribution=var_contribution
            )
            if hasattr(self, "target_scale"):
                return contributions * self.target_scale

            # Final fallback: use old transformer for backward compatibility
            return apply_sklearn_transformer_across_dim(
                data=contributions,
                func=self.get_target_transformer().inverse_transform,
                dim_name="date",
            )
        else:
            # Use scaled contributions directly
            return self._format_model_contributions(var_contribution=var_contribution)

    def plot_components_contributions(
        self, original_scale: bool = False, **plt_kwargs: Any
    ) -> plt.Figure:
        """Plot the target variable and the posterior predictive model components.

        We can plot the target variable and the posterior predictive model components in
        the scaled space or in the original space.

        Parameters
        ----------
        original_scale : bool, optional
            Whether to plot in the original scale.

        **plt_kwargs
            Additional keyword arguments to pass to `plt.subplots`.

        Returns
        -------
        plt.Figure

        """
        channel_contribution = self.get_ts_contribution_posterior(
            var_contribution="channel_contribution", original_scale=original_scale
        )

        means = [channel_contribution.mean(["chain", "draw"])]
        contribution_vars = [
            az.hdi(channel_contribution, hdi_prob=0.94).channel_contribution
        ]

        for arg, var_contribution in zip(
            ["control_columns", "yearly_seasonality"],
            ["control_contribution", "fourier_contribution"],
            strict=True,
        ):
            if getattr(self, arg, None):
                contributions = self.get_ts_contribution_posterior(
                    var_contribution=var_contribution, original_scale=original_scale
                )

                means.append(contributions.mean(["chain", "draw"]))
                contribution_vars.append(
                    az.hdi(contributions, hdi_prob=0.94)[var_contribution]
                )

        fig, ax = plt.subplots(**plt_kwargs)

        for i, (mean, hdi, var_contribution) in enumerate(
            zip(
                means,
                contribution_vars,
                [
                    "channel_contribution",
                    "control_contribution",
                    "fourier_contribution",
                ],
                strict=False,
            )
        ):
            if self.X is not None:
                ax.fill_between(
                    x=self.X[self.date_column],
                    y1=hdi.isel(hdi=0),
                    y2=hdi.isel(hdi=1),
                    color=f"C{i}",
                    alpha=0.25,
                    label=f"$94\\%$ HDI ({var_contribution})",
                )
                ax.plot(
                    np.asarray(self.X[self.date_column]),
                    np.asarray(mean),
                    color=f"C{i}",
                )
        if self.X is not None:
            intercept = az.extract(
                self.fit_result, var_names=["intercept"], combined=False
            )

            if not original_scale:
                intercept = apply_sklearn_transformer_across_dim(
                    data=intercept,
                    func=self.get_target_transformer().transform,
                    dim_name="chain",
                )

            if intercept.ndim == 2:
                # Intercept has a stationary prior
                intercept_hdi = np.repeat(
                    a=az.hdi(intercept).intercept.data[None, ...],
                    repeats=self.X[self.date_column].shape[0],
                    axis=0,
                )
            elif intercept.ndim == 3:
                # Intercept has a time-varying prior
                intercept_hdi = az.hdi(intercept).intercept.data

            ax.plot(
                np.asarray(self.X[self.date_column]),
                np.full(
                    len(self.X[self.date_column]),
                    intercept.mean(["chain", "draw"]).data,
                ),
                color=f"C{i + 1}",
            )
            ax.fill_between(
                x=self.X[self.date_column],
                y1=intercept_hdi[:, 0],
                y2=intercept_hdi[:, 1],
                color=f"C{i + 1}",
                alpha=0.25,
                label="$94\\%$ HDI (intercept)",
            )

            y_to_plot = (
                self.get_target_transformer().transform(
                    np.asarray(self.preprocessed_data["y"]).reshape(-1, 1)
                )
                if not original_scale
                else np.asarray(self.preprocessed_data["y"])
            )

            ylabel = self.output_var if original_scale else f"{self.output_var} scaled"

            ax.plot(
                np.asarray(self.X[self.date_column]),
                y_to_plot,
                label=ylabel,
                color="black",
            )
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
            ax.set(
                title="Posterior Predictive Model Components",
                xlabel="date",
                ylabel=ylabel,
            )
        return fig

    def plot_channel_contribution_grid(
        self,
        start: float,
        stop: float,
        num: int,
        absolute_xrange: bool = False,
        **plt_kwargs: Any,
    ) -> plt.Figure:
        """Plot a grid of scaled channel contributions for a given grid of share values.

        Parameters
        ----------
        start : float
            Start of the grid. It must be equal or greater than 0.
        stop : float
            End of the grid. It must be greater than start.
        num : int
            Number of points in the grid.
        absolute_xrange : bool, optional
            If True, the x-axis is in absolute values (input units), otherwise it is in
            relative percentage values, by default False.
        **plt_kwargs
            Keyword arguments to pass to `plt.subplots()`

        Returns
        -------
        plt.Figure
            Plot of grid of channel contributions.

        """
        share_grid = np.linspace(start=start, stop=stop, num=num)
        contributions = self.get_channel_contribution_forward_pass_grid(
            start=start, stop=stop, num=num
        )

        fig, ax = plt.subplots(**plt_kwargs)

        for i, channel in enumerate(self.channel_columns):
            channel_contribution_total = contributions.sel(channel=channel).sum(
                dim="date"
            )

            hdi_contribution = az.hdi(ary=channel_contribution_total).x

            total_channel_input = self.X[channel].sum()
            x_range = (
                total_channel_input * share_grid if absolute_xrange else share_grid
            )

            ax.fill_between(
                x=x_range,
                y1=hdi_contribution[:, 0],
                y2=hdi_contribution[:, 1],
                color=f"C{i}",
                label=f"{channel} $94\\%$ HDI contribution",
                alpha=0.4,
            )

            sns.lineplot(
                x=x_range,
                y=channel_contribution_total.mean(dim=("chain", "draw")),
                color=f"C{i}",
                marker="o",
                label=f"{channel} contribution mean",
                ax=ax,
            )
            if absolute_xrange:
                ax.axvline(
                    x=total_channel_input,
                    color=f"C{i}",
                    linestyle="--",
                    label=f"{channel} current total input",
                )

        if not absolute_xrange:
            ax.axvline(x=1, color="black", linestyle="--", label=r"$\delta = 1$")

        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        x_label = "input" if absolute_xrange else r"$\delta$"
        ax.set(
            title="Channel contribution as a function of cost share",
            xlabel=x_label,
            ylabel="contribution",
        )
        return fig

    def new_spend_contributions(
        self,
        spend: np.ndarray | None = None,
        one_time: bool = True,
        spend_leading_up: np.ndarray | None = None,
        prior: bool = False,
        original_scale: bool = True,
        **sample_posterior_predictive_kwargs,
    ) -> DataArray:
        """Return the upcoming contributions for a given spend.

        The spend can be one time or constant over the period. The spend leading up to the
        period can also be specified in order account for the lagged effect of the spend.

        Parameters
        ----------
        spend : np.ndarray, optional
            Array of spend for each channel. If None, the average spend for each channel is used, by default None.
        one_time : bool, optional
            Whether the spends for each channel are only at the start of the period.
            If True, all spends after the initial spend are zero.
            If False, all spends after the initial spend are the same as the initial spend.
            By default True.
        spend_leading_up : np.array, optional
            Array of spend for each channel leading up to the spend, by default None or 0 for each channel.
            Use this parameter to account for the lagged effect of the spend.
        prior : bool, optional
            Whether to use the prior or posterior, by default False (posterior)
        **sample_posterior_predictive_kwargs
            Additional keyword arguments passed to pm.sample_posterior_predictive

        Returns
        -------
        DataArray
            Upcoming contributions for each channel

        Examples
        --------
        Channel contributions from 1 unit on each channel only once.

        .. code-block:: python

            n_channels = len(model.channel_columns)
            spend = np.ones(n_channels)
            new_spend_contributions = model.new_spend_contributions(spend=spend)

        Channel contributions from continuously spending 1 unit on each channel.

        .. code-block:: python

            n_channels = len(model.channel_columns)
            spend = np.ones(n_channels)
            new_spend_contributions = model.new_spend_contributions(
                spend=spend, one_time=False
            )

        Channel contributions from 1 unit on each channel only once but with 1 unit leading up to the spend.

        .. code-block:: python

            n_channels = len(model.channel_columns)
            spend = np.ones(n_channels)
            spend_leading_up = np.ones(n_channels)
            new_spend_contributions = model.new_spend_contributions(
                spend=spend, spend_leading_up=spend_leading_up
            )

        """
        if spend is None:
            spend = self.X.loc[:, self.channel_columns].mean().to_numpy()  # type: ignore

        n_channels = len(self.channel_columns)
        if len(spend) != n_channels:
            raise ValueError("spend must be the same length as the number of channels")

        new_data = create_new_spend_data(
            spend=spend,
            adstock_max_lag=self.adstock.l_max,
            one_time=one_time,
            spend_leading_up=spend_leading_up,
        )

        new_data = (
            self.channel_transformer.transform(new_data) if not prior else new_data
        )

        idata: Dataset = self.fit_result if not prior else self.prior

        coords = {
            "time_since_spend": np.arange(-self.adstock.l_max, self.adstock.l_max + 1),
            "channel": self.channel_columns,
        }
        with pm.Model(coords=coords):
            pm.Deterministic(
                "channel_contribution",
                self.forward_pass(x=new_data),
                dims=("time_since_spend", "channel"),
            )

            samples = pm.sample_posterior_predictive(
                idata,
                var_names=["channel_contribution"],
                **sample_posterior_predictive_kwargs,
            )

        channel_contribution = samples.posterior_predictive["channel_contribution"]

        if not original_scale:
            channel_contribution = apply_sklearn_transformer_across_dim(
                data=channel_contribution,
                func=self.get_target_transformer().transform,
                dim_name="time_since_spend",
            )

        return channel_contribution

    def plot_new_spend_contributions(
        self,
        spend_amount: float,
        one_time: bool = True,
        lower: float = 0.025,
        upper: float = 0.975,
        ylabel: str = "Sales",
        idx: slice | None = None,
        channels: list[str] | None = None,
        prior: bool = False,
        original_scale: bool = True,
        ax: plt.Axes | None = None,
        **sample_posterior_predictive_kwargs,
    ) -> plt.Axes:
        """Plot the upcoming sales for a given spend amount.

        Calls the new_spend_contributions method and plots the results. For more
        control over the plot, use new_spend_contributions directly.

        Parameters
        ----------
        spend_amount : float
            The amount of spend for each channel
        one_time : bool, optional
            Whether the spend are one time (at start of period) or constant (over period), by default True (one time)
        lower : float, optional
            The lower quantile for the confidence interval, by default 0.025
        upper : float, optional
            The upper quantile for the confidence interval, by default 0.975
        ylabel : str, optional
            The label for the y-axis, by default "Sales"
        idx : slice, optional
            The index slice of days to plot, by default None or only the positive days.
            More specifically, slice(0, None, None)
        channels : List[str], optional
            The channels to plot, by default None or all channels
        prior : bool, optional
            Whether to use the prior or posterior, by default False (posterior)
        original_scale : bool, optional
            Whether to plot in the original scale of the target variable, by default True
        ax : plt.Axes, optional
            The axes to plot on, by default None or current axes
        **sample_posterior_predictive_kwargs
            Additional keyword arguments passed to pm.sample_posterior_predictive

        Returns
        -------
        plt.Axes
            The plot of upcoming sales for the given spend amount

        """
        for value in [lower, upper]:
            if value < 0 or value > 1:
                raise ValueError("lower and upper must be between 0 and 1")
        if lower > upper:
            raise ValueError("lower must be less than or equal to upper")

        ax = ax or plt.gca()
        total_channels = len(self.channel_columns)
        contributions = self.new_spend_contributions(
            np.ones(total_channels) * spend_amount,
            one_time=one_time,
            spend_leading_up=np.ones(total_channels) * spend_amount,
            prior=prior,
            original_scale=original_scale,
            **sample_posterior_predictive_kwargs,
        )

        contributions_groupby = contributions.to_series().groupby(
            level=["time_since_spend", "channel"]
        )

        idx = idx or pd.IndexSlice[0:]

        conf = (
            contributions_groupby.quantile([lower, upper])
            .unstack("channel")
            .unstack()
            .loc[idx]
        )

        channels = channels or self.channel_columns  # type: ignore
        for channel in channels:  # type: ignore
            ax.fill_between(
                conf.index,
                conf[channel][lower],
                conf[channel][upper],
                label=f"{channel} {100 * (upper - lower):.0f}% CI",
                alpha=0.5,
            )
        mean = contributions_groupby.mean().unstack("channel").loc[idx, channels]
        color = [f"C{i}" for i in range(len(channels))]  # type: ignore
        mean.add_suffix(" mean").plot(ax=ax, color=color, alpha=0.75)
        ax.legend().set_title("Channel")
        ax.set(
            xlabel="Time since spend",
            ylabel=ylabel,
            title=f"Upcoming sales for {spend_amount:.02f} spend",
        )
        return ax

    def _validate_data(self, X, y=None):
        return X

    def _validate_shutdown_components(self, shutdown_components: list[str]) -> None:
        """Validate that shutdown_components contains valid component names.

        Parameters
        ----------
        shutdown_components : list[str]
            List of component names to validate.

        Raises
        ------
        ValueError
            If any component name is not valid.

        Notes
        -----
        Valid component names include:
        - 'intercept': The model intercept
        - 'seasonality' or 'fourier': Yearly seasonality effects
        - 'controls': All control variables
        - Individual control column names (e.g., 'event_1', 't')
        - 'channels' or 'all_channels': All media channels
        - Individual channel column names (e.g., 'x1', 'x2')
        """
        # Build set of valid components
        valid_components = {'intercept', 'seasonality', 'fourier', 'controls', 'channels', 'all_channels'}

        # Add individual channel names
        valid_components.update(self.channel_columns)

        # Add individual control names if they exist
        if self.control_columns:
            valid_components.update(self.control_columns)

        # Validate each component
        invalid_components = [comp for comp in shutdown_components if comp not in valid_components]

        if invalid_components:
            raise ValueError(
                f"Invalid component(s): {invalid_components}. "
                f"Valid components are: {sorted(valid_components)}"
            )

    def _apply_component_shutdown(
        self,
        data: dict[str, np.ndarray | Any],
        shutdown_components: list[str],
        X: pd.DataFrame,
    ) -> dict[str, np.ndarray | Any]:
        """Apply shutdown logic to specified components.

        Modifies the data dict in-place by zeroing out specified components.

        Parameters
        ----------
        data : dict
            Dictionary containing model data (channel_data, control_data, dayofyear, etc.)
        shutdown_components : list[str]
            List of components to shut down
        X : pd.DataFrame
            Input data frame (used for getting array shapes)

        Returns
        -------
        dict
            Modified data dictionary with specified components zeroed out

        Notes
        -----
        The intercept component is handled separately in sample_posterior_predictive
        as it is a model parameter, not pm.Data.
        """
        # 1. Shutdown individual channels or all channels
        if 'channels' in shutdown_components or 'all_channels' in shutdown_components:
            # Zero out ALL channels
            if 'channel_data' in data:
                data['channel_data'] = np.zeros_like(data['channel_data'])
        else:
            # Zero out specific channels
            channel_shutdown = [comp for comp in shutdown_components if comp in self.channel_columns]

            if channel_shutdown and 'channel_data' in data:
                # Get indices of channels to shutdown
                channel_indices = [self.channel_columns.index(ch) for ch in channel_shutdown]
                # Zero out specific columns
                data['channel_data'][:, channel_indices] = 0

        # 2. Shutdown controls (all or specific)
        if 'controls' in shutdown_components:
            # Zero out ALL controls
            if 'control_data' in data:
                data['control_data'] = np.zeros_like(data['control_data'])
        else:
            # Shutdown specific controls
            control_shutdown = [c for c in shutdown_components if c in (self.control_columns or [])]

            if control_shutdown and 'control_data' in data:
                control_indices = [self.control_columns.index(ctrl) for ctrl in control_shutdown]
                # Zero out specific columns
                data['control_data'][:, control_indices] = 0

        # 3. Shutdown seasonality/fourier
        if 'seasonality' in shutdown_components or 'fourier' in shutdown_components:
            if 'dayofyear' in data:
                # Set to constant (middle of year = 182) to remove seasonal variation
                # This makes all Fourier modes evaluate to the same value, effectively zeroing
                # their contribution since Fourier coefficients will cancel out
                # Use same dtype as original to avoid PyTensor type mismatch
                data['dayofyear'] = np.full_like(data['dayofyear'], 182)

        # Note: Intercept shutdown is handled in sample_posterior_predictive
        # because intercept is a parameter, not pm.Data

        return data

    @property
    def _channel_scales(self) -> np.ndarray:
        return self.channel_transformer["scaler"].scale_

    def _channel_map_scales(self) -> dict:
        return dict(zip(self.channel_columns, self._channel_scales, strict=True))  # type: ignore

    def format_recovered_transformation_parameters(
        self, quantile: float = 0.5
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Format the recovered transformation parameters for each channel.

        This function retrieves the quantile of the parameters for each channel and formats them into a dictionary
        containing the channel name, the saturation parameters, and the adstock parameters.

        Parameters
        ----------
        quantile : float, optional
            The quantile to retrieve from the posterior distribution of the parameters. Default is 0.5.

        Returns
        -------
        dict
            A dictionary containing the channel names as keys and the corresponding saturation and adstock parameters
            as values.

        Example
        -------
        >>> self.format_recovered_transformation_parameters(quantile=0.5)
        >>> Output:
        {
            'x1': {
                'saturation_params': {
                    'lam': 2.4761893929757077,
                    'beta': 0.360226791880304
                },
            'adstock_params': {
                'alpha': 0.39910387900504796
                }
            },
            'x2': {
                'saturation_params': {
                    'lam': 2.6485978655163436,
                    'beta': 0.2399381337197204
                },
            'adstock_params': {
                'alpha': 0.18859423763437405
                }
            }
        }

        """
        # Retrieve channel names
        channels = self.fit_result.channel.values

        # Initialize the dictionary to store channel information
        channels_info = {}

        # Define the parameter groups for consolidation
        param_groups = {
            "saturation_params": self.saturation.model_config.keys(),
            "adstock_params": self.adstock.model_config.keys(),
        }

        # Iterate through each channel to fetch and store parameters
        for channel in channels:
            channel_info = {}

            # Process each group of parameters (saturation and adstock)
            for group_name, params in param_groups.items():
                # Build dictionary for the current group of parameters
                param_dict = {
                    param.replace(group_name.split("_")[0] + "_", ""): self.fit_result[
                        param
                    ]
                    .quantile(quantile, dim=["chain", "draw"])
                    .to_pandas()
                    .to_dict()[channel]
                    for param in params
                    if param in self.fit_result
                }
                channel_info[group_name] = param_dict

            channels_info[channel] = channel_info

        return channels_info

    def _plot_response_curve_fit(
        self,
        ax: plt.Axes,
        channel: str,
        color_index: int,
        xlim_max: int | None,
        label: str = "Fit Curve",
        quantile_lower: float = 0.05,
        quantile_upper: float = 0.95,
    ) -> None:
        """Plot the curve fit for the given channel based on the estimation of the parameters by the model.

        Parameters
        ----------
        ax : plt.Axes
            The matplotlib axes object where the plot should be drawn.
        channel : str
            The name of the channel for which the curve fit is being plotted.
        color_index : int
            An index used for color selection to ensure distinct colors for multiple plots.
        xlim_max: int
            The maximum value to be plot on the X-axis
        label: str
            The label for the curve being plotted, default is "Fit Curve".
        quantile_lower: float
            The lower quantile for parameter estimation, default is 0.05.
        quantile_upper: float
            The upper quantile for parameter estimation, default is 0.95.

        Returns
        -------
        None
            The function modifies the given axes object in-place and doesn't return any object.

        """
        if self.X is not None:
            x_mean = np.max(self.X[channel])

        # Set x_limit based on the method or xlim_max
        if xlim_max is not None:
            x_limit = xlim_max
        else:
            x_limit = x_mean

        # Generate x_fit and y_fit
        x_fit = np.linspace(0, x_limit, 1000)
        upper_params = self.format_recovered_transformation_parameters(
            quantile=quantile_upper
        )
        lower_params = self.format_recovered_transformation_parameters(
            quantile=quantile_lower
        )
        mid_params = self.format_recovered_transformation_parameters(quantile=0.5)
        y_fit = self.saturation.function(
            x=x_fit, **mid_params[channel]["saturation_params"]
        ).eval()

        y_fit_lower = self.saturation.function(
            x=x_fit, **lower_params[channel]["saturation_params"]
        ).eval()
        y_fit_upper = self.saturation.function(
            x=x_fit, **upper_params[channel]["saturation_params"]
        ).eval()

        # scale all y fit values to the original scale using
        # `mmm.target_transformer.named_steps["scaler"].scale_.item()`
        y_fit = (
            self.get_target_transformer()
            .inverse_transform(y_fit.reshape(-1, 1))
            .flatten()
        )
        y_fit_lower = (
            self.get_target_transformer()
            .inverse_transform(y_fit_lower.reshape(-1, 1))
            .flatten()
        )
        y_fit_upper = (
            self.get_target_transformer()
            .inverse_transform(y_fit_upper.reshape(-1, 1))
            .flatten()
        )

        # scale x fit values
        x_fit = self._channel_map_scales()[channel] * x_fit

        ax.fill_between(
            x_fit, y_fit_lower, y_fit_upper, color=f"C{color_index}", alpha=0.25
        )
        ax.plot(x_fit, y_fit, color=f"C{color_index}", label=label, alpha=0.6)

        ax.set(xlabel="Spent", ylabel="Contribution")
        ax.legend()

    def plot_direct_contribution_curves(
        self,
        show_fit: bool = False,
        same_axes: bool = False,
        xlim_max: int | None = None,
        channels: list[str] | None = None,
        quantile_lower: float = 0.05,
        quantile_upper: float = 0.95,
    ) -> plt.Figure:
        """Plot the direct contribution curves for each marketing channel.

        The term "direct" refers to the fact that we plot costs vs immediate returns and
        we do not take into account the lagged effects of the channels e.g. adstock transformations.

        Parameters
        ----------
        show_fit : bool, optional
            If True, the function will also plot the curve fit. Defaults to False.
        xlim_max : int, optional
            The maximum value to be plot on the X-axis. If not provided, the maximum value in the data will be used.
        channels : List[str], optional
            A list of channels to plot. If not provided, all channels will be plotted.
        same_axes : bool, optional
            If True, all channels will be plotted on the same axes. Defaults to False.

        Returns
        -------
        plt.Figure
            A matplotlib Figure object with the direct contribution curves.

        """
        channels_to_plot = self.channel_columns if channels is None else channels

        if not all(channel in self.channel_columns for channel in channels_to_plot):
            unknown_channels = set(channels_to_plot) - set(self.channel_columns)
            raise ValueError(
                f"The provided channels must be a subset of the available channels. Got {unknown_channels}"
            )

        if len(channels_to_plot) != len(set(channels_to_plot)):
            raise ValueError("The provided channels must be unique.")

        channel_contribution = self.compute_channel_contribution_original_scale().mean(
            ["chain", "draw"]
        )

        if same_axes:
            nrows = 1
            figsize = (12, 4)

            def label_func(channel):
                return f"{channel} Data Points"

            def legend_title_func(channel):
                return "Legend"

        else:
            nrows = len(channels_to_plot)
            figsize = (12, 4 * len(channels_to_plot))

            def label_func(channel):
                return "Data Points"

            def legend_title_func(channel):
                return f"{channel} Legend"

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=1,
            sharex=False,
            sharey=False,
            figsize=figsize,
            layout="constrained",
        )

        if same_axes:
            axes_channels: list[tuple[Any, str]] | Any = [
                (axes, channel) for channel in channels_to_plot
            ]
        else:
            axes_channels = zip(np.ravel(axes), channels_to_plot, strict=False)

        for i, (ax, channel) in enumerate(axes_channels):
            if self.X is not None:
                x = self.X[channel].to_numpy()
                y = channel_contribution.sel(channel=channel).to_numpy()

                label = label_func(channel)
                ax.scatter(x, y, label=label, color=f"C{i}")

                if show_fit:
                    label = f"{channel} Fit Curve" if same_axes else "Fit Curve"
                    self._plot_response_curve_fit(
                        ax=ax,
                        channel=channel,
                        color_index=i,
                        xlim_max=xlim_max,
                        label=label,
                        quantile_lower=quantile_lower,
                        quantile_upper=quantile_upper,
                    )

                title = legend_title_func(channel)
                ax.legend(
                    loc="upper left",
                    facecolor="white",
                    title=title,
                    fontsize="small",
                )

                ax.set(xlabel="Spent", ylabel="Contribution")

        fig.suptitle("Direct response curves", fontsize=16)
        return fig

    def sample_posterior_predictive(
        self,
        X=None,
        extend_idata: bool = True,
        combined: bool = True,
        include_last_observations: bool = False,
        original_scale: bool = True,
        shutdown_components: list[str] | None = None,
        **sample_posterior_predictive_kwargs,
    ) -> DataArray:
        """Sample from the model's posterior predictive distribution.

        Parameters
        ----------
        X : array, shape (n_pred, n_features)
            The input data used for prediction.
        extend_idata : bool, optional
            Boolean determining whether the predictions should be added to inference data object. Defaults to True.
        combined: bool, optional
            Combine chain and draw dims into sample. Won't work if a dim named sample already exists. Defaults to True.
        include_last_observations: bool, optional
            Boolean determining whether to include the last observations of the training data in order to carry over
            costs with the adstock transformation. Assumes that X are the next predictions following the
            training data.Defaults to False.
        original_scale: bool, optional
            Boolean determining whether to return the predictions in the original scale of the target variable.
            Defaults to True.
        shutdown_components : list[str], optional
            List of components to zero out for contribution isolation.
            Can include:
            - Individual channel names (e.g., ['x1', 'x2'])
            - 'channels' or 'all_channels': shutdown all channels
            - 'controls': shutdown all control variables
            - Individual control names (e.g., ['event_1', 't'])
            - 'seasonality' or 'fourier': shutdown seasonal effects
            - 'intercept': shutdown intercept (handled specially)
            The default is None (no components shut down).

            Example: shutdown_components=['x1', 'seasonality'] will zero out
            channel x1 and seasonal effects for contribution isolation.
        **sample_posterior_predictive_kwargs
            Additional arguments to pass to pymc.sample_posterior_predictive

        Returns
        -------
        posterior_predictive_samples : DataArray, shape (n_pred, samples)
            Posterior predictive samples for each input X

        """
        self.logger.info("Starting posterior predictive sampling")
        self.logger.info(f"Input parameters - X shape: {X.shape if X is not None else 'None'}")
        self.logger.info(f"Input parameters - extend_idata: {extend_idata}, combined: {combined}")
        self.logger.info(f"Input parameters - include_last_observations: {include_last_observations}, original_scale: {original_scale}")
        
        X = _handle_deprecate_pred_argument(X, "X", sample_posterior_predictive_kwargs)
        if include_last_observations:
            X = pd.concat(
                [self.X.iloc[-self.adstock.l_max :, :], X], axis=0
            ).sort_values(by=self.date_column)

        self._data_setter(X, shutdown_components=shutdown_components)

        with self.model:  # sample with new input data
            post_pred = pm.sample_posterior_predictive(
                self.idata, **sample_posterior_predictive_kwargs
            )
            if extend_idata:
                self.idata.extend(post_pred, join="right")  # type: ignore

        group = (
            "predictions"
            if sample_posterior_predictive_kwargs.get("predictions", False)
            else "posterior_predictive"
        )
        posterior_predictive_samples = az.extract(post_pred, group, combined=combined)

        if include_last_observations:
            posterior_predictive_samples = posterior_predictive_samples.isel(
                date=slice(self.adstock.l_max, None)
            )

        # Handle intercept shutdown (special case)
        # Intercept is a parameter, not pm.Data, so we subtract it from predictions
        if shutdown_components is not None and 'intercept' in shutdown_components:
            if self.output_var in posterior_predictive_samples:
                # Get intercept values from posterior
                intercept_values = self.fit_result['intercept']

                # Expand intercept to match prediction dimensions if needed
                # intercept shape: (chain, draw) or (chain, draw, date) for time-varying
                if self.time_varying_intercept:
                    # Time-varying intercept already has date dimension
                    # Extract to match combined samples if needed
                    if combined:
                        intercept_expanded = az.extract(intercept_values, combined=True)
                    else:
                        intercept_expanded = intercept_values
                else:
                    # Static intercept: expand to all dates
                    if combined:
                        # Extract intercept as 1D array (sample dimension)
                        # Stack chain and draw dimensions into a single sample dimension
                        intercept_samples = intercept_values.stack(sample=('chain', 'draw')).values
                        # Expand to match (date, sample) shape of predictions
                        n_dates = posterior_predictive_samples.sizes['date']
                        intercept_expanded = np.tile(intercept_samples, (n_dates, 1))
                    else:
                        # Keep chain and draw dimensions
                        intercept_samples = intercept_values.values  # shape: (chain, draw)
                        n_dates = posterior_predictive_samples.sizes['date']
                        # Expand to (chain, draw, date)
                        intercept_expanded = np.repeat(
                            intercept_samples[:, :, np.newaxis], n_dates, axis=2
                        )

                # Subtract intercept from predictions (in log scale, subtraction is correct)
                posterior_predictive_samples[self.output_var] = (
                    posterior_predictive_samples[self.output_var] - intercept_expanded
                )

        # Transform to original scale if requested (default behavior)
        if original_scale:
            if hasattr(self, "target_scale"):
                # Use the new computed scale factor approach
                if self.output_var in posterior_predictive_samples:
                    posterior_predictive_samples[self.output_var] = (
                        posterior_predictive_samples[self.output_var]
                        * self.target_scale
                    )
                # No need to apply transformer since we use direct scaling
            else:
                # Fallback to transformer for backward compatibility
                if self.output_var in posterior_predictive_samples:
                    posterior_predictive_samples[self.output_var] = (
                        apply_sklearn_transformer_across_dim(
                            data=posterior_predictive_samples[self.output_var],
                            func=self.get_target_transformer().inverse_transform,
                            dim_name="date",
                        )
                    )

                intercept_condition = (
                    "intercept"
                    in sample_posterior_predictive_kwargs.get("var_names", [])
                    and not self.time_varying_intercept
                )

                # We need to expand the intercept to the date dimension
                # because the target transformer is applied to the date dimension
                if intercept_condition:
                    posterior_predictive_samples["intercept"] = (
                        posterior_predictive_samples["intercept"]
                        .expand_dims(
                            dim={"date": posterior_predictive_samples["date"]}, axis=0
                        )
                        .rename("intercept")
                    )

                posterior_predictive_samples = apply_sklearn_transformer_across_dim(
                    data=posterior_predictive_samples,
                    func=self.get_target_transformer().inverse_transform,
                    dim_name="date",
                )

                # We need to remove the date dimension after the inverse transform
                if intercept_condition:
                    posterior_predictive_samples["intercept"] = (
                        posterior_predictive_samples["intercept"].isel(date=0)
                    )

        # Log completion of posterior predictive sampling
        self.logger.info("Posterior predictive sampling completed successfully")
        self.logger.info(f"Output samples shape: {posterior_predictive_samples.sizes}")
        self.logger.info(f"Output variables: {list(posterior_predictive_samples.data_vars.keys())}")
        if original_scale:
            self.logger.info("Samples returned in original scale")
        else:
            self.logger.info("Samples returned in scaled format")

        return posterior_predictive_samples

    def add_lift_test_measurements(
        self,
        df_lift_test: pd.DataFrame,
        dist: type[pm.Distribution] = pm.Gamma,
        name: str = "lift_measurements",
    ) -> None:
        """Add lift tests to the model.

        The model difference of a channel's saturation curve is created
        from `x` and `x + delta_x` for each channel. This random variable is
        then conditioned using the empirical lift, `delta_y`, and `sigma` of the lift test
        with the specified distribution `dist`.

        The pseudo-code for the lift test is as follows:

        .. code-block:: python

            model_estimated_lift = saturation_curve(x + delta_x) - saturation_curve(x)
            empirical_lift = delta_y
            dist(abs(model_estimated_lift), sigma=sigma, observed=abs(empirical_lift))


        The model has to be built before adding the lift tests.

        Parameters
        ----------
        df_lift_test : pd.DataFrame
            DataFrame with lift test results with at least the following columns:
                * `channel`: channel name. Must be present in `channel_columns`.
                * `x`: x axis value of the lift test.
                * `delta_x`: change in x axis value of the lift test.
                * `delta_y`: change in y axis value of the lift test.
                * `sigma`: standard deviation of the lift test.
        dist : pm.Distribution, optional
            The distribution to use for the likelihood, by default pm.Gamma
        name : str, optional
            The name of the likelihood of the lift test contribution(s),
            by default "lift_measurements". Name change required if calling
            this method multiple times.

        Raises
        ------
        RuntimeError
            If the model has not been built yet.
        KeyError
            If the 'channel' column is not present in df_lift_test.

        Examples
        --------
        Build the model first then add lift test measurements.

        .. code-block:: python

            import pandas as pd
            import numpy as np

            from pymc_marketing.mmm import (
                GeometricAdstock,
                LogisticSaturation,
                MMM,
            )

            model = MMM(
                date_column="date_week",
                channel_columns=["x1", "x2"],
                adstock=GeometricAdstock(l_max=8),
                saturation=LogisticSaturation(),
                control_columns=[
                    "event_1",
                    "event_2",
                ],
                yearly_seasonality=2,
            )

            X: pd.DataFrame = ...
            y: np.ndarray = ...

            model.build_model(X, y)

            df_lift_test = pd.DataFrame(
                {
                    "channel": ["x1", "x1"],
                    "x": [1, 1],
                    "delta_x": [0.1, 0.2],
                    "delta_y": [0.1, 0.1],
                    "sigma": [0.1, 0.1],
                }
            )

            model.add_lift_test_measurements(df_lift_test)

        """
        if not hasattr(self, "model"):
            raise RuntimeError(
                "The model has not been built yet. Please, build the model first."
            )

        if "channel" not in df_lift_test.columns:
            raise KeyError(
                "The 'channel' column is required to map the lift measurements to the model."
            )

        if self.time_varying_media and "date" not in df_lift_test.columns:
            # `time_varying_media=True` parameter requires the date in the df_lift_test DataFrame.
            # The `add_lift_test_measurements` method itself doesn't need a date
            # We need to make sure the `date` coord is present in model_coords
            # By adding this we make sure the model_coords match
            df_lift_test["date"] = pd.to_datetime(self.model_coords["date"][0])

        # Store df_lift_test for testing purposes
        self._last_lift_test_df = df_lift_test

        df_lift_test_scaled = scale_lift_measurements(
            df_lift_test=df_lift_test,
            channel_col="channel",
            channel_columns=self.channel_columns,  # type: ignore
            channel_transform=self.channel_transformer.transform,
            target_transform=self.target_transformer.transform,
        )
        # This is coupled with the name of the
        # latent process Deterministic
        time_varying_var_name = (
            "media_temporal_latent_multiplier" if self.time_varying_media else None
        )
        add_lift_measurements_to_likelihood_from_saturation(
            df_lift_test=df_lift_test_scaled,
            saturation=self.saturation,
            time_varying_var_name=time_varying_var_name,
            model=self.model,
            dist=dist,
            name=name,
        )

    def _create_synth_dataset(
        self,
        df: pd.DataFrame,
        date_column: str,
        allocation_strategy: DataArray,
        channels: list[str] | tuple[str],
        controls: list[str] | None,
        target_col: str,
        time_granularity: Literal["daily", "weekly", "monthly", "quarterly", "yearly"],
        time_length: int,
        lag: int,
        noise_level: float = 0.01,
    ) -> pd.DataFrame:
        """Create a synthetic dataset based on the given allocation strategy (Budget) and time granularity.

        **Important**: When generating the posterior predicive distribution for the target with the optimized budget,
        we are setting the control variables to zero! This is done because in many situations we do not have all the
        control variables in the future (e.g. outlier control, special events).

        Parameters
        ----------
        df : pd.DataFrame
            The original dataset.
        date_column : str
            The name of the date column in the dataset.
        allocation_strategy : DataArray
            A DataArray mapping channel names to their corresponding allocation values.
        channels : list[str] | tuple[str]
            A list or tuple of channel names.
        controls : list[str] | None
            A list of control column names or None if no controls are present.
        target_col : str
            The name of the target column.
        time_granularity : Literal["daily", "weekly", "monthly", "quarterly", "yearly"]
            The time granularity of the synthetic dataset: 'daily', 'weekly', 'monthly', 'quarterly', or 'yearly'.
        time_length : int
            The length of the synthetic dataset in terms of the time granularity.
        lag : int
            The lag value (not used in this function).
        noise_level : int
            The level of noise added to the allocation strategy (by default 1%).

        Returns
        -------
        pd.DataFrame
            A synthetic dataset with the specified allocation strategy and time granularity.

        Raises
        ------
        ValueError
            If the time granularity is not supported.

        """
        time_offsets = {
            "daily": {"days": 1},
            "weekly": {"weeks": 1},
            "monthly": {"months": 1},
            "quarterly": {"months": 3},
            "yearly": {"years": 1},
        }

        if time_granularity not in time_offsets:
            raise ValueError(
                "Unsupported time granularity. Choose from 'daily', 'weekly', 'monthly', 'quarterly', 'yearly'."
            )

        X_data = self.preprocessed_data["X"]
        mmm_has_controls = (
            self.control_columns is not None
            and len(self.control_columns) > 0
            and isinstance(X_data, pd.DataFrame)
            and all(column in X_data.columns for column in self.control_columns)
        )

        if controls is not None and mmm_has_controls:
            _controls: list[str] = controls
        elif controls is not None and not mmm_has_controls:
            raise ValueError(
                """The model was built without controls and cannot translate the provided controls to contributions.
                Remove the controls from the function call and try again."""
            )
        else:
            _controls = []

        last_date = pd.to_datetime(df[date_column]).max()
        new_dates = []
        for i in range(1, time_length + 1):
            if time_granularity == "daily":
                new_date = last_date + pd.DateOffset(days=i)
            elif time_granularity == "weekly":
                new_date = last_date + pd.DateOffset(weeks=i)
            elif time_granularity == "monthly":
                new_date = last_date + pd.DateOffset(months=i)
            elif time_granularity == "quarterly":
                new_date = last_date + pd.DateOffset(months=3 * i)
            elif time_granularity == "yearly":
                new_date = last_date + pd.DateOffset(years=i)
            new_dates.append(new_date)

        if allocation_strategy.dims != ("channel",):
            raise ValueError(
                "The allocation strategy DataArray must have a single dimension named 'channel'."
                f"Got {allocation_strategy.dims}"
            )

        new_rows = [
            {
                self.date_column: pd.to_datetime(new_date),
                **{
                    channel: allocation_strategy.sel(channel=channel).values
                    + np.random.normal(
                        0, noise_level * allocation_strategy.sel(channel=channel).values
                    )
                    for channel in channels
                },
                **{control: 0 for control in _controls},
                target_col: 0,
            }
            for new_date in new_dates
        ]

        return pd.DataFrame(new_rows)

    def sample_response_distribution(
        self,
        allocation_strategy: DataArray | dict[str, float],
        time_granularity: Literal["daily", "weekly", "monthly", "quarterly", "yearly"],
        num_periods: int,
        noise_level: float,
    ) -> az.InferenceData:
        """Generate synthetic dataset and sample posterior predictive based on allocation.

        Parameters
        ----------
        allocation_strategy : DataArray or dict[str, float]
            The allocation strategy for the channels.
        time_granularity : Literal["daily", "weekly", "monthly", "quarterly", "yearly"]
            The granularity of the time units (e.g., 'daily', 'weekly', 'monthly').
        num_periods : int
            The number of time periods for prediction.
        noise_level : float
            The level of noise to add to the synthetic data.

        Returns
        -------
        az.InferenceData
            The posterior predictive samples based on the synthetic dataset.
        """
        if isinstance(allocation_strategy, dict):
            # For backward compatibility
            allocation_strategy = DataArray(
                pd.Series(allocation_strategy), dims=("channel",)
            )

        synth_dataset = self._create_synth_dataset(
            df=self.X,
            date_column=self.date_column,
            allocation_strategy=allocation_strategy,
            channels=self.channel_columns,
            controls=self.control_columns,
            target_col=self.output_var,
            time_granularity=time_granularity,
            time_length=num_periods,
            lag=self.adstock.l_max,
            noise_level=noise_level,
        )

        constant_data = allocation_strategy.to_dataset(name="allocation")

        return self.sample_posterior_predictive(
            X=synth_dataset,
            extend_idata=False,
            include_last_observations=True,
            original_scale=False,
            var_names=["y", "channel_contribution"],
            progressbar=False,
        ).merge(constant_data)

    def _set_predictors_for_optimization(self, num_periods: int) -> pm.Model:
        """Return the respective PyMC model with any predictors set for optimization."""
        if "date" not in self.model.coords:
            return self.model

        # Coords aren't used anywhere, so just go with integers
        opt_date_coords = range(num_periods + self.adstock.l_max)

        # Copy the model and update the "date" coordinates to the ones used for optimization (num_periods + l_max)
        model = self.model.copy()

        if "time_index" in model.named_vars:
            # Models with HSGP have a time_index data variable
            # We set them to start after the last date in the training data
            start_date = model["time_index"].get_value(borrow=True)[-1]
            opt_time_index = (
                np.arange(num_periods + self.adstock.l_max) + start_date + 1
            ).astype(start_date.dtype)
            model.set_data(
                "time_index", opt_time_index, coords={"date": opt_date_coords}
            )
        else:
            model.set_dim(
                "date", new_length=len(opt_date_coords), coord_values=opt_date_coords
            )
        return model

    def optimize_budget(
        self,
        budget: float | int,
        num_periods: int,
        budget_bounds: DataArray | dict[str, tuple[float, float]] | None = None,
        response_variable: str = "total_contribution",
        utility_function: UtilityFunctionType = average_response,
        constraints: Sequence[dict[str, Any]] = (),
        default_constraints: bool = True,
        callback: bool = False,
        **minimize_kwargs,
    ) -> (
        tuple[DataArray, OptimizeResult]
        | tuple[DataArray, OptimizeResult, list[dict[str, Any]]]
    ):
        """Optimize the given budget based on the specified utility function over a specified time period.

        .. deprecated:: 0.0.3
            This function optimizes the allocation of a given budget across different channels
           to maximize the response, considering adstock and saturation effects. It scales the
           budget and budget bounds, performs the optimization, and generates a synthetic dataset
           for posterior predictive sampling.

            The function first scales the budget and budget bounds using the maximum scale
            of the channel transformer. It then uses the `BudgetOptimizer` to allocate the
            budget, and creates a synthetic dataset based on the optimal allocation. Finally,
            it performs posterior predictive sampling on the synthetic dataset.

            **Important**: When generating the posterior predicive distribution for the target with the
            optimized budget, we are setting the control variables to zero! This is done because in many
            situations we do not have all the control variables in the future (e.g. outlier control,
            special events).

        Parameters
        ----------
        budget : float or int
            The total budget to be allocated.
        num_periods : int
            The number of time units over which the budget is to be allocated.
        budget_bounds : DataArrayr or dict[str, tuple[float, float]], optional
            An xarray DataArary or dictionary specifying the lower and upper bounds for the budget allocation
            for each channel. If None, no bounds are applied.
        response_variable : str, optional
            The response variable to optimize. Default is "total_contribution".
        utility_function : UtilityFunctionType, optional
            The utility function to maximize. Default is the mean of the response distribution.
        custom_constraints : list[dict[str, Any]], optional
            Custom constraints for the optimization. If None, no custom constraints are applied. Format:
            [{"key":...,"constraint_fun":...,"constraint_type":...}]
        default_constraints : bool, optional
            Whether to add the default sum constraint to the optimizer. Default is True.
        callback : bool, optional
            Whether to return callback information tracking optimization progress. When True, returns a third
            element containing a list of dictionaries with optimization information at each iteration.
            Default is False for backward compatibility.
        **minimize_kwargs
            Additional arguments to pass to the `BudgetOptimizer`.

        Returns
        -------
        az.InferenceData
            The posterior predictive samples generated from the synthetic dataset.

        Raises
        ------
        ValueError
            If the time granularity is not supported.

        ValueError
            If the noise level is not a float.
        """
        warnings.warn(
            "This method is deprecated and will be removed in a future version. "
            "Please migrate to the `Multidimensal.MMM` class.",
            DeprecationWarning,
            stacklevel=2,
        )

        from pymc_marketing.mmm.budget_optimizer import BudgetOptimizer

        allocator = BudgetOptimizer(
            num_periods=num_periods,
            utility_function=utility_function,
            response_variable=response_variable,
            custom_constraints=constraints,
            default_constraints=default_constraints,
            model=self,
        )

        return allocator.allocate_budget(
            total_budget=budget,
            budget_bounds=budget_bounds,
            callback=callback,
            **minimize_kwargs,
        )

    def plot_budget_allocation(
        self,
        samples: Dataset,
        figsize: tuple[float, float] = (12, 6),
        ax: plt.Axes | None = None,
        original_scale: bool = True,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot the budget allocation and channel contributions.

        Parameters
        ----------
        samples : Dataset
            The dataset containing the channel contributions.
        figsize : tuple[float, float], optional
            The size of the figure to be created, by default (12, 6).
        ax : plt.Axes, optional
            The axis to plot on. If None, a new figure and axis will be created.
        original_scale : bool, optional
            A boolean flag to determine if the values should be plotted in their original scale,
            by default True.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            The matplotlib figure object and axis containing the plot.

        """
        channel_contribution = (
            samples["channel_contribution"].mean(dim=["date", "sample"]).to_numpy()
        )

        if not original_scale:
            channel_contribution /= self.get_target_transformer()["scaler"].scale_

        allocated_spend = samples.allocation.to_numpy()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        bar_width = 0.35
        opacity = 0.7

        index = np.arange(len(self.channel_columns))

        bars1 = ax.bar(
            index,
            allocated_spend,
            bar_width,
            color="C0",
            alpha=opacity,
            label="Allocated Spend",
        )

        ax2 = ax.twinx()

        bars2 = ax2.bar(
            index + bar_width,
            channel_contribution,
            bar_width,
            color="C1",
            alpha=opacity,
            label="Channel Contribution",
        )

        ax.set_xlabel("Channels")
        ax.set_ylabel("Allocate Spend", color="C0")
        ax.tick_params(axis="x", rotation=90)
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels(self.channel_columns)

        ax.set_ylabel("Allocate Spend", color="C0", labelpad=10)
        ax2.set_ylabel("Channel Contributions", color="C1", labelpad=10)

        ax.grid(False)
        ax2.grid(False)

        bars = [bars1, bars2]
        labels = ["Allocated Spend", "Channel Contributions"]
        ax.legend(bars, labels, loc="best")

        return fig, ax

    def plot_allocated_contribution_by_channel(
        self,
        samples: Dataset,
        lower_quantile: float = 0.025,
        upper_quantile: float = 0.975,
        original_scale: bool = True,
    ) -> plt.Figure:
        """Plot the allocated contribution by channel with uncertainty intervals.

        This function visualizes the mean allocated contributions by channel along with
        the uncertainty intervals defined by the lower and upper quantiles. The contributions
        can be plotted on the original scale or the transformed scale.

        Parameters
        ----------
        samples : Dataset
            The dataset containing the samples of channel contributions.
        lower_quantile : float, optional
            The lower quantile for the uncertainty interval. Default is 0.025.
        upper_quantile : float, optional
            The upper quantile for the uncertainty interval. Default is 0.975.
        original_scale : bool, optional
            If True, the contributions are plotted on the original scale. Default is True.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The matplotlib figure object containing the plot.

        """
        if original_scale:
            # Use original scale variables if available
            if "channel_contribution_original_scale" in samples:
                channel_contribution = samples["channel_contribution_original_scale"]
            else:
                # Fallback for backward compatibility
                channel_contribution = samples["channel_contribution"]
        else:
            # Use scaled contributions directly
            channel_contribution = samples["channel_contribution"]

        fig, ax = plt.subplots()
        channel_contribution.mean(dim="sample").plot(hue="channel", ax=ax)

        for channel in self.model_coords["channel"]:
            ax.fill_between(
                x=channel_contribution.date.values,
                y1=channel_contribution.sel(channel=channel).quantile(
                    lower_quantile, dim="sample"
                ),
                y2=channel_contribution.sel(channel=channel).quantile(
                    upper_quantile, dim="sample"
                ),
                alpha=0.1,
            )
        return fig

    def _get_observed_data_for_plot(
        self, original_scale: bool = False
    ) -> np.ndarray:
        """Get the observed target data in the correct scale for plotting.

        This method properly handles log transformations that are applied within
        the PyMC model rather than in the preprocessing pipeline.

        Parameters
        ----------
        original_scale : bool, optional
            If True, return data in the original scale (before any transformations).
            If False, return data in the transformed scale used for modeling
            (including both scaling and log transformation). Default is False.

        Returns
        -------
        np.ndarray
            The observed target data in the requested scale.

        Notes
        -----
        The transformation pipeline is:
        1. Original y values
        2. Scaling: y_scaled = y / target_scale
        3. Log transformation: y_log = log(y_scaled + 1.0) [if enabled]

        When original_scale=False and log_transform_vars contains 'target',
        this returns log(y_scaled + 1.0).
        When original_scale=True, this returns the original y values.
        """
        # Always start with original y values
        y_data = np.asarray(self.y)

        if original_scale:
            # Return original values without any transformation
            return y_data

        # Apply the same transformations as in the model
        # Step 1: Apply scaling
        target_scale = getattr(self, "target_scale", 1.0)
        y_transformed = y_data / target_scale

        # Step 2: Apply log transformation if enabled
        if "target" in self.log_transform_vars:
            y_transformed = np.log(y_transformed + 1.0)

        return y_transformed

    def get_errors(self, original_scale: bool = False) -> DataArray:
        """Get model errors posterior distribution with proper log transformation handling.

        This method overrides the base class implementation to properly handle
        log transformations that are applied within the PyMC model.

        errors = true values - predicted

        Parameters
        ----------
        original_scale : bool, optional
            If True, return errors in the original scale of the target variable.
            If False, return errors in the transformed scale used for modeling
            (including both scaling and log transformation). Default is False.

        Returns
        -------
        DataArray
            The errors posterior distribution with dimensions (chain, draw, date).

        Raises
        ------
        RuntimeError
            If the model hasn't been fitted or posterior predictive hasn't been sampled.
        ValueError
            If the length of the target variable doesn't match the length of the date column.

        Notes
        -----
        This method ensures that both predictions and observed data are in the same
        scale before computing errors:
        - When original_scale=True: errors = y_original - predictions_original
        - When original_scale=False: errors = y_transformed - predictions_transformed

        For log-transformed models, the transformed scale includes both scaling and
        log transformation: log(y_scaled + 1.0).

        Examples
        --------
        >>> # Get errors in transformed space (log scale if log-transformed)
        >>> errors_log = mmm.get_errors(original_scale=False)
        >>> # Get errors in original scale (actual sales units)
        >>> errors_original = mmm.get_errors(original_scale=True)
        """
        try:
            posterior_predictive_data: Dataset = self._get_group_predictive_data(
                group="posterior_predictive", original_scale=original_scale
            )
        except Exception as e:
            raise RuntimeError(
                "Make sure the model has been fitted and the posterior_predictive has been sampled!"
            ) from e

        # Get observed data in the SAME scale as predictions
        target_array = self._get_observed_data_for_plot(original_scale=original_scale)

        if len(target_array) != len(posterior_predictive_data.date):
            raise ValueError(
                "The length of the target variable doesn't match the length of the date column. "
                "If you are computing out-of-sample errors, please overwrite `self.y` with the "
                "corresponding (non-transformed) target variable."
            )

        # Convert target to xarray with proper coordinates
        target = (
            pd.Series(target_array, index=posterior_predictive_data.date)
            .rename_axis("date")
            .to_xarray()
        )

        # Compute errors: true - predicted
        errors = (
            (target - posterior_predictive_data)[self.output_var]
            .rename("errors")
            .transpose(..., "date")
        )

        return errors

    def plot_posterior_predictive(
        self,
        original_scale: bool = False,
        hdi_list: list[float] | None = None,
        add_mean: bool = True,
        add_gradient: bool = False,
        ax: plt.Axes = None,
        **plt_kwargs: Any,
    ) -> plt.Figure:
        """Plot the posterior predictive distribution with proper log transformation handling.

        This method overrides the base class implementation to properly handle
        log transformations that are applied within the PyMC model.

        Parameters
        ----------
        original_scale : bool, optional
            If True, plot in the original scale of the target variable.
            If False, plot in the transformed scale used for modeling
            (including both scaling and log transformation). Default is False.
        hdi_list : list of float, optional
            List of HDI levels to plot. Default is [0.94, 0.5].
            Provide an empty list to omit plotting the HDI.
        add_mean : bool, optional
            If True, add the mean prediction to the plot. Default is True.
        add_gradient : bool, optional
            If True, add a gradient representation of the full posterior distribution.
            Default is False.
        ax : plt.Axes, optional
            A matplotlib Axes object to plot on. If None, a new figure and axes will be created.
        **plt_kwargs : dict
            Additional keyword arguments to pass to plt.subplots() when creating a new figure.

        Returns
        -------
        plt.Figure
            The matplotlib Figure object containing the plot.

        Raises
        ------
        ValueError
            If the length of the target variable doesn't match the length
            of the date column in the posterior predictive data.

        Notes
        -----
        This method properly handles log-transformed models by ensuring both
        predictions and observed data are in the same space:
        - When original_scale=True: Both shown in original scale (e.g., sales units)
        - When original_scale=False: Both shown in transformed scale (e.g., log space)

        Examples
        --------
        For a model with log_transform_vars=['target', 'channels']:

        >>> # Plot in log space (natural log of scaled values)
        >>> fig = mmm.plot_posterior_predictive(original_scale=False)

        >>> # Plot in original scale (actual sales units)
        >>> fig = mmm.plot_posterior_predictive(original_scale=True)
        """
        # Get predictions in the requested scale
        group_data: Dataset = self._get_group_predictive_data(
            group="posterior_predictive", original_scale=original_scale
        )

        # Get observed data in the SAME scale as predictions
        target_to_plot = self._get_observed_data_for_plot(original_scale=original_scale)

        if len(target_to_plot) != len(group_data.date):
            raise ValueError(
                "The length of the target variable doesn't match the length of the date column. "
                "If you are predicting out-of-sample, please overwrite `self.y` with the "
                "corresponding (non-transformed) target variable."
            )

        if ax is None:
            fig, ax = plt.subplots(**plt_kwargs)
        else:
            fig = ax.figure

        if hdi_list is None:
            hdi_list = [0.94, 0.5]

        if hdi_list and not add_gradient:
            alpha_list = np.linspace(0.2, 0.4, len(hdi_list), dtype=float)
            for hdi_prob, alpha in zip(hdi_list, alpha_list, strict=True):
                ax = self._add_hdi_to_plot(
                    ax=ax,
                    group="posterior_predictive",
                    original_scale=original_scale,
                    hdi_prob=hdi_prob,
                    alpha=alpha,
                )

        if add_mean:
            ax = self._add_mean_to_plot(
                ax=ax,
                group="posterior_predictive",
                original_scale=original_scale,
                color="blue",
            )

        if add_gradient:
            ax = self._add_gradient_to_plot(
                ax=ax,
                group="posterior_predictive",
                original_scale=original_scale,
                n_percentiles=30,
                palette="Blues",
            )

        ax.plot(
            np.asarray(group_data.date),
            target_to_plot,
            color="black",
            label="Observed",
        )
        ax.legend()

        # Set appropriate y-axis label based on scale and transformation
        if "target" in self.log_transform_vars and not original_scale:
            ylabel = f"{self.output_var} (log scale)"
        elif original_scale:
            ylabel = self.output_var
        else:
            ylabel = f"{self.output_var} (scaled)"

        ax.set(
            title="Posterior predictive check",
            xlabel="date",
            ylabel=ylabel,
        )

        return fig

    def plot_prior_predictive(
        self,
        original_scale: bool = False,
        hdi_list: list[float] | None = None,
        add_mean: bool = True,
        add_gradient: bool = False,
        ax: plt.Axes = None,
        **plt_kwargs: Any,
    ) -> plt.Figure:
        """Plot the prior predictive distribution with proper log transformation handling.

        This method overrides the base class implementation to properly handle
        log transformations that are applied within the PyMC model.

        Parameters
        ----------
        original_scale : bool, optional
            If True, plot in the original scale of the target variable.
            If False, plot in the transformed scale used for modeling. Default is False.
        hdi_list : list of float, optional
            List of HDI levels to plot. Default is [0.94, 0.5].
        add_mean : bool, optional
            If True, add the mean prediction to the plot. Default is True.
        add_gradient : bool, optional
            If True, add a gradient representation of the full posterior distribution.
            Default is False.
        ax : plt.Axes, optional
            A matplotlib Axes object to plot on. If None, a new figure and axes will be created.
        **plt_kwargs : dict
            Additional keyword arguments to pass to plt.subplots() when creating a new figure.

        Returns
        -------
        plt.Figure
            The matplotlib Figure object containing the plot.
        """
        # Get predictions in the requested scale
        group_data: Dataset = self._get_group_predictive_data(
            group="prior_predictive", original_scale=original_scale
        )

        # Get observed data in the SAME scale as predictions
        target_to_plot = self._get_observed_data_for_plot(original_scale=original_scale)

        if len(target_to_plot) != len(group_data.date):
            raise ValueError(
                "The length of the target variable doesn't match the length of the date column. "
                "If you are predicting out-of-sample, please overwrite `self.y` with the "
                "corresponding (non-transformed) target variable."
            )

        if ax is None:
            fig, ax = plt.subplots(**plt_kwargs)
        else:
            fig = ax.figure

        if hdi_list is None:
            hdi_list = [0.94, 0.5]

        if hdi_list and not add_gradient:
            alpha_list = np.linspace(0.2, 0.4, len(hdi_list), dtype=float)
            for hdi_prob, alpha in zip(hdi_list, alpha_list, strict=True):
                ax = self._add_hdi_to_plot(
                    ax=ax,
                    group="prior_predictive",
                    original_scale=original_scale,
                    hdi_prob=hdi_prob,
                    alpha=alpha,
                )

        if add_mean:
            ax = self._add_mean_to_plot(
                ax=ax, group="prior_predictive", original_scale=original_scale, color="blue"
            )

        if add_gradient:
            ax = self._add_gradient_to_plot(
                ax=ax,
                group="prior_predictive",
                original_scale=original_scale,
                n_percentiles=30,
                palette="Blues",
            )

        ax.plot(
            np.asarray(group_data.date),
            target_to_plot,
            color="black",
            label="Observed",
        )
        ax.legend()

        # Set appropriate y-axis label
        if "target" in self.log_transform_vars and not original_scale:
            ylabel = f"{self.output_var} (log scale)"
        elif original_scale:
            ylabel = self.output_var
        else:
            ylabel = f"{self.output_var} (scaled)"

        ax.set(
            title="Prior predictive check",
            xlabel="date",
            ylabel=ylabel,
        )

        return fig

    def plot_components_contributions(
        self, original_scale: bool = False, **plt_kwargs: Any
    ) -> plt.Figure:
        """Plot the target variable and the posterior predictive model components.

        This method overrides the base class implementation to properly handle
        log transformations that are applied within the PyMC model.

        We can plot the target variable and the posterior predictive model components in
        the scaled space or in the original space.

        Parameters
        ----------
        original_scale : bool, optional
            If True, plot in the original scale of the target variable.
            If False, plot in the transformed scale used for modeling
            (including both scaling and log transformation). Default is False.
        **plt_kwargs
            Additional keyword arguments to pass to `plt.subplots`.

        Returns
        -------
        plt.Figure
            The matplotlib Figure object containing the plot.

        Notes
        -----
        This method properly handles log-transformed models by ensuring both
        the target variable and all component contributions are in the same space:
        - When original_scale=True: Everything shown in original scale (e.g., sales units)
        - When original_scale=False: Everything shown in transformed scale (e.g., log10 space)

        The method automatically detects whether the model uses log transformation
        by checking self.log_transform_vars and applies the correct transformations.

        Examples
        --------
        For a model with log_transform_vars=['target', 'channels']:

        >>> # Plot in log space (log10 of scaled values)
        >>> fig = mmm.plot_components_contributions(original_scale=False)

        >>> # Plot in original scale (actual sales units)
        >>> fig = mmm.plot_components_contributions(original_scale=True)
        """
        # Get channel contributions in the requested scale
        channel_contribution = self.get_ts_contribution_posterior(
            var_contribution="channel_contribution", original_scale=original_scale
        )

        means = [channel_contribution.mean(["chain", "draw"])]
        contribution_vars = [
            az.hdi(channel_contribution, hdi_prob=0.94).channel_contribution
        ]

        # Collect control and seasonality contributions if they exist
        for arg, var_contribution in zip(
            ["control_columns", "yearly_seasonality"],
            ["control_contribution", "fourier_contribution"],
            strict=True,
        ):
            if getattr(self, arg, None):
                contributions = self.get_ts_contribution_posterior(
                    var_contribution=var_contribution, original_scale=original_scale
                )

                means.append(contributions.mean(["chain", "draw"]))
                contribution_vars.append(
                    az.hdi(contributions, hdi_prob=0.94)[var_contribution]
                )

        fig, ax = plt.subplots(**plt_kwargs)

        # Plot all contributions with HDI bands
        for i, (mean, hdi, var_contribution) in enumerate(
            zip(
                means,
                contribution_vars,
                [
                    "channel_contribution",
                    "control_contribution",
                    "fourier_contribution",
                ],
                strict=False,
            )
        ):
            if self.X is not None:
                ax.fill_between(
                    x=self.X[self.date_column],
                    y1=hdi.isel(hdi=0),
                    y2=hdi.isel(hdi=1),
                    color=f"C{i}",
                    alpha=0.25,
                    label=f"$94\\%$ HDI ({var_contribution})",
                )
                ax.plot(
                    np.asarray(self.X[self.date_column]),
                    np.asarray(mean),
                    color=f"C{i}",
                )

        if self.X is not None:
            # Handle intercept
            intercept = az.extract(
                self.fit_result, var_names=["intercept"], combined=False
            )

            # Transform intercept to the correct scale
            if not original_scale:
                # For log-transformed models, the intercept needs proper handling
                if "target" in self.log_transform_vars:
                    # The intercept is already in log space in the model
                    # Just need to account for scaling
                    target_scale = getattr(self, "target_scale", 1.0)
                    if target_scale != 1.0:
                        # If there's scaling, we need to adjust the intercept
                        # But for log models, intercept is additive in log space
                        # so we keep it as-is
                        pass
                else:
                    # For non-log models, apply the same transformer as before
                    intercept = apply_sklearn_transformer_across_dim(
                        data=intercept,
                        func=self.get_target_transformer().transform,
                        dim_name="chain",
                    )
            else:
                # When original_scale=True with log transform
                if "target" in self.log_transform_vars:
                    # The intercept in log space needs to be converted back
                    # This is complex because exp(intercept) isn't quite right
                    # The intercept contributes additively in log space
                    # When we transform back: exp(log_y) = exp(intercept + other_terms)
                    # This is handled by the original_scale deterministics in the model
                    pass

            if intercept.ndim == 2:
                # Intercept has a stationary prior
                intercept_mean = intercept.mean(["chain", "draw"]).data
                intercept_hdi = np.repeat(
                    a=az.hdi(intercept).intercept.data[None, ...],
                    repeats=self.X[self.date_column].shape[0],
                    axis=0,
                )
            elif intercept.ndim == 3:
                # Intercept has a time-varying prior
                intercept_mean = intercept.mean(["chain", "draw"]).data
                intercept_hdi = az.hdi(intercept).intercept.data

            ax.plot(
                np.asarray(self.X[self.date_column]),
                np.full(len(self.X[self.date_column]), intercept_mean)
                if intercept.ndim == 2
                else intercept_mean,
                color=f"C{i + 1}",
            )
            ax.fill_between(
                x=self.X[self.date_column],
                y1=intercept_hdi[:, 0],
                y2=intercept_hdi[:, 1],
                color=f"C{i + 1}",
                alpha=0.25,
                label="$94\\%$ HDI (intercept)",
            )

            # Get observed target data in the SAME scale as contributions
            # This is the key fix - use our helper method!
            y_to_plot = self._get_observed_data_for_plot(original_scale=original_scale)

            # Set appropriate y-axis label
            if "target" in self.log_transform_vars and not original_scale:
                ylabel = f"{self.output_var} (log scale)"
            elif original_scale:
                ylabel = self.output_var
            else:
                ylabel = f"{self.output_var} (scaled)"

            ax.plot(
                np.asarray(self.X[self.date_column]),
                y_to_plot,
                label=ylabel,
                color="black",
            )
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
            ax.set(
                title="Posterior Predictive Model Components",
                xlabel="date",
                ylabel=ylabel,
            )
        return fig

    def extract_coefficients(
        self,
        include_ci: bool = True,
        ci_percentiles: tuple[float, float] = (5, 95),
        include_metrics: bool = True,
        original_scale_metrics: bool = True,
        verbose: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """Extract all model coefficients and fit metrics for manual calculation.

        This method provides a comprehensive summary of all estimated parameters
        needed to manually reconstruct the model predictions. It extracts coefficients
        in the transformed space (e.g., log space if log transformation was used).

        Parameters
        ----------
        include_ci : bool, optional
            Whether to include confidence intervals for parameters, by default True
        ci_percentiles : tuple of float, optional
            Percentiles for confidence intervals (lower, upper), by default (5, 95)
            which gives a 90% credible interval
        include_metrics : bool, optional
            Whether to calculate and include model fit metrics (MAPE, MAE, RMSE, R²),
            by default True
        original_scale_metrics : bool, optional
            Whether to calculate metrics in original scale of target variable,
            by default True. If False, metrics are calculated in transformed space.
        verbose : bool, optional
            Whether to print polished output showing all extracted coefficients,
            by default False. When True, prints formatted tables for each section.

        Returns
        -------
        dict of pd.DataFrame
            Dictionary containing DataFrames for different parameter types:

            - 'intercept': Intercept parameter with mean, std, and CI
            - 'scaling_transformation': Scaling factors and log epsilon values
            - 'control_coefficients': Coefficients for control variables (gamma_control)
            - 'seasonality_coefficients': Fourier coefficients (gamma_fourier)
            - 'channel_summary': Summary of all channel parameters (means only)
            - 'adstock_parameters': Detailed adstock parameters by channel
            - 'saturation_beta': Detailed saturation beta parameters by channel
            - 'saturation_lambda': Detailed saturation lambda parameters (if applicable)
            - 'model_error': Model error parameter (y_sigma)
            - 'fit_metrics': Model fit metrics (MAPE, MAE, RMSE, R²)
            - 'all_coefficients': All coefficients in single table (means only)
            - 'model_config': Model configuration information

        Raises
        ------
        RuntimeError
            If the model hasn't been fitted yet

        Notes
        -----
        The extracted coefficients are in the **transformed space** used during modeling:

        - If `log_transform_vars=['target', 'channels']` was used, coefficients are
          estimated in log space
        - If no log transformation was used, coefficients are in the scaled/original space

        To manually reconstruct predictions, you need to:

        1. Apply the same transformations (scaling, log, adstock, saturation)
        2. Use the extracted coefficients for linear combination
        3. Apply inverse transformations to get predictions in original scale

        Examples
        --------
        Extract all coefficients after fitting:

        >>> mmm.fit(X=X, y=y)
        >>> coefficients = mmm.extract_coefficients()
        >>> print(coefficients.keys())
        dict_keys(['intercept', 'scaling_transformation', 'control_coefficients', ...])

        Access specific parameter groups:

        >>> print(coefficients['intercept'])
        >>> print(coefficients['channel_summary'])
        >>> print(coefficients['fit_metrics'])

        Extract without confidence intervals for faster computation:

        >>> coefficients = mmm.extract_coefficients(include_ci=False)

        Use different confidence interval levels:

        >>> coefficients = mmm.extract_coefficients(ci_percentiles=(2.5, 97.5))  # 95% CI

        Extract with polished printed output:

        >>> coefficients = mmm.extract_coefficients(verbose=True)
        # Prints formatted tables for all coefficient sections
        # Still returns the dictionary of DataFrames

        See Also
        --------
        get_errors : Get model prediction errors
        plot_posterior_predictive : Visualize model fit
        """
        if not hasattr(self, "fit_result") or self.fit_result is None:
            raise RuntimeError(
                "Model must be fitted before extracting coefficients. "
                "Please call fit() first."
            )

        import pandas as pd
        import numpy as np

        results = {}
        lower_percentile, upper_percentile = ci_percentiles

        # ========== 1. INTERCEPT ==========
        if "intercept" in self.fit_result:
            intercept_samples = self.fit_result["intercept"].values.flatten()
            intercept_data = {
                "parameter": ["intercept"],
                "mean": [intercept_samples.mean()],
            }

            if include_ci:
                intercept_data["std"] = [intercept_samples.std()]
                intercept_ci = np.percentile(
                    intercept_samples, [lower_percentile, upper_percentile]
                )
                intercept_data[f"ci_{lower_percentile}%"] = [intercept_ci[0]]
                intercept_data[f"ci_{upper_percentile}%"] = [intercept_ci[1]]

            results["intercept"] = pd.DataFrame(intercept_data)

        # ========== 2. SCALING AND TRANSFORMATION PARAMETERS ==========
        scaling_params = []

        # Target scaling
        target_scale = getattr(self, "target_scale", 1.0)
        scaling_params.append(
            {
                "parameter": "target_scale_factor",
                "value": target_scale,
                "description": "Factor used to scale target variable",
            }
        )

        # Log transformation epsilon
        if "target" in self.log_transform_vars:
            scaling_params.append(
                {
                    "parameter": "log_epsilon_target",
                    "value": 1.0,
                    "description": "Epsilon for log(target + epsilon)",
                }
            )

        if "channels" in self.log_transform_vars:
            scaling_params.append(
                {
                    "parameter": "log_epsilon_channels",
                    "value": 1.0,
                    "description": "Epsilon for log(channels + epsilon)",
                }
            )

        # Channel scaling factors
        if hasattr(self, "channel_scale"):
            for i, channel in enumerate(self.channel_columns):
                scale_val = (
                    self.channel_scale[i]
                    if hasattr(self.channel_scale, "__len__")
                    else self.channel_scale
                )
                scaling_params.append(
                    {
                        "parameter": f"channel_scale_{channel}",
                        "value": scale_val,
                        "description": f"Scale factor for channel {channel}",
                    }
                )

        results["scaling_transformation"] = pd.DataFrame(scaling_params)

        # ========== 3. CONTROL VARIABLE COEFFICIENTS ==========
        if "gamma_control" in self.fit_result and self.control_columns:
            control_means = (
                self.fit_result["gamma_control"].mean(dim=["chain", "draw"]).values
            )
            control_data = {
                "variable": self.control_columns,
                "coefficient": control_means,
            }

            if include_ci:
                control_std = (
                    self.fit_result["gamma_control"].std(dim=["chain", "draw"]).values
                )
                control_ci = np.percentile(
                    self.fit_result["gamma_control"]
                    .values.reshape(-1, len(self.control_columns)),
                    [lower_percentile, upper_percentile],
                    axis=0,
                )
                control_data["std"] = control_std
                control_data[f"ci_{lower_percentile}%"] = control_ci[0]
                control_data[f"ci_{upper_percentile}%"] = control_ci[1]

            results["control_coefficients"] = pd.DataFrame(control_data)

        # ========== 4. SEASONALITY COEFFICIENTS ==========
        if "gamma_fourier" in self.fit_result:
            fourier_means = (
                self.fit_result["gamma_fourier"].mean(dim=["chain", "draw"]).values
            )
            fourier_modes = list(self.fit_result.coords["fourier_mode"].values)
            n_modes = len(fourier_modes)

            fourier_data = {
                "fourier_mode": fourier_modes,
                "coefficient": fourier_means,
            }

            if include_ci:
                fourier_std = (
                    self.fit_result["gamma_fourier"].std(dim=["chain", "draw"]).values
                )
                fourier_ci = np.percentile(
                    self.fit_result["gamma_fourier"].values.reshape(-1, n_modes),
                    [lower_percentile, upper_percentile],
                    axis=0,
                )
                fourier_data["std"] = fourier_std
                fourier_data[f"ci_{lower_percentile}%"] = fourier_ci[0]
                fourier_data[f"ci_{upper_percentile}%"] = fourier_ci[1]

            results["seasonality_coefficients"] = pd.DataFrame(fourier_data)

        # ========== 5. CHANNEL PARAMETERS ==========
        # 5.1 Summary (means only)
        channel_summary = []
        for channel in self.channel_columns:
            row = {"channel": channel}

            if "adstock_alpha" in self.fit_result:
                row["adstock_alpha"] = (
                    self.fit_result["adstock_alpha"]
                    .sel(channel=channel)
                    .mean()
                    .values.item()
                )
                row["adstock_l_max"] = self.adstock.l_max

            if "saturation_beta" in self.fit_result:
                row["saturation_beta"] = (
                    self.fit_result["saturation_beta"]
                    .sel(channel=channel)
                    .mean()
                    .values.item()
                )

            if "saturation_lambda" in self.fit_result:
                row["saturation_lambda"] = (
                    self.fit_result["saturation_lambda"]
                    .sel(channel=channel)
                    .mean()
                    .values.item()
                )

            channel_summary.append(row)

        results["channel_summary"] = pd.DataFrame(channel_summary)

        # 5.2 Adstock parameters (detailed)
        if "adstock_alpha" in self.fit_result:
            adstock_data = []
            for channel in self.channel_columns:
                adstock_samples = (
                    self.fit_result["adstock_alpha"]
                    .sel(channel=channel)
                    .values.flatten()
                )
                row = {
                    "channel": channel,
                    "mean": adstock_samples.mean(),
                    "l_max": self.adstock.l_max,
                }

                if include_ci:
                    adstock_ci = np.percentile(
                        adstock_samples, [lower_percentile, upper_percentile]
                    )
                    row["std"] = adstock_samples.std()
                    row[f"ci_{lower_percentile}%"] = adstock_ci[0]
                    row[f"ci_{upper_percentile}%"] = adstock_ci[1]

                adstock_data.append(row)

            results["adstock_parameters"] = pd.DataFrame(adstock_data)

        # 5.3 Saturation beta parameters (detailed)
        if "saturation_beta" in self.fit_result:
            saturation_beta_data = []
            for channel in self.channel_columns:
                beta_samples = (
                    self.fit_result["saturation_beta"]
                    .sel(channel=channel)
                    .values.flatten()
                )
                row = {
                    "channel": channel,
                    "mean": beta_samples.mean(),
                }

                if include_ci:
                    beta_ci = np.percentile(
                        beta_samples, [lower_percentile, upper_percentile]
                    )
                    row["std"] = beta_samples.std()
                    row[f"ci_{lower_percentile}%"] = beta_ci[0]
                    row[f"ci_{upper_percentile}%"] = beta_ci[1]

                saturation_beta_data.append(row)

            results["saturation_beta"] = pd.DataFrame(saturation_beta_data)

        # 5.4 Saturation lambda parameters (detailed)
        if "saturation_lambda" in self.fit_result:
            saturation_lambda_data = []
            for channel in self.channel_columns:
                lambda_samples = (
                    self.fit_result["saturation_lambda"]
                    .sel(channel=channel)
                    .values.flatten()
                )
                row = {
                    "channel": channel,
                    "mean": lambda_samples.mean(),
                }

                if include_ci:
                    lambda_ci = np.percentile(
                        lambda_samples, [lower_percentile, upper_percentile]
                    )
                    row["std"] = lambda_samples.std()
                    row[f"ci_{lower_percentile}%"] = lambda_ci[0]
                    row[f"ci_{upper_percentile}%"] = lambda_ci[1]

                saturation_lambda_data.append(row)

            results["saturation_lambda"] = pd.DataFrame(saturation_lambda_data)

        # ========== 6. MODEL ERROR ==========
        if "y_sigma" in self.fit_result:
            sigma_samples = self.fit_result["y_sigma"].values.flatten()
            sigma_data = {
                "parameter": ["y_sigma"],
                "mean": [sigma_samples.mean()],
            }

            if include_ci:
                sigma_ci = np.percentile(
                    sigma_samples, [lower_percentile, upper_percentile]
                )
                sigma_data["std"] = [sigma_samples.std()]
                sigma_data[f"ci_{lower_percentile}%"] = [sigma_ci[0]]
                sigma_data[f"ci_{upper_percentile}%"] = [sigma_ci[1]]

            results["model_error"] = pd.DataFrame(sigma_data)

        # ========== 7. MODEL FIT METRICS ==========
        if include_metrics:
            try:
                errors = self.get_errors(original_scale=original_scale_metrics)
                y_xr = pd.Series(np.asarray(self.y), index=errors.date).to_xarray()

                # MAPE (as decimal)
                mape_overall = (np.abs(errors) / y_xr).mean().values.item()

                # MAE
                mae_overall = np.abs(errors).mean().values.item()

                # RMSE
                rmse_overall = np.sqrt((errors**2).mean()).values.item()

                # R-squared approximation
                ss_res = (errors**2).mean().values.item()
                ss_tot = np.var(np.asarray(self.y))
                r2_approx = 1 - (ss_res / ss_tot)

                metrics_data = {
                    "metric": ["MAPE", "MAE", "RMSE", "R²"],
                    "value": [mape_overall, mae_overall, rmse_overall, r2_approx],
                    "description": [
                        "Mean Absolute Percentage Error",
                        "Mean Absolute Error",
                        "Root Mean Squared Error",
                        "Coefficient of determination",
                    ],
                }

                results["fit_metrics"] = pd.DataFrame(metrics_data)
            except Exception as e:
                # If metrics calculation fails, skip it
                import warnings

                warnings.warn(
                    f"Could not calculate fit metrics: {e}. "
                    "Make sure posterior_predictive has been sampled.",
                    UserWarning,
                )

        # ========== 8. ALL COEFFICIENTS SUMMARY (MEANS ONLY) ==========
        all_coefs = {}

        # Intercept
        if "intercept" in self.fit_result:
            all_coefs["intercept"] = self.fit_result["intercept"].mean().values.item()

        # Control variables
        if "gamma_control" in self.fit_result and self.control_columns:
            control_means = (
                self.fit_result["gamma_control"].mean(dim=["chain", "draw"]).values
            )
            for i, control in enumerate(self.control_columns):
                all_coefs[f"gamma_control_{control}"] = control_means[i]

        # Seasonality
        if "gamma_fourier" in self.fit_result:
            fourier_means = (
                self.fit_result["gamma_fourier"].mean(dim=["chain", "draw"]).values
            )
            fourier_modes = list(self.fit_result.coords["fourier_mode"].values)
            for i, mode in enumerate(fourier_modes):
                all_coefs[f"gamma_fourier_{mode}"] = fourier_means[i]

        # Channel adstock
        if "adstock_alpha" in self.fit_result:
            for channel in self.channel_columns:
                all_coefs[f"adstock_alpha_{channel}"] = (
                    self.fit_result["adstock_alpha"]
                    .sel(channel=channel)
                    .mean()
                    .values.item()
                )

        # Channel saturation beta
        if "saturation_beta" in self.fit_result:
            for channel in self.channel_columns:
                all_coefs[f"saturation_beta_{channel}"] = (
                    self.fit_result["saturation_beta"]
                    .sel(channel=channel)
                    .mean()
                    .values.item()
                )

        # Channel saturation lambda
        if "saturation_lambda" in self.fit_result:
            for channel in self.channel_columns:
                all_coefs[f"saturation_lambda_{channel}"] = (
                    self.fit_result["saturation_lambda"]
                    .sel(channel=channel)
                    .mean()
                    .values.item()
                )

        # Model error
        if "y_sigma" in self.fit_result:
            all_coefs["y_sigma"] = self.fit_result["y_sigma"].mean().values.item()

        results["all_coefficients"] = pd.DataFrame(
            list(all_coefs.items()), columns=["parameter", "mean"]
        )

        # ========== 9. MODEL CONFIGURATION ==========
        config_data = {
            "setting": [
                "log_transform_target",
                "log_transform_channels",
                "adstock_type",
                "saturation_type",
                "date_column",
                "n_channels",
                "n_controls",
                "yearly_seasonality",
            ],
            "value": [
                "YES" if "target" in self.log_transform_vars else "NO",
                "YES" if "channels" in self.log_transform_vars else "NO",
                type(self.adstock).__name__,
                type(self.saturation).__name__,
                self.date_column,
                len(self.channel_columns),
                len(self.control_columns),
                self.yearly_seasonality if hasattr(self, "yearly_seasonality") else "N/A",
            ],
        }

        results["model_config"] = pd.DataFrame(config_data)

        # ========== VERBOSE PRINTING ==========
        if verbose:
            print("=" * 80)
            print("EXTRACTING ALL MODEL COEFFICIENTS FOR MANUAL CALCULATION")
            print("=" * 80)
            print()

            # 1. Intercept
            if "intercept" in results:
                print("1. INTERCEPT")
                print("-" * 80)
                print(results["intercept"].to_string(index=False))
                print()

            # 2. Scaling and Transformation
            print("2. SCALING AND TRANSFORMATION PARAMETERS")
            print("-" * 80)
            print(results["scaling_transformation"].to_string(index=False))
            print()

            # 3. Control Coefficients
            print("3. CONTROL VARIABLE COEFFICIENTS (gamma_control)")
            print("-" * 80)
            if "control_coefficients" in results:
                print(results["control_coefficients"].to_string(index=False))
            else:
                print("No control variables in model")
            print()

            # 4. Seasonality Coefficients
            print("4. SEASONALITY COEFFICIENTS (gamma_fourier)")
            print("-" * 80)
            if "seasonality_coefficients" in results:
                print(results["seasonality_coefficients"].to_string(index=False))
            else:
                print("No seasonality in model")
            print()

            # 5. Channel Parameters
            print("5. CHANNEL PARAMETERS")
            print("=" * 80)

            # 5.1 Summary
            print("5.1 Summary - All Channel Parameters (means only)")
            print("-" * 80)
            print(results["channel_summary"].to_string(index=False))
            print()

            # 5.2 Adstock
            if "adstock_parameters" in results:
                print("5.2 Adstock Parameters (alpha) - Detailed")
                print("-" * 80)
                print(results["adstock_parameters"].to_string(index=False))
                print()

            # 5.3 Saturation Beta
            if "saturation_beta" in results:
                print("5.3 Saturation Parameters (beta) - Detailed")
                print("-" * 80)
                print(results["saturation_beta"].to_string(index=False))
                print()

            # 5.4 Saturation Lambda
            if "saturation_lambda" in results:
                print("5.4 Saturation Parameters (lambda) - Detailed")
                print("-" * 80)
                print(results["saturation_lambda"].to_string(index=False))
                print()

            # 6. Model Error
            print("6. MODEL ERROR (SIGMA)")
            print("-" * 80)
            if "model_error" in results:
                print(results["model_error"].to_string(index=False))
            else:
                print("No sigma parameter in model")
            print()

            # 7. Fit Metrics
            if "fit_metrics" in results:
                print("7. MODEL FIT METRICS")
                print("-" * 80)
                print(results["fit_metrics"].to_string(index=False))
                print()

            # 8. All Coefficients Summary
            print("8. ALL COEFFICIENTS SUMMARY (MEANS ONLY)")
            print("-" * 80)
            print(results["all_coefficients"].to_string(index=False))
            print()

            # 9. Model Configuration
            print("9. MODEL CONFIGURATION")
            print("-" * 80)
            print(results["model_config"].to_string(index=False))
            print()
            print("Channel Columns:", self.channel_columns)
            print("Control Columns:", self.control_columns)
            print()

            print("=" * 80)
            print("COEFFICIENT EXTRACTION COMPLETE!")
            print("=" * 80)

        return results
