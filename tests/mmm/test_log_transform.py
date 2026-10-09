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

"""Tests for log_transform_vars parameter and serialization."""

import json
import tempfile
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pytest

from pymc_marketing.mmm.components.adstock import GeometricAdstock
from pymc_marketing.mmm.components.saturation import LogisticSaturation
from pymc_marketing.mmm.mmm import MMM

seed: int = sum(map(ord, "pymc_marketing"))
rng: np.random.Generator = np.random.default_rng(seed=seed)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    n = 100
    date_data = pd.date_range(start="2020-01-01", periods=n, freq="W")

    data = pd.DataFrame(
        {
            "date": date_data,
            "channel_1": rng.integers(low=10, high=400, size=n),
            "channel_2": rng.integers(low=5, high=50, size=n),
            "target": rng.integers(low=100, high=1000, size=n),
        }
    )

    X = data[["date", "channel_1", "channel_2"]]
    y = data["target"]

    return X, y


@pytest.fixture
def mmm_config():
    """Basic MMM configuration."""
    return {
        "date_column": "date",
        "channel_columns": ["channel_1", "channel_2"],
        "adstock": GeometricAdstock(l_max=4),
        "saturation": LogisticSaturation(),
        "validate_data": False,
    }


class TestLogTransformVarsParameter:
    """Test the log_transform_vars parameter functionality."""

    def test_parameter_declaration(self, mmm_config):
        """Test that log_transform_vars parameter can be set."""
        mmm_none = MMM(**mmm_config, log_transform_vars=None)
        assert mmm_none.log_transform_vars == []

        mmm_channels = MMM(**mmm_config, log_transform_vars=['channels'])
        assert mmm_channels.log_transform_vars == ['channels']

        mmm_both = MMM(**mmm_config, log_transform_vars=['target', 'channels'])
        assert mmm_both.log_transform_vars == ['target', 'channels']

    def test_default_value(self, mmm_config):
        """Test that log_transform_vars defaults to empty list."""
        mmm = MMM(**mmm_config)
        assert mmm.log_transform_vars == []

    def test_model_builds_with_channels_log_transform(self, mmm_config, sample_data):
        """Test that model builds successfully with log_transform_vars=['channels']."""
        X, y = sample_data

        mmm = MMM(**mmm_config, log_transform_vars=['channels'])
        mmm.build_model(X, y)

        # Check that model was built
        assert mmm.model is not None

        # Check that channel log transform variable exists
        assert "channel_data_log_transformed" in mmm.model.named_vars

    def test_model_builds_with_target_log_transform(self, mmm_config, sample_data):
        """Test that model builds successfully with log_transform_vars=['target']."""
        X, y = sample_data

        mmm = MMM(**mmm_config, log_transform_vars=['target'])
        mmm.build_model(X, y)

        # Check that model was built
        assert mmm.model is not None

        # Check that target log transform variable exists
        assert "target_log_transformed" in mmm.model.named_vars

    def test_model_builds_with_both_log_transforms(self, mmm_config, sample_data):
        """Test that model builds successfully with both target and channels log transformed."""
        X, y = sample_data

        mmm = MMM(**mmm_config, log_transform_vars=['target', 'channels'])
        mmm.build_model(X, y)

        # Check that model was built
        assert mmm.model is not None

        # Check that both log transform variables exist
        assert "target_log_transformed" in mmm.model.named_vars
        assert "channel_data_log_transformed" in mmm.model.named_vars

    def test_model_builds_without_log_transform(self, mmm_config, sample_data):
        """Test that model builds successfully with no log transformation."""
        X, y = sample_data

        mmm = MMM(**mmm_config, log_transform_vars=None)
        mmm.build_model(X, y)

        # Check that model was built
        assert mmm.model is not None

        # Check that log transform variables do NOT exist
        assert "target_log_transformed" not in mmm.model.named_vars
        assert "channel_data_log_transformed" not in mmm.model.named_vars


class TestLogTransformVarsSerialization:
    """Test serialization and deserialization of log_transform_vars parameter."""

    def test_serialization_to_attrs(self, mmm_config, sample_data):
        """Test that log_transform_vars is serialized to inference data attrs."""
        X, y = sample_data

        # Test with ['channels']
        mmm_channels = MMM(**mmm_config, log_transform_vars=['channels'])
        mmm_channels.build_model(X, y)

        # Get attrs
        attrs = mmm_channels.create_idata_attrs()

        # Check that log_transform_vars is in attrs
        assert "log_transform_vars" in attrs
        assert json.loads(attrs["log_transform_vars"]) == ['channels']

        # Test with ['target', 'channels']
        mmm_both = MMM(**mmm_config, log_transform_vars=['target', 'channels'])
        mmm_both.build_model(X, y)

        attrs = mmm_both.create_idata_attrs()
        assert "log_transform_vars" in attrs
        assert json.loads(attrs["log_transform_vars"]) == ['target', 'channels']

        # Test with None
        mmm_none = MMM(**mmm_config, log_transform_vars=None)
        mmm_none.build_model(X, y)

        attrs = mmm_none.create_idata_attrs()
        assert "log_transform_vars" in attrs
        assert json.loads(attrs["log_transform_vars"]) == []

    def test_deserialization_from_attrs(self, mmm_config):
        """Test that log_transform_vars is deserialized from attrs."""
        # Test with all required attrs
        full_attrs_channels = {
            "model_config": json.dumps({}),
            "date_column": json.dumps("date"),
            "control_columns": json.dumps(None),
            "channel_columns": json.dumps(["channel_1", "channel_2"]),
            "adstock": json.dumps({"type": "geometric", "l_max": 4}),
            "saturation": json.dumps({"type": "logistic"}),
            "adstock_first": json.dumps(True),
            "yearly_seasonality": json.dumps(None),
            "validate_data": json.dumps(False),
            "sampler_config": json.dumps({}),
            "log_transform_vars": json.dumps(["channels"]),
        }

        # This should not raise an error
        kwargs = MMM.attrs_to_init_kwargs(full_attrs_channels)
        assert kwargs["log_transform_vars"] == ["channels"]

        # Test with both
        full_attrs_both = {
            "model_config": json.dumps({}),
            "date_column": json.dumps("date"),
            "control_columns": json.dumps(None),
            "channel_columns": json.dumps(["channel_1", "channel_2"]),
            "adstock": json.dumps({"type": "geometric", "l_max": 4}),
            "saturation": json.dumps({"type": "logistic"}),
            "adstock_first": json.dumps(True),
            "yearly_seasonality": json.dumps(None),
            "validate_data": json.dumps(False),
            "sampler_config": json.dumps({}),
            "log_transform_vars": json.dumps(["target", "channels"]),
        }

        kwargs = MMM.attrs_to_init_kwargs(full_attrs_both)
        assert kwargs["log_transform_vars"] == ["target", "channels"]

    def test_save_and_load_preserves_parameter(self, mmm_config, sample_data):
        """Test that save/load cycle preserves log_transform_vars parameter."""
        X, y = sample_data

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test_mmm.nc"

            # Create and fit model with log_transform_vars=['target', 'channels']
            mmm_original = MMM(**mmm_config, log_transform_vars=['target', 'channels'])
            mmm_original.build_model(X, y)

            # Sample from prior to create inference data
            with mmm_original.model:
                mmm_original.prior_predictive_samples = mmm_original.sample_prior_predictive(
                    samples=10, random_seed=seed
                )

            # Save
            mmm_original.save(str(file_path))

            # Load
            mmm_loaded = MMM.load(str(file_path), check=False)

            # Check that parameter is preserved
            assert mmm_loaded.log_transform_vars == ['target', 'channels']

            # Check attrs
            assert json.loads(mmm_loaded.idata.attrs["log_transform_vars"]) == ['target', 'channels']

    def test_save_and_load_with_channels_only(self, mmm_config, sample_data):
        """Test save/load with log_transform_vars=['channels']."""
        X, y = sample_data

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test_mmm_channels.nc"

            # Create and fit model with log_transform_vars=['channels']
            mmm_original = MMM(**mmm_config, log_transform_vars=['channels'])
            mmm_original.build_model(X, y)

            # Sample from prior
            with mmm_original.model:
                mmm_original.prior_predictive_samples = mmm_original.sample_prior_predictive(
                    samples=10, random_seed=seed
                )

            # Save
            mmm_original.save(str(file_path))

            # Load
            mmm_loaded = MMM.load(str(file_path), check=False)

            # Check that parameter is preserved
            assert mmm_loaded.log_transform_vars == ['channels']

    def test_backward_compatibility_missing_attr(self, mmm_config):
        """Test that missing log_transform_vars attr defaults to None (empty list)."""
        # Create attrs without log_transform_vars (old model)
        attrs_old = {
            "model_config": json.dumps({}),
            "date_column": json.dumps("date"),
            "control_columns": json.dumps(None),
            "channel_columns": json.dumps(["channel_1", "channel_2"]),
            "adstock": json.dumps({"type": "geometric", "l_max": 4}),
            "saturation": json.dumps({"type": "logistic"}),
            "adstock_first": json.dumps(True),
            "yearly_seasonality": json.dumps(None),
            "validate_data": json.dumps(False),
            "sampler_config": json.dumps({}),
            # log_transform_vars is missing
        }

        kwargs = MMM.attrs_to_init_kwargs(attrs_old)

        # Should default to None
        assert kwargs["log_transform_vars"] is None


class TestApplyScalingSerialization:
    """Test that the apply_scaling serialization bug is fixed."""

    def test_apply_scaling_in_deserialization(self, mmm_config):
        """Test that apply_scaling is properly deserialized."""
        # Create attrs with apply_scaling
        attrs = {
            "model_config": json.dumps({}),
            "date_column": json.dumps("date"),
            "control_columns": json.dumps(None),
            "channel_columns": json.dumps(["channel_1", "channel_2"]),
            "adstock": json.dumps({"type": "geometric", "l_max": 4}),
            "saturation": json.dumps({"type": "logistic"}),
            "adstock_first": json.dumps(True),
            "yearly_seasonality": json.dumps(None),
            "validate_data": json.dumps(False),
            "sampler_config": json.dumps({}),
            "apply_scaling": json.dumps(False),
            "fit_intercept": json.dumps(True),
        }

        kwargs = MMM.attrs_to_init_kwargs(attrs)

        # apply_scaling should be in kwargs and set to False
        assert "apply_scaling" in kwargs
        assert kwargs["apply_scaling"] is False

    def test_save_and_load_preserves_apply_scaling(self, mmm_config, sample_data):
        """Test that save/load cycle preserves apply_scaling parameter."""
        X, y = sample_data

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test_mmm_scaling.nc"

            # Create model with apply_scaling=False
            mmm_original = MMM(**mmm_config, apply_scaling=False)
            mmm_original.build_model(X, y)

            # Sample from prior
            with mmm_original.model:
                mmm_original.prior_predictive_samples = mmm_original.sample_prior_predictive(
                    samples=10, random_seed=seed
                )

            # Save
            mmm_original.save(str(file_path))

            # Load
            mmm_loaded = MMM.load(str(file_path), check=False)

            # Check that parameter is preserved
            assert mmm_loaded.apply_scaling is False

            # Check attrs
            assert json.loads(mmm_loaded.idata.attrs["apply_scaling"]) is False


class TestIntegration:
    """Integration tests combining multiple parameters."""

    def test_all_parameters_together(self, mmm_config, sample_data):
        """Test model with apply_scaling=False, fit_intercept=False, log_transform_vars=['target', 'channels']."""
        X, y = sample_data

        mmm = MMM(
            **mmm_config,
            apply_scaling=False,
            fit_intercept=False,
            log_transform_vars=['target', 'channels'],
        )

        mmm.build_model(X, y)

        assert mmm.model is not None
        assert mmm.apply_scaling is False
        assert mmm.fit_intercept is False
        assert mmm.log_transform_vars == ['target', 'channels']

    def test_serialization_all_parameters(self, mmm_config, sample_data):
        """Test serialization of all parameters together."""
        X, y = sample_data

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test_mmm_all.nc"

            # Create model with specific settings
            mmm_original = MMM(
                **mmm_config,
                apply_scaling=False,
                fit_intercept=False,
                log_transform_vars=['target', 'channels'],
            )
            mmm_original.build_model(X, y)

            # Sample from prior
            with mmm_original.model:
                mmm_original.prior_predictive_samples = mmm_original.sample_prior_predictive(
                    samples=10, random_seed=seed
                )

            # Save
            mmm_original.save(str(file_path))

            # Load
            mmm_loaded = MMM.load(str(file_path), check=False)

            # Check all parameters are preserved
            assert mmm_loaded.apply_scaling is False
            assert mmm_loaded.fit_intercept is False
            assert mmm_loaded.log_transform_vars == ['target', 'channels']
