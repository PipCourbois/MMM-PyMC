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

class PerChannelDistributionFactory(VariableFactory):
    """
    Creates different distribution types for each channel.
    
    Parameters
    ----------
    priors_list : list of Prior
        One Prior object per channel. Each can have a different distribution type.
    """
    
    def __init__(self, priors_list: list):
        self.priors_list = priors_list
        self.dims = ('channel',)
    
    def create_variable(self, name: str) -> pt.TensorVariable:
        """Create PyMC variables by stacking individual channel distributions."""
        variables = []
        for i, prior in enumerate(self.priors_list):
            var = prior.create_variable(f'{name}_ch{i}')
            variables.append(var)
        return pt.stack(variables, axis=0)
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        return {
            "class": "PerChannelDistributionFactory",
            "priors": [prior.to_dict() for prior in self.priors_list]
        }
    
    def __repr__(self):
        dist_types = [prior.distribution for prior in self.priors_list]
        return f"PerChannelDistributionFactory({dist_types})"