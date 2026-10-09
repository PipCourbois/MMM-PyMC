


import pandas as pd
from typing import Dict, Any


def load_and_standardize_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load data from a config-specified source and standardize the target column name to 'y'.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing 'data' key with:
        - 'source': URL or path to the CSV file
        - 'columns': Dictionary with 'date_column' and 'target_column' keys
    
    Returns
    -------
    pd.DataFrame
        Loaded dataframe with target column standardized to 'y'
    
    Examples
    --------
    >>> config = {
    ...     'data': {
    ...         'source': 'data.csv',
    ...         'columns': {
    ...             'date_column': 'date',
    ...             'target_column': 'sales'
    ...         }
    ...     }
    ... }
    >>> data = load_and_standardize_data(config)
    """
    # Load the data from config-specified source
    data_config = config['data']
    url = data_config['source']
    data = pd.read_csv(url, parse_dates=[data_config['columns']['date_column']])
    
    # GUARDRAIL: Standardize target column name to 'y'
    target_col = data_config['columns']['target_column']
    if target_col != 'y':
        data = data.rename(columns={target_col: 'y'})
        print(f"✓ Renamed target column '{target_col}' → 'y'")
    else:
        print(f"✓ Target column is already named 'y'")
    
    print(f"Data shape: {data.shape}")
    display(data.head())
    
    return data