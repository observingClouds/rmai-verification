import numpy as np
import xarray as xr
import itertools
from typing import List, Dict

def broadcast_nans(list_of_datasets):
# Step 1: Iterate over all pairs of datasets
    for dsA, dsB in itertools.combinations(list_of_datasets, 2):
        # Find the shared coordinates for all dimensions
        common_coords = {
            dim: sorted(set(dsA[dim].values) & set(dsB[dim].values))
            for dim in dsA.dims
        }
        
        # Iterate over all variables
        for var in dsA.data_vars:
            if var in dsB:  # Ensure both datasets have the variable
                # Select the data at common coordinates
                selA = dsA[var].sel(**common_coords)
                selB = dsB[var].sel(**common_coords)

                # Compute NaN mask for shared coordinates
                nan_mask = selA.isnull() | selB.isnull()

                # Apply NaN mask back to both datasets
                dsA[var].loc[common_coords] = dsA[var].sel(**common_coords).where(~nan_mask)
                dsB[var].loc[common_coords] = dsB[var].sel(**common_coords).where(~nan_mask)

def concat_dict_along_keys(dict_of_datasets : Dict[str,xr.Dataset], dim : str) -> xr.Dataset:
    coords = list(dict_of_datasets.keys())
    combined_ds = xr.concat(
        dict_of_datasets.values(), 
        dim=xr.Variable(dim, coords)
    )
    return combined_ds


def prep_config(full_config : Dict, sub_config_key : str) -> Dict[str,str]:
    # Create a copy without the config_key
    result = {k: v for k, v in full_config.items() if k != sub_config_key}
    
    # If the config_key exists and contains a dictionary
    if sub_config_key in full_config and isinstance(full_config[sub_config_key], dict):
        sub_config = full_config[sub_config_key]
        
        # Update the main level keys with values from user_config
        for key, value in sub_config.items():
            result[key] = value
            
    return result