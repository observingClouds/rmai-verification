from typing import Dict, List
import logging
import xarray as xr
import numpy as np

from ..datastores.base import BaseDataStore

LOG = logging.getLogger(__name__)


def align_reference_times(datastores : Dict[str, BaseDataStore]) -> None:
    """Align reference times across multiple datastores.

    This function finds the intersection of reference times across all non-observational datastores
    and updates each datastore to only include those common reference times.

    Args:
        datastores (Dict[str, BaseDataStore]): Dictionary of datastores to align, where keys are store
            identifiers and values are BaseDataStore instances.

    Returns:
        None: The datastores are modified in-place.
    """
    ref_times = []
    for store in datastores.values():
        if not store.is_observation:
            ref_times.append(store.data["reference_time"].values)
    ref_times = list(set(ref_times[0]).intersection(*ref_times))
    for store in datastores.values():
        if not store.is_observation:
            store.select_reference_times(ref_times)

def align_valid_times(datastores : Dict[str, BaseDataStore]) -> None:
    """Aligns valid times across multiple datastores.

    This function ensures temporal alignment between forecasts and observations by taking
    the union of all forecast valid times and aligning all re-arranging all observation datastores
    to align with the union of the forecast valid_times.
    
    Parameters
    ----------
    datastores : Dict[str, BaseDataStore]
        Dictionary containing datastore objects with their respective names as keys.
        Each datastore should inherit from BaseDataStore.

    Returns
    -------
    None
        The function modifies the datastores in-place.
    
    Raises
    ------
    NotImplementedError
        When no forecasts are found in the datastores.
    """

    all_fcst_valid_times  = _get_all_fcst_valid_times(datastores)
    if all_fcst_valid_times is not None:
        for store in datastores.values():
            if store.is_observation:
                store.select_valid_times(all_fcst_valid_times)
    else:
        LOG.warning("No forecasts found in datastores, homogenizing valid_time on observations")
        raise NotImplementedError

def _get_all_fcst_valid_times(datastores : Dict[str, BaseDataStore]) -> xr.DataArray:
    """Get the union of all forecast valid times in de the datastore dictionary

    Helper function to merge all forecast valid times from the datastores. This Union 
    can be used to selected the valid dates from the observations datastores.

    Parameters
    ----------
    datastores : Dict[str, BaseDataStore]
        Dictionary containing datastore objects with their respective names as keys.
        Each datastore should inherit from BaseDataStore.

    Returns
    -------
    xr.DataArray
        DataArray with the merged valid times with dimensions
        [reference_time, lead_time]
    """

    valid_times = []
    for store in datastores.values():
        if not store.is_observation:
            valid_times.append(
                store.valid_times.rename("valid_times")
            )
    if len(valid_times) == 0:
        return None
    else:
        all_valid_times = xr.merge(valid_times)
        return all_valid_times["valid_time"]




