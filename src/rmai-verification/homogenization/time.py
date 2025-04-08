from typing import Dict, List
import logging
import xarray as xr
import numpy as np

from datastores.base import BaseDataStore

LOG = logging.getLogger(__name__)


def reference_times(datastores : Dict[str, BaseDataStore]):
    ref_times = []
    for store in datastores.values():
        if not store.is_observation:
            ref_times.append(store.data["reference_time"].values)
    ref_times = list(set(ref_times[0]).intersection(*ref_times))
    for store in datastores.values():
        if not store.is_observation:
            store.select_reference_times(ref_times)

def _get_all_fcst_valid_times(datastores : Dict[str, BaseDataStore]) -> xr.DataArray:
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

def valid_times(datastores : Dict[str, BaseDataStore], reference_datastore : str):
    all_fcst_valid_times  = _get_all_fcst_valid_times(datastores)
    if all_fcst_valid_times is not None:
        for name, store in datastores.items():
            if store.is_observation:
                if name == reference_datastore:
                    unique_fcst_valid_times = np.unique(
                        all_fcst_valid_times.values.ravel()
                    )
                    store.select_valid_times(unique_fcst_valid_times)
                else:
                    store.select_valid_times(all_fcst_valid_times)
    else:
        LOG.warning("No forecasts found in datastores, homogenizing valid_time on observations")
        raise NotImplementedError




