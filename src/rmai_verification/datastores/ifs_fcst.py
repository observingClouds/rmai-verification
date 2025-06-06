import xarray as xr
import cfgrib
import dask
import numpy as np
import logging

from numpy.typing import NDArray
from typing import List, Tuple, Dict, Union

from .base import GridDataStore, FcstDataStore

LOG = logging.getLogger(__name__)

DROP_VARS = ["surface"]
             
class IfsForecast(GridDataStore, FcstDataStore):
    def __init__(self,
                 files: List[str],
                 variables: Union[List[str], Tuple[str], set] = None,
                 mf_kwargs: Dict[str,str] = dict()
                ):
        LOG.info("Initializing IfsForecast datastore")

        self._files = files
        self._mf_kwargs = mf_kwargs
        self._stacked: bool = True

        data = xr.open_mfdataset(
            self._files,
            combine="nested",
            concat_dim="time",
            chunks={
                "time" : 1,
                "step": -1,
                "values": -1
            },
            **self._mf_kwargs
        )
        
        data.coords["longitude"] = (data.coords["longitude"] + 180.) % 360. -180.

        self._data = data.rename_dims(
            time="reference_time",
            step="lead_time",
            values="grid_index"
        ).rename_vars(
            time="reference_time",
            step="lead_time"
        ).drop_vars(
            ["number","surface"]
        )

        if variables:
            self.select_variables(variables)
        
        LOG.info("Finished initializing IfsForecast datastore")
        
    def unstack(self):
        LOG.warning("Unstacking of IfsForecast datastores not supported yet")
        pass





def _preprocess(ds: xr.Dataset) -> xr.Dataset:
    ds_pruned = ds.rename_dims(
        time="reference_time",
        step="lead_time",
        values="grid_index"
    ).rename_vars(
        time="reference_time",
        step="lead_time"
    ).drop_vars(
        ["number","surface"]
    )
    return ds_pruned