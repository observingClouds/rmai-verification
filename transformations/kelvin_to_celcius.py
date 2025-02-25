import numpy as np
import xarray as xr
import logging
from typing import List

LOG = logging.getLogger(__name__)

CORRECTION = 273.15

class KelvinToCelcius():
    def __init__(self, fields: str | List[str] = "2t" , inverse: bool = False):
        if isinstance(fields, str):
            self.fields = [fields]
        else:
            self.fields = fields
        self.correction = np.sign(inverse-0.5) * CORRECTION

    def execute(self,input_ds : xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
        if isinstance(input_ds, xr.DataArray):
            LOG.debug("Transforming an xr.DataArray")
            crs = input_ds.attrs["crs"]
            input_ds = xr.where(input_ds["variable"].isin(self.fields), input_ds + self.correction, input_ds)          
            input_ds.attrs["crs"] = crs
        else:
            LOG.debug("Transforming an xr.Dataset")
            for field in self.fields:
                input_ds[field] = input_ds[field] + self.correction
        
        return input_ds
