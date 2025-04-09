import xarray as xr
import cartopy.crs as ccrs

import logging
from typing import List

LOG = logging.getLogger(__name__)

DEFAULTS = dict()


class Interpolator():
    def __init__(self,output_ds,interp_kwargs = dict()):
        LOG.info("Initializing interpolator")
        self.output_ds = output_ds
        self.kwargs = interp_kwargs

    def execute(self,input_ds):
        assert "crs" in input_ds.attrs, "Input dataset must have a CRS attribute"
        crs = input_ds.attrs["crs"]
        xyz = crs.transform_points(
            x=self.output_ds["longitude"].values,
            y=self.output_ds["latitude"].values,
            src_crs=ccrs.PlateCarree() 
        )
        x = xr.DataArray(
            xyz[:,0],
            dims="code"
        ).assign_coords(code=self.output_ds["code"])
        
        y = xr.DataArray(
            xyz[:,1],
            dims="code"
        ).assign_coords(code=self.output_ds["code"])
        output_ds = input_ds.interp(
            x=x,
            y=y,
            **self.kwargs
        ).drop_vars(
            ["longitude","latitude"]
        ).assign_coords(
            longitude=self.output_ds["longitude"],
            latitude=self.output_ds["latitude"]
        ).load()
        return output_ds
