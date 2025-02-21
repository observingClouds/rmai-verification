from .data import *
import xarray as xr
SAVERS = {
    "netcdf": save_as_netcdf,
    "zarr" : save_as_zarr,
    "verif" : save_as_verif
}
import logging

LOG = logging.getLogger(__name__)

def save_dataset(dataset : xr.Dataset, type : str, path : str, **kwargs): 
    save = SAVERS[type]
    save(dataset, path, **kwargs)


