import xarray as xr
import logging

LOG = logging.getLogger(__name__)


def save_as_netcdf(ds : xr.Dataset | xr.DataArray, path : str, **kwargs):
    LOG.info(f"Saving data as netCDF-file at {path}")
    ds.to_netcdf(path, **kwargs)

def save_as_zarr(ds : xr.Dataset | xr.DataArray, path : str, **kwargs):
    LOG.info(f"Saving data as zarr-file at {path}")
    ds.to_zarr(path, **kwargs)

def save_as_verif(ds : xr.Dataset | xr.DataArray, path : str):
    LOG.error("Saving to verif-format not yet supported")
    raise NotImplementedError