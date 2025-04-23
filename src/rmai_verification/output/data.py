import xarray as xr
import logging

LOG = logging.getLogger(__name__)


def save_as_netcdf(ds : xr.Dataset | xr.DataArray, path : str, **kwargs) -> None:
    """Save an xarray dataset or array as a netCDF file.
    Args:
        ds (xarray.Dataset | xarray.DataArray): The xarray dataset or array to save
        path (str): Path where the netCDF file will be saved
        **kwargs: Additional keyword arguments passed to xarray's to_netcdf() method
    Returns:
        None
    Examples:
        >>> ds = xarray.Dataset(...)
        >>> save_as_netcdf(ds, "output.nc")
    """

    LOG.info(f"Saving data as netCDF-file at {path}")
    ds.to_netcdf(path, **kwargs)

def save_as_zarr(ds : xr.Dataset | xr.DataArray, path : str, **kwargs) -> None:
    """Save an xarray dataset or array as a zarr file.
    Args:
        ds (xarray.Dataset | xarray.DataArray): The xarray dataset or array to save
        path (str): Path where the zarr file will be saved
        **kwargs: Additional keyword arguments passed to xarray's to_zarr() method
    Returns:
        None
    Examples:
        >>> ds = xarray.Dataset(...)
        >>> save_as_zarr(ds, "output.zarr")
    """
    LOG.info(f"Saving data as zarr-file at {path}")
    ds.to_zarr(path, **kwargs)

def save_as_verif(ds : xr.Dataset | xr.DataArray, path : str) -> None:
    """Save an xarray dataset or array in the verif format.
    Args:
        ds (xarray.Dataset | xarray.DataArray): The xarray dataset or array to save
        path (str): Path where the verif file will be saved
    Returns:
        None
    Examples:
        >>> ds = xarray.Dataset(...)
        >>> save_as_verif(ds, "output.verif")
    """ 
    LOG.error("Saving to verif-format not yet supported")
    raise NotImplementedError