import xarray as xr
import cartopy.crs as ccrs

import logging
from typing import List, Dict

LOG = logging.getLogger(__name__)

DEFAULTS = dict()


class XrInterpolator():
    """
    Interpolates a GridDataStore to the points of a PointDataStore using xarray's built-in interpolation methods.
    This means that the input dataset must have a coordinate system defined in the attributes and that
    the GridDataStore must have a grid mapping to be able to unstack the GridDataStore.
    """
    def __init__(self, output_ds: xr.Dataset | xr.DataArray ,interp_kwargs : Dict[str,str] = dict()) -> None:
        """Initialize the interpolator object.
        Args:
          output_ds (xarray.Dataset | xarray.DataArray): The target dataset/array to interpolate to.
            This defines the target grid for interpolation.
          interp_kwargs (Dict[str,str], optional): Dictionary of keyword arguments to pass to the 
            interpolation function. Defaults to empty dict.
        Returns:
          None
        Notes:
          The interpolator is initialized with a target dataset/array that defines the desired
          output grid. Additional interpolation parameters can be provided via interp_kwargs.
        """
        
        LOG.info("Initializing interpolator")
        self.output_ds = output_ds
        self.kwargs = interp_kwargs

    def execute(self, input_ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
        """Interpolates input dataset to output coordinates using the given CRS.
        
        This method performs spatial interpolation of the input dataset onto a new coordinate
        system defined by the output dataset's longitude/latitude coordinates.
        
        Args:
          input_ds (xr.Dataset | xr.DataArray): Input dataset/array to interpolate. 
            Must have a CRS attribute.
        
        Returns:
          xr.Dataset | xr.DataArray: Interpolated dataset/array with coordinates matching 
            the output dataset's longitude/latitude.
        
        Raises:
          AssertionError: If input dataset does not have a CRS attribute.
        Notes:
          - The input dataset must have a CRS (Coordinate Reference System) attribute
          - Interpolation is performed using the coordinates of the target transformed to the input CRS
          - Original longitude/latitude coordinates are dropped and replaced with output coordinates
          - Additional interpolation parameters can be passed via self.kwargs
        """
        
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
