import xarray as xr
import cartopy.crs as ccrs

import logging
from typing import List, Dict

import numpy as np
import xarray as xr
import dask.array as dda

from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

LOG = logging.getLogger(__name__)

DEFAULTS = dict()


def interpolate_block(block : xr.DataArray, triangulation: Delaunay, target_points: np.ndarray):
    data = block.values #data # shape = (.., npoints)
    original_shape = data.shape[:-1]
    data_flat = data.reshape(-1, data.shape[-1]) # shape = (ndim1 * ndim2 * ... , npoints)

    _interpolated = []
    for row in data_flat:
        interpolator = LinearNDInterpolator(triangulation, row)
        _interpolated.append(interpolator(target_points))

    interpolated_flat = np.stack(_interpolated)

    interpolated = interpolated_flat.reshape(*original_shape, target_points.shape[0])

    new_dims = block.dims[:-1] + ("point_index", )
    new_coords = {dim: block.coords[dim] for dim in block.dims[:-1]}

    return xr.DataArray(interpolated, dims=new_dims, coords=new_coords)

def interpolate_dataset(ds: xr.Dataset, target_lons: np.ndarray, target_lats: np.ndarray) -> xr.Dataset:
    if 'latitude' not in ds.coords or 'longitude' not in ds.coords:
        raise KeyError("Dataset must have 'latitude' and 'longitude' coordinates.")
    elif len(ds.coords["latitude"].shape) != 1 or len(ds.coords["longitude"].shape) != 1: 
        raise ValueError("Dataset must have 'latitude' and 'longitude' as 1D coordinates.")
  
  
    source_points = np.column_stack((ds['latitude'].values, ds['longitude'].values))
    target_points = np.column_stack((target_lats, target_lons))
  
    # point_dim_name = target_lons.dims[0]
    # point_dim = target_lons[point_dim_name].values

    n_target = len(target_points)

    LOG.info("Calculating Delaunay-triangulation")
    triangulation = Delaunay(source_points)
   
    out_vars = {}
    for var in ds.data_vars:
        LOG.info(f"Interpolating variable '{var}'")
        da = ds[var]

        if da.dims[-1] != "grid_index":
            LOG.warning(f"Skipping variable '{var}' - doesn't end with spatial dimension grid_index")
            continue
      
        leading_dims = da.dims[:-1]

        # Build template
        # Get chunking info for leading dims
        tmp_shape = tuple(
            da.sizes[d] for d in leading_dims
        ) + (n_target,)
    
        if isinstance(da.data, dda.Array):  # If it's a Dask array
            dim_to_chunks = dict(zip(da.dims, da.chunks))
        else:
            dim_to_chunks = {dim: (da.sizes[dim],) for dim in da.dims}
    
        tmp_chunks = tuple(
            dim_to_chunks[dim] if dim in dim_to_chunks else (da.sizes[dim],)
            for dim in leading_dims
        ) + ((n_target,), )
 
        # Create a dask array template matching the chunking pattern
        tmp = dda.empty(shape=tmp_shape, chunks=tmp_chunks, dtype=da.dtype)
        tmp = xr.DataArray(
            tmp,
            dims=leading_dims + ("point_index", ),
            coords={d: da.coords[d].load() for d in leading_dims}
        )
        #     latitude=("point_index",target_lats),
        #     longitude=("point_index",target_lons)
        #)

        da_interp = da.map_blocks(
            lambda block: interpolate_block(
                block,
                triangulation,
                target_points,
            ),
            template=tmp
        ).assign_coords(point_index=np.arange(n_target))

        out_vars[var] = da_interp
  
    return xr.Dataset(out_vars).compute()
        

class DelaunayInterpolator():
    """
    Interpolates a GridDataStore to the points of a PointDataStore using Delauny-triangulation function.
    This means that the both input and output datasets must have 'latitude' and 'longitude' coordinates  
    """
    def __init__(self, output_ds: xr.Dataset | xr.DataArray, interp_kwargs: Dict[str,str] = dict()) -> None:
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
        LOG.info("Initializing Delaunay interpolater")
        self.output_ds = output_ds
        self.kwargs = interp_kwargs
        method = self.kwargs.get("method", "linear")
        if  method != "linear":
            LOG.error("Delaunay interpolation only supports linear interpolation")
            raise KeyError(f"method: {method}")


    def execute(self, input_ds: xr.Dataset) -> xr.Dataset:
        target_lons = self.output_ds["longitude"].values
        target_lats = self.output_ds["latitude"].values
        ds_interpolated = interpolate_dataset(
            input_ds,
            target_lons,
            target_lats,
        )
        ds_code = ds_interpolated.assign_coords(
            code=("point_index", self.output_ds.code.values)
        ).swap_dims(
            point_index="code"
        ).drop_vars("point_index")

        aux_coords = {key: value for key, value in self.output_ds.coords.items() 
        if (key not in ds_code.coords) & (key != "valid_time")}
        
        ds_final = ds_code.assign_coords(aux_coords)
        assert (self.output_ds.latitude.values == ds_final.latitude.sel(code=self.output_ds.code)).all()
        return ds_final

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
        
        LOG.info("Initializing Xarray-based interpolator")
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
        ).compute()
        return output_ds
