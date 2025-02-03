import pandas as pd
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import logging
from .grids import GRIDS, PROJECTIONS

LOG = logging.getLogger(__name__)

def create_cartopy_crs(projection,projection_kws):
    assert projection in PROJECTIONS, f"Projection {projection} not supported (yet)."
    
    # - Get the cartopy projection (crs)
    projection = PROJECTIONS[projection]
    kwargs = projection_kws.copy()

    # - Move globe keywords to different dictionary
    globe = kwargs.pop("globe", None)
    if globe:
        globe = ccrs.Globe(**globe)
    
    crs = projection(globe=globe, **kwargs)
    return crs


def create_multiindex(ds, x="x", y="y", dim_to_multiindex="index", **kwargs):
    nx = kwargs.get("nx")
    ny = kwargs.get("ny")
    lon_ll, lat_ll = kwargs.get("lower_left")
    lon_ur, lat_ur = kwargs.get("upper_right")
    crs = ds.attrs["crs"]

    assert dim_to_multiindex in ds.dims, f"Dimension {dim_to_multiindex} not found in the dataset"
    assert ds.sizes[dim_to_multiindex] == nx * ny, f"Proposed grid dimensions ({ny}, {nx}) do not match the length of {dim_to_multiindex}: {ds.sizes[dim_to_multiindex]}"

    x_ll, y_ll = crs.transform_point(
        x=lon_ll,
        y=lat_ll,
        src_crs=ccrs.PlateCarree()
    )
    x_ur, y_ur = crs.transform_point(
        x=lon_ur,
        y=lat_ur,
        src_crs=ccrs.PlateCarree()
    )
    
    if "thinning" in ds.attrs:
        thinning_factor = ds.attrs["thinning"]
    else:
        thinning_factor = 1
    x_values = np.linspace(x_ll,x_ur,nx,endpoint=False)[::thinning_factor]
    y_values = np.linspace(y_ll,y_ur,ny,endpoint=False)[::thinning_factor]
    
    mindex = pd.MultiIndex.from_product(
        [y_values, x_values],
        names=[y,x]
    )  
    return mindex

def add_xy(ds,grid):
    # If native domain is a string, get the specifications from the pre-defined mappings
    if isinstance(grid, str):
        assert grid in GRIDS, f"Grid {grid} not supported, please provide a dictionary with the specifications"
        grid = GRIDS[grid]

    # - Get the cartopy projection (crs)
    projection = grid["projection"]

    # - Get the crs-keywords
    projection_kwargs = grid["projection_kws"].copy()

    # - Get the grid-keywords
    grid_kwargs = grid["grid_kws"].copy()
    
    # Create Coordinate Reference Systems (crs)
    crs = create_cartopy_crs(
        projection,
        projection_kwargs
    )
    ds_new = ds.copy()
    ds_new.attrs["crs"] = crs
 
    assert "lower_left" in grid_kwargs and "upper_right" in grid_kwargs, f"Longitudes and latitudes of lower left and/or upper right conrer are missing"    
    LOG.info("Calculating x and y values from the extent")
    multi_index = create_multiindex(ds_new,**grid_kwargs) 
    multi_coordinates = xr.Coordinates.from_pandas_multiindex(
        multi_index, "index")
    ds_xy = ds_new.assign_coords(multi_coordinates)
    return ds_xy


   # if "latitude" in ds and "longitude" in ds:
    #     LOG.info("Calculating x and y values from longitude and latitude")
    #     xy = crs.transform_points(
    #         src_crs=ccrs.PlateCarree(),
    #         y=ds["latitude"].values,
    #         x=ds["longitude"].values,
    #     )
    #     x = xy[:,0]
    #     y = xy[:,1]
    #     ds_new = ds.assign_coords(
    #         x=("index", x, 
    #             dict(
    #                 units="meter",
    #                 description="x-coordinate",
    #             )
    #         ),
    #         y=("index", y,
    #             dict(
    #                 units="meter",
    #                 description="y-coordinate",
    #             )
    #         )
    #     )
    #     return ds_new
