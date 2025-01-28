import cartopy.crs as ccrs
import pandas as pd
import numpy as np
import xarray as xr
import yaml

with open("grids.yaml") as stream:
    try:
        GRIDS = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

EXTENTS={name: GRIDS[name].pop('extent') for name in GRIDS}
MAPPINGS=GRIDS

PROJECTIONS = {
    "lcc" : ccrs.LambertConformal,  
}

def list_to_grid(ds, extent, dim="values"):
    # If extent is a string, get the specifications from the pre-defined extents
    if isinstance(extent, str):
        assert extent in EXTENTS, f"Extent {extent} not supported, please provide a dictionary with the specifications"
        extent = dict(EXTENTS[extent])
    nx=extent['nx']
    ny=extent['ny']
    assert dim in ds.dims, f"The dimension {dim} you want to grid is not in the dataset."
    assert ds.sizes[dim] == nx * ny, f"Proposed grid dimensions ({ny}, {nx}) do not match the length of {dim}: {ds.dims[dim]}"
    
    if "thinning" in ds.attrs:
        thinning_factor = ds.attrs["thinning"]
    else:
        thinning_factor = 1
    mindex = pd.MultiIndex.from_product(
        [np.arange(0,ny)*thinning_factor, np.arange(0,nx)*thinning_factor],
        names=["y","x"]
    )
    
    mindex_coords = xr.Coordinates.from_pandas_multiindex(
        mindex, dim)
    return ds.assign_coords(mindex_coords).unstack()

def get_cartopy_crs(grid_mapping):
        # If native domain is a string, get the specifications from the pre-defined mappings
    if isinstance(grid_mapping, str):
        assert grid_mapping in MAPPINGS, f"Grid mapping {grid_mapping} not supported, please provide a dictionary with the specifications"
        grid_mapping = dict(MAPPINGS[grid_mapping])
    # Build the cartopy CRS    
    assert grid_mapping["projection"] in PROJECTIONS, f"Projection {grid_mapping['projection']} not supported (yet)."
    projection = PROJECTIONS[grid_mapping["projection"]]
    kwargs = dict(grid_mapping["projection_kws"])
    globe = kwargs.pop("globe", None)
    if globe:
        globe = ccrs.Globe(**globe)
    crs = projection(globe=globe, **kwargs)
    return crs

def map_grid(dataset, grid_mapping):
    # Cartopy CRS
    crs = get_cartopy_crs(grid_mapping)

    # Transform the lat-lons to native format
    assert "longitude" in dataset.coords and "latitude" in dataset.coords, "dataset must contain longitudes and latitudes, building domain from domain edges not implemented (yet)."
    longitude = dataset.longitude.values
    latitude = dataset.latitude.values
    xy = crs.transform_points(
        src_crs=ccrs.PlateCarree(),
        y=latitude,
        x=longitude
    )
    x = xy[0,:,0]
    y = xy[:,0,1]

    # Assign as new coordinates
    #TODO: not all coordinates have meters as unit!
    dataset = dataset.assign_coords(
        x=(
            "x",
            x,
            dict(
                units="meter",
                description="x-coordinate",
            )
        ),
        y=(
            "y",
            y,
            dict(
                units="meter",
                description="y-coordinate",
            )
        )
    )

    # Add CRS information to the attributes, usefull for regridding an plotting
    dataset.attrs["grid_mapping"] = crs
    return dataset

def to_grid(ds, grid):
    if isinstance(grid, str):
        grid = { 'extent' : grid,
                 'grid_mapping' : grid }
    ds = list_to_grid(ds, grid['extent'])
    return map_grid(ds, grid['grid_mapping'])
        

def interpolate_to_latlon(ds, lon, lat):
    crs=ds.grid_mapping
    x, y = crs.transform_point(lon, lat,
        src_crs=ccrs.PlateCarree() )
    return ds.interp(x = x, y = y)