import anemoi.datasets
import dask
import xarray as xr
import pandas as pd
import numpy as np

LOAD_REGISTRY=dict(
    zarr = lambda fname, fc_time, kwargs : zarr_to_xarray(fname, fc_time=fc_time, lead_time=True, **kwargs),
    inference = lambda fname, fc_time, kwargs : inference_to_xarray(fname, fc_time, lead_time=True, **kwargs)
    )

SAVE_REGISTRY=dict(
    zarr = lambda ds, fname : ds.to_zarr(fname),
    netcdf = lambda ds, fname : ds.to_netcdf(fname)
    )

COORDS = dict(
    longitude="longitudes",
    latitude="latitudes",
    valid_time="dates"
)
DROP = [
    "count",
    "has_nans",
    "maximum",
    "minimum",
    "mean",
    "squares",
    "sums",
    "stdev",
    "longitudes",
    "latitudes",
    "dates"
]

def load_data(fname, fc_time, _type, kwargs):
    assert _type in LOAD_REGISTRY, f"The datatype {_type} is not (yet) supported."
    return LOAD_REGISTRY[_type](fname, fc_time, kwargs)

def save_data(ds,fname, _type):
    assert _type in SAVE_REGISTRY, f"The datatype {_type} is not (yet) supported."
    SAVE_REGISTRY[_type](ds,fname)

def list_to_grid(ds, nx, ny, dim="values"):
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

def valid_to_lead_time(ds):
    assert "valid_time" in ds.coords and "reference_time" in ds.coords, "Need a valid_time and reference_time coordinate"
    ds = ds.assign_coords(
        lead_time=("time", ds.valid_time.data - ds.reference_time.data)
    )
    ds = ds.swap_dims({"time":"lead_time"})

    # Make valid_time a 2 dimensional array for easier merging
    ds = ds.assign_coords(
        valid_time=(
            ["reference_time","lead_time"],
            ds.valid_time.data[np.newaxis,:]
        )
    )
    return ds

def anemoi_datasets(
        filename,
        coords=COORDS,
        drop=DROP,
        thinning=1):

    # Open the file
    ds = xr.open_zarr(filename,consolidated=False)

    # Assign the coordinates
    for key, value in coords.items():
        ds = ds.assign_coords({key : ds[value]})
    
    # Add the variables as a coordinate
    ds = ds.assign_coords(variable=ds.attrs["variables"])
    
    # Drop unused variables
    ds = ds.drop_vars(drop)

    # Remove the ensemble dimension
    ds = ds.isel(ensemble=0)

    # Convert list to grid
    ds = list_to_grid(
        ds=ds,
        nx=ds.attrs["field_shape"][1],
        ny=ds.attrs["field_shape"][0],
        dim="cell"
    )

    # Thinning
    if thinning > 1:
        ds = ds.isel(x=slice(0,None,thinning),y=slice(0,None,thinning))
        ds.attrs["thinnig"] = thinning

    # Transform the a dataset with 1 dataarray per variable
    ds = ds.data.to_dataset(dim="variable")
    
    # Make valid_time the main time dimension
    #TODO: Fix non-nanosecond precision warning
    ds = ds.swap_dims({"time":"valid_time"})
    return ds
  

    
def anemoi_inference(
        filename, 
        reference_time=None, 
        nx=None, 
        ny=None, 
        dataset_attrs=None,
        lead_time=True):
    
    # Open the file
    ds = xr.open_dataset(filename)
    
    # Add additional attributes
    if dataset_attrs:
        for key, value in dataset_attrs.items():
            ds.attrs[key] = value
    
    # Get the total number of gridpoints
    ngridpoints = ds.sizes["values"]

    # Try to figure out the grid-dimensions
    if not nx and not ny:
        print("Warning: No nx and ny provided, trying to build a square grid")
        nx = ny =  np.sqrt(ngridpoints)
    elif not nx:
        nx = ngridpoints/ny
    elif not ny:
        ny = ngridpoints/nx

    # Check if nx and ny are integers
    assert nx.is_integer() and ny.is_integer(), "'nx' and 'ny' must be integers."

    # Convert list to grid
    ds=list_to_grid(
        ds=ds,
        nx=nx,
        ny=ny,
        dim="values"
    )

    # Add a reference time
    if not reference_time:
        print("Warning: No reference_time provided, calculating it from the time coordinate")
        reference_time = ds["time"].data[0] - np.diff(ds["time"].data)[0]
    elif not np.issubdtype(reference_time, np.datetime64):
        reference_time = np.datetime64(reference_time)
    
    ds = ds.expand_dims(reference_time=[reference_time])
    ds.reference_time.attrs["standard_name"] = "forecast_reference_time"

    # Rename the time coordinate to valid_time
    ds = ds.rename_vars({"time": "valid_time"})

    # Add leadtimes and set as dimension
    if lead_time:
        ds = valid_to_lead_time(ds)


        
    return(ds)
    

    






    

