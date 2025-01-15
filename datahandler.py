import xarray as xr
import pandas as pd
import numpy as np
from projections import map_grid
from datetime import datetime

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


## UTILS ##
def get_loader(type):
    assert type in LOAD_REGISTRY, f"The datatype {type} is not (yet) supported."
    return LOAD_REGISTRY[type]

def get_saver(type):
    assert type in SAVE_REGISTRY, f"The datatype {type} is not (yet) supported."
    return SAVE_REGISTRY[type]

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

def load_model(**kwargs):
    # Get the model specific loader
    loader = get_loader(kwargs.pop("type"))

    # Get the path-format
    path_fmt = kwargs.pop("path")

    # Get all the dates information
    dates = kwargs.pop("dates")
    start = np.datetime64(dates["start"])
    end = np.datetime64(dates["end"])
    value = dates["frequency"][:-1]
    unit = dates["frequency"][-1]
    frequency= np.timedelta64(value,unit)

    # Start loop over dates
    date = start
    model = []
    while date <= end:
        kwargs["reference_time"] = date.astype('datetime64[ns]')
        print(f"    - {date}")
        # Some path-formatting (Can probably be done better)
        date_dt = date.astype(datetime)
        path = path_fmt.format(
            yyyy=date_dt.strftime("%Y"),
            yy=date_dt.strftime("%y"),
            mm=date_dt.strftime("%m"),
            dd=date_dt.strftime("%d"),
            HH=date_dt.strftime("%H"),
            MM=date_dt.strftime("%M"),
            SS=date_dt.strftime("%S"),
        )
        model.append(
            loader(
                filename=path,
                **kwargs
            )
        )
        date += frequency
    model = xr.concat(model,dim="reference_time")
    return(model)


## LOADERS ##
def anemoi_datasets(
        filename,
        valid_times=None,
        coords=COORDS,
        drop=DROP,
        thinning=1,
        grid_mapping=None):
    


    # Open the file
    ds = xr.open_zarr(filename,consolidated=False)

    # Assign the coordinates
    for key, value in coords.items():
        ds = ds.assign_coords({key : ds[value]})
    
    # Add the variables as a coordinate
    ds = ds.assign_coords(
        variable=ds.attrs["variables"]
    )
    
    # Drop unused variables
    ds = ds.drop_vars(drop)

    # Remove the ensemble dimension
    ds = ds.isel(ensemble=0)
 
    #FIXME: For now we have to convert the longitudes and latitudes to np.float32
    # to match those of anemoi-inference netCDFs
    # Convert the coordinates to np.float32 
    ds["longitude"] = ds["longitude"].astype(np.float32)
    ds["latitude"] = ds["latitude"].astype(np.float32)

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

    # Map grid
    if grid_mapping:
        ds = map_grid(ds, grid_mapping)


    # Transform the a dataset with 1 dataarray per variable
    ds = ds.data.to_dataset(dim="variable")
    
    # Make valid_time the main time dimension
    #TODO: Fix non-nanosecond precision warning
    ds = ds.swap_dims({"time":"valid_time"})

    # Select only needed valid_times
    if valid_times is not None:
        ds = ds.sel(valid_time=valid_times)

    return ds
  
def anemoi_inference(
        filename,
        reference_time=None,
        reshape = False, 
        nx=None, 
        ny=None, 
        dataset_attrs=None,
        grid_mapping=None,
        lead_time=True):
    
    # We need to set the chunksize to tell xarray to use Dask
    chunks = {
        "time" : -1,
        "values" : -1
    }

    # Open the file
    ds = xr.open_dataset(filename,chunks=chunks)
    
    # set latitude and longitude as coordinates
    ds = ds.assign_coords(latitude=ds.latitude).assign_coords(longitude=ds.longitude)
#    ds = ds[vars]
    # Add additional attributes
    if dataset_attrs:
        for key, value in dataset_attrs.items():
            ds.attrs[key] = value

    if reshape:    
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
        assert int(nx) == nx and int(ny) == ny, "'nx' and 'ny' must be integers."

        # Convert list to grid
        ds=list_to_grid(
            ds=ds,
            nx=nx,
            ny=ny,
            dim="values"
        )

        # Map grid
        if grid_mapping:
            ds = map_grid(ds, grid_mapping)

    # Add a reference time
    if not reference_time:
        print("Warning: No reference_time provided, using first valid time")
        reference_time = ds["time"].data[0]
    elif not isinstance(reference_time,np.datetime64):
        reference_time = np.datetime64(reference_time)
    
    ds = ds.expand_dims(reference_time=[reference_time])
    ds.reference_time.attrs["standard_name"] = "forecast_reference_time"

    # Rename the time coordinate to valid_time
    ds = ds.rename_vars({"time": "valid_time"})

    # Add leadtimes and set as dimension
    if lead_time:
        ds = valid_to_lead_time(ds)
        
    return(ds)

## SAVERS ##
def save_to_netcdf(ds, filepath, **kwargs):
    ds.to_netcdf(filepath, **kwargs)

def save_to_zarr(ds, filepath, **kwargs):
    ds.to_zarr(filepath, **kwargs)


## REGISTRIES ##
LOAD_REGISTRY = {
    "anemoi-inference" : anemoi_inference,
    "anemoi-datasets" : anemoi_datasets
}

SAVE_REGISTRY = {
    "netcdf" : save_to_netcdf,
    "zarr" : save_to_zarr
}




    

