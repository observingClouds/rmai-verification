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

def load_data(fname, fc_time, _type, kwargs):
    assert _type in LOAD_REGISTRY, f"The datatype {_type} is not (yet) supported."
    return LOAD_REGISTRY[_type](fname, fc_time, kwargs)

def save_data(ds,fname, _type):
    assert _type in SAVE_REGISTRY, f"The datatype {_type} is not (yet) supported."
    SAVE_REGISTRY[_type](ds,fname)

def list2grid(ds, grid_info={"y":1069,"x":1069},dim="values"):
    assert len(grid_info) == 2, ("Only gridding to 2 dimensional grids is currently supported."
                                "'grid_info' should specify exactly 2 dimensions.")
    names=list(grid_info)
    n0=grid_info[names[0]]
    n1=grid_info[names[1]]
    assert dim in ds.dims, f"The dimension {dim} you want to grid is not in the dataset."
    assert len(ds.coords[dim]) == n0*n1, "Proposed grid size doesn't match."
    # For now we can just use vectors from 0:n0, 0:n1 since we don't have to
    # regrid yet. In the future this multiIndex should contain the 
    # actual lambert coordinates!!
    mindex = pd.MultiIndex.from_product(
        [range(0,n0), range(0,n1)],
        names=names
    )
    mindex_coords = xr.Coordinates.from_pandas_multiindex(
        mindex, dim)
    return ds.assign_coords(mindex_coords).unstack()

def xr_thinning(ds,thinning):
    assert 'x' in ds.coords and 'y' in ds.coords, "The dataset needs to have 'x' and 'y' coordinates."
    xs=ds.coords['x'].values
    ys=ds.coords['y'].values
    xs=xs[::thinning]
    ys=ys[::thinning]
    return ds.sel(x=xs,y=ys)

def valid_to_lead(ds, fc_time):
    assert 'valid_time' in ds.dims, 'valid_time is not a dimension'
    lts=[valid_time - fc_time for valid_time in ds.valid_time.values]
    ds=ds.assign_coords({'valid_time': lts})
    return ds.rename({'valid_time' : 'lead_time'})

COORDS = dict(
    longitude="longitudes",
    latitude="latitudes",
    valid_date="dates"
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

def zarr_to_xarray(file, fc_time="",
                   coords=COORDS,
                   drop=DROP,
                   thinning=1,
                   lead_time=False):
    ds = xr.open_zarr(file,consolidated=False)
    for key, value in coords.items():
        ds = ds.assign_coords({key : ds[value]})

    ds = ds.assign_coords(variable=ds.attrs["variables"])
    ds = ds.drop_vars(drop)
    ds = ds.isel(ensemble=0)

    grid_info=dict(y=ds.attrs["field_shape"][0],
                   x=ds.attrs["field_shape"][1])
    ds=list2grid(ds,grid_info=grid_info,dim="cell")
    ds=xr_thinning(ds,thinning)
    da=ds.data
    ds=da.to_dataset(dim='variable')
    ds=ds.assign_coords({'time':ds['valid_date'].values})
    ds=ds.rename({'time':'valid_time'}).drop_vars('valid_date')
    ds=ds.expand_dims({'fc_time': [fc_time]})
    if lead_time:
       ds = valid_to_lead(ds, fc_time) 
    return ds

def inference_to_xarray(file,fc_time,
                        lx=None, ly=None, 
                        thinning=1, lead_time = False):
    ds = xr.open_dataset(file)
    a = len(ds['values'])
    if lx is None and ly is None:
        lx = np.sqrt(a)
        ly = np.sqrt(a)
    elif lx is None:
        lx = a/ly
    elif ly is None:
        ly = a/lx
    assert lx == int(lx) and ly == int(ly), "'lx' and 'ly' must be integers."
    grid_info=dict(y = int(ly),
                   x = int(lx))
    ds=list2grid(ds,grid_info=grid_info)
    xs=[x*thinning for x in ds.x.values]
    ys=[y*thinning for y in ds.y.values]
    ds=ds.assign_coords({'x':xs,'y':ys})
    ds=ds.rename({'time':'valid_time'})
    ds=ds.expand_dims({'fc_time' : [fc_time]})
    if lead_time:
        ds = valid_to_lead(ds, fc_time)
    return ds 