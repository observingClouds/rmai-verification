import anemoi.datasets
import dask
import xarray as xr
import pandas as pd

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

def zarr_to_xarray(file):
    ds = xr.open_zarr(file,consolidated=False)
    variables = anemoi.datasets.open_dataset(file).variables
    field_shape = anemoi.datasets.open_dataset(file).field_shape #(y,x)
    for key, value in COORDS.items():
        ds = ds.assign_coords({key : ds[value]})
    ds = ds.assign_coords(variable=variables)
    ds = ds.drop_vars(DROP)
    ds = ds.isel(ensemble=0)
    mindex = pd.MultiIndex.from_product(
        [range(0,field_shape[0]),range(0,field_shape[1])],
        names=["y","x"]
    )
    mindex_coords = xr.Coordinates.from_pandas_multiindex(
        mindex, 'cell')
    ds = ds.assign_coords(mindex_coords).unstack()
    return(ds)
