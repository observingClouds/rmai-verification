import numpy as np
import xarray as xr

from data.anemoi_inference import DROP_VARS
import dask

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

def preprocess(dataset):

    # Add coordinates
    coords = {key: dataset[value].load() for key, value in COORDS.items()}
    for key in ("latitude","longitude"):
        coords[key] = coords[key].astype(np.float32)
    coords["variable"] = dataset.attrs["variables"]

    ds_coords = dataset.assign_coords(coords)

    # Drop unused variables and remove ensemble dimension
    ds_pruned = ds_coords.drop_vars(
        DROP
    ).isel(
        ensemble=0
    ).drop_sel(
        variable=DROP_VARS
    ).swap_dims(
        {"time":"valid_time"}
    ).rename_dims(
        {"cell":"values"}
    )

    ds_dataset = ds_pruned["data"]

    return ds_dataset


def load(filename,rename_dict=None,valid_time=None):
    ds = xr.open_zarr(filename,consolidated=False,chunks="auto")
    ds_preprocessed = preprocess(ds)
    if rename_dict is not None:
        new_names = [np.str_(rename_dict[var]) if var in rename_dict.keys() else var for var in ds_preprocessed["variable"].values]
        ds_renamed = ds_preprocessed.assign_coords(variable=("variable",new_names))
    else:
        ds_renamed = ds_preprocessed
    if valid_time is None:
        ds_final =  ds_renamed
    else:
        valid_time = [time for time in valid_time if time in ds_preprocessed["valid_time"]]
        ds_validtime = ds_renamed.sel(valid_time=valid_time)
        ds_final = ds_validtime
    ds_final.attrs['spatial_dimension'] = 'values'
    return ds_final

