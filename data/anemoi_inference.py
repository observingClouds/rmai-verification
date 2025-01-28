import numpy as np
import xarray as xr
from datetime import datetime
import os
# import tqdm

DROP_VARS = [
    "cos_julian_day",
    "cos_latitude",
    "cos_local_time",
    "cos_longitude",
    "insolation",
    "sin_julian_day",
    "sin_latitude",
    "sin_local_time",
    "sin_longitude",
]

def preprocess(dataset,reference_time=None):
    keep_vars = [var for var in dataset if var not in DROP_VARS]
    ds_pruned = dataset[keep_vars]
    ds_latlon = ds_pruned.assign_coords(
        latitude=ds_pruned.latitude,
        longitude=ds_pruned.longitude
    )

    if not reference_time:
        reference_time = dataset["time"].data[0]
    ds_reftime = ds_latlon.expand_dims(
        reference_time=[reference_time]
    )
    ds_reftime[
        "reference_time"
    ].attrs["standard_name"] = "forecast_reference_time"
    ds_validtime = ds_reftime.rename_vars(
        {"time" : "valid_time"}
    )
    ds_leadtime =  ds_validtime.assign_coords(
        lead_time=(
            "time",
            ds_validtime["valid_time"].data -
                ds_validtime["reference_time"].data
        )
    )
    ds_swapped = ds_leadtime.swap_dims({"time":"lead_time"})
    ds_final = ds_swapped.assign_coords(
        valid_time=(
            ["reference_time", "lead_time"],
            ds_swapped["valid_time"].data[np.newaxis,:]
        )
    )
    return ds_final

def load(dates, path_fmt, **kwargs):
    # List all files
    filenames = []
    for date in dates:
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
        if not os.path.isfile(path):
            print(f"Warning no file for date {date_dt.strftime('%Y%m%d %H:%M')}, skipping file")
        else:
            filenames.append(path)
    #prep metadata
    engine = kwargs.get("engine","h5netcdf")
    combine = kwargs.get("combine","nested")
    parallel = kwargs.get("parallel",True)
    concat_dim = kwargs.get("concat_dim","reference_time")
    data_vars = kwargs.get("data_vars","minimal")
    chunks = kwargs.get(
        "chunks",
        {"reference_time":5,"time":11, "values": 1142761}
    )
    # go get them data
    ds = xr.open_mfdataset(
        filenames,
        preprocess=preprocess,
        engine=engine,
        combine=combine,
        concat_dim=concat_dim,
        data_vars=data_vars,
        chunks=chunks,
        parallel=parallel
    )
    ds.attrs['spatial_dimension'] = 'values'
    return ds

    
