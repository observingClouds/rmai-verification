import xarray as xr
import numpy as np
import logging

from .base import GridDatastore
from grids.grid_mapping import add_xy
LOG = logging.getLogger(__name__)

DROP_VARS = [
    "latitude",
    "longitude",
    "time",
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

MF_KWARGS = {
    "engine":"h5netcdf",
    "combine":"by_coords",
    "parallel":True,
    "concat_dim": None,
    "data_vars":"minimal"
}


class AnemoiInference(GridDatastore):

    def __init__(self,config):
        LOG.info("Initializing AnemoiInferenceDataStore")
        # Add the files to the class
        self._files = config["files"] #FIXME
        self._vars = config.get("variables", None)
        self._mapping = config.get("mapping", None)
        self._unstacked = False
        self._observation = False

        # Add the xr.open_mfdataset kwargs
        self._mf_kwargs = dict()
        for key, value in MF_KWARGS.items():
            self._mf_kwargs[key]=config.get(key,value)
    
        # open a single dataset to infer some properties
        ds = xr.open_dataset(self._files[0])

        # Set the dimension names
        self._dim_names = ("reference_time", "valid_time", "grid_index")
    
        # Set the dimensions
        self._dims = self._set_dims(ds)

        # Get the longitudes and latitude for 
        self._longitudes = ds["longitude"].data
        self._latitudes = ds["latitude"].data

        self._lead_times = self._set_lead_times(ds)

        ds.close()


        self._data = self._open()
        if self._vars:
            self.select_vars(self._vars)
        else:
            self._vars = list(self._data.keys())
        
        if self._mapping:
            self._data = add_xy(self.data,self._mapping)


    def _set_dims(self,ds): #FIXME: this no longer holds when we unstack
        n_reference_time = len(self._files)
        n_valid_time = ds.sizes["time"]
        n_index = ds.sizes["values"]
        return (n_reference_time, n_valid_time, n_index)
    
    def _set_lead_times(self,ds):
        return (ds["time"]- ds["time"][0]).data

    def dim_names(self):
        return self._dim_names
    
    def dims(self):
        return self._dims

    def vars(self):
        return self._vars
    
    def data(self):
        return self._data
        
    def unstacked(self):
        return self._unstacked
    
    def observation(self):
        return self._observation
    
    def _open(self):
        ds = xr.open_mfdataset(
            self._files,
            preprocess=_preprocess,
            chunks={
                "reference_time" : 1,
                "time": -1,
                "values": -1
            },
            **self._mf_kwargs,
        )
        ds_coords = ds.assign_coords(
            {
                "lead_time": ("lead_time", self._lead_times),
                "grid_index": ("grid_index", np.arange(self._dims[2])),
                "valid_time": (
                    ["reference_time", "lead_time"],
                    ds["reference_time"].data[:,np.newaxis] + \
                        self._lead_times[np.newaxis,:]
                ),
                "longitude" : ("grid_index", self._longitudes),
                "latitude": ("grid_index", self._latitudes),
            }
        )
        ds_coords.attrs["is_forecast"] = True
        
        return ds_coords
    
    def select_variables(self,vars):
        self._data = self._data[vars]
        self._vars = vars

    def unstack(self,mapping=None):
        if self._unstacked:
            pass

        if mapping != None:
            if self._mapping != None:
                LOG.error("Dataset already contains a mapping")
            else:
                self._mapping = mapping
                self._data = add_xy(self._data,mapping)
        elif self._mapping == None:
            LOG.error("No grid mapping found!")
            raise ValueError
        ds_unstacked = self._data.unstack()
        ds_transposed = ds_unstacked.transpose(
            "reference_time",
            "lead_time",
            "x",
            "y"
        )
        self._data = ds_transposed
        self._unstacked = True
            

    
def _preprocess(ds):
    reference_time = ds["time"].data[0]
    
    ds_pruned = ds.drop_vars(DROP_VARS)
    ds_reftime = ds_pruned.expand_dims(
        reference_time=[reference_time]
    )
    ds_reftime[
        "reference_time"
    ].attrs["standard_name"] = "forecast_reference_time"

    ds_renamed = ds_reftime.rename_dims(
        {
            "values":"grid_index",
            "time":"lead_time"
        }
    )

    return ds_renamed


    
    



    
            


# def preprocess(dataset,reference_time=None):
#     keep_vars = [var for var in dataset if var not in DROP_VARS]
#     ds_pruned = dataset[keep_vars]
#     ds_latlon = ds_pruned.assign_coords(
#         latitude=ds_pruned.latitude,
#         longitude=ds_pruned.longitude
#     )

#     if not reference_time:
#         reference_time = dataset["time"].data[0]
#     ds_reftime = ds_latlon.expand_dims(
#         reference_time=[reference_time]
#     )
#     ds_reftime[
#         "reference_time"
#     ].attrs["standard_name"] = "forecast_reference_time"
#     ds_validtime = ds_reftime.rename_vars(
#         {"time" : "valid_time"}
#     )
#     ds_leadtime =  ds_validtime.assign_coords(
#         lead_time=(
#             "time",
#             ds_validtime["valid_time"].data -
#                 ds_validtime["reference_time"].data
#         )
#     )
#     ds_swapped = ds_leadtime.swap_dims({"time":"lead_time"})
#     ds_final = ds_swapped.assign_coords(
#         valid_time=(
#             ["reference_time", "lead_time"],
#             ds_swapped["valid_time"].data[np.newaxis,:]
#         )
#     )
#     return ds_final

# def load(dates, path_fmt, **kwargs):
#     # List all files
#     filenames = []
#     for date in dates:
#         date_dt = date.astype(datetime)
#         path = path_fmt.format(
#         yyyy=date_dt.strftime("%Y"),
#         yy=date_dt.strftime("%y"),
#         mm=date_dt.strftime("%m"),
#         dd=date_dt.strftime("%d"),
#         HH=date_dt.strftime("%H"),
#         MM=date_dt.strftime("%M"),
#         SS=date_dt.strftime("%S"),
#         )
#         if not os.path.isfile(path):
#             print(f"Warning no file for date {date_dt.strftime('%Y%m%d %H:%M')}, skipping file")
#         else:
#             filenames.append(path)
#     #prep metadata
#     engine = kwargs.get("engine","h5netcdf")
#     combine = kwargs.get("combine","nested")
#     parallel = kwargs.get("parallel",True)
#     concat_dim = kwargs.get("concat_dim","reference_time")
#     data_vars = kwargs.get("data_vars","minimal")
#     chunks = kwargs.get(
#         "chunks",
#         {"reference_time":5,"time":11, "values": 1142761}
#     )
#     # go get them data
#     ds = xr.open_mfdataset(
#         filenames,
#         preprocess=preprocess,
#         engine=engine,
#         combine=combine,
#         concat_dim=concat_dim,
#         data_vars=data_vars,
#         chunks=chunks,
#         parallel=parallel
#     )
#     ds.attrs['spatial_dimension'] = 'values'
#     return ds

    
