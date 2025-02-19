import xarray as xr
import numpy as np
import logging

from .base import GridDataStore, FcstDataStore
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


class AnemoiInference(GridDataStore,FcstDataStore):
    def __init__(self, files, variables=None, mapping=None, mf_kwargs=dict()):
        LOG.info("Initializing AnemoiInference datastore")
        # Add the files to the class
        self._files = files #FIXME
        self._mapping = mapping
        self._stacked = True


        # Add the xr.open_mfdataset kwargs
        self._mf_kwargs = dict()
        for key, value in MF_KWARGS.items():
            self._mf_kwargs[key]=mf_kwargs.get(key,value)
    
        # open a single dataset to infer some properties
        ds = xr.open_dataset(self._files[0])

        # Get the longitudes and latitude 
        self._longitudes = ds["longitude"].data
        self._latitudes = ds["latitude"].data

        # Get the lead times
        self._lead_times = _set_lead_times(ds)

        ds.close()


        self._data = self._open()
        if variables:
            self.select_vars(self._vars)
        
        if self._mapping:
            self._data = add_xy(self._data,self._mapping)

        self._dims = dict(self._data.sizes)
        LOG.info("Finished initializing AnemoiInference datastore")

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
                "grid_index": ("grid_index", np.arange(ds.sizes["grid_index"])),
                "valid_time": (
                    ["reference_time", "lead_time"],
                    ds["reference_time"].data[:,np.newaxis] + \
                        self._lead_times[np.newaxis,:]
                ),
                "longitude" : ("grid_index", self._longitudes),
                "latitude": ("grid_index", self._latitudes),
            }
        )
        ds_coords.attrs["is_observation"] = False
        
        return ds_coords
    
    def select_variables(self,vars):
        new_data = self._data[vars]
        self._data = new_data
        self._vars = vars
    
    def select_reference_times(self,reference_times):
        new_data = self._data.sel(reference_time=reference_times)
        self._data = new_data

    def select_lead_times(self,lead_times):
        new_data = self._data.sel(lead_time=lead_times)
        self._data = new_data


    def unstack(self,mapping=None):
        LOG.debug(f"Start unstacking, stacked state is currently: {self._stacked}")
        if not self._stacked:
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
        self._stacked = False
            

def _set_lead_times(ds):
        return (ds["time"]- ds["time"][0]).data

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
