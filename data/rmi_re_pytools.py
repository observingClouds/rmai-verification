import xarray as xr
import numpy as np
import logging

from .base import PointDatastore
from utils.utils import load_yaml

LOG = logging.getLogger(__name__)

class RmiRePytoolsForecast(PointDatastore):

    def __init__(self,config):
        LOG.info("Initialzing RmiRePytoolsForecast datastore")
        self._files = config["files"]
        self._model = config["model"]
        self._station_info = load_yaml(config["station_info"])
        self._vars = config.get("variables",None)
        self._observation = False

        self._dim_names = ("reference_time","lead_time","point_index")
        self._data = self._open()

        self._dims = tuple(self._data.sizes.values())

        if self._vars:
            self.select_vars(self._vars)
        else:
            self._vars = list(self._data.keys())

            
    def dim_names(self):
        return self._dim_names
    
    def dims(self):
        return self._dims

    def vars(self):
        return self._vars
    
    def data(self):
        return self._data
        
    def observation(self):
        return self._observation

    def select_variables(self, vars):
        self._data = self._data[vars]
        self._vars = vars

    def _open(self):
        ds = xr.open_dataset(self._files).sel(model=self._model).drop_vars("model")
        ds_postproc = _postprocess_fcst(ds)
        aux_coords = dict(
            code=("point_index", [self._station_info[station]["code"] for station in ds_postproc["station"].values]),
            longitude=("point_index", [self._station_info[station]["lon"] for station in ds_postproc["station"].values]),
            latitude=("point_index", [self._station_info[station]["lat"] for station in ds_postproc["station"].values])
        )
        ds_info = ds_postproc.assign_coords(aux_coords)
        ds_index = ds_info.set_xindex("code")
        return ds_index


class RmiRePytoolsObservation(PointDatastore):

    def __init__(self,config):
        LOG.info("Initialzing RmiRePytoolsForecast datastore")
        self._files = config["files"]
        self._model = config["model"]
        self._station_info = load_yaml(config["station_info"])
        self._vars = config.get("variables",None)
        self._observation = True

        self._dim_names = ("valid_time","point_index")
        self._data = self._open()

        self._dims = tuple(self._data.sizes.values())

        if self._vars:
            self.select_vars(self._vars)
        else:
            self._vars = list(self._data.keys())

            
    def dim_names(self):
        return self._dim_names
    
    def dims(self):
        return self._dims

    def vars(self):
        return self._vars
    
    def data(self):
        return self._data
        
    def observation(self):
        return self._observation

    def select_variables(self, vars):
        self._data = self._data[vars]
        self._vars = vars

    def _open(self):
        ds = xr.open_dataset(self._files).sel(model=self._model).drop_vars("model")
        ds_postproc = _postprocess_obs(ds)
        aux_coords = dict(
            code=("point_index", [self._station_info[station]["code"] for station in ds_postproc["station"].values]),
            longitude=("point_index", [self._station_info[station]["lon"] for station in ds_postproc["station"].values]),
            latitude=("point_index", [self._station_info[station]["lat"] for station in ds_postproc["station"].values])
        )
        ds_info = ds_postproc.assign_coords(aux_coords)
        ds_index = ds_info.set_xindex("code")
        return ds_index
        

def _postprocess_fcst(ds):
    ds_reftime = ds.assign_coords(
        reference_time=ds["date"]+ds["run"].astype("timedelta64[h]"),
        lead_time = ds["lead_time"].astype("timedelta64[h]")
    )
    ds_stacked = ds_reftime.stack(
        combined=["date", "run"]
    ).swap_dims(
        {"combined":"reference_time"}
    ).drop_vars(
        ["combined","date","run"]
    ).transpose(
        "reference_time","lead_time","station"
    ).rename_dims(
        {"station":"point_index"}
    )

    ds_valid = ds_stacked.assign_coords(
        valid_time=ds_stacked["reference_time"]+ds_stacked["lead_time"]
    )

    return ds_valid

def _postprocess_obs(ds):
    ds_valid = _postprocess_fcst(ds)
    ds_dropped = ds_valid.stack(
        combined=["reference_time", "lead_time"]
    ).swap_dims(
        {"combined":"valid_time"}
    ).drop_vars(
        ["combined","reference_time","lead_time"]
    ).drop_duplicates(
        dim="valid_time"
    )
    return ds_dropped