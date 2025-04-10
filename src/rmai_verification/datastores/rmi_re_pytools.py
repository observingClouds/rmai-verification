import xarray as xr
import numpy as np
import logging
from typing import List

from .base import PointDataStore, FcstDataStore, ObsDataStore
from ..utils.utils import load_yaml

LOG = logging.getLogger(__name__)

class RmiRePytoolsForecast(PointDataStore, FcstDataStore):

    def __init__(self, files : str, model : str, station_info : str, variables : List[str] = None):
        LOG.info("Initialzing RmiRePytoolsForecast datastore")
        self._files = files
        self._model = model
        self._station_info = load_yaml(station_info)
        self._data = self._open()

        if variables:
            self.select_vars(self._vars)
        
    def select_variables(self, vars):
        new_data = self._data[vars]
        self._data = new_data

    def select_reference_times(self,reference_times):
        new_data = self._data.sel(reference_time=reference_times)
        self._data = new_data

    def select_lead_times(self,lead_times):
        new_data = self._data.sel(lead_time=lead_times)
        self._data = new_data
    
    def _open(self):
        ds = xr.open_dataset(self._files).sel(model=self._model).drop_vars("model")
        ds_postproc = _postprocess_fcst(ds)
        aux_coords = dict(
            code=("point_index", [self._station_info[station]["code"] for station in ds_postproc["station"].values]),
            longitude=("point_index", [self._station_info[station]["lon"] for station in ds_postproc["station"].values]),
            latitude=("point_index", [self._station_info[station]["lat"] for station in ds_postproc["station"].values])
        )
        ds_info = ds_postproc.assign_coords(aux_coords)
        ds_index = ds_info.swap_dims({"point_index":"code"})
        return ds_index


class RmiRePytoolsObservation(PointDataStore, ObsDataStore):

    def __init__(self,files : str, model : str, station_info : str, variables : List[str] = None):
        LOG.info("Initialzing RmiRePytoolsObservation datastore")
        self._files = files
        self._model = model
        self._station_info = load_yaml(station_info)
        
        self._data = self._open()

        if variables:
            self.select_vars(self._vars)
        
    def select_variables(self, vars):
        new_data = self._data[vars]
        self._data = new_data

    def select_valid_times(self,valid_times):
        new_data = self._data.sel(valid_time=valid_times)
        self._data = new_data

    def _open(self):
        ds = xr.open_dataset(self._files).sel(model=self._model).drop_vars("model")
        ds_postproc = _postprocess_obs(ds)
        aux_coords = dict(
            code=("point_index", [self._station_info[station]["code"] for station in ds_postproc["station"].values]),
            longitude=("point_index", [self._station_info[station]["lon"] for station in ds_postproc["station"].values]),
            latitude=("point_index", [self._station_info[station]["lat"] for station in ds_postproc["station"].values])
        )
        ds_info = ds_postproc.assign_coords(aux_coords)
        ds_index = ds_info.swap_dims({"point_index":"code"})
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