import xarray as xr
import numpy as np
import logging
from typing import List

from .base import PointDataStore, FcstDataStore, ObsDataStore
from ..utils.utils import load_yaml

LOG = logging.getLogger(__name__)

class RmiRePytoolsForecast(PointDataStore, FcstDataStore):
    """Datastore-class to represent forecast netCDF-files produced by RMI's renewable-energy pytools"""
    def __init__(self, files : str, model : str, station_info : str, variables : List[str] = None) -> None:
        """Initialize the RmiRePytoolsForecast datastore.
        This constructor sets up an RmiRePytoolsForecast instance by loading data files and optionally
        mapping coordinates and selecting variables.

        Args:
            files (str): file path to load data from.
            model (str): model name to select from the dataset.
            station_info (str): path to the station information YAML file.
            variables (List[str], optional): Variables to select from the dataset.
                If None, all variables are loaded. Defaults to None.
        
        Returns:
            None
        """
        LOG.info("Initializing RmiRePytoolsForecast datastore")
        self._files = files
        self._model = model
        self._station_info = load_yaml(station_info)
        self._data = self._open()

        if variables:
            self.select_variables(variables)
         
    def _open(self) -> xr.Dataset:
        """Open the dataset and postprocess it.
        This method loads the dataset from the specified files, selects the model,
        and applies postprocessing.
        
        Returns:
            xr.Dataset: The postprocessed dataset.
        """
        
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
    """Datastore-class to represent observation netCDF-files produced by RMI's renewable-energy pytools"""
    def __init__(self,files : str, model : str, station_info : str, variables : List[str] = None) -> None:
        """Initialize the RmiRePytoolsObservation datastore.
        This constructor sets up an RmiRePytoolsObservation instance by loading data files and optionally
        mapping coordinates and selecting variables.

        Args:
            files (str): file path to load data from.
            model (str): model name to select from the dataset.
            station_info (str): path to the station information YAML file.
            variables (List[str], optional): Variables to select from the dataset.
                If None, all variables are loaded. Defaults to None.
        
        Returns:
            None
        """
        LOG.info("Initialzing RmiRePytoolsObservation datastore")
        self._files = files
        self._model = model
        self._station_info = load_yaml(station_info)
        
        self._data = self._open()

        if variables:
            self.select_variables(variables)
        
    def _open(self) -> xr.Dataset:
        """Open the dataset and postprocess it.
        This method loads the dataset from the specified files, selects the model,
        and applies postprocessing.

        Returns:
            xr.Dataset: The postprocessed dataset.
        """
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
        

def _postprocess_fcst(ds : xr.Dataset) -> xr.Dataset:
    """Postprocess the forecast dataset.
    This method stacks the dataset, assigns coordinates for reference time and lead time,
    and renames dimensions.

    Args:
        ds (xr.Dataset): The input dataset to be postprocessed.

    Returns:
        xr.Dataset: The postprocessed dataset with assigned coordinates and renamed dimensions.
    """
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

def _postprocess_obs(ds : xr.Dataset) -> xr.Dataset:
    """Postprocess the observation dataset.
    This method stacks the dataset, assigns coordinates for valid_time and
    renames dimensions.

    Args:
        ds (xr.Dataset): The input dataset to be postprocessed.
    
    Returns:
        xr.Dataset: The postprocessed dataset with assigned coordinates and renamed dimensions.
    """
    
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