import xarray as xr
import numpy as np
import logging

from numpy.typing import NDArray
from typing import List, Tuple, Dict, Union

from .base import GridDataStore, FcstDataStore
from ..grids.grid_mapping import add_xy

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
    def __init__(self, 
                 files: Union[str, List[str], xr.Dataset, List[xr.Dataset], None] = None,
                 filename_or_obj: Union[str, List[str], xr.Dataset, List[xr.Dataset], None] = None,
                 variables: Union[List[str],Tuple[str],set] = None,
                 mapping: Union[Dict[str,str],str] = None,
                 mf_kwargs: Dict[str,str] = dict()
                 ):
        LOG.info("Initializing AnemoiInference datastore")

        if filename_or_obj is not None and files is not None:
            raise ValueError("Provide only one of `files` or `filename_or_obj`")
        if filename_or_obj is None:
            filename_or_obj = files
        if filename_or_obj is None:
            raise ValueError("Missing input data, please provide `files` or `filename_or_obj`")

        self._filename_or_obj = filename_or_obj
        self._mapping: Union[Dict[str],str] = mapping
        self._stacked: bool = True

        # Add the xr.open_mfdataset kwargs
        self._mf_kwargs = dict()
        for key, value in MF_KWARGS.items():
            self._mf_kwargs[key]=mf_kwargs.get(key,value)
        for key, value in mf_kwargs.items():
            if key not in self._mf_kwargs.keys():
                self._mf_kwargs[key] = value
        
        # Open a single dataset to infer some properties.
        if isinstance(self._filename_or_obj, list):
            if len(self._filename_or_obj) == 0:
                raise ValueError("filename_or_obj must not be an empty list")
            if isinstance(self._filename_or_obj[0], xr.Dataset):
                ds = self._filename_or_obj[0]
                close_ds = False
            else:
                ds = xr.open_dataset(self._filename_or_obj[0], engine=self._mf_kwargs["engine"])
                close_ds = True
        elif isinstance(self._filename_or_obj, xr.Dataset):
            ds = self._filename_or_obj
            close_ds = False
        else:
            ds = xr.open_dataset(self._filename_or_obj, engine=self._mf_kwargs["engine"])
            close_ds = True

        self._input_format = _infer_input_format(ds)
        if self._input_format == "structured":
            self._stacked = False

        # Get the longitudes and latitude 
        self._longitudes = ds["longitude"].data
        self._latitudes = ds["latitude"].data

        # Get the lead times
        self._lead_times = _calc_lead_times(ds)

        if close_ds:
            ds.close()


        self._data = self._open()
        if variables:
            self.select_variables(variables)
        
        if self._mapping and self._stacked:
            self._data = add_xy(self._data,self._mapping)
        elif self._mapping and not self._stacked:
            LOG.warning("Ignoring `mapping` for already structured input data")

        LOG.info("Finished initializing AnemoiInference datastore")

    def _open(self):
        """
        Opens and processes multiple NetCDF datasets into an xarray Dataset with
        assigned coordinates and attributes.
        This method uses `xarray.open_mfdataset` to open multiple NetCDF files,
        preprocesses them, and assigns additional coordinates such as lead time,
        grid index, valid time.

        Returns:
            xarray.Dataset: The processed dataset with assigned coordinates and
            attributes.
        """
        if isinstance(self._filename_or_obj, list):
            if len(self._filename_or_obj) > 0 and isinstance(self._filename_or_obj[0], xr.Dataset):
                datasets = [_preprocess(ds, self._input_format) for ds in self._filename_or_obj]
                ds = xr.concat(datasets, dim="reference_time")
            else:
                ds = xr.open_mfdataset(
                    self._filename_or_obj,
                    preprocess=lambda x: _preprocess(x, self._input_format),
                    chunks={
                        "reference_time" : 1,
                        "time": -1,
                        "values": -1
                    },
                    **self._mf_kwargs,
                )
        elif isinstance(self._filename_or_obj, xr.Dataset):
            ds = _preprocess(self._filename_or_obj, self._input_format)
        else:
            ds = xr.open_mfdataset(
                [self._filename_or_obj],
                preprocess=lambda x: _preprocess(x, self._input_format),
                chunks={
                    "reference_time" : 1,
                    "time": -1,
                    "values": -1
                },
                **self._mf_kwargs,
            )
        if self._input_format == "structured":
            ds_coords = ds.assign_coords(
                {
                    "lead_time": ("lead_time", self._lead_times),
                    "valid_time": (
                        ["reference_time", "lead_time"],
                        ds["reference_time"].data[:,np.newaxis] + self._lead_times[np.newaxis,:]
                    ),
                }
            )
        else:
            ds_coords = ds.assign_coords(
                {
                    "lead_time": ("lead_time", self._lead_times),
                    "grid_index": ("grid_index", np.arange(ds.sizes["grid_index"])),
                    "valid_time": (
                        ["reference_time", "lead_time"],
                        ds["reference_time"].data[:,np.newaxis] + self._lead_times[np.newaxis,:]
                    ),
                    "longitude" : ("grid_index", self._longitudes),
                    "latitude": ("grid_index", self._latitudes),
                }
            )
        ds_coords.attrs["is_observation"] = False
        return ds_coords

    def unstack(self,mapping: Union[str, Dict[str,str]] = None):
        """Unstacks the dataset from a stacked format to a grid format.

        This method checks if the dataset is currently stacked and if so, it
        unstacks it. If a mapping is provided, it will be used to add x and y
        coordinates to the dataset. If the dataset is already unstacked it
        will do nothing.

        Args:
            mapping (Union[Dict[str,str],str], optional): A mapping to add x and y
                coordinates to the dataset. Defaults to None.

        Returns:
            None
        """
        LOG.debug(f"Start unstacking, stacked state is currently: {self._stacked}")
        if not self._stacked:
            return

        if mapping != None:
            if self._mapping != None:
                LOG.error("Dataset already contains a mapping")
                raise ValueError
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
            "y",
            "x"
        )
        self._data = ds_transposed
        self._stacked = False
            

def _calc_lead_times(ds: xr.Dataset | xr.DataArray) -> NDArray[np.timedelta64]:
        """
        Calculate the lead times from a dataset.

        This function computes the lead times by subtracting the first time value 
        in the dataset from all other time values. The result is returned as a 
        data array.

        Args:
            ds (xarray.Dataset): The input dataset containing a "time" coordinate.

        Returns:
            numpy.ndarray: An array of lead times relative to the first time value.
        """
        if "lead_time" in ds.coords:
            return ds["lead_time"].data

        if "time" not in ds.coords:
            raise ValueError("Cannot infer lead_time: dataset must contain `time` or `lead_time` coordinate")

        reference_time = _infer_reference_time(ds)
        return (ds["time"].data - reference_time).astype("timedelta64[ns]")

def _preprocess(ds: xr.Dataset | xr.DataArray, input_format: str) -> xr.Dataset:
    """
    Preprocess the dataset by dropping unnecessary variables and renaming dimensions.
    This function drops specified variables from the dataset and renames dimensions
    to standard names. It also expands the reference time dimension and assigns
    attributes to the reference time variable.

    Args:
        ds (xarray.Dataset): The input dataset to preprocess.

    Returns:
        xarray.Dataset: The preprocessed dataset with dropped variables and renamed dimensions.
    """

    if input_format == "structured":
        return _preprocess_structured(ds)
    return _preprocess_unstructured(ds)


def _infer_input_format(ds: xr.Dataset | xr.DataArray) -> str:
    if "values" in ds.dims:
        return "unstructured"
    if "x" in ds.dims and "y" in ds.dims:
        return "structured"
    raise ValueError("Unsupported Anemoi inference format, expected dims containing `values` or (`y`, `x`)")


def _infer_reference_time(ds: xr.Dataset | xr.DataArray) -> np.datetime64:
    if "forecast_reference_time" in ds:
        return np.asarray(ds["forecast_reference_time"].data).reshape(-1)[0].astype("datetime64[ns]")
    if "reference_time" in ds.coords:
        return np.asarray(ds["reference_time"].data).reshape(-1)[0].astype("datetime64[ns]")
    return np.asarray(ds["time"].data).reshape(-1)[0].astype("datetime64[ns]")


def _preprocess_unstructured(ds: xr.Dataset | xr.DataArray) -> xr.Dataset:
    reference_time = _infer_reference_time(ds)

    ds_pruned = ds.drop_vars(DROP_VARS, errors="ignore")
    ds_reftime = ds_pruned.expand_dims(reference_time=[reference_time])
    ds_reftime["reference_time"].attrs["standard_name"] = "forecast_reference_time"

    ds_renamed = ds_reftime.rename_dims(
        {
            "values":"grid_index",
            "time":"lead_time"
        }
    )
    return ds_renamed


def _preprocess_structured(ds: xr.Dataset | xr.DataArray) -> xr.Dataset:
    reference_time = _infer_reference_time(ds)

    ds_work = ds.drop_vars(["forecast_reference_time"], errors="ignore")
    if "latitude" in ds_work.data_vars:
        ds_work = ds_work.assign_coords(latitude=ds_work["latitude"])
    if "longitude" in ds_work.data_vars:
        ds_work = ds_work.assign_coords(longitude=ds_work["longitude"])

    if "time" in ds_work.dims:
        lead_times = (ds_work["time"].data - reference_time).astype("timedelta64[ns]")
        ds_work = ds_work.rename({"time": "lead_time"})
        ds_work = ds_work.assign_coords(lead_time=("lead_time", lead_times))
    elif "lead_time" not in ds_work.coords:
        raise ValueError("Structured input must contain a `time` dimension or `lead_time` coordinate")

    ds_work = ds_work.expand_dims(reference_time=[reference_time])
    ds_work["reference_time"].attrs["standard_name"] = "forecast_reference_time"

    extra_dims = [dim for dim in ds_work.dims if dim not in ["reference_time", "lead_time"]]
    return ds_work.transpose("reference_time", "lead_time", *extra_dims)
