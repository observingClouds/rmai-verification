import numpy as np
import xarray as xr
import logging

from .base import GridDataStore, ObsDataStore
from grids.grid_mapping import add_xy
from datastores.anemoi_inference import DROP_VARS
from transformations.rename import Renamer
from transformations.uv_to_speed import UVToSpeed


LOG = logging.getLogger(__name__)

COORDS = dict(
    longitude="longitudes",
    latitude="latitudes",
    valid_time="dates"
)

# DROP= [
#     "count",
#     "has_nans",
#     "maximum",
#     "minimum",
#     "mean",
#     "squares",
#     "sums",
#     "stdev",
#     "longitudes",
#     "latitudes",
#     "dates"
# ]


class AnemoiDatasets(GridDataStore, ObsDataStore):
    def __init__(self, files, variables=None, mapping=None):
        LOG.info("Initializing AnemoiDataset datastore")
        self._files = files
        self._mapping = mapping
        self._stacked = True

        # Open the dataset
        self._data = self._open()

        if variables:
            self.select_vars(variables)

        if self._mapping:
            self._data = add_xy(self._data,self._mapping)

    @property
    def dims(self):
        dims = super().dims.copy()
        _ = dims.pop("variable",None)
        return dims
        
    @property
    def vars(self):
        return self._data["variable"].values


    @property
    def data(self):
        LOG.info("Transforming anemoi-datasets xr.DataArray to xr.Dataset, this might take some time.")
        data = self._data.to_dataset(dim="variable")
        return data
        
    def select_variables(self, vars):
        new_data = self._data.sel(variable=vars)
        self._data = new_data
        self._vars = vars

    def select_valid_times(self,valid_times):
        new_data = self._data.sel(valid_time=valid_times)
        self._data = new_data

    def unstack(self,mapping=None):
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

        #FIXME: in some edge-cases an anemoi-datasets can have ref and leadtime
        if "valid_time" in self._data.dims:
            dims = ["valid_time", "x", "y"]
        else:
            dims = ["reference_time", "lead_time", "x", "y"]

        ds_transposed = ds_unstacked.transpose(*dims,...)
        self._data = ds_transposed
        self._unstacked = True

    def transform(self, transformation):
        match transformation:
            case Renamer():
                LOG.debug("Using AnemoiDatasets specific Renamer transformation")
                new_names = []
                for variable in self._data["variable"].values:
                    name = None
                    for new_name, old_names in transformation.rename_dict.items():
                        if variable in old_names:
                            name = new_name
                    if name == None:
                        name = variable.astype(str)
                    new_names.append(name)
                new_data = self._data.assign_coords(
                    {
                        "variable": ("variable", new_names)
                    }
                )
                self._data = new_data
            case UVToSpeed():
                LOG.debug("Using AnemoiDatasets specific UVToSpeed transformation")
                speed = np.sqrt(self._data.sel(variable=transformation.u_wind)**2 + self._data.sel(variable=transformation.v_wind)**2)
                speed = speed.expand_dims("variable").assign_coords(variable=[transformation.wind_speed])
                new_data = xr.concat([speed,self._data], dim="variable",combine_attrs="drop_conflicts")
                self._data = new_data

            case _:
                super().transform(transformation)

    def _open(self):
        ds = xr.open_zarr(self._files,consolidated=False,chunks="auto")
        ds_postproc = _postprocess(ds)
        return ds_postproc
    

def _postprocess(dataset):

    # Add coordinates
    coords = {key: dataset[value].astype("datetime64[ns]").load() if key == "valid_time" else dataset[value].load() for key, value in COORDS.items()}
    for key in ("latitude","longitude"):
        coords[key] = coords[key].astype(np.float32)
    coords["variable"] = dataset.attrs["variables"]
    coords["valid_time"] = coords["valid_time"].astype("datetime64[ns]")
    ds_coords = dataset.assign_coords(coords)

    # Drop unused variables and remove ensemble dimension
    drop_vars = [var for var in DROP_VARS if var in coords["variable"]]
    
    ds_pruned = ds_coords["data"].isel(
        ensemble=0
    ).drop_sel(
        variable=drop_vars
    ).swap_dims(
        {"time":"valid_time"}
    ).rename(
        {"cell":"grid_index"}
    )
    return ds_pruned