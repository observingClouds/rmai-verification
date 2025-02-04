import numpy as np
import xarray as xr
import logging

from .base import GridDatastore
from grids.grid_mapping import add_xy
from data.anemoi_inference import DROP_VARS


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


class AnemoiDatasets(GridDatastore):

    def __init__(self,config):
        LOG.info("Initializing AnemoiDataset datastore")
        self._files = config["files"]
        self._vars = config.get("variables", None)
        self._mapping = config.get("mapping", None)
        self._unstacked = False
        self._observation = True

        # Set the dimension names        
        self._dim_names = ("valid_time","grid_index")

        # Open the dataset
        self._data = self._open()

        # Set the dimensions
        self._dims = (self._data.sizes["valid_time"],
                      self._data.sizes["grid_index"] )
        
        if self._vars:
            self.select_vars(self._vars)
        else:
            self._vars = self._data["variable"].values

        if self._mapping:
            self._data = add_xy(self.data,self._mapping)

        
    def dim_names(self):
        return self._dim_names
    
    def dims(self):
        return self._dims

    def vars(self):
        return self._vars
    
    def data(self):
        data = self._data.to_dataset(dim="variable")
        return data
        
    def unstacked(self):
        return self._unstacked
    
    def observation(self):
        return self._observation

    def select_variables(self, vars):
        self._data = self._data.sel(variable=vars)
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
            "valid_time",
            "x",
            "y",
            "variable"
        )
        self._data = ds_transposed
        self._unstacked = True

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