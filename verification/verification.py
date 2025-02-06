import numpy as np
from datetime  import datetime
import os 
import logging

from utils.time import to_timedelta64
from data import get_datastore
from transformations import get_transformation
from data.base import GridDatastore, PointDatastore

LOG = logging.getLogger(__name__)

POINT_COORDS = ["latitude", "longitude"]

VERIF_VARS = ["2t", "10s"]

class Verification():
    def __init__(self,config):
        self._config = config
        self._start = np.datetime64(config["dates"]["start"])
        self._end = np.datetime64(config["dates"]["end"])
        self._frequency = to_timedelta64(config["dates"]["frequency"])
        self._datastores = self._init_datastores(config["datastores"].copy())
        self._reference_datastore = config["verification"]["reference_datastore"]
        self._apply_transformation()

    def verify(self):
        self._unify_datastores()
        data = self._set_common_grid_or_points()
        return data



    def _init_datastores(self,config):
        stores = dict()

        for name, config in config.items():
            LOG.info(f"Initializing {name}")
            config["files"] = get_filenames(
                config["path"],
                self._start,
                self._end,
                self._frequency
            )
            type = get_datastore(config.pop("type"))
            store = type(config)
            stores[name]=store
        return stores
    
    def _apply_transformation(self):
        for transformation, trans_options in self._config["transformations"].items():
            trans_options = trans_options.copy()
            datastores = trans_options.pop("datastores",self._datastores.keys())
            transformer = get_transformation(transformation)
            transformation = transformer(trans_options)
            for datastore in datastores:
                self._datastores[datastore].transform(transformation)


    def _unify_datastores(self):
        self._select_variables()
        self._select_common_reference_times()
        self._select_valid_times()

    def _select_variables(self):
        variables = self._config["verification"].get("variables",VERIF_VARS)
        for datastore in self._datastores.values():
            datastore.select_variables(variables)

    def _select_common_reference_times(self):
        ref_times = []
        for datastore in self._datastores.values():
            if not datastore.observation():
                ref_times.append(datastore.data()["reference_time"].values)
        ref_times = list(set(ref_times[0]).intersection(*ref_times))
        for datastore in self._datastores.values():
            if not datastore.observation():
                datastore.select_reference_times(ref_times)
               
    def _valid_time_union(self):
        valid_times = []
        for name, datastore in self._datastores.items():
            if not datastore.observation():
                valid_times.append(list(datastore.data()["valid_time"].values.ravel()))
        valid_times = list(set().union(*valid_times))
        return valid_times
        
    def _select_valid_times(self):
        valid_times = self._valid_time_union()
        for datastore in self._datastores.values():
            if datastore.observation():
                datastore.select_valid_times(valid_times)

    def _set_common_grid_or_points(self):
        #FIXME: Datastores should have a .coords() classmethod
        datastores = self._datastores.copy()
        reference = datastores.pop(self._reference_datastore)
        common_data = dict()
        if isinstance(reference,GridDatastore):
            pass
            transformer = get_transformation("regrid")
            key = "regridding"
        elif isinstance(reference, PointDatastore):
            transformer = get_transformation("interpolate")
            interp_kwargs = self._config.get("interpolation",dict())
            transformation = transformer(
                reference.data(),
                interp_kwargs
            )
            for name, store in datastores.items():
                if isinstance(store, PointDatastore):
                    #TODO: Check if all reference codes are in Datastore
                    _data = store.data().sel(code=reference.data()["code"].values)
                
                elif isinstance(store, GridDatastore):
                    for coord in POINT_COORDS:
                        assert coord in list(reference.data().coords.keys()), f"Coordinate {coord} missing from the reference datastore {self._reference_datastore}"
                    if not store.unstacked():
                        store.unstack()
                    _data = transformation.execute(store.data())
                common_data[name] = _data


                    
                    #_data = transformation(
                    #    input_ds = store.data()

                
        else:
            LOG.error("Datatore class not supported")            
        return common_data
        



        

    

def get_filenames(path_fmt,start,end,frequency):
    filenames = []
    date = start
    while date <= end:
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
        if not os.path.exists(path):
            LOG.warning(f"No file found for date {date_dt.strftime('%Y%m%d %H:%M')}, skipping file")
            pass
        else:
            filenames.append(path)
        date += frequency
    filenames = list(set(filenames))
    if len(filenames) == 1: 
        filenames = filenames[0]
    return filenames

