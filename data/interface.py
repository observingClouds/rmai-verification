from data.anemoi_datasets import AnemoiDatasets
from data.anemoi_inference import AnemoiInference
from data.rmi_re_pytools import RmiRePytoolsForecast, RmiRePytoolsObservation


_datastores = {
    "anemoi-inference": AnemoiInference,
    "anemoi-datasets" : AnemoiDatasets,
    "rmi-re-pytools-fc": RmiRePytoolsForecast,
    "rmi-re-pytools-obs": RmiRePytoolsObservation
}

def get_datastore(name):
    try:
        return _datastores[name]
    except KeyError:
        raise ValueError(
            f"Unknown loader: {name}\n"
            + "The available loaders are:"
            + str(list(_datastores.keys()))
        ) 

