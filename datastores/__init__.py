from datastores.anemoi_datasets import AnemoiDatasets
from datastores.anemoi_inference import AnemoiInference
from datastores.rmi_re_pytools import RmiRePytoolsForecast, RmiRePytoolsObservation
from datastores.base import PointObservations

DATASTORES = {
    "anemoi-inference": AnemoiInference,
    "anemoi-datasets" : AnemoiDatasets,
    "rmi-re-pytools-fc": RmiRePytoolsForecast,
    "rmi-re-pytools-obs": RmiRePytoolsObservation,
    "point-observations" : PointObservations
}
