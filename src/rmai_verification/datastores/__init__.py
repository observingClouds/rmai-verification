from .anemoi_datasets import AnemoiDatasets
from .anemoi_inference import AnemoiInference
from .rmi_re_pytools import RmiRePytoolsForecast, RmiRePytoolsObservation
from .base import PointObservations

DATASTORES = {
    "anemoi-inference": AnemoiInference,
    "anemoi-datasets" : AnemoiDatasets,
    "rmi-re-pytools-fc": RmiRePytoolsForecast,
    "rmi-re-pytools-obs": RmiRePytoolsObservation,
    "point-observations" : PointObservations
}
