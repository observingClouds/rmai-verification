import numpy as np
from typing import Dict, List

from ..utils.time import to_timedelta64
from ..utils.files import get_filenames

from ..datastores import DATASTORES
from ..transformations import TRANSFORMATIONS
from ..datastores.base import BaseDataStore


def load_datastores(datastores : Dict[str, Dict], start_date : str, end_date : str, frequency : str) -> Dict:
    stores = dict()

    start_date = np.datetime64(start_date)
    end_date = np.datetime64(end_date)
    frequency = to_timedelta64(frequency)

    for name, config in datastores.items():
        files = get_filenames(
            path_fmt=config.pop("path"),
            start=start_date,
            end=end_date,
            frequency=frequency,
        )
        store = DATASTORES[config.pop("type")]

        stores[name] = store(files=files,**config)

    return stores

def apply_transformations(datastores : Dict[str, BaseDataStore], transformations: Dict[str, Dict]):
    for transformation, config in transformations.items():
        active_stores = config.pop("datastores", datastores.keys())
        
        transformer = TRANSFORMATIONS[transformation]
        transformation = transformer(**config)
        for store in active_stores:
            datastores[store].transform(transformation)


def select_variables(datastores : Dict[str, BaseDataStore], variables : List[str]):
    for store in datastores.values():
        store.select_variables(variables)








    