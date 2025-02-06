from .interpolate import Interpolator
from .rename import Renamer
from .uv_to_speed import UVToSpeed

_transformations = {
    "interpolate": Interpolator,
    "rename": Renamer,
    "uv_to_speed": UVToSpeed
}

def get_transformation(name):
    try:
        return _transformations[name]
    except KeyError:
        raise ValueError(
            f"Unknown loader: {name}\n"
            + "The available transformations are:"
            + str(list(_transformations.keys()))
        ) 
