from .rename import Renamer
from .uv_to_speed import UVToSpeed
from .kelvin_to_celcius import KelvinToCelcius

TRANSFORMATIONS = {
    "rename": Renamer,
    "uv_to_speed": UVToSpeed,
    "kelvin_to_celcius": KelvinToCelcius
}