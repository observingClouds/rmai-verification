import xarray as xr
import numpy as np
from projections import to_grid, interpolate_to_latlon
from utils import load_yaml

def post_processor(config, forecasts, observations):
    if len(config) == 0:
        return (forecasts, observations)
    data_list=[observations.expand_dims({'model':['TEMP']})]
    if 'model' in forecasts.dims:
        data_list.append(forecasts)
    ds=xr.concat(data_list,dim='model')
    for name, kwargs in config.items():
        processor = get_processor(name)
        ds = processor(ds, **kwargs)
    return (ds.drop_sel(model=['TEMP']), ds.sel(model='TEMP').drop_vars('model'))


## PROCESSORS

def interpolate(ds, grid = None, locations = None ):
    ds = to_grid(ds, grid)
    if isinstance(locations, str):
        locations = load_yaml(locations)
    dss=[]
    for location, info in locations.items():
        lat=info['lat']
        lon=info['lon']
        dss.append(interpolate_to_latlon(ds,lon,lat).expand_dims({'location':[location]}))
    ds = xr.concat(dss,dim='location')
    ds.attrs['locations'] = locations
    ds.attrs['spatial_dimension'] = 'location'
    return ds.chunk(dict(location=-1))


def uv_to_s(ds, levels = [], drop_uv = False):
    uvs_pairs=[]
    for level in levels:
        if level == 10:
            uvs_pairs.append(('10u', '10v', '10s'))
        else:
            uvs_pairs.append((f"u_{level}", f"v_{level}", f"s_{level}" ))
    for uvs_pair in uvs_pairs:
        ds[uvs_pair[2]] = np.sqrt(ds[uvs_pair[0]]**2+ds[uvs_pair[1]]**2)
        if drop_uv:
            del ds[uvs_pair[0]]
            del ds[uvs_pair[1]]
    return ds

def t_C_to_K(ds, levels = []):
    ts=[]
    for level in levels:
        if level == 2:
            ts.append('2t')
        else:
            ts.append(f"t_{level}")
    for t in ts:
        ds[t] = ds[t]+273.15
    return ds

## REGISTRY ##
PROCESSOR_REGISTRY = {
    "interpolation" : interpolate,
    "uv-to-s" : uv_to_s,
    "t-C-to-K": t_C_to_K
}

def get_processor(name):
    assert name in PROCESSOR_REGISTRY, f"The post-processor {name} is not (yet) supported."
    return PROCESSOR_REGISTRY[name]