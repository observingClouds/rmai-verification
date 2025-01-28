from data import anemoi_inference, anemoi_datasets, rmi_re_pytools


_loaders = {
    "anemoi-inference": anemoi_inference.load,
    "anemoi-datasets" : anemoi_datasets.load,
    "rmi-re-pytools-fc": rmi_re_pytools.load_fc,
    "rmi-re-pytools-obs": rmi_re_pytools.load_obs
}

def get_loader(name):
    try:
        return _loaders[name]
    except KeyError:
        raise ValueError(
            f"Unknown loader: {name}\n"
            + "The available loaders are:"
            + str(list(_loaders.keys()))
        ) 

