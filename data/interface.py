from data import anemoi_inference, anemoi_datasets


_loaders = {
    "anemoi-inference": anemoi_inference.load,
    "anemoi-datasets" : anemoi_datasets.load
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

