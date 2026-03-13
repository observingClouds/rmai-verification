import numpy as np
import xarray as xr

from rmai_verification.datastores.anemoi_datasets import AnemoiDatasets


def _make_test_dataset(num_cells=2, num_variables=2):
    """Create a minimal in-memory dataset mimicking anemoi-datasets structure."""
    # Define dims
    ensemble = [0]
    time = np.array(["2000-01-01T00:00:00"], dtype="datetime64[ns]")
    cell = np.arange(num_cells)

    variables = [f"v{i+1}" for i in range(num_variables)]

    data = np.zeros((len(ensemble), len(time), len(cell), len(variables)))

    ds = xr.Dataset(
        data_vars={
            "data": (("ensemble", "time", "cell", "variable"), data)
        },
        coords={
            "ensemble": ensemble,
            "time": time,
            "cell": cell,
            "longitudes": ("cell", np.linspace(0, 1, len(cell))),
            "latitudes": ("cell", np.linspace(0, 1, len(cell))),
            "dates": ("time", time),
        },
        attrs={"variables": variables},
    )
    return ds


def test_anemoidatasets_accepts_xarray_dataset():
    ds = _make_test_dataset()
    ds_obj = AnemoiDatasets(filename_or_obj=ds)

    # The output should be a Dataset with variables matching the `variables` attr.
    assert set(ds_obj.vars) == set(ds.attrs["variables"])
    assert "valid_time" in ds_obj.dims
    assert "grid_index" in ds_obj.dims


def test_anemoidatasets_accepts_list_of_xarray_datasets():
    ds1 = _make_test_dataset()
    ds2 = _make_test_dataset()
    ds_obj = AnemoiDatasets(filename_or_obj=[ds1, ds2])

    # Concatenation along valid_time should at least double the length
    assert ds_obj.dims["valid_time"] == 2
    assert set(ds_obj.vars) == set(ds1.attrs["variables"])
