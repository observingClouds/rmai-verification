from typing import List
import scores
import xarray as xr

def rmse(fcst: xr.Dataset | xr.DataArray, obs: xr.Dataset | xr.DataArray, avg_dim: List[str], skipna: bool = True) -> xr.Dataset | xr.DataArray:
    return scores.continuous.rmse(fcst, obs, reduce_dims=avg_dim)

def mse(fcst: xr.Dataset | xr.DataArray, obs: xr.Dataset | xr.DataArray, avg_dim: List[str], skipna: bool = True) -> xr.Dataset | xr.DataArray:
    return scores.continuous.mse(fcst, obs, reduce_dims=avg_dim)

def bias(fcst: xr.Dataset | xr.DataArray, obs: xr.Dataset | xr.DataArray, avg_dim: List[str], skipna: bool = True) -> xr.Dataset | xr.DataArray:
    return scores.continuous.additive_bias(fcst, obs, reduce_dims=avg_dim)

