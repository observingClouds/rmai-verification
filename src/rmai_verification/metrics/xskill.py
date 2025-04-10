import xskillscore as xs

def rmse(obs, fcst, avg_dim, skipna=True):
    return xs.rmse(obs, fcst, avg_dim, skipna=skipna)

def mse(obs, fcst, avg_dim, skipna=True):
    return xs.mse(obs, fcst, avg_dim, skipna=skipna)

def bias(obs, fcst, avg_dim, skipna=True):
    return xs.me(obs, fcst, avg_dim, skipna=skipna)

