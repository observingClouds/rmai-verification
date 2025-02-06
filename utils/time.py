import numpy as np

def to_timedelta64(freq):
    value = freq[:-1]
    unit = freq[-1]
    return np.timedelta64(value,unit)