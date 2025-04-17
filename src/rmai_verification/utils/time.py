import numpy as np

def to_timedelta64(freq: str) -> np.timedelta64:
    """
    Convert a frequency string to a numpy timedelta64 object.
    The frequency string should be in the format of a number followed by a time unit,
    e.g. '1D', '2H', '3M', etc.
    The time unit can be one of the following:
    - 'Y' for years
    - 'M' for months
    - 'W' for weeks
    - 'D' for days
    - 'h' for hours
    - 'm' for minutes
    - 's' for seconds
    - 'ms' for milliseconds
    Parameters
    ----------
    freq : str
        The frequency string to convert.
    
    Returns
    -------
    np.timedelta64
        The converted numpy timedelta64 object.
    """
    value = freq[:-1]
    unit = freq[-1]
    return np.timedelta64(value,unit)