import numpy as np
import xarray as xr
import logging
from typing import List

LOG = logging.getLogger(__name__)

CORRECTION = 273.15

class KelvinToCelcius():
    """Convert temperature data between Kelvin and Celsius scales.
    This class implements a transformation to convert temperature values between Kelvin
    and Celsius scales. It can operate on both xarray DataArrays and Datasets.
    Parameters
    ----------
    fields : str or List[str], default="2t"
        The field(s) to transform. Can be a single field name as string or a list of field names.
        These should correspond to temperature variables in the dataset.
    inverse : bool, default=False
        If False, converts from Kelvin to Celsius.
        If True, converts from Celsius to Kelvin.
    Methods
    -------
    execute(input_ds : xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset
        Performs the temperature conversion on the input data.
    Examples
    --------
    >>> transformer = KelvinToCelcius(fields=['2t', 'temp'])
    >>> transformed_data = transformer.execute(input_dataset)
    Notes
    -----
    The transformation uses the standard conversion between Kelvin and Celsius:
    - Kelvin to Celsius: T(°C) = T(K) - 273.15
    - Celsius to Kelvin: T(K) = T(°C) + 273.15
    """

    def __init__(self, fields: str | List[str] = "2t" , inverse: bool = False):
        if isinstance(fields, str):
            self.fields = [fields]
        else:
            self.fields = fields
        self.correction = np.sign(inverse-0.5) * CORRECTION

    def execute(self,input_ds : xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
        """Execute the Kelvin to Celsius transformation on input data.
        This method converts temperature values from Kelvin to Celsius by subtracting
        273.15 (stored in self.correction) from the specified temperature fields.
        Parameters
        ----------
        input_ds : xr.DataArray or xr.Dataset
            Input data containing temperature values in Kelvin. Can be either a DataArray
            with multiple variables or a Dataset with temperature fields.
        Returns
        -------
        xr.DataArray or xr.Dataset
            Input data with temperature values converted to Celsius for the specified fields.
            The input type (DataArray or Dataset) is preserved in the output.
        Notes
        -----
        - For DataArrays, the transformation is applied to all values where the "variable"
          coordinate matches the specified fields
        - For Datasets, the transformation is applied directly to the specified field variables
        - All attributes of the input data are preserved
        """
        
        if isinstance(input_ds, xr.DataArray):
            LOG.debug("Transforming an xr.DataArray")
            input_ds = xr.where(input_ds["variable"].isin(self.fields), input_ds + self.correction, input_ds, keep_attrs=True)          
        else:
            LOG.debug("Transforming an xr.Dataset")
            for field in self.fields:
                input_ds[field] = input_ds[field] + self.correction
        
        return input_ds
