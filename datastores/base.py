import abc
import xarray as xr
import numpy.typing as npt
from typing import Dict, List, Any


class BaseDataStore(abc.ABC):
    """
    Base class
    """
    @property
    def dims(self) -> Dict[str, int]:
        """Get the names and size of the dimensions of the DataStore
        
        Returns:
            Dict: the dimension names and sisze of the data
        """
        return dict(self._data.sizes)

    @property
    def vars(self) -> List[str]:
        """Returns the variables in the datastore
        
        Returns:
            List: Variables in the datastore
        """
        return list(self._data.keys())
    
    @property
    def longitudes(self) -> npt.NDArray:
        """Returns the longitudes of the datastore
        
        Returns
            ndarray: 
        """
        return self._data["longitude"].values

    @property
    def latitudes(self) -> npt.NDArray:
        """Returns the latitudes of the datastore
        
        Returns
            ndarray: 
        """
        return self._data["latitude"].values

    @property
    def valid_times(self) -> xr.DataArray:
        """Returns the valid times of the datastore
        
        Returns
            xr.DataArray: for ObsDatastore a 1D data-array, 
            for FcstDataStore a 2D data-array 
        """
        return self._data["valid_time"]


    @property
    def is_observation(self) -> bool:
        """Returns True if the datastore contains observations, 
        False if it contains forecasts

        Returns:
            bool: True for observation datastore, False for forecast datastore
        """
        return self._is_observation

    @property
    def is_point(self) -> bool:
        """Returns True if the datastore contains point-based data, 
        False if it contains grid-based data

        Returns:
            bool: True for point-based datastore, False for grid-based datastore
        """
        return self._is_point

    @property
    def data(self) -> xr.Dataset | xr.DataArray:
        """Returns the xr.Dataset or xr.DataArray in a prediscribed format

        Returns:
            xr.Dataset or xr.DataArray
        """
        return self._data

    @abc.abstractmethod
    def select_variables(self,variables):
        """Selects variables from the data

        Returns: 
            None
        """
        pass
    
    def transform(self,transformation):
        new_data = transformation.execute(self._data)
        self._data = new_data

    
class GridDataStore(BaseDataStore):

    _is_point: bool = False

    @property
    def is_stacked(self) -> bool:
        """Returns true if the data is stacked (1-D)
        
        Returns:
            Boolean: True if data in the GridDataStore is stacked (1-D)
        """
        return self._stacked
    
    
    @abc.abstractmethod
    def unstack(self):
        """Unstack the 1D spatial dimension to 2D

        Returns:
            None
        """
        pass
    
    # @abc.abstractmethod
    # def stack(self):
    #     """Stack the 2D spatial dimension to 1D
        
    #     Returns:
    #         None
    #     """
    #     pass

class PointDataStore(BaseDataStore):
    _is_point: bool = True


class ObsDataStore(BaseDataStore):
    _is_observation: bool = True

    @property
    def valid_times(self):
        return self._data["valid_time"].values

    @abc.abstractmethod
    def select_valid_times(self,valid_times: List | xr.DataArray):
        """Subsets the data in the datastore to only contain 
        the selected valid_times

        Returns:
            None
        """
        pass

class FcstDataStore(BaseDataStore):
    _is_observation: bool = False

    @abc.abstractmethod
    def select_reference_times(self,reference_times: List | xr.DataArray):
        """Subsets the data in the datastore to only contain 
        the selected reference times

        Returns:
            None
        """
        pass

    @abc.abstractmethod
    def select_lead_times(self,lead_times: List | xr.DataArray):
        """Subsets the data in the datastore to only contain 
        the selected lead times

        Returns:
            None
        """
        pass
    
