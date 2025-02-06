import abc

class BaseDataStore(abc.ABC):
    """
    Base class
    """
    @property
    @abc.abstractmethod
    def dim_names(self):
        """Get the dimension names of the DataStore

        Returns:
            List[str]: The names of the variables
        """
        pass
    
    @property
    @abc.abstractmethod
    def dims(self):
        """Get the dimensions of the DataStore
        
        Returns:
            tuple: the dimensions of the data
        """
        pass

    @property
    @abc.abstractmethod
    def vars(self):
        """Returns the variables in the datastore
        
        Returns:
            List: Variables in the datastore
        """
        pass

    @property
    @abc.abstractmethod
    def observation(self):
        """Returns True if observations datastore
        
        Observation datastores only contain valid_times,
        while forecast (also) contain reference_time and lead_times.

        Returns:
            Boolean: True if datastore contains observations
        """

    @property
    @abc.abstractmethod
    def data(self):
        """Returns the xr.Dataset or xr.DataArray in a prediscribed format

        Returns:
            xr.Dataset or xr.DataArray
        """
        pass

    @abc.abstractmethod
    def select_variables(self,variables):
        """Selects variables from the data

        Returns: 
            None
        """
        pass
    
    def transform(self,transformation):
        self._data = transformation.execute(self._data)

    
class GridDatastore(BaseDataStore):
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

class PointDatastore(BaseDataStore):
    @abc.abstractmethod
    def longitudes(self):
        """Return the longitudes of the points
        
        Returns: 
            List: list with longitudes of the points in the datastore
        """
        pass

    @abc.abstractmethod
    def latitudes(self):
        """Return the latitudes of the points
        
        Returns: 
            List: list with latitudes of the points in the datastore
        """
        pass


    # @abc.abstractmethod
    # def load(self):
    #     """(Lazy) load the data and return xr.Dataset or xr.DataArray"""
    #     pass
    
