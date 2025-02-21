# Datastores
## Main class
### `BaseDataStore`
- `dims`: returns a dictionary with where the keys are the names of the dimensions and the values are their respective size

- `vars`: returns a list with all the variables present in the datastore

- `is_observation`: returns `True` if datastore represents observations, `False` if it represents forecats

- `is_point`: returns `True` if datastore represents point-data, False if it represents gridded data

- `data`: returns the actual data in a predescribed format, see `ObsDataStore`,`FcstDataStore`, `GridDataStore` and `PointDataStore` for more information

- `select_variables(variables)`: update the internal state so that `data` will only return `variables`

## Subclasses Determining the spatial component
### `GridDataStore(BaseDataStore)`
The spatial component of a `GridDataStore` can be 1-D (stacked) or 2-D.
A stacked `GridDataStore` should have:
- spatial dimension: `[grid_index]`
- spatial coordinates : `[grid_index(grid_index), longitude(grid_index), latitude(grid_index)]`

An unstacked `GridDataStore` should have:
- spatial dimensions: `[x,y]`
- spatial coorinates `[longitude(x,y), latitude(x,y), x(x), y(y)]`, where `x` and `y` are the coordinates in the reference projection.

Following additional methods must be implemented

- `stacked`: return True if the spatial dimension is 1-D (stacked) or 2-D (unstacked)

- `unstack`: convert the 1D spatial dimension to 2D.

### `PointDataStore(BaseDataStore)`
The spatial component of a `PointDataStore` is always 1-D and should have
- spatial dimension: `[point_index]`
- spatial coordinates: `[point_index(point_index), longitude(point_index), latitude(point_index), code(point_index)]`

## Subclasses Determining the temporal component
### `ObsDataStore(BaseDataStore)`
The temporal component of a `ObsDataStore` is always 1-D and should have
- temporal dimension: `[valid_time]`
- temporal coordinates: `[valid_time(valid_time)]`

Follwing additional methods must be implemented

- select_valid_times(valid_times): subset the data to only contain the specified valid times


### `FcstDataStore(BaseDataStore)`
The temporal component of a `FcstDataStore` is always 2-D and should have
- temporal dimesions: `[reference_time,lead_time]`
- temporal coordinates: `[reference_time(reference_time),lead_time(lead_time),valid_time(reference_time,lead_time)]`

