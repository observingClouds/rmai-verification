import cartopy.crs as ccrs
import yaml

with open("grid_mappings.yaml") as stream:
    try:
        MAPPINGS = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

PROJECTIONS = {
    "lcc" : ccrs.LambertConformal,  
}

def get_cartopy_crs(grid_mapping):
        # If native domain is a string, get the specifications from the pre-defined mappings
    if isinstance(grid_mapping, str):
        assert grid_mapping in MAPPINGS, f"Grid mapping {grid_mapping} not supported, please provide a dictionary with the specifications"
        grid_mapping = dict(MAPPINGS[grid_mapping])
    # Build the cartopy CRS    
    assert grid_mapping["projection"] in PROJECTIONS, f"Projection {grid_mapping['projection']} not supported (yet)."
    projection = PROJECTIONS[grid_mapping["projection"]]
    kwargs = dict(grid_mapping["projection_kws"])
    globe = kwargs.pop("globe", None)
    if globe:
        globe = ccrs.Globe(**globe)
    crs = projection(globe=globe, **kwargs)
    return crs

def map_grid(dataset, grid_mapping):
    # Cartopy CRS
    crs = get_cartopy_crs(grid_mapping)

    # Transform the lat-lons to native format
    assert "longitude" in dataset.coords and "latitude" in dataset.coords, "dataset must contain longitudes and latitudes, building domain from domain edges not implemented (yet)."
    longitude = dataset.longitude.values
    latitude = dataset.latitude.values
    xy = crs.transform_points(
        src_crs=ccrs.PlateCarree(),
        y=latitude,
        x=longitude
    )
    x = xy[0,:,0]
    y = xy[:,0,1]

    # Assign as new coordinates
    #TODO: not all coordinates have meters as unit!
    dataset = dataset.assign_coords(
        x=(
            "x",
            x,
            dict(
                units="meter",
                description="x-coordinate",
            )
        ),
        y=(
            "y",
            y,
            dict(
                units="meter",
                description="y-coordinate",
            )
        )
    )

    # Add CRS information to the attributes, usefull for regridding an plotting
    dataset.attrs["grid_mapping"] = crs
    return dataset