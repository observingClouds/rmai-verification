import cartopy.crs as ccrs

# Predefined grids
GRIDS = dict(
  cerra=dict(
    projection="lcc",
    projection_kws=dict(
      globe=dict(
        semimajor_axis=6371229.0,
        semiminor_axis=6371229.0,
      ),
      central_longitude=8.0,
      central_latitude=50.0,
      standard_parallels=[50.0, 50.0],
    ),
    grid_kws=dict(
      lower_left=(-17.4859, 20.2923),
      upper_right=(74.1051, 63.7695),
      delta_x=5500.0,
      delta_y=5500.0,
      nx=1069,
      ny=1069
    ),
  ),
  carra_west=dict(
    projection="lcc",
    projection_kws=dict(
        globe=dict(
            semimajor_axis=6371229.0,
            semiminor_axis=6371229.0,
        ),
        central_longitude=324.0,
        central_latitude=72.0,
        standard_parallels=[72.0, 72.0],
    ),
    grid_kws=dict(
        lower_left=(-57.09699999999995, 55.80999999999999),
        upper_right=(37.63491110786931, 77.83194769476704),
        delta_x=2500.0,
        delta_y=2500.0,
        nx=1069,
        ny=1269
    ),
  ),
  carra_east=dict(
    projection="lcc",
    projection_kws=dict(
        globe=dict(
            semimajor_axis=6371229.0,
            semiminor_axis=6371229.0,
        ),
        central_longitude=326.0,
        central_latitude=80.0,
        standard_parallels=[80.0, 80.0],
    ),
    grid_kws=dict(
        lower_left=(-19.407999999999994, 70.13499999999999),
        upper_right=(64.60843181916762, 67.33009041873429),
        delta_x=2500.0,
        delta_y=2500.0,
        nx=789,
        ny=989
    ),
  ),
)

PROJECTIONS=dict(
    lcc=ccrs.LambertConformal,
    latlon=ccrs.PlateCarree,
    PlateCarree=ccrs.PlateCarree,
    Mercator=ccrs.Mercator,
    Orthographic=ccrs.Orthographic
)