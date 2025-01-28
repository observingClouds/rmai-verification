import xarray as xr
import numpy as np

def valid_to_lead_time(ds):
    assert "valid_time" in ds.coords and "reference_time" in ds.coords, "Need a valid_time and reference_time coordinate"
    ds = ds.assign_coords(
        lead_time=("time", ds.valid_time.data - ds.reference_time.data)
    )
    ds = ds.swap_dims({"time":"lead_time"})

    # Make valid_time a 2 dimensional array for easier merging
    ds = ds.assign_coords(
        valid_time=(
            ["reference_time","lead_time"],
            ds.valid_time.data[np.newaxis,:]
        )
    )
    return ds

def _load(filename,
                    model, runs=None,
                    rename_dict=None ):
    ds=xr.load_dataset(filename).sel(model=model)
    if runs:
        ds = ds.sel(run=runs)
    if rename_dict:
        ds = ds.rename(rename_dict)

    lts=[np.timedelta64(lt,'h') for lt in  ds.lead_time.values]
    runs=[np.timedelta64(int(r),'h') for r in  ds.run.values]
    ds = ds.assign_coords({'lead_time': lts, 'run': runs})
    ds = ds.stack(reference_time=('date','run'))
    rts=[rt[0]+rt[1] for rt in ds.reference_time.values]
    return ds.assign_coords({'reference_time':rts})

def load_obs(filename,
                    model=[],
                    valid_times=None,
                    rename_dict=None,
                    spatial_dimension='location',
                    ):
    ds=_load(filename,model,runs=['00'],rename_dict=rename_dict)
    # transform to valid time
    lts=[lt for lt in ds.lead_time.values if lt < np.timedelta64(24,'h')]
    ds = ds.sel(lead_time=lts)
    ds = ds.stack(valid_time=('reference_time','lead_time'))
    vts=[vt[0]+vt[1] for vt in ds.valid_time.values]
    ds = ds.assign_coords({'valid_time':vts})
    # Select only needed valid_times
    if valid_times is not None:
        ds = ds.sel(valid_time=valid_times)
    ds.attrs['spatial_dimension']=spatial_dimension
    return ds

def load_fc(dates, filename,
                    model=[],
                    rename_dict=None,
                    spatial_dimension='location'
                    ):
    ds=_load(filename,model,rename_dict=rename_dict)
    ds = ds.sel(reference_time=dates)
    ref_times = ds.reference_time.values
    dss=[]
    for ref_time in ref_times:
        ds_0=ds.sel(reference_time=[ref_time])
        vts=[ref_time+lt for lt in ds.lead_time.values]
        ds_0 = ds_0.assign_coords({'valid_time':vts})
        ds_0=valid_to_lead_time(ds_0)
        dss.append(ds_0)
    ds=xr.concat(dss,dim="reference_time")
    ds.attrs['spatial_dimension']=spatial_dimension
    return ds