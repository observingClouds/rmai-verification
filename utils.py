import yaml
import xarray as xr
import numpy as np

### yaml files ###
def write_yaml(x,fn,sort=False):
    with open (fn,'w') as f:
        yaml.dump(x,f,sort_keys=False)

def load_yaml(fn):
    with open(fn,'r') as f:
        return yaml.safe_load(f)
    
### time stuff ##
## utils

def to_timedelta64(s):
    value = s[:-1]
    unit = s[-1]
    return np.timedelta64(value,unit)

def time_range(cfg, delta=False):
    times=[]
    start = cfg['start']
    end = cfg['end']
    freq = to_timedelta64(cfg['frequency'])
    
    if not delta:
        start = np.datetime64(start)
        end = np.datetime64(end)
        missing = [np.datetime64(m) for m in cfg.get('missing',[])]
    else:
        start = to_timedelta64(start)
        end = to_timedelta64(end)
        missing = [to_timedelta64(m) for m in cfg.get('missing',[])]
    
    time = start
    while time <= end:
        times.append(time)
        time += freq
    times = [time for time in times if not time in missing]
    return times

def ref_lead_to_valid(ref_times,lead_times):
    vts=np.array([[rt+lt for lt in lead_times] for rt in ref_times])
    da=xr.DataArray(vts, dims=['reference_time', 'lead_time'], coords = {'reference_time' : ref_times, 'lead_time' : lead_times})
    return da

def prepare_time(time_cfg):
    reference_times = time_range(time_cfg['reference_times'])
    lead_times = time_range(time_cfg['lead_times'], delta=True)
    valid_times = ref_lead_to_valid(reference_times, lead_times)
    return {'reference_time': reference_times, 'lead_time' : lead_times, 'valid_time' : valid_times}