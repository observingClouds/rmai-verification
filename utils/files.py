from datetime  import datetime
import yaml
import argparse
import os 
import logging

LOG = logging.getLogger(__name__)

def get_filenames(path_fmt,start,end,frequency):
    filenames = []
    date = start
    while date <= end:
        date_dt = date.astype(datetime)
        path = path_fmt.format(
            yyyy=date_dt.strftime("%Y"),
            yy=date_dt.strftime("%y"),
            mm=date_dt.strftime("%m"),
            dd=date_dt.strftime("%d"),
            HH=date_dt.strftime("%H"),
            MM=date_dt.strftime("%M"),
            SS=date_dt.strftime("%S"),
        )
        if not os.path.exists(path):
            LOG.warning(f"No file found for date {date_dt.strftime('%Y%m%d %H:%M')}, skipping file")
            pass
        else:
            filenames.append(path)
        date += frequency
    filenames = list(set(filenames))
    if len(filenames) == 1: 
        filenames = filenames[0]
    return filenames

def write_yaml(x,fn,sort=False):
    with open (fn,'w') as f:
        yaml.dump(x,f,sort_keys=False)

def load_yaml(fn):
    with open(fn,'r') as f:
        return yaml.safe_load(f)
    

def yaml_file_type(filename):
    """Custom argparse type for YAML file validation."""
    if not os.path.isfile(filename):
        raise argparse.ArgumentTypeError(f"File '{filename}' does not exist.")
    if not (filename.endswith(".yaml") or filename.endswith(".yml")):
        raise argparse.ArgumentTypeError("Config file must have a .yaml or .yml extension.")
    
    # Try to load YAML file to check for syntax errors
    try:
        with open(filename, "r") as f:
            yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise argparse.ArgumentTypeError(f"Invalid YAML file: {e}")

    return filename  # Return the valid filename