from datetime  import datetime
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