import argparse
import numpy as np
import xarray as xr
from data import get_loader
from scores_registry import compute
from datahandler import get_saver
from visualization import plot_variables_overview
from post_processing import post_processor
from utils import load_yaml
from datetime import datetime
import os
#import dask.distributed
from dask.distributed import Client, LocalCluster

def prepare_dates(date_cfg):
    dates=[]
    start = np.datetime64(date_cfg["start"])
    end = np.datetime64(date_cfg["end"])
    value = date_cfg["frequency"][:-1]
    unit = date_cfg["frequency"][-1]
    frequency= np.timedelta64(value,unit)
    date = start
    files = []
    while date <= end:
        dates.append(date.astype(datetime))
        date += frequency
    return dates

class DataGroup():
    def __init__(self, group_dict, fc_config, obs_config, dates, variables):
        self.data_names=group_dict.pop('data')
        self.post_processing=group_dict
        self.fc_config={ name : kwargs for name, kwargs in fc_config.items() if name in self.data_names}
        self.obs_config={ name : kwargs for name, kwargs in obs_config.items() if name in self.data_names}
        self.dates = dates
        self.variables = variables

    def load_forecasts(self):
        models = []
        #TODO rewrite in seperate function
        for model, kwargs in self.fc_config.items():
            loader = kwargs.pop("type")
            load_model = get_loader(loader)
            # Get the path-format
            path_fmt = kwargs.pop("path")

            ## below no longer needed, dates read at higher level
            # Get all the dates information
            # dates = kwargs.pop("dates")
            # start = np.datetime64(dates["start"])
            # end = np.datetime64(dates["end"])
            # value = dates["frequency"][:-1]
            # unit = dates["frequency"][-1]
            # frequency= np.timedelta64(value,unit)

            # Get reshaping information
            reshape = kwargs.pop("reshape",False)
            nx = kwargs.pop("nx",None)
            ny = kwargs.pop("ny",None)
            grid_mapping = kwargs.pop("grid_mapping",None)

            # List all files
            files = []
            for date_dt in self.dates:
                #date_dt = date.astype(datetime)
                path = path_fmt.format(
                    yyyy=date_dt.strftime("%Y"),
                    yy=date_dt.strftime("%y"),
                    mm=date_dt.strftime("%m"),
                    dd=date_dt.strftime("%d"),
                    HH=date_dt.strftime("%H"),
                    MM=date_dt.strftime("%M"),
                    SS=date_dt.strftime("%S"),
                )
                if not os.path.isfile(path):
                    print(f"Warning no file for date {date_dt.strftime('%Y%m%d %H:%M')}, skipping file")
                else:
                    files.append(path)

            # Actual loading
            print(f"Loading model: {model}")
            data = load_model(files)
            if 'model' not in data.dims:
                data = data.expand_dims(model=[model])
            models.append(data)
        # Use merge to not throw away variables that are not common?
        self.forecasts = xr.merge(models) 

    def load_observations(self):
        observations = self.obs_config
        print(observations)
        assert len(observations) == 1, "Loading of multiple observation types per data group not implemented yet"
        (obs_name, kwargs) = next(iter(observations.items()))
        # Get the model specific loader
        loader = get_loader(kwargs.pop("type"))

        # Get the path
        path = kwargs.pop("path")
        self.observations = loader(
            filename=path,
        #   valid_time=self.forecasts["valid_time"], Not working ATM
            **kwargs
            )

    #   Keep only those variables that are needed
    def unify(self):
        # Observations is a data-array
        if "variable" in self.observations.dims:
            common_vars = list(set(self.forecasts).intersection(set(self.observations["variable"].values)))
            if not self.variables == "all":
                common_vars = [var for var in common_vars if var in self.variables]
            obs_reduced = self.observations.sel(
                valid_time=self.forecasts["valid_time"],
                variable = common_vars
            ).to_dataset(
                dim="variable"
            ).chunk(
                {
                    "reference_time": 2,
                    "lead_time": -1,
                    "values": -1
                }
            )
        # Observations is a dataset
        else:
            common_vars = list(set(self.forecasts).intersection(set(self.observations)))
            if not self.variables == "all":
                common_vars = [var for var in common_vars if var in self.variables]
            obs_reduced = self.observations["commom_vars"].sel(
                valid_time=self.forecasts["valid_time"]
            ).chunk(
                {
                    "reference_time": 2,
                    "lead_time": -1,
                    "values": -1
                }
            )
        print(common_vars)
        fcst_reduced = self.forecasts[common_vars].chunk(
            {
                "reference_time": 2,
                "lead_time": -1,
                "values": -1
            }
        )
        #TODO: Warn if there are asked variables not common
        self.forecasts, self.observations =  (fcst_reduced, obs_reduced)

    def post_process(self):
        self.forecasts, self.observations = post_processor(self.post_processing, self.forecasts, self.observations)


    def load(self):
        self.load_forecasts()
        self.load_observations()
        self.unify()
        self.post_process()

class VerificationSuite():
    def   __init__(self,fn):
        self.config = load_yaml(fn)


    def prepare_workflow(self):
        # create list of dates from config
        self.dates=prepare_dates(self.config['dates'])
        # get list of variables if specified
        self.variables=self.config.get('variables','all')
        # set up data groups
        self.data_groups=self.get_data_groups()
        # set up verification
        self.verification = self.config['verification']
        # set up visualization
        self.visualization = self.config['visualization']
        # set up output
        self.output = self.config['output']
        
    def get_data_groups(self):
        fc_config=self.config.get('forecasts',{})
        obs_config=self.config.get('observations',{})
        
        data_types=list(fc_config)+list(obs_config)
        
        post_processing_groups=self.config.get('post_processing',[])
        post_processed_data=[]
        data_groups=[]
        
        for group_dict in post_processing_groups:
            post_processed_data += group_dict['data']
            data_groups.append(DataGroup(group_dict,fc_config,obs_config,self.dates,self.variables))
            
        trivial_data=[data for data in data_types if data not in post_processed_data]
        if trivial_data:
            trivial_group = DataGroup({'data':trivial_data},fc_config,obs_config,self.dates,self.variables)
            data_groups.append(trivial_group)
        
        return data_groups
    
    def load_data(self):
        fcs=[]
        obs=[]
        for data_group in self.data_groups:
            data_group.load()
            fcs.append(data_group.forecasts)
            obs.append(data_group.observations)
        self.forecasts=xr.merge(fcs)
        self.observations=xr.merge(obs)
        
    
    def compute_scores(self):
        spatial_dimension = self.forecasts.attrs.get('spatial_dimension', 'values')
        verification_type = self.config["verification"].get("type","temporal")
        if verification_type == "temporal":
            self.avg_dims = [spatial_dimension]
        elif verification_type == "spatial":
            self.avg_dims = ["reference_time"]
        else:
            raise KeyError(f"Verification type {verification_type} not supported (yet).")
        #TODO: Unify metrics to upper or lower case
        self.metrics = self.config["verification"].get("metrics",["RMSE", "BIAS"])        
        # chunking = dict()
        # for dim in self.avg_dims:
        #     chunking[dim] = -1
        _scores = []

        for metric in self.metrics:
            print(f"Computing {metric}.")
            def map_compute(model):
                #TODO fix chunking issue
                ds = compute(
                    self.observations,
                    model,
                    metric,
                    dim=self.avg_dims)
                return ds
            
            _score = self.forecasts.groupby("model").map(map_compute)
            _score = _score.expand_dims(dim={"metric": [metric]})
            _scores.append(_score)
        self.scores = xr.concat(_scores,dim="metric").compute()
    
    def save_scores(self):
        output = self.config.get("output", None)
        if not output:
            pass
        saver = get_saver(output.get("type",'netcdf'))
        path = output.get("path","scores.nc")
        print(f"Saving scores to {path}")
        self.scores.attrs["config"] = str(self.config)
        saver(self.scores,path)

    def plot_scores(self):
        visualization = self.config.get("visualization",None)
        if not visualization:
            pass
   
        metrics = visualization.pop("metrics",self.metrics)
        if metrics == "all":
            metrics = self.metrics
        for metric in metrics:
            print(f"Plotting metric: {metric}.")
            plot_variables_overview(self.scores, metric, **visualization)
        
            
    def run(self):
        self.prepare_workflow()
        self.load_data()
        self.compute_scores()
        self.save_scores()
        self.plot_scores()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='run a verification suite',
                    description="Takes a .yaml configuration and runs some verification.", epilog='The end')

    parser.add_argument('-y', '--yaml')

    args = parser.parse_args()
    cfg_fn=f"{args.yaml}.yaml"

    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=1,
        processes=True,
    )
    client = Client(cluster)
    vs=VerificationSuite(cfg_fn)
    vs.run()
    client.close()
    cluster.close()
