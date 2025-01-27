import yaml
import argparse
import numpy as np
import xarray as xr
from data import get_loader
from scores_registry import compute
from datahandler import get_saver
from visualization import plot_variables_overview
from datetime import datetime
import os
import dask.distributed
from dask.distributed import Client, LocalCluster

### yaml files ###
def write_yaml(x,fn,sort=False):
    with open (fn,'w') as f:
        yaml.dump(x,f,sort_keys=False)

def load_yaml(fn):
    with open(fn,'r') as f:
        return yaml.safe_load(f)

class VerificationSuite():
    def   __init__(self,fn):
        self.config = load_yaml(fn)

    def load_forecasts(self):
        forecasts = self.config["forecasts"]
        models = []
        #TODO rewrite in seperate function
        for model, kwargs in forecasts.items():
            loader = kwargs.pop("type")
            load_model = get_loader(loader)
            # Get the path-format
            path_fmt = kwargs.pop("path")

            # Get all the dates information
            dates = kwargs.pop("dates")
            start = np.datetime64(dates["start"])
            end = np.datetime64(dates["end"])
            value = dates["frequency"][:-1]
            unit = dates["frequency"][-1]
            frequency= np.timedelta64(value,unit)

            # Get reshaping information
            reshape = kwargs.pop("reshape",False)
            nx = kwargs.pop("nx",None)
            ny = kwargs.pop("ny",None)
            grid_mapping = kwargs.pop("grid_mapping",None)

            # List all files
            date = start
            files = []
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
                if not os.path.isfile(path):
                    print(f"Warning no file for date {date_dt.strftime('%Y%m%d %H:%M')}, skipping file")
                else:
                    files.append(path)
                date += frequency

            # Actual loading
            print(f"Loading model: {model}")
            data = load_model(files)
            data = data.expand_dims(model=[model])
            models.append(data)
        # Use merge to not throw away variables that are not common?
        self.forecasts = xr.merge(models) 

    def load_observations(self):
        observations = self.config["observations"]
        assert len(observations) == 1, "Verification against multiple obersvation types not implemented yet"
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
            if not self.config["verification"].get("variables","all") == "all":
                common_vars = [var for var in common_vars if var in self.config["verification"]["variables"]]
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
            if not self.config["verification"].get("variables","all") == "all":
                common_vars = [var for var in common_vars if var in self.config["verification"]["variables"]]
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
        return (fcst_reduced, obs_reduced)
    
    def compute_scores(self):
        fcst, obs = self.unify()
        verification_type = self.config["verification"].get("type","temporal")
        if verification_type == "temporal":
            self.avg_dims = ["values"]
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
                    obs,
                    fcst,
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
        self.load_forecasts()
        self.load_observations()
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
