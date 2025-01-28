import argparse
import numpy as np
import xarray as xr
from data import get_loader
from scores_registry import compute
from datahandler import get_saver
from visualization import plot_variables_overview
from post_processing import post_processor
from utils import load_yaml, prepare_time
from datetime import datetime
import os
#import dask.distributed
from dask.distributed import Client, LocalCluster

# utils
  

def get_variables(data):
        # data is a data-array
        if "variable" in data.dims:
            vrs=list(data["variable"].values)
        # data is a dataset
        else:
            vrs=list(data)
        return vrs
    
def reduce_and_chunck(data, times, vrs):
    spatial_dimension = data.attrs['spatial_dimension']
    # data is a data-array
    if "variable" in data.dims:
        data_reduced = data.sel(variable = vrs).to_dataset(dim="variable")
    # data is a dataset
    else: 
        data_reduced = data[vrs]
    return data_reduced.sel(**times
            ).chunk(
            {
                "reference_time": 2,
                "lead_time": -1,
                spatial_dimension: -1
                }
            )


## Data Group
class DataGroup():
    def __init__(self, group_dict, fc_config, obs_config, times, variables):
        self.data_names=group_dict.pop('data')
        self.post_processing=group_dict
        self.fc_config={ name : kwargs for name, kwargs in fc_config.items() if name in self.data_names}
        self.obs_config={ name : kwargs for name, kwargs in obs_config.items() if name in self.data_names}
        self.times = times
        self.variables = variables

    def load_forecasts(self):
        models = []
        #TODO rewrite in seperate function
        for model, kwargs in self.fc_config.items():
            loader = kwargs.pop("type")
            load_model = get_loader(loader)
            # Get the path-format
            path_fmt = kwargs.pop("path")

            # Actual loading
            print(f"Loading model: {model}")
            data = load_model(self.times['reference_time'], path_fmt, **kwargs)
            if 'model' not in data.dims:
                data = data.expand_dims(model=[model])
            models.append(data)
        # Use merge to not throw away variables that are not common?
        self.forecasts = xr.merge(models) 

    def load_observations(self):
        self.observations = xr.Dataset()
        observations = self.obs_config
        if len(observations) > 0:
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
        vrs=self.variables
        fc_vars = get_variables(self.forecasts)
        obs_vars = get_variables(self.observations)
        if vrs != 'all':
            fc_vars = [v for v in fc_vars if v in vrs]
            obs_vars = [v for v in obs_vars if v in vrs]
        elif len(fc_vars) > 0 and len(obs_vars) > 0:
            fc_vars = [ v for v in fc_vars if v in obs_vars]
            obs_vars = [ v for v in obs_vars if v in fc_vars]
        
        fc_times = { key : value for key, value in self.times.items() if key in ['reference_time', 'lead_time']}
        obs_times = { key : value for key, value in self.times.items() if key in ['valid_time']}
        
        if fc_vars:
            self.forecasts = reduce_and_chunck(self.forecasts, fc_times, fc_vars)
        if obs_vars:
            self.observations = reduce_and_chunck(self.observations, obs_times, obs_vars)

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
        time_info = { key : val for key,val in self.config.items() if key in ['reference_times', 'lead_times'] }
        self.times = prepare_time(time_info)
        # get list of variables if specified
        self.variables=self.config.get('variables','all')
        # set up data groups
        self.data_groups=self.get_data_groups()
        # set up verification
        self.verification = self.config['verification']
        # set up visualization
        self.visualization = self.config.get("visualization")
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
            data_groups.append(DataGroup(group_dict,fc_config,obs_config,self.times,self.variables))
            
        trivial_data=[data for data in data_types if data not in post_processed_data]
        if trivial_data:
            trivial_group = DataGroup({'data':trivial_data},fc_config,obs_config,self.times,self.variables)
            data_groups.append(trivial_group)
        
        return data_groups
    
    def load_data(self):
        fcs=[]
        obs=[]
        common_vars=[]
        for data_group in self.data_groups:
            data_group.load()
            fc=data_group.forecasts
            ob=data_group.observations
            fc_vars = list(fc)
            ob_vars = list(ob)
            if fc_vars:
                if common_vars:
                    common_vars = [v for v in common_vars if v in fc_vars]
                else:
                    common_vars = fc_vars
                fcs.append(fc)
            if ob_vars:
                ob_name = list(data_group.obs_config)[0] #this needs to be changed if we ever allow more than one obs per data_group
                ob = ob.expand_dims({'name':[ob_name]})
                if common_vars:
                    common_vars = [v for v in common_vars if v in ob_vars]
                else:
                    common_vars = ob_vars
                obs.append(ob)
        self.forecasts=xr.merge(fcs)[common_vars]
        self.observations=xr.merge(obs)[common_vars]
        
    
    def compute_scores(self):
        spatial_dimension = self.forecasts.attrs.get('spatial_dimension', 'values')

        fcs = self.forecasts
        obs = self.observations
        
        # handle case with multiple observation types
        obs_names=list(obs.coords['name'].values)

        if len(obs_names) > 1:
            ref_model = self.verification.get('reference_model')
            assert ref_model in obs_names, 'When more than one set of observations is provided, a reference model needs to be specified.'
            obs_2_fc = obs.drop_sel(name = ref_model)
            obs_2_fc = obs_2_fc.rename({'name':'model'})
            fcs = xr.merge([fcs,obs_2_fc])
            obs = obs.sel(name = ref_model)
        
        obs = obs.drop_vars('name')

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
                    obs,
                    model,
                    metric,
                    dim=self.avg_dims)
                return ds
            
            _score = fcs.groupby("model").map(map_compute)
            _score = _score.expand_dims(dim={"metric": [metric]})
            _scores.append(_score)
        self.scores = xr.concat(_scores,dim="metric").compute()
    
    def save_scores(self):
        output = self.config.get("output")
        if output:
            saver = get_saver(output.get("type",'netcdf'))
            path = output.get("path","scores.nc")
            print(f"Saving scores to {path}")
            self.scores.attrs["config"] = str(self.config)
            saver(self.scores,path)

    def plot_scores(self):
        visualization = self.visualization
        if visualization:
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
