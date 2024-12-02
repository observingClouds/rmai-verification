import yaml
import argparse
import numpy as np
import xarray as xr
from scores_registry import compute
from datahandler import load_model, get_loader, get_saver

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
        for model, kwargs in forecasts.items():
            print(f"Loading model: {model}")
            data = load_model(**kwargs)
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
            valid_times=self.forecasts.valid_time,
            **kwargs
        )

    def unify_variables(self):
        common_vars = set(self.forecasts).intersection(set(self.observations))
        self.observations = self.observations[list(common_vars)]
        self.forecasts = self.forecasts[list(common_vars)]
        
    
    def compute_scores(self):
        _scores = []
        for score, dim in self.config['scores'].items():
            print(f"Computing {score}.")
            def map_compute(model):
                #TODO fix chunking issue
                ds = compute(self.observations.chunk({dim[0]:-1}), model, score, dim=dim)
                return ds
            _score = self.forecasts.groupby("model").map(map_compute)
            _score = _score.expand_dims(dim={"score": [score]})
            _scores.append(_score)
        self.scores = xr.concat(_scores,dim="score")
    
    def save_scores(self):
        output = self.config.get("output", None)
        if not output:
            pass
        else:
            saver = get_saver(output.get("type",'netcdf'))
            path = output.get("path","scores.nc")
            print(f"Saving scores to {path}")
            self.scores.attrs["config"] = str(self.config)
            saver(self.scores,path)

    
    def run(self):
        self.load_forecasts()
        self.load_observations()
        self.unify_variables()
        self.compute_scores()
        self.save_scores()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='run a verification suite',
                    description="Takes a .yaml configuration and runs some verification.", epilog='The end')

    parser.add_argument('-y', '--yaml')

    args = parser.parse_args()
    cfg_fn=f"{args.yaml}.yaml"

    vs=VerificationSuite(cfg_fn)
    vs.run()
