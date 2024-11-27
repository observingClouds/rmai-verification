import yaml
import argparse
import numpy as np
import xarray as xr
from scores_registry import compute
from datahandler import load_data, save_data, get_loader
from datetime import datetime

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

    def load_model(self,**kwargs):
        # Get the model specific loader
        loader = get_loader(kwargs.pop("type"))

        # Get the path-format
        path_fmt = kwargs.pop("path")

        # Get all the dates information
        dates = kwargs.pop("dates")
        start = np.datetime64(dates["start"])
        end = np.datetime64(dates["end"])
        value = dates["frequency"][:-1]
        unit = dates["frequency"][-1]
        frequency= np.timedelta64(value,unit)

        # Start loop over dates
        date = start
        model = []
        while date <= end:
            kwargs["reference_time"] = date.astype('datetime64[ns]')
            print(f"    - {date}")
            # Some path-formatting (Can probably be done better)
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
            model.append(
                loader(
                    filename=path,
                    **kwargs
                )
            )
            date += frequency
        model = xr.concat(model,dim="reference_time")
        return(model)

    def load_forecasts(self):
        forecasts = self.config["forecasts"]
        models = []
        for model, kwargs in forecasts.items():
            print(f"Loading model: {model}")
            data = self.load_model(**kwargs)
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
        
    
    def compute_scores2(self):
        for score, dim in self.config['scores'].items():
            print(f"Computing {score}.")
            def map_compute(model):
                #TODO fix chunking issue
                ds = compute(self.observations.chunk({dim[0]:-1}), model, score, dim=dim)
                return ds
            ds_score = self.forecasts.groupby("model").map(map_compute)
            print(ds_score)


    def load_dataset(self):
        dss_models=[]
        first_instance=True
        for model, model_cfg in self.config['models'].items():
            dss_fc_time=[]
            pth=model_cfg['path']
            type=model_cfg['type']
            kwargs=model_cfg.get('kwargs',{})
            for time, name in model_cfg['filenames'].items():
                print(f"Prepping data for {model}, fc_time {time}.")
                fn=f"{pth}/{name}"
                fc_time=np.datetime64(time)
                ds=load_data(fn,fc_time,type,kwargs)
                if first_instance:
                    lts = ds.lead_time.values
                    vrs = list(ds)
                    first_instance = False
                dss_fc_time.append(ds)
            ds=xr.concat(dss_fc_time,dim='fc_time')
            lts = [lt for lt in lts if lt in ds.lead_time.values]
            vrs = [v for v in vrs if v in list(ds)]
            dss_models.append(ds.expand_dims({'model': [model]}))
        dss_models = [ds[vrs].sel(lead_time=lts) for ds in dss_models]
        self.dataset=xr.concat(dss_models, dim='model')
        print(f"Data ready")

    def compute_scores(self):
        self.load_dataset()
        ref_model=self.config['reference_model']
        ds=self.dataset
        models=ds.model.values
        assert ref_model in models, f"The reference model {ref_model} is not among the listed models."
        fc_models=[model for model in models if model != ref_model]
        ds_ref=ds.sel(model=ref_model).expand_dims({'model' : fc_models})
        ds_fc=ds.sel(model=fc_models)
        dss_scores=[]
        for score, dim in self.config['scores'].items():
            print(f"Computing {score}.")
            ds_score=compute(ds_ref, ds_fc, score, dim=dim)
            dss_scores.append( ds_score.expand_dims({'score':[score]}) )
        self.scores=xr.concat(dss_scores,dim='score')
        print('All scores computed')
    
    def run(self):
        self.compute_scores()
        output = self.config.get('output', None)
        if output is not None:
            _type=output.get('type','netcdf')
            path=output.get('path','scores.nc')
            print(f"Saving scores to {path}")
            ds=self.scores
            ds.attrs={'verification_suite_config' : str(self.config)}
            save_data(ds,path,_type)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='run a verification suite',
                    description="Takes a .yaml configuration and runs some verification.", epilog='The end')

    parser.add_argument('-y', '--yaml')

    args = parser.parse_args()
    cfg_fn=f"{args.yaml}.yaml"

    vs=VerificationSuite(cfg_fn)
    vs.run()
