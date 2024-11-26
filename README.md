A number of scores for a number of forecasts can be computed as specified in `some_suite_config.yaml` by running
```bash
python verification_suite.py -y some_suite_config
```
On Lumi this can be tested using `example_suite.yaml` which is contained in this repository. On other servers some paths will need to be changed. In a bit more than 2 minutes a file `scores.nc` of around 600MB should be generated. 
The scores contained in this file can be inspected and plotted using `plot_scores.ipynb` which is also included in the repository. 
