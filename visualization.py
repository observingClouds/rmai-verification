import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import xskillscore as xs
import xarray as xr
import numpy as np

DEFAULT_VARS = ["z_500","t_500","u_500","v_500"]

## UTILS ##
def get_plotter(type):
    assert type in PLOT_REGISTRY, f"The datatype {type} is not (yet) supported."
    return PLOT_REGISTRY[type]


def add_ci(dataset, grid, **kwargs):
    if "capsize" not in kwargs:
        kwargs["capsize"] = 5
    color_mapping = dict()
    for i, entry in enumerate(grid.fig.legends[0].texts):
        model = entry.get_text()
        color = grid.fig.legends[0].get_lines()[i].get_color()
        color_mapping[model] = color
    x = grid.data.loc[grid.name_dicts.flat[0]].lead_time.values

    for i, ax in enumerate(grid.axs.flat):
        for model, color in color_mapping.items():
            if grid.name_dicts.flat[i]:
                y = grid.data.loc[grid.name_dicts.flat[i]].sel(model=model).data
                ci = dataset.sel(grid.name_dicts.flat[i]).sel(model=model).data
                ax.errorbar(x=x, y=y, yerr=ci, color=color, marker="o", **kwargs)
    return grid
        
## PLOT FUNCTIONS ##
def plot_variables_overview(scores, metric, **kwargs):

    vars = kwargs.pop("variables","base")
    confidence_intervals = kwargs.pop("confidence_intervals",False)
    
    # Select the needed variables
    if isinstance(vars, str):
        assert vars in ["base", "all"], f"Vars keyword {vars} not supported"
        if vars == "base":
            data = scores[DEFAULT_VARS]
        else:
            data = scores
    else:
        data = scores[vars]
    data = data.sel(metric=metric)
    
    dims = list(scores.sizes.keys())
    
    # Some consistency checking
    if "x" in dims:
        assert "y" in dims, "The dataset only has 1 spatial dimension, y is missing"
    if "y" in dims:
        assert "x" in dims, "The dataset only has 1 spatial dimension, x is missing"
    
    if "x" in dims or "y" in dims:
        # Prepare for spatial overview
        plotter = get_plotter("spatial")
        plot_dims = ["x","y","model"]
        avg_dims = [item for item in dims if item not in plot_dims]
        assert len(avg_dims) == 1, "More than 1 avg_dims found!"
        avg_dim = avg_dims[0]
        data = scores.group_by("model").mean(avg_dim)
        plotter = get_plotter("spatial")
    else:
        # Prepare for temporal overview
        # FIXME: avg_dim is now fixed to "reference_time"
        # You could also want to plot things w.r.t. the time of the day for instance
        # Should we include these options here is this to specific?
        plotter = get_plotter("temporal")
        avg_dim = "reference_time"
        if confidence_intervals:
            data = xs.resample_iterations(
                data,
                iterations=1000,
                dim=avg_dim,
                replace=True
            ).compute()

            _mean = data.mean([avg_dim,"iteration"]).to_dataarray(dim="variable")
            _confidence_interval = data.mean(
                avg_dim
            ).quantile(
                q=[0.05,0.95],
                dim="iteration"
            ).to_dataarray(
                dim="variable"
            )
            data = xr.Dataset(
                {
                    "mean" : _mean,
                    "confidence_interval": _confidence_interval,
                }
            )
        else:
            _mean = data.mean(avg_dim).to_dataarray(dim="variable")
            data = xr.Dataset({"mean": _mean})
    
    plotter(data,**kwargs)

def plot_spatial_overview(data, **kwargs):
    pass
    #data.to_dataarray.plot(x="x",y="y")

def plot_temporal_overview(data, **kwargs):

    rows_per_page = kwargs.pop("rows_per_page",2)
    cols_per_page = kwargs.pop("cols_per_page",3)
    defaults = dict(
        figsize=(17,10),
        aspect=16/9,
        marker="o",
    )
    for key, value in defaults.items():
        if key not in kwargs:
            kwargs[key] = value
        
    plots_per_page = rows_per_page * cols_per_page
    total_plots = len(data["mean"])
    total_pages = -(-total_plots // plots_per_page) # Ceiling division (thanks chatGPT)

    xticks = data["lead_time"].values.astype(np.float64)
    xticklabels = data["lead_time"].values/np.timedelta64(1,"h")
    if "confidence_interval" in data.keys():
        ci = data["confidence_interval"]
    else:
        ci = None
    data = data["mean"]

    filename = f"{data['metric'].values}-overview.pdf"
    with PdfPages(filename) as pdf:
        for page in range(total_pages):
            # Determine the subset of data to plot on this page
            start = page * plots_per_page
            end = min(start + plots_per_page, total_plots)
            g = data.isel(
                variable=slice(start, end)
            ).plot(
                    x="lead_time",
                    hue="model",
                    col="variable",
                    col_wrap=cols_per_page,
                    sharey=False,
                    **kwargs,
            )
            if ci is not None:
                g = add_ci(ci,g)

            # Nice xticks
            for ax in g.axes.flat:  # Loop over all subplot axes
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)
            
            g.set_xlabels("lead time [h]")
            g.set_ylabels(f"{data['metric'].values}")

            # Save the current page to the PDF
            pdf.savefig(g.fig)
            plt.close(g.fig)  # Close the figure to avoid overlapping plots

## REGISTRIES ##
PLOT_REGISTRY = {
    "temporal" : plot_temporal_overview,
    "spatial" : plot_spatial_overview,
}


    
    

# def plot_leadtime(ds,**kwargs):
#     assert len(ds.dims) == 2, "plot_timeseries works only with 2D data"
#     if isinstance(ds, xr.DataArray):
#         ds.plot(hue="model")
#         ax = plt.gca()
#         ax.set_title(ds.name)
#         ax.set_ylabel(f"{ds.score.data} []")
#         if "lead_time" in ds.coords:
#             xticks = ds.lead_time.values.astype(np.float64)
#             xticklabels = ds.lead_time.values/np.timedelta64(1,"h")
#             ax.set_xticks(xticks)
#             ax.set_xticklabels(xticklabels)
#             ax.set_xlabel("lead time [h]")
#     elif isinstance(ds, xr.Dataset):
#         g = ds.to_dataarray().plot(col="variable",col_wrap=4,sharey=False)
#         g_


    