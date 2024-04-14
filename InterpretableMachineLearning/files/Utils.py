import numpy as np
from mlxtend.plotting import plot_decision_regions


def create_decision_plot(
        X,
        y,
        model,
        feature_index,
        feature_names,
        X_highlight,
        filler_feature_values,
        filler_feature_ranges=None,
        ax=None,
):
    if feature_names is None:
        feature_names = feature_index

    if list(np.arange(X.shape[1])) != list(filler_feature_values.keys()):
        fillers = {}
        fillers.update({X.columns.get_loc(key): vals \
                        for key, vals in filler_feature_values.items()})
        filler_feature_values = fillers

    filler_feature_keys = list(filler_feature_values.keys())
    feature_index = [X.columns.get_loc(k) for k in feature_index if k not in filler_feature_keys]
    filler_values = {k: filler_feature_values[k] for k in filler_feature_values.keys() if k not in feature_index}

    new_x = X.to_numpy()
    if filler_feature_ranges is None:
        filler_vals = np.array(list(filler_feature_values.values()))
        filler_rangs = np.vstack([np.abs(np.amax(new_x, axis=0) - filler_vals), \
                                  np.abs(np.amin(new_x, axis=0) - filler_vals)])
        filler_rangemax = np.amax(filler_rangs, axis=0)
        filler_rangemax = list(np.where(filler_rangemax == 0, 1, filler_rangemax))
        filler_feature_ranges = {i: v for i, v in enumerate(filler_rangemax)}

    filler_ranges = {k: filler_feature_ranges[k] for k in filler_feature_ranges.keys() if k not in feature_index}

    ax = plot_decision_regions(
        new_x,
        y.to_numpy(),
        clf=model,
        feature_index=feature_index,
        X_highlight=X_highlight,
        filler_feature_values=filler_values,
        filler_feature_ranges=filler_ranges,
        scatter_kwargs={'s': 48, 'edgecolor': None, 'alpha': 0.7},
        contourf_kwargs={'alpha': 0.2}, legend=2, ax=ax)
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])

    return ax