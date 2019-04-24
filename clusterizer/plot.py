import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def axis_is_in_datetime_format(axis):
    """Used to check whether a previous plot on the `ax` was a time series.
    Note that `axis` is not the same as `ax`, but rather the x or y axis of `ax` (_axes_).
    """
    return type(axis.get_major_formatter()) == pd.plotting._converter.PandasAutoDateFormatter


def draw_location_time_scatter(circuit, ax=None, dot_size_to_charge_ratio=1e4, dot_colors="black"):
    """Draw a location (x) vs time (y) scatter plot.

    :param circuit: Circuit object containing PD series to plot
    :type circuit: class:`clusterizer.circuit.Circuit`

    :param ax: Axes to draw on. Defaults to `plt.gca()`
    :type ax: class:`matplotlib.axes.Axes`, optional

    :param dot_size_to_charge_ratio: Conversion factor: picocoulomb/pixel. Set to `None` to draw all PDs the same size.
    :type dot_size_to_charge_ratio: float, optional

    :param dot_colors: A single color or a list of colors to use as dot colors.
    :type dot_colors: color, optional

    """
    if ax is None:
        ax = plt.gca()

    locations = circuit.pd['Location in meters (m)'][circuit.pd_occured]
    times = circuit.pd['Date/time (UTC)'][circuit.pd_occured]
    charges = circuit.pd['Charge (picocoulomb)'][circuit.pd_occured]
    if dot_size_to_charge_ratio is None:
        ax.scatter(x=locations, y=times, s=1, c=dot_colors)
    else:
        ax.scatter(x=locations, y=times, s=charges/dot_size_to_charge_ratio, c=dot_colors, label="Circuit {0}".format(circuit.circuitnr))
    ax.set_xlabel("Location (m)")
    ax.set_ylabel("Date")


def draw_location_hist(circuit, weigh_charges=False, ax=None, bins=None, color='black'):
    """Draw a histogram of PD locations.

    :param circuit: Circuit object containing PD series to plot
    :type circuit: class:`clusterizer.circuit.Circuit`

    :param weigh_charges: When set to `True`, PD charges are accumulated. otherwise PD occurences are counted.
    :type weigh_charges: bool, optional

    :param bins: When set to `None` (the default value), uniformly spaced bins of width 4 meters are used. If `bins` is a sequence, it defines a monotonically increasing array of bin edges, including the rightmost edge, allowing for non-uniform bin widths.
    :type bins: Union[list,numpy.ndarray], optional

    :param ax: Axes to draw on. Defaults to `plt.gca()`
    :type ax: class:`matplotlib.axes.Axes`, optional

    :param color: Bar color
    :type color: color, optional

    :return: Array of histogram values, corresponding to accumulated bin content.
    :rtype: numpy.ndarray
    """
    if ax is None:
        ax = plt.gca()
    if bins is None:
        bins = np.arange(0, circuit.circuitlength+4.0, 4.0)
    hist_weights = None
    if weigh_charges:
        hist_weights = circuit.pd["Charge (picocoulomb)"]

    counts, _, _ = ax.hist(circuit.pd["Location in meters (m)"], weights=hist_weights, bins=bins, color=color, label="Circuit {0}".format(circuit.circuitnr))
    ax.set_xlabel("Location (m)")
    ax.set_ylabel("Number of PDs")


def overlay_warnings(circuit, ax=None, opacity=.3, line_width=None, add_to_legend=True):
    """Draw colored lines for every warning in the circuit. Useful when the same axis was used to draw a location time scatter plot.

    :param circuit: Circuit object containing warning series to plot
    :type circuit: class:`clusterizer.circuit.Circuit`

    :param ax: Axes to draw on. Defaults to `plt.gca()`
    :type ax: class:`matplotlib.axes.Axes`, optional

    :param opacity: Fill opacity (1=opaque; 0=invisible)
    :type opacity: float, optional

    :param line_width: Width of warning line. Defaults to 1% of circuit length.
    :type line_width: float, optional

    :param add_to_legend: Label warning colors?
    :type add_to_legend: bool, optional
    """
    if ax is None:
        ax = plt.gca()

    # TODO: warn user when overlaying an empty plot
    show_date = axis_is_in_datetime_format(ax.yaxis)

    if line_width is None:
        line_width = circuit.circuitlength * 0.01

    warningcolors = {'1': 'yellow', '2': 'orange', '3': 'red', 'N': 'green'}
    colors_added_to_legend = set()
    for i, w in circuit.warning.sort_values(by=['SCG warning level (1 to 3 or Noise)']).iterrows():
        loc = w["Location in meters (m)"]
        dates = [w["Start Date/time (UTC)"], w["End Date/time (UTC)"]]
        # Using str key in dict instead of int to support 'Noise' warning
        level = str(w["SCG warning level (1 to 3 or Noise)"])
        label = "Warning " + level if (level not in colors_added_to_legend and add_to_legend) else None
        colors_added_to_legend.add(level)
        if show_date:
            # Draw a single line
            ax.plot([loc, loc], dates, linewidth=line_width, c=warningcolors[level], alpha=opacity, solid_capstyle="butt", label=label)
        else:
            # Draw a complete vertical line
            ax.axvline(loc, linewidth=line_width, c=warningcolors[level], alpha=opacity, solid_capstyle="butt", label=label)


def overlay_cluster_collection(clusters, ax=None, color=None, opacity=.3):
    """Draw shaded rectangles matching the cluster dimensions. Useful when the same axis was used to draw a location time scatter plot.

    :param circuit: Cluster objects with time or location bounds defined.
    :type circuit: list of class:`clusterizer.cluster.Cluster`

    :param ax: Axes to draw on. Defaults to `plt.gca()`
    :type ax: class:`matplotlib.axes.Axes`, optional

    :param color: Fill color
    :type color: color, optional

    :param opacity: Fill opacity (1=opaque; 0=invisible)
    :type opacity: float, optional
    """
    for c in clusters:
        overlay_cluster(c, ax, color, opacity)


def overlay_cluster(cluster, ax=None, color=None, opacity=.3, label=None):
    """Draw a shaded rectangle matching the cluster dimensions. Useful when the same axis was used to draw a location time scatter plot.

    :param circuit: Cluster object with time or location bounds defined.
    :type circuit: class:`clusterizer.cluster.Cluster`

    :param ax: Axes to draw on. Defaults to `plt.gca()`
    :type ax: class:`matplotlib.axes.Axes`, optional

    :param color: Fill color
    :type color: color, optional

    :param opacity: Fill opacity (1=opaque; 0=invisible)
    :type opacity: float, optional

    :param label: Label to add to the legend
    :type label: str, optional
    """
    if ax is None:
        ax = plt.gca()

    # TODO: warn user when overlaying an empty plot
    show_date = cluster.time_range is not None and axis_is_in_datetime_format(ax.yaxis)

    loc = list(cluster.location_range)
    dates = cluster.time_range
    if show_date:
        overlay_boolean_series([True, True], loc=loc, ax=ax, y1=dates[0], y2=dates[1], color=color, opacity=opacity, label=label)
    else:
        overlay_boolean_series([True, True], loc=loc, ax=ax, color=color, opacity=opacity, label=label)


def overlay_boolean_series(values, loc=None, ax=None, y1=None, y2=None, color='yellow', opacity=.3, label=None):
    """Overlay a series of vertical colored stripes at locations in `loc` where the corresponding element of `values` is truthy.

    :param loc: Locations to draw boolean values. (If the plot contains binned data, `bins[:-1]` would be a logical choice.) Defaults to equal division of the x-axis.
    :type loc: Union[list,numpy.ndarray], optional

    :param ax: Axes to draw on. Defaults to `plt.gca()`
    :type ax: class:`matplotlib.axes.Axes`, optional

    :param y1: Lower edge of shaded area. Defaults to `ax.get_ylim()[0]`.
    :type y1: Union[float, list, numpy.ndarray], optional

    :param y2: Upper edge of shaded area. Defaults to `ax.get_ylim()[1]`.
    :type y2: Union[float, list, numpy.ndarray], optional

    :param color: Fill color
    :type color: color, optional

    :param opacity: Fill opacity (1=opaque; 0=invisible)
    :type opacity: float, optional

    :param label: Label to add to the legend
    :type label: str, optional
    """
    if ax is None:
        ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    y_lower = ymin if y1 is None else y1
    y_upper = ymax if y2 is None else y2

    if loc is None:
        loc = np.linspace(xmin, xmax, num=len(values))

    ax.fill_between(loc, y1=y_lower, y2=y_upper, where=values, color=color, alpha=opacity, label=label)

    # Omdat het gekleurde gebied misschien doorloopt tot de boven- en onderkanten van het plotgebied, wordt het plotgebied door matplotlib automatisch vergroot om dit te laten passen. Dit doen we ongedaan:
    ax.set_ylim(ymin, ymax)
