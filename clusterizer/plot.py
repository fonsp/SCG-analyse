import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def axis_is_in_datetime_format(axis):
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
    :type dot_colors: color

    """
    if ax is None:
        ax = plt.gca()

    locations = circuit.pd['Location in meters (m)'][circuit.pd_occured]
    times = circuit.pd['Date/time (UTC)'][circuit.pd_occured]
    charges = circuit.pd['Charge (picocoulomb)'][circuit.pd_occured]
    if dot_size_to_charge_ratio is None:
        ax.scatter(x=locations, y=times, s=1, c=dot_colors)
    else:
        ax.scatter(x=locations, y=times, s=charges/dot_size_to_charge_ratio, c=dot_colors)
    ax.set_xlabel("Location")
    ax.set_ylabel("Date")
    ax.set_title("Circuit {0}".format(circuit.circuitnr))


def overlay_warnings(circuit, ax=None):
    """Draw colored lines for every warning in the circuit. Useful when the same axis was used to draw a location time scatter plot.

    :param circuit: Circuit object containing warning series to plot
    :type circuit: class:`clusterizer.circuit.Circuit`

    :param ax: Axes to draw on. Defaults to `plt.gca()`
    :type ax: class:`matplotlib.axes.Axes`, optional
    """
    if ax is None:
        ax = plt.gca()

    # TODO: warn user when overlaying an empty plot
    show_date = axis_is_in_datetime_format(ax.yaxis)

    warningcolors = {'1': 'yellow', '2': 'orange', '3': 'red', 'N': 'green'}
    for i, w in circuit.warning.iterrows():
        loc = w["Location in meters (m)"]
        dates = [w["Start Date/time (UTC)"], w["End Date/time (UTC)"]]
        # Using str key in dict to support 'Noise' warning
        level = str(w["SCG warning level (1 to 3 or Noise)"])
        if show_date:
            ax.plot([loc, loc], dates, linewidth=8, c=warningcolors[level], alpha=.7, solid_capstyle="butt")
        else:
            ax.axvline(loc, linewidth=8, c=warningcolors[level], alpha=.7)


def overlay_cluster(cluster, ax=None, color=None):
    """Draw a shaded rectangle matching the cluster dimensions. Useful when the same axis was used to draw a location time scatter plot.

    :param circuit: Cluster object with time or location bounds defined.
    :type circuit: class:`clusterizer.cluster.Cluster`

    :param ax: Axes to draw on. Defaults to `plt.gca()`
    :type ax: class:`matplotlib.axes.Axes`, optional

    :param color: Fill color
    :type color: color
    """
    if ax is None:
        ax = plt.gca()

    # TODO: warn user when overlaying an empty plot
    show_date = cluster.time_range is not None and axis_is_in_datetime_format(ax.yaxis)

    loc = cluster.location_range
    dates = cluster.time_range
    if show_date:
        overlay_boolean_series(loc, True, ax, y1=dates[0], y2=dates[1], color=color)
    else:
        overlay_boolean_series(loc, True, ax, color=color)


def overlay_boolean_series(loc, values, ax=None, y1=None, y2=None, color='yellow', opacity=.3):
    if ax is None:
        ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    y1 = ymin if y1 is None else y1
    y2 = ymax if y2 is None else y2
    ax.fill_between(loc, y1=y1, y2=y2, where=values, color=color, alpha=opacity)


def draw_location_hist(circuit, ax=None, bins=None, color='black'):
    if ax is None:
        ax = plt.gca()
    if bins is None:
        bins = np.arange(0, circuit.circuitlength, 4)
    counts, _, _ = ax.hist(circuit.pd["Location in meters (m)"], bins=bins, color=color)
    ax.set_xlabel("Location (m)")
    ax.set_ylabel("Number of PDs")
