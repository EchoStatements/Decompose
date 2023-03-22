"""
A collection of tools to plot bias-variance-diversity curves, points on a simplex and other figures from the paper.

Functions are designed so that many settings can be adjusted via arguments passed to them, though the easiest way to
set many things (e.g., axis ticks, gridlines, etc) is to use standard matplotlib function calls on the axis objects themselves.
"""
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import matplotlib.lines as lines
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib
from sklearn.metrics import zero_one_loss, mean_squared_error

from decompose import BVDExperiment

def _create_decomp_style_dict():
    """
    Parameters
    ----------
    None

    Returns
    -------
    style_dict : dict
        The default style dict for plotting the bias-variance-diversity and bias-variance decomposition_class
        in matplotlib.

    """
    style_dict = {}
    style_dict["average_bias"] = {"linestyle": "--",
                                  "linewidth": 4,
                                  "color": "tab:blue"}

    style_dict["average_variance"] = {"linestyle": "dashdot",
                                      "linewidth": 4,
                                      "color": "tab:orange"}

    style_dict["ensemble_bias"] = {"linestyle": "--",
                                      "linewidth": 4,
                                      "color": "cornflowerblue"}

    style_dict["ensemble_variance"] = {"linestyle": ":",
                                       "linewidth": 4,
                                       "color": "orange"}

    style_dict["diversity"] = {"linestyle": "-",
                               "linewidth": 4,
                               "color": "tab:green"}

    style_dict["ensemble_risk"] = {"linestyle": "solid",
                                  "linewidth": 4,
                                  "color": "indianred"}

    style_dict["expected_member_loss"] = {"linestyle": "-.",
                                  "linewidth": 4,
                                  "color": "indianred"}
    return style_dict

def _create_error_style_dict():
    """

    Returns
    -------
    style_dict : dict
        Default style dict for error plots

    """
    style_dict = {}
    style_dict["test_error"] = {"linestyle": "-",
                                 "linewidth": 1,
                                 "color": "blue"}

    style_dict["train_error"] = {"linestyle": "-.",
                                "linewidth": 1,
                                "color": "dimgrey"}

    style_dict["member_test_error"] = {"linestyle": "--",
                                       "linewidth": 1,
                                       "color": "midnightblue"}

    style_dict["member_train_error"] = {"linestyle": ":",
                                       "linewidth": 1,
                                       "color": "black"}
    return style_dict


def plot_bvd(results,
             bias=True,
             variance=True,
             ensemble_bias=False,
             ensemble_variance=False,
             diversity=True,
             ensemble_risk=True,
             average_risk=False,
             title=None,
             legend=None,
             all_ticks=False,
             x_label=None,
             y_label=None,
             axes=None,
             style_changes={},
             label_size=None,
             title_size=None,
             show_legend=True,
             log_scale=False,
             test_split=0,
             y_lims=None,
             integer_x=False,
             xvalues=None):
    """
    Plots biases, variances and diversity all on the same plot directly from the results object.

    Parameters
    ----------
    results : ResultsObject
        Results object of experiment for which bias-variance-(diversity) decomposition_class is to be plotted. Results dict
        must include keys "parameter_name" and "parameter_values", as well as keys for the quantities which have
        been selected to be plotted.
    bias : boolean, optional (default=True)
        Determines whether the average bias of ensemble members is plotted
    variance : boolean, optional (default=True)
        Determines whether the average variance of the ensemble members is plotted
    ensemble_bias : boolean, optional (default=False)
        Determines if the ensemble bias is plotted
    ensemble_variance : boolean, optional (default=False)
        Determines if the ensemble variance is plotted
    diversity : boolean, optional (default=True)
        Determines whether the diversity is plotted
    ensemble_risk : boolean, optional (default=True)
        Determines if the NLL of the ensemble average is plotted
    average_risk: boolean, optional (default=False)
        Determines if the average NLL of the ensemble models is plotted
    title : string, optional (default=None)
        The figure's title. If None, the figure is produced without a title
    legend : list of strings, optional (default=None)
        If default_legend is None, the default names are used for the plotted curves. Otherwise, default_legend should be a list
        of strings of the same length as the number of curves which are to be plotted, the ordering of the curves is
        as follows: average bias, average variance, ensemble bias, ensemble variance, diversity, ensemble KL
    all_ticks : boolean, optional (default=False)
        whether there should be a tick for each value in experiment_object.param_range. Default behaviour if False
        is matplotlib's standard x-ticks
    x_label : string, optional (default=None)
        If not None, gives the text to be used on the x axis. If None, experiment_object.parameter_name is used
    y_label : string, optional (default=None)
        The label on the y-axis of the plot, if None, label is left blank
    style_changes: dict, optional (default={})
        The entries of this dictionary overwrite/augment the properties of lines
        determined by the default style dictionary. For example `style_changes = {"average_bias": {"alpha" : 0.1}}`
        sets the opacity of the "average_bias" curve to 0.1, while keeping all other properties of all lines the same
        as the default values.
    label_size : int or string, optional (default=None)
        The matplotlib font size to be given to matplotlib for the axis labels (either integer or matplotlib
        fontsize string (e.g. "large"). If None, the default matplotlib value for label font size is used.
    title_size : int or string, optional (default=None)
        The matplotlib font size to be given to matplotlib for the figure title (either integer or matplotlib
        fontsize string (e.g. "large"). If None, the default matplotlib value for label font size is used.
    show_legend : boolean, optional (default=True)
        Determines whether legend is visible on plot
    log_scale : boolean, optional (default=False)
        Determines whether y-axis should have use log-scale
    integer_x : boolean, optional (default=False)
        If True, ticks on the x-axis are forced to have integer values
    y_lims: tuple or None, optional (default=None)
        Set the limits on the upper and lower limits of the y axis. Argument should be a tuple of the form (lower_limit,
        upper_limit).

    Returns
    -------
    ax - Axes
        The axes on which the lines are plotted
    """

    if isinstance(results, BVDExperiment):
        results = results.results_object

    if axes is None:
        fig, ax = plt.subplots()
    else:
        ax = axes
    legend_strings = []
    xvalues = results.parameter_values if xvalues is None else xvalues

    if integer_x:
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    style_dict = _create_decomp_style_dict()
    for key, value in style_changes.items():
        try:
            style_dict[key] = {**style_dict[key], **style_changes[key]}
        except KeyError:
            print(f"style_dict has no entry {key}")

    if all_ticks:
        ax.set_xticks(xvalues, results.parameter_values)

    if ensemble_risk:
        ax.plot(xvalues, results.ensemble_risk[:, test_split], **style_dict["ensemble_risk"], label="expected risk")
        legend_strings.append("expected risk")

    if average_risk:
        ax.plot(xvalues, results.average_bias[:, test_split] + results.average_variance[:, test_split],
                **style_dict["expected_member_loss"], label="average member risk")
        legend_strings.append("average member risk")

    if bias:
        # If average bias does't exist, we assume its an effect decompose and try to get the average bias-effect
        average_bias = results.average_bias if hasattr(results, "average_bias") else results.average_bias_effect
        legend_string = "average bias + noise" if hasattr(results, "average_bias") else "average bias-effect"
        ax.plot(xvalues, average_bias[:, test_split], **style_dict["average_bias"], label=legend_string)
        legend_strings.append(legend_string)

    if variance:
        # If average variance does't exist, we assume its an effect decompose and try to get the average variance-effect
        average_variance = results.average_variance if hasattr(results, "average_variance") else results.average_variance_effect
        legend_string = "average variance" if hasattr(results, "average_variance") else "average variance-effect"
        ax.plot(xvalues, average_variance[:, test_split], **style_dict["average_variance"],
                label=legend_string)
        legend_strings.append(legend_string)

    if ensemble_bias:
        # If ensemble_bias does't exist, we assume its an effect decompose and try to get the ensemble bias-effect
        ensemble_bias = results.ensemble_bias if hasattr(results, "ensemble_bias") else results.ensemble_bias_effect
        legend_string = "ensemble bias" if hasattr(results, "ensemble_bias") else "ensemble bias-effect"
        ax.plot(xvalues, ensemble_bias[:, test_split], **style_dict["ensemble_bias"], label=legend_string)
        legend_strings.append(legend_string)

    if ensemble_variance:
        # If ensemble_variance does't exist, we assume its an effect decompose and try to get the ensemble variance-effect
        ensemble_variance = results.ensemble_variance if hasattr(results, "ensemble_variance") else results.ensemble_variance_effect
        legend_string = "ensemble variance" if hasattr(results, "ensemble_variance") else "ensemble variance-effect"
        ax.plot(xvalues, ensemble_variance[:, test_split], **style_dict["ensemble_variance"], label=legend_string)
        legend_strings.append(legend_string)

    if diversity:
        # If diversity does't exist, we assume its an effect decompose and try to get the diversity-effect
        diversity = results.diversity if hasattr(results, "diversity") else results.diversity_effect
        legend_string = "diversity" if hasattr(results, "diversity") else "diversity-effect"
        ax.plot(xvalues, diversity[:, test_split], **style_dict["diversity"], label=legend_string)
        legend_strings.append(legend_string)

    label_size = label_size if label_size is not None else "medium"

    if show_legend:
        if legend is None:
            ax.legend(legend_strings)
        else:
            ax.legend(legend)
    ax.grid()
    if x_label is None:
        ax.set_xlabel(results.parameter_name, fontsize=label_size)
    else:
        ax.set_xlabel(x_label, fontsize=label_size)
    if title is not None:
        title_size = title_size if title_size is not None else "large"
        ax.set_title(title, fontsize=title_size)

    y_label = y_label if y_label is not None else ""
    ax.set_ylabel(y_label, fontsize=label_size)
    if log_scale:
        ax.set_yscale("log")
    if y_lims is not None:
        if hasattr(y_lims, "len"):
            ax.set_ylim(*y_lims)
        else:
            ax.set_ylim(y_lims)
    return ax

def plot_bv(results,
            bias=True,
            variance=True,
            risk=True,
            title=None,
            all_ticks=False,
            x_label=None,
            y_label=None,
            axes=None,
            style_changes={},
            label_size=None,
            title_size=None,
            show_legend=True,
            log_scale=False,
            test_split=0,
            integer_x=False,
            y_lims=None,
            xvalues=None):
    """
    Plots the bias and variance from a ResultsObject

    Parameters
    ----------
    results : dict
        Results dict of experiment for which bias-variance-(diversity) decomposition_class is to be plotted. Results dict
        must include keys "parameter_name" and "parameter_values", as well as keys for the quantities which have
        been selected to be plotted.
    bias : boolean, optional (default=True)
        Determines whether the average bias of model is plotted
    variance : boolean, optional (default=True)
        Determines whether the average variance of the model is plotted
    risk: boolean, optional (default=False)
        Determines if the NLL of the model is plotted
    title : string, optional (default=None)
        The figure's title. If None, the figure is produced without a title
    legend : list of strings, optional (default=None)
        If default_legend is None, the default names are used for the plotted curves. Otherwise, default_legend should be a list
        of strings of the same length as the number of curves which are to be plotted, the ordering of the curves is
        as follows: average bias, average variance, ensemble bias, ensemble variance, diversity, ensemble KL
    all_ticks : boolean, optional (default=False)
        whether there should be a tick for each value in experiment_object.param_range. Default behaviour if False
        is matplotlib's standard x-ticks
    x_label : string, optional (default=None)
        If not None, gives the text to be used on the x axis. If None, experiment_object.parameter_name is used
    y_label : string, optional (default=None)
        The label on the y-axis of the plot, if None, label is left blank
    style_changes: dict, optional (default={})
        The entries of this dictionary overwrite/augment the properties of lines
        determined by the default style dictionary. For example `style_changes = {"average_bias": {"alpha" : 0.1}}`
        sets the opacity of the "average_bias" curve to 0.1, while keeping all other properties of all lines the same
        as the default values.
    label_size : int or string, optional (default=None)
        The matplotlib font size to be given to matplotlib for the axis labels (either integer or matplotlib
        fontsize string (e.g. "large"). If None, the default matplotlib valueo for label font size is used.
    title_size : int or string, optional (default=None)
        The matplotlib font size to be given to matplotlib for the figure title (either integer or matplotlib
        fontsize string (e.g. "large"). If None, the default matplotlib valueo for label font size is used.
    show_legend : boolean, optional (default=True)
        Determines whether legend is visible on plot
    log_scale : boolean, optional (default=False)
        Determines whether y-axis should have use log-scale
    integer_x : boolean, optional (default=False)
        If True, ticks on the x-axis are forced to have integer values


    Returns
    -------
    ax : Axes
        Axes on which the line are plotted.
    """
    legend = []
    if risk:
        legend.append("expected risk")
    if bias:
        legend.append("bias + noise")
    if variance:
        legend.append("variance")
    return plot_bvd(results, bias=False, variance=False, ensemble_risk=risk,
                    ensemble_bias=bias, ensemble_variance=variance,
                    diversity=False, title=title, all_ticks=all_ticks,
                    legend=legend, x_label=x_label, y_label=y_label, axes=axes,
                    style_changes=style_changes, label_size=label_size,
                    title_size=title_size, show_legend=show_legend, log_scale=log_scale,
                    test_split=test_split, y_lims=y_lims, integer_x=integer_x, xvalues=xvalues)


def plot_errors(results,
                training=True,
                test=True,
                member_training=False,
                member_test=False,
                title=None,
                title_size=None,
                label_size=None,
                legend=None,
                all_ticks=False,
                y_label=None,
                x_label=None,
                axes=None,
                style_changes={},
                test_split=0,
                integer_x=False,
                xvalues=None):
    """
    Plot the average train and test errors of the ensemble over all bootstraps

    Parameters
    ----------
    results : dict
        Results dict of experiment for which bias-variance-(diversity) decomposition_class is to be plotted. Results dict
        must include keys "decomposition_class", "parameter_name", and "parameter_values", as well as keys for the
        quantities which have been selected to be plotted.
    training : boolean, optional (default=True)
        Whether training error is plotted
    test : boolean, optional (default=False)
        Whether test error is plotted
    member_training : boolean, optional (default=False)
    member_test : boolean, optional (default=False)
    title : string, optional (default=None)
    title_size : string or int, optional (default=None)
    label_size : string or int, optional (default=None)
    legend : list of strings, optional (default=None)
    all_ticks: boolean, optional (default=True)
        Whether there should be a tick for each value in experiment_object.parameter_values. If false, matplotlib handles ticks
        in its default manner

    y_label : string, optional (default=None)
        If not None, gives the text to be used on the y axis. If None, either "zero-one decomposition_class" or mean squared error
        is used for classification and regression, respectively.
    x_label : string, optional (default=None)
        If not None, gives the text to be used on the x axis. If None, experiment_object.parameter_name is used
    style_change : dict, optional (default=None)
        Dictionary containing a entries which overwrite the default style for the plot
    test_split: int, optional (default=None)
        The test split on which the curve is to be plotted. By default, test_split is 1 if there are multiple test splits,
        or is 0 if 0 is the only split available.
    integer_x : boolean, optional (default=False)
        If True, ticks on the x-axis are forced to have integer values
    """
    if isinstance(results, BVDExperiment):
        results = results.results_object

    # If we have multiple test splits, the default is to use the second,
    # using the first to plot the decomposition_class. If we only have a single
    # split, it is used for both purposes
    if test_split is None:
        if results.test_error.shape[1] > 1:
            test_split = 1
        else:
            test_split = 0

    if axes is None:
        fig, ax = plt.subplots()
    else:
        ax = axes

    if integer_x:
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    x_values = results.parameter_values if xvalues is None else xvalues
    if all_ticks:
        ax.set_xticks(x_values, results.parameter_values)

    style_dict = _create_error_style_dict()
    style_dict = {**style_dict, **style_changes}

    default_legend = []
    if test:
        default_legend.append("test")
        ax.plot(x_values, results.test_error[:, test_split], **style_dict["test_error"])
    if training:
        default_legend.append("train")
        ax.plot(x_values, results.train_error, **style_dict["train_error"])
    if member_test:
        default_legend.append("member test")
        ax.plot(x_values, results.member_test_error[:, test_split], **style_dict["member_test_error"])
    if member_training:
        default_legend.append("member train")
        ax.plot(x_values, results.member_train_error[:, test_split], **style_dict["member_test_error"])

    if legend is None:
        legend = default_legend
    ax.legend(legend)
    ax.grid()
    ax.set_ylim(0)
    label_size = label_size if label_size is not None else "medium"
    if x_label is None:
        ax.set_xlabel(results.parameter_name, fontsize=label_size)
    else:
        ax.set_xlabel(x_label, fontsize=label_size)
    if y_label is None:
        if results.loss_func == zero_one_loss:
            ax.set_ylabel("zero-one loss", fontsize=label_size)
        elif results.loss_func == mean_squared_error:
            ax.set_ylabel("mean squared error", fontsize=label_size)
    else:
        ax.ylabel(y_label, fontsize=label_size)

    if title is not None:
        title_size = title_size if title_size is not None else "large"
        ax.set_title(title, fontsize=title_size)
    return ax


def _project_onto_simplex(points):
    """
    Helper function to find the coordinates on a 2-d plane to make it look like points in 3d have been projected onto
    2-d simplex

    Parameters
    ----------
    points : ndarray of shape (n_points, 3)
        The coordinates of points in 3d space to be projected on to the 2-d simplex

    Returns
    -------
    tripts : ndarray of shape (n_points, 2)
        The coordinates of the points in 2d so as to appear projected on a 2-d simplex

    """

    # Convert points one at a time
    if len(points.shape) == 1:
        assert points.shape[0] == 3
        points = points.reshape([1, 3])
    tripts = np.zeros((points.shape[0], 2))
    for idx in range(points.shape[0]):
        # Init to triangle centroid
        x = 1.0 / 2
        y = 1.0 / (2 * np.sqrt(3))
        # Vector 1 - bisect out of lower left vertex 
        p1 = points[idx, 0]
        x = x - (1.0 / np.sqrt(3)) * p1 * np.cos(np.pi / 6)
        y = y - (1.0 / np.sqrt(3)) * p1 * np.sin(np.pi / 6)
        # Vector 2 - bisect out of lower right vertex  
        p2 = points[idx, 1]  
        x = x + (1.0 / np.sqrt(3)) * p2 * np.cos(np.pi / 6)
        y = y - (1.0 / np.sqrt(3)) * p2 * np.sin(np.pi / 6)        
        # Vector 3 - bisect out of top vertex
        p3 = points[idx, 2]
        y = y + (1.0 / np.sqrt(3) * p3)
      
        tripts[idx,:] = (x,y)

    return tripts
 

def plot_simplex_3d(points, centroid, ax=None, target=None):
    """

    Parameters
    ----------
    points : ndarray of shape (n_points, 3)
        n_points points which are to be drawn onto the simplex
    centroid : ndarray of shape (3,)
        centroid point
    ax : Axes (default=None)
        The axes on which the simplex is to be drawn.
    target : ndarray of shape (3,) (default=None)
        The target point. If None, no target is drawn.

    Returns
    -------

    """
    if ax is None:
        fig, ax = plt.subplot()

    vtx = np.array([[1,0,0], [0,1,0],[0,0,1]])
    tri = a3.art3d.Poly3DCollection([vtx], zorder=-1)
    # tri.set_color(colors.rgb2hex((0, 0.3, .8)))
    tri.set_color(colors.rgb2hex((54/255., 192/255., 16/255., 0.5)))
    tri.set_edgecolor("darkslategray")
    tri.set_facecolor((0, 0.8, 0.1, 0.5))
    line = a3.art3d.Line3D([0.5, 1/3.], [0.5, 1/3.], [0, 1/3.], linestyle=":", c="g")
    ax.add_line(line)
    line = a3.art3d.Line3D([0.5, 1/3.], [0, 1/3.], [0.5, 1/3.], linestyle=":", c="g")
    ax.add_line(line)
    line = a3.art3d.Line3D([0, 1/3.], [0.5, 1/3.], [.5, 1/3.], linestyle=":", c="g")
    ax.add_line(line)
    ax.add_collection3d(tri)
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="deepskyblue", alpha=1., zorder=1)
    ax.scatter(centroid[0], centroid[1], centroid[2], marker="*", color="blue", edgecolors="black", s=200, alpha=1., zorder=1000)
    if target is not None:
        ax.scatter(target[0], target[1], target[2], marker="*", color="gold", edgecolors="black", s=200, alpha=1., zorder=1000)
    ax.view_init(elev=16., azim=13.)
    
    # Add text
    ax.text(0.1, 0.9, -0.05, "class 2")
    ax.text(1.1, 0.1, -0.05, "class 1")
    ax.text(0., 0., 1.05, "class 3")
    return ax
    
def plot_simplex_2d(points,
                    centroid,
                    axes=None,
                    faux_3d=False,
                    target=None,
                    legend=True,
                    green_edges=False,
                    point_kwargs={},
                    show_text=True):
    """

    Parameters
    ----------
    points : ndarray of shape (n_points, 3)
        n_points points which are to be drawn onto the simplex
    centroid : ndarray of shape (3,)
        centroid point
    axes : Axes (default=None)
        The axes on which the simplex is to be drawn.
    faux_3d : boolean (default=False)
        If True, isometric gridlines are drawn to give a 3d effect
    legend : boolean (default=True)
        Determines whether a legend is draw
    green_edges : boolean (default=False)
        If true, green edges are drawn around the simplex
    point_kwargs : dict (default= {})
        keyword arguments to be passed to scatter() when points are drawn
    show_text : boolean (default=True)
        If true, class labels are shown in the corners of the simplex

    Returns
    -------
    axes : Axes
        The axes on which the simplex has been drawn

    """

    if axes is None:
        fig, axes = plt.subplots()
       
    # The easiest part is plotting the points, so we get that out of the way first, 
    # then worry about making it look pretty. 
    points = _project_onto_simplex(points)
    centroid = _project_onto_simplex(centroid)

    # default point style dict
    point_style = {"color" : "royalblue", "alpha" : .75, "zorder" : 3, "s": 10 }
    # Merge default and passed dictionaries
    point_style = {**point_style, **point_kwargs}
    axes.scatter(points[:, 0], points[:, 1], **point_style)
    axes.scatter(centroid[:, 0], centroid[:, 1], marker="*", color="deepskyblue", edgecolors="black", s=400, alpha=1., zorder=10)
    if target is not None:
        target = _project_onto_simplex(target)
        # axes.scatter(target[:, 0], target[:, 1], marker="o", color=(177/255., 247/255., 158/255., 0.5), edgecolors="black", s=200, zorder=10)
        axes.scatter(target[:, 0], target[:, 1], marker="o", color="gold", edgecolors="black", s=120, alpha=1., zorder=5)


    # Class borders
    axes.add_line(lines.Line2D([.25, 0.5], [np.sqrt(3)/4., 1 / (2 * np.sqrt(3))], linestyle=":", c="g", zorder=2))
    axes.add_line(lines.Line2D([.75, 0.5], [np.sqrt(3)/4., 1 / (2 * np.sqrt(3))], linestyle=":", c="g", zorder=2))
    axes.add_line(lines.Line2D([0.5, 0.5], [0, 1 / (2 * np.sqrt(3))], linestyle=":", c="g", zorder=2))

    triangle = patches.Polygon(np.array([[0., 0], [0.5, np.sqrt(3)/2], [1, 0]]),
                               ec="b", linewidth=5.)
    triangle_collection = PatchCollection([triangle])
    # triangle_collection.set_color((0, 0.8, 0.1, 0.5))
    triangle_collection.set_color((177/255., 247/255., 158/255., 0.5))
    # opaque version of same color
    #
    triangle_collection.set_color("#d8fbce")

    if green_edges:
        # triangle_collection.set_edgecolor((72/255, 255/255., 26/255, 1.))
        triangle_collection.set_edgecolor("darkslategray")
        triangle_collection.set_linewidth(3)
    else:
        triangle_collection.set_edgecolor((0, 0, 0, 1.))
    axes.add_collection(triangle_collection)
    axes.set_xlim(-.1, 1.1)
    axes.set_ylim(-.1, 1.1)
    if show_text:
        axes.text(-.05, -.15, "class 1")
        axes.text(.7, -.15, "class 2")
        axes.text(.4, np.sqrt(3)/2 + .05, "class 3")
    
    if faux_3d:
        #origin lines
        axes.add_line(lines.Line2D([0.5, 0.5], [np.sqrt(3)/2, 1.5], linestyle="-", c="black", zorder=-1))
        axes.add_line(lines.Line2D([1, 1.4], [0, .4 * (-1/np.sqrt(3))], linestyle="-", c="black", zorder=-1))
        axes.add_line(lines.Line2D([0, -1.4], [0, -1.4 * (1/np.sqrt(3))], linestyle="-", c="black", zorder=-1))
        # Corners to centre
        axes.add_line(lines.Line2D([0, 0.5], [0, 1 / (2 * np.sqrt(3))], linestyle="-", c="black", alpha=.1, zorder=1))
        axes.add_line(lines.Line2D([1, 0.5], [0, 1 / (2 * np.sqrt(3))], linestyle="-", c="black", alpha=.1, zorder=1))
        axes.add_line(lines.Line2D([0.5, 0.5], [np.sqrt(3)/2, 1 / (2 * np.sqrt(3))], linestyle="-", c="black", alpha=.1, zorder=1))
 
        
        # Grid lines
        for x in np.arange(-.2, 1.3, 0.1):
            if x < 0.5:
                y_min = x / (np.sqrt(3))
            else:
                y_min = -x / (np.sqrt(3)) + 1 / (np.sqrt(3))
            
            # add vertical lines
            axes.add_line(lines.Line2D([x, x], [y_min, 1.4], linestyle="-", c="lightgray", zorder=-10))
            # sloping down lines
            if x < 0.5:
                axes.add_line(lines.Line2D([x, x + 1.4], [y_min, y_min + 1.4 * (-1/np.sqrt(3))], linestyle="-", c="lightgray", zorder=-10))
            # sloping up lines
            if x > 0.5:
                axes.add_line(lines.Line2D([x, x - 1.4], [y_min, y_min - 1.4 * (1/np.sqrt(3))], linestyle="-", c="lightgray", zorder=-10))
        
        for y in np.arange(1/(2*np.sqrt(3)), 1.6, .2 / ( np.sqrt(3))):
            axes.add_line(lines.Line2D([0.5, 0.5 + 1.4], [y, y + 1.4 * (-1/np.sqrt(3))], linestyle="-", c="lightgray", zorder=-10))
            axes.add_line(lines.Line2D([0.5, 0.5 - 1.4], [y, y - 1.4 * (1/np.sqrt(3))], linestyle="-", c="lightgray", zorder=-10))
        
    axes.set_xticks([])
    axes.set_yticks([])
    if legend:
        legend_elements = [lines.Line2D([0], [0], marker="*", markersize=15, color="w", markerfacecolor='blue', markeredgecolor="black", label='centroid')]
        if target is not None:
            legend_elements.append(lines.Line2D([0], [0], marker="*", markersize=15, color="w", markerfacecolor='gold', markeredgecolor="black", label='target'))
        axes.legend(handles=legend_elements, loc="upper left", prop={'size':7})
    return axes
    
def plot_2d_and_3d_simplex(points, centroid, faux_3d=False, target=None,
                           legend=False):
    """
    Plots 2d and 3d simplex diagrams side-by-side. See `plot_simplex_2d` and `plot_simplex_3d` for descriptions of
    parameters.

    """
    fig = plt.figure(figsize=[10.8, 5.4])
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, aspect="equal")
    plot_simplex_3d(points, centroid, ax1, target=target)
    plot_simplex_2d(points, centroid, ax2, faux_3d=faux_3d, target=target, legend=legend)
 
def plot_dartboard(points, axes=None, legend=True):
    """
    Plots collection of points and their mean on a dartboard figure.

    Parameters
    ----------
    points : ndarray of shape (n_points, 2)
        The points to be drawn
    axes : Axes
        The axes on which dartboard is to be plotted
    legend : boolean (default=True)
        Whether to draw a legend for the dartboard

    Returns
    -------

    """

    if axes is None:
        fig, axes = plt.subplot()

    centroid = points.mean(axis=0)
    # The easiest part is plotting the points, so we get that out of the way first,
    # then worry about making it look pretty.

    axes.scatter(points[:, 0], points[:, 1], color="royalblue", alpha=1., zorder=5, s=10)
    axes.scatter(centroid[0], centroid[1], marker="*", color="deepskyblue", edgecolors="black", s=200, alpha=1., zorder=10)
    # axes.scatter(0, 0, marker="o", color="gold", edgecolors="black", s=400, alpha=.8, zorder=2, linewidth=2)
    # axes.scatter(0, 0, marker="$\\bigotimes$", color="black", edgecolors="black", s=150, zorder=3, linewidth=.4)
    axes.scatter(0, 0, marker="o", color="gold", edgecolors="black", s=200, zorder=2, linewidth=1)

    # Class borders
    # axes.add_line(lines.Line2D([.25, 0.5], [np.sqrt(3)/4., 1 / (2 * np.sqrt(3))], linestyle=":", c="g", zorder=1))
    # axes.add_line(lines.Line2D([.75, 0.5], [np.sqrt(3)/4., 1 / (2 * np.sqrt(3))], linestyle=":", c="g", zorder=1))
    # axes.add_line(lines.Line2D([0.5, 0.5], [0, 1 / (2 * np.sqrt(3))], linestyle=":", c="g", zorder=1))

    circle = patches.Circle(np.array([0., 0]), 1,
                               ec="b", linewidth=5., zorder=-1)
    circle_collection = PatchCollection([circle])
    circle_collection.set_color((177/255., 247/255., 158/255., 0.4))
    circle_collection.set_edgecolor((0, 0, 0, 1.))
    axes.add_collection(circle_collection)


    circle2 = patches.Circle(np.array([0., 0]), 0.5,
                               ec="b", linewidth=5., zorder=0)
    circle_collection = PatchCollection([circle2])
    circle_collection.set_color((0, 0., 0, 0.))
    circle_collection.set_edgecolor((0, 0, 0, 1.))
    axes.add_collection(circle_collection)

    axes.set_xlim(-1.1, 1.1)
    axes.set_ylim(-1.1, 1.1)

    axes.set_xticks([])
    axes.set_yticks([])
    if legend:
        """
        legend_elements = [lines.Line2D([0], [0], marker="*", markersize=15, color="w", markerfacecolor='blue', markeredgecolor="black", label='centroid')]
        if target is not None:
            legend_elements.append(lines.Line2D([0], [0], marker="*", markersize=15, color="w", markerfacecolor='gold', markeredgecolor="black", label='target'))
        axes.legend(handles=legend_elements, loc="upper left", prop={'size':7})
        """
    return axes
