import numpy as np
import matplotlib.pyplot as plt
from matplotlib import lines, patches
from pyFAI.units import RADIAL_UNITS 
from bokeh.plotting import figure 
from bokeh.models import ColorBar, LinearColorMapper, HoverTool, ColumnDataSource  
from bokeh.palettes import Viridis256 
from bokeh.models.annotations import Label

from bayfai.geometry import pix2q, radial_profile

def plot_radial_integration(q, profile, error, calibrant=None, label=None, ax=None):
    """
    Plot the radial integration of a powder image

    Parameters
    ----------
    q : np.array
        Array of q values
    profile : np.array
        Array of intensity values
    error : np.array
        Array of intensity errors if provided
    calibrant : Calibrant
        Calibrant object
    label : str
        Name of the curve
    ax : plt.Axes
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    unit = RADIAL_UNITS["q_A^-1"]
    if error is not None:
        ax.errorbar(q, profile, error, label=label)
    else:
        ax.plot(q, profile, label=label, color="black", linewidth=0.8)

    if label:
        ax.legend(fontsize=8)
    if calibrant and unit:
        x_values = calibrant.get_peaks(unit)
        if x_values is not None:
            for x in x_values:
                line = lines.Line2D(
                    [x, x],
                    ax.axis()[2:4],
                    color="red",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.7,
                )
                ax.add_line(line)

    ax.set_title("Radial Profile", fontsize=6)
    if unit:
        ax.set_xlabel(unit.label, fontsize=6)
    ax.set_ylabel("Intensity", fontsize=6)
    ax.tick_params(axis="x", labelsize=4)
    ax.tick_params(axis="y", labelsize=4)

def plot_score_distance_scan(scan, distances, thrsh, ax):
    """
    Plot the score scan over distance

    Parameters
    ----------
    scan : dict
        Scan data containing scores and threshold
    distances : np.array
        Array of distances
    thrsh : float
        Threshold value
    ax : plt.Axes
        Matplotlib axes
    """
    scores = scan["score"]
    ax.plot(distances, scores)
    ax.axhline(
        thrsh,
        color="red",
        linestyle="--",
        label=f"Threshold score: {thrsh}",
    )
    ax.legend(fontsize=6)
    ax.set_xlabel("Distance (m)", fontsize=6)
    ax.set_ylabel("Score", fontsize=6)
    ax.tick_params(axis="x", labelsize=4)
    ax.tick_params(axis="y", labelsize=4)
    ax.set_title("Number of Control Points vs Distance", fontsize=6)

def plot_residual_distance_scan(scan, distances, refined_dist, ax):
    """
    Plot the residual scan over distance

    Parameters
    ----------
    scan : dict
        Scan data containing residuals
    distances : np.array
        Array of distances
    refined_dist : float
        Refined distance
    ax : plt.Axes
        Matplotlib axes
    """
    residuals = scan["residual"]
    ax.plot(distances, residuals)
    best_dist = distances[scan["index"]]
    ax.axvline(
        best_dist,
        color="green",
        linestyle="--",
        label=f"Best distance (m): {best_dist:.3f}",
    )
    ax.axvline(
        refined_dist,
        color="red",
        linestyle="--",
        label=f"Refined distance (m): {refined_dist:.3f}",
    )
    ax.legend(fontsize=6)
    ax.set_yscale("log")
    ax.set_xlabel("Distance (m)", fontsize=6)
    ax.set_ylabel("Residual", fontsize=6)
    ax.tick_params(axis="x", labelsize=4)
    ax.tick_params(axis="y", labelsize=4)
    ax.set_title("Residual vs Distance", fontsize=6)

def plot_hist_and_compute_stats(powder, exp, run, ax):
    """
    Plot histogram of pixel intensities and compute statistics

    Parameters
    ----------
    powder : np.ndarray
        Powder image
    exp : str
        Experiment name
    run : int
        Run number
    ax : plt.Axes
        Matplotlib axes
    """
    threshold = np.mean(powder) + 3 * np.std(powder)
    nice_pix = powder < threshold
    mean = np.mean(powder[nice_pix])
    std_dev = np.std(powder[nice_pix])
    _ = ax.hist(
        powder[nice_pix],
        bins=200,
        color="skyblue",
        edgecolor="black",
        alpha=0.7,
        label="Pixel Intensities",
        orientation="horizontal",
    )
    ax.axhline(
        mean,
        color="red",
        linestyle="--",
        label=f"Mean ({mean:.2f})",
    )
    ax.axhline(
        mean + std_dev,
        color="orange",
        linestyle="--",
        label=f"Mean + Std Dev ({mean + std_dev:.2f})",
    )
    ax.axhline(
        mean + 2 * std_dev,
        color="green",
        linestyle="--",
        label=f"Mean + 2 Std Dev ({mean + 2 * std_dev:.2f})",
    )
    ax.axhline(
        np.percentile(powder[nice_pix], 95),
        color="purple",
        linestyle=":",
        linewidth=1.5,
        label=f"95 th Percentile ({np.percentile(powder[nice_pix], 95):.2f})",
    )
    ax.set_xlim([0, 100000])
    ax.set_ylim([0, mean + 3 * std_dev])
    ax.set_ylabel("Pixel Intensity", fontsize=6)
    ax.set_xlabel("Frequency", fontsize=6)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.tick_params(axis="y", labelsize=4)
    ax.set_title(
        f"Histogram of Pixel Intensities \n for {exp} run {run}", fontsize=6
    )
    ax.legend(fontsize=6)

def create_interactive_powder(powder, detector, distance, wavelength):
    """
    Create an interactive powder image with control points and calibrated rings.

    Parameters
    ----------
    powder : np.ndarray
        Powder image
    detector : PyFAI(Detector)
        Corrected PyFAI detector object
    distance : float
        Refined distance
    """
    y, x, z = detector.calc_cartesian_positions()
    if z is None:
        z = np.zeros_like(x)
    z += distance

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    if xmin < 0 and ymin < 0 and xmax > 0 and ymax > 0:
        xlim = (xmin * 1.1, xmax * 1.1)
        ylim = (ymin * 1.1, ymax * 1.1)
    elif xmin < 0 and ymin < 0 and xmax < 0 and ymax < 0:
        xlim = (xmin * 1.1, xmax * 0.9)
        ylim = (ymin * 1.1, ymax * 0.9)
    elif xmin < 0 and ymin > 0 and xmax > 0 and ymax > 0:
        xlim = (xmin * 1.1, xmax * 1.1)
        ylim = (ymin * 0.9, ymax * 1.1)
    elif xmin > 0 and ymin < 0 and xmax > 0 and ymax > 0:
        xlim = (xmin * 0.9, xmax * 1.1)
        ylim = (ymin * 1.1, ymax * 1.1)
    elif xmin < 0 and ymin < 0 and xmax > 0 and ymax < 0:
        xlim = (xmin * 1.1, xmax * 1.1)
        ylim = (ymin * 1.1, ymax * 0.9)
    elif xmin < 0 and ymin < 0 and xmax < 0 and ymax > 0:
        xlim = (xmin * 1.1, xmax * 0.9)
        ylim = (ymin * 1.1, ymax * 1.1)
    elif xmin < 0 and ymin > 0 and xmax < 0 and ymax > 0:
        xlim = (xmin * 1.1, xmax * 0.9)
        ylim = (ymin * 0.9, ymax * 1.1)
    elif xmin > 0 and ymin < 0 and xmax > 0 and ymax < 0:
        xlim = (xmin * 0.9, xmax * 1.1)
        ylim = (ymin * 1.1, ymax * 0.9)
    elif xmin > 0 and ymin > 0 and xmax > 0 and ymax > 0:
        xlim = (xmin * 0.9, xmax * 1.1)
        ylim = (ymin * 0.9, ymax * 1.1)

    p = figure(
        title=f"Run {run} - {detname} - {calibrant_name}",
        x_axis_label="X-axis (m)",
        y_axis_label="Y-axis (m)",
        width=1200,
        height=1200,
        match_aspect=True,
        x_range=xlim,
        y_range=ylim,
    )

    vmin, vmax = np.percentile(powder, 5), np.percentile(powder, 95)
    color_mapper = LinearColorMapper(palette=Viridis256, low=vmin, high=vmax)

    source = ColumnDataSource(
        data={"x": x.ravel(), "y": y.ravel(), "intensity": powder.ravel()}
    )

    _ = p.circle(
        x="x",
        y="y",
        size=1,
        color={"field": "intensity", "transform": color_mapper},
        line_color=None,
        source=source,
    )

    color_bar = ColorBar(
        color_mapper=color_mapper, width=8, location=(0, 0), title="Intensity"
    )
    p.add_layout(color_bar, "right")

    tth = calibrant.get_2th()
    x = np.reshape(x, detector.raw_shape)
    y = np.reshape(y, detector.raw_shape)
    z = np.reshape(z, detector.raw_shape)

    for i in range(detector.n_modules):
        ttha = np.arctan2(np.sqrt(x[i] * x[i] + y[i] * y[i]), z[i])
        p.contour(
            x=x[i],
            y=y[i],
            z=ttha,
            levels=tth,
            line_color="red",
            line_width=3,
            line_dash="dashed",
        )

    cx, cy = 0, 0
    d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    closest_pixel_index = np.argmin(d)
    closest_pixel = d.flatten()[closest_pixel_index]
    closest_q = pix2q(closest_pixel, distance, wavelength)
    closest_resol = 2 * np.pi / closest_q

    furthest_pixel_index = np.argmax(d)
    furthest_pixel = d.flatten()[furthest_pixel_index]
    furthest_q = pix2q(furthest_pixel, distance, wavelength)
    furthest_resol = 2 * np.pi / furthest_q

    d_left = abs(cx - xmin)
    d_right = abs(cx - xmax)
    d_bottom = abs(cy - ymin)
    d_top = abs(cy - ymax)
    border_distances = [d_left, d_right, d_bottom, d_top]
    border_pixel = max(border_distances)
    border_q = pix2q(border_pixel, distance)
    border_resol = 2 * np.pi / border_q
    border_2_q = pix2q(border_pixel / 2, distance)
    border_2_resol = 2 * np.pi / border_2_q

    circles_data = [
        (closest_pixel, closest_resol),
        (furthest_pixel, furthest_resol),
        (border_pixel, border_resol),
        (border_pixel / 2, border_2_resol),
    ]

    for radius, resol in circles_data:
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = cx + radius * np.cos(theta)
        circle_y = cy + radius * np.sin(theta)
        p.line(
            circle_x, circle_y, line_color="green", line_dash="dashed", line_width=3
        )
        text_x = cx + radius / np.sqrt(2)
        text_y = cy + radius / np.sqrt(2)

        label_annotation = Label(
            x=text_x,
            y=text_y,
            text=f"{resol:.3f} Å",
            text_color="red",
            text_font_size="16pt",
        )
        p.add_layout(label_annotation)

    hover = HoverTool(
        tooltips=[
            ("x", "@x{0.000}"),
            ("y", "@y{0.000}"),
            ("Intensity", "@intensity{0.0}"),
        ]
    )
    p.add_tools(hover)

    p.title.text_font_size = "12pt"
    p.xaxis.axis_label_text_font_size = "10pt"
    p.yaxis.axis_label_text_font_size = "10pt"
    p.xaxis.major_label_text_font_size = "8pt"
    p.yaxis.major_label_text_font_size = "8pt"

    return (
        p,
        closest_q,
        closest_resol,
        furthest_q,
        furthest_resol,
        border_q,
        border_resol,
    )

def create_diagnostics_panel(
    powder,
    detector,
    distance,
    low_resolution=None,
    high_resolution=None,
    border_resolution=None,
    plot="",
):
    """
    Create a diagnostics panel with the results of the Bayesian Optimization.

    Parameters
    ----------
    powder : np.ndarray
        Powder image
    detector : PyFAI(Detector)
        Corrected PyFAI detector object
    distance : float
        Refined distance
    low_resolution : float, optional
        Lowest resolution value, if available
    high_resolution : float, optional
        Highest resolution value, if available
    border_resolution : float, optional
        Border resolution value, if available
    plot : str
        Path to save plot
    """
    fig = plt.figure(figsize=(6, 9), dpi=100)
    nrow, ncol = 3, 2
    irow, icol = 0, 0

    # Labelling experiment and run number
    ax1 = plt.subplot2grid((nrow, ncol), (irow, icol))
    rect = patches.Rectangle(
        (0, 0),
        1,
        1,
        transform=ax1.transAxes,
        color="lightgrey",
        alpha=0.3,
    )
    ax1.add_patch(rect)
    ax1.text(
        0.05,
        0.9,
        f"Experiment {exp}",
        ha="left",
        va="center",
        fontsize=8,
    )
    ax1.text(0.05, 0.8, f"Run {run}", ha="left", va="center", fontsize=8)
    ax1.text(
        0.05, 0.7, f"Detector {det_name}", ha="left", va="center", fontsize=8
    )
    ax1.text(
        0.05,
        0.6,
        f"Calibrant {calibrant_name}",
        ha="left",
        va="center",
        fontsize=8,
    )
    ax1.text(
        0.05,
        0.5,
        f"Distance = {distance:.4f} m",
        ha="left",
        va="center",
        fontsize=8,
    )
    if low_resolution is not None:
        ax1.text(
            0.05,
            0.4,
            f"{'Low-q Resolution':<30}",
            ha="left",
            va="center",
            fontsize=8,
            color="black",
        )
        ax1.text(
            0.50,
            0.4,
            f"{low_resolution:.3f} \u00c5",
            ha="left",
            va="center",
            fontsize=8,
            color="red",
        )
        ax1.text(
            0.05,
            0.3,
            f"{'Border Resolution':<30}",
            ha="left",
            va="center",
            fontsize=8,
            color="black",
        )
        ax1.text(
            0.50,
            0.3,
            f"{border_resolution:.3f} \u00c5",
            ha="left",
            va="center",
            fontsize=8,
            color="red",
        )
        ax1.text(
            0.05,
            0.2,
            f"{'Corner Resolution':<30}",
            ha="left",
            va="center",
            fontsize=8,
            color="black",
        )
        ax1.text(
            0.50,
            0.2,
            f"{high_resolution:.3f} \u00c5",
            ha="left",
            va="center",
            fontsize=8,
            color="red",
        )
    ax1.axis("off")
    icol += 1

    # Plotting histogram of pixel intensities
    ax2 = plt.subplot2grid((nrow, ncol), (irow, icol))
    plot_hist_and_compute_stats(powder, exp, run, ax2)
    icol = 0
    irow += 1

    # Plotting radial profiles with peaks
    ax3 = plt.subplot2grid((nrow, ncol), (irow, icol), colspan=2)
    profile, radii = radial_profile(powder, detector)
    q = pix2q(radii, distance)
    plot_radial_integration(
        q, profile, error=None, calibrant=calibrant, ax=ax3
    )
    irow += 1

    # Plotting score scan over distance
    ax5 = plt.subplot2grid((nrow, ncol), (irow, icol))
    plot_score_distance_scan(distances, ax5)
    icol += 1

    # Plotting residual scan over distance
    ax6 = plt.subplot2grid((nrow, ncol), (irow, icol))
    plot_residual_distance_scan(distances, distance, ax6)

    fig.tight_layout()

    if plot != "":
        fig.savefig(plot, dpi=100)
    return fig
