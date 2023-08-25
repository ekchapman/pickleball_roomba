import fire
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import collections
import numpy as np

from cluster import cluster_segments
from optimize import find_vanishing_points


COLORS = {
    'blue':    '#377eb8',
    'orange':  '#ff7f00',
    'green':   '#4daf4a',
    'pink':    '#f781bf',
    'brown':   '#a65628',
    'purple':  '#984ea3',
    'gray':    '#999999',
    'yellow':  '#dede00',
    'red':     '#e41a1c',
}

COLOR_NAMES = list(COLORS.keys())


def main(
    image_path: str,
    blur_kernel_size: int = 13,
    canny_threshold_low: float = 50,
    canny_threshold_high: float = 100,
    canny_appeture_size: int = 3,
    hough_rho: float = 1,
    hough_theta: float = np.pi / 180,
    hough_threshold: float = 125,
    hough_min_lin_length: float = 25,
    hough_max_line_gap: float = 15
) -> None:
    """Given an image, detect straight lines and vanishing points.

    Except image_path, all args are pass-throughs to the corresponding openCV
    function.

    Script to perform and visualize the following processing to an image:
        1. Gaussian blur
        2. Canny edge detection
        3. Hough line detection
        4. Agglomerative clustering on ^
        5. "Fuse" similar lines from ^
        6. Find two vanishing points (VP) and assign lines from ^ to one VP

    Args:
        image_path: Path to the image.
    """
    im = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
    im_blur = cv.GaussianBlur(im, (blur_kernel_size, blur_kernel_size), 0)
    im_edges = cv.Canny(im_blur, canny_threshold_low,
                        canny_threshold_high, canny_appeture_size)
    # Formatted s.t.: lines[i] = x1, y1, x2, y2
    hough_lines = cv.HoughLinesP(
        image=im_edges,
        rho=hough_rho,
        theta=hough_theta,
        threshold=hough_threshold,
        minLineLength=hough_min_lin_length,
        maxLineGap=hough_max_line_gap,
    )[:, 0, :]

    cluster_out = cluster_segments(hough_lines)
    avg_lines = cluster_out["average_lines"]
    opt_out = find_vanishing_points(avg_lines, im.shape[0], im.shape[1])
    avg_lines_classes = opt_out["classes"]
    vp_a = opt_out["vp_a"]
    vp_b = opt_out["vp_b"]

    plt.figure()
    plt.imshow(im)
    plt.title("original")

    plt.figure()
    plt.imshow(im_blur)
    plt.title("blurred")

    plt.figure()
    plt.imshow(im_edges, cmap="Greys")
    plt.title("edges")

    _, ax1 = plt.subplots()
    ax1.imshow(im_edges, cmap="Greys")
    plot_segments(ax1, hough_lines, colors=["red"], linewidths=[2])
    plt.title("hough")

    _, ax2 = plt.subplots()
    ax2.imshow(im_edges, cmap="Greys")
    plot_segments(ax2, avg_lines, colors=["red"], linewidths=[2])
    plt.title("grouped hough")

    _, ax3 = plt.subplots()
    ax3.imshow(im)
    colors = [COLOR_NAMES[cl] for cl in avg_lines_classes]
    plot_lines(ax3, avg_lines, colors)
    plot_segments(ax3, avg_lines, colors=colors, linewidths=3)
    ax3.scatter(
        x=[vp_a[0], vp_b[0]], y=[vp_a[1], vp_b[1]],
        s=1000, marker='+', color="red"
    )

    plt.show()


def plot_segments(ax: plt.Axes, lines: np.ndarray, **lc_kwargs) -> None:
    """Plot the line segments on a matplotlib plot.

    Args:
        ax: One of the outputs from plt.subplots() ¯\_(ツ)_/¯
        lines: The lines to plot. Line i is defined by lines[i] =
            x1, y1, x2, y2.
        lc_kwargs: Keyword args to pass down to LineCollection, e.g. colors.
    """
    pts = np.array([((x1, y1), (x2, y2)) for (x1, y1, x2, y2) in lines])
    lc = collections.LineCollection(pts, **lc_kwargs)
    ax.add_collection(lc)


def plot_lines(ax: plt.Axes, lines: np.ndarray, colors: list[str]) -> None:
    """Plot the infinite lines on a matplotlib plot."""
    for line, color in zip(lines, colors):
        ax.axline(line[:2], line[2:], color=color, linewidth=0.5)


if __name__ == "__main__":
    fire.Fire(main)
