import numpy as np
from scipy.optimize import minimize


def find_vanishing_points(lines: np.ndarray, im_width: int, im_height: int) -> dict:
    """Assign each of the input lines to one of two vanishing points and find said VPs.

    Assumes two vanishing points and omits all verticalish lines.

    Args:
        lines: (N, 4) Each ith line is lines[i] = (x1, y1, x2, y2)
        im_width: The width of the image
        im_height: The height of the image

    Returns:
        A dict of ndarrays:
            vp_a: (2,) A vanishing point, in pixels
            vp_b: (2,) The other vanishing point, in pixels
            classes: (N,) A boolean array indicating whether each ith line belongs to
                vanishing point a (True) or vanishing point b (False).
    """
    is_vertical = np.array([
        abs(get_polar_line(line)[1]) < np.deg2rad(10) for line in lines])

    non_vertical_lines = lines[np.logical_not(is_vertical)]
    pts = np.array([
        ((x1, y1), (x2, y2)) for (x1, y1, x2, y2) in non_vertical_lines])

    # Initial guesses
    vp_a0 = np.array([-0.25 * im_width, im_height / 2])
    vp_b0 = np.array([1.25 * im_width, im_height / 2])
    classes0 = np.array([0.5] * len(lines))
    x0 = np.concatenate((vp_a0, vp_b0, classes0))

    vp_bounds = np.array([[-im_width, im_width], [-im_height, im_height]] * 2) * 4
    class_bounds = np.array([[0, 1]] * len(lines))
    bounds = np.concatenate((vp_bounds, class_bounds))

    min_out = minimize(
        fun=cost, x0=x0, bounds=bounds, args=(pts,), options={"disp": True})

    vp_a = min_out.x[:2].astype(int)
    vp_b = min_out.x[2:4].astype(int)
    non_vert_classes = (min_out.x[4:] > 0.5).astype(int)

    classes = np.empty((len(lines),), dtype=int)
    classes[is_vertical] = -1
    classes[np.logical_not(is_vertical)] = non_vert_classes
    assert not np.any(np.isnan(classes))

    return dict(vp_a=vp_a, vp_b=vp_b, classes=classes)


def cost(x, pts):
    """Cost function for scipy.optimize.minimize to classify lines based on the
    vanishing point and locate the vanishing point.

    Args:
        x: (4 + N,)
            vp_a = x[:2]
            vp_b = x[2:4]
            The remainder are N weights from 0 to 1 which assign a given
            line to one of (up to) two vanishing points.
        pts: (N,2,2) An array of N lines defined by (p1=(x1, y1), p2=(x2, y2))
    """
    # vanishing points 1 and 2
    vp_a = x[:2]
    vp_b = x[2:4]

    weights = x[4:]

    costs_a = get_arclength_correction(vp_a, pts) ** 2 * weights ** 2
    costs_b = get_arclength_correction(vp_b, pts) ** 2 * (1 - weights) ** 2

    return (np.sum(costs_a) + np.sum(costs_b)) / len(pts)


def get_arclength_correction(p: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Compute the minimum angle * r needed to rotate each pts[i] about the center s.t.
    it passes through p.


    Args:
        p: (2,) The (possible) vanishing point
        pts: (N,2,2) The lines for which to compute errors
    """
    # I know some of this could be vectorized, but... eh
    arclengths = []
    for pt1, pt2 in pts:
        p_pivot = np.mean([pt1, pt2], axis=0)
        v1 = (p - p_pivot) / np.linalg.norm(p - p_pivot)
        v2 = (pt1 - pt2) / np.linalg.norm(pt1 - pt2)
        theta = cliparccos(v1 @ v2)
        theta = min(theta, abs(np.pi - theta))
        arclengths.append(np.linalg.norm(p - p_pivot) * theta)
    
    return np.array(arclengths)


def get_polar_line(line: np.ndarray):
    """From the inputted line segment, get the collinear line in polar (Hough).

    Args:
        line: (4,) Line defined by two endpoints: x1, y1, x2, y2

    Returns:
        r, theta.
    """

    x1, y1, x2, y2 = line

    # Edge case: If slope is 0 or infinity
    if x1 == x2:
        return abs(y1 - y2), 0
    if y1 == y2:
        return abs(x1 - x2), np.pi / 2

    # Find the line per y = mx + b
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    # find pi, the intersection between input line and origin
    xi = b / (1 / m - m)
    yi = 1 / m * xi
    pi = (xi, yi)

    return np.linalg.norm(pi), np.arctan(yi / xi)


def is_vertical(line: np.ndarray, threshold: float) -> bool:
    """Return whether a line is vertical within some radian threshold."""
    theta = get_polar_line(line)[1]


def cliparccos(x):
    return np.arccos(np.clip(x, -1, 1))