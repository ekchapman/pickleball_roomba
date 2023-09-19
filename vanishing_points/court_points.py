import numpy as np
from itertools import permutations


# Court dimensions can be found online. See:
# https://usapickleball.org/what-is-pickleball/court-diagram/temporary-court-setup/
COURT_POINTS = np.array((
    (0, 0, 0), (10, 0, 0), (20, 0, 0),  # baseline
    (0, 15, 0), (10, 15, 0), (20, 15, 0),  # kitchen line
    (0, 22, 0), (20, 22, 0),  # mid-line (at net)
    (0, 29, 0), (10, 29, 0), (20, 29, 0),  # far kitchen line
    (0, 44, 0), (10, 44, 0), (20, 44, 0),  # far baseline
    (-1, 22, 3), (21, 22, 3),  # net "line" (strip of white across the top of most pball nets)
)) * 0.3048  # feet -> meters


def get_segments():
    """Get all possible painted line segments between any points:"""
    segments = set()
    xs = [0, 10, 20]
    ys = [0, 15, 22, 29, 44]
    for x in xs:
        for y1, y2 in permutations(ys, r=2):
            segments.add((x, y1, 0, x, y2, 0))
    for y in ys:
        for x1, x2 in permutations(xs, r=2):
            segments.add((x1, y, 0, x2, y, 0))

    # There is no line under the net
    segments.remove((0, 22, 0, 20, 22, 0))
    segments.remove((20, 22, 0, 0, 22, 0))

    # There is no centerline across the kitchen
    segments.remove((10, 15, 0, 10, 29, 0))
    segments.remove((10, 29, 0, 10, 15, 0))

    # Manually add the net tape "line"
    segments.add((-1, 22, 3, 21, 22, 3))
    segments.add((21, 22, 3, -1, 22, 3))

    return np.array(list(segments)) * 0.3048  # feet -> meters

COURT_SEGMENTS = get_segments()
