import numpy as np

"""The graph of keypoints on one side of a pickleball court.

A node "a" through "i" describes a graph as shown below, and an edge
exists between two nodes if a line can be drawn between them.

    o------o------o
    |      |      |
    |      |      |
    |      |      |
    g------h------i
    |             |
    |             |
    d------e------f
    |      |      |
    |      |      |
    |      |      |
    a------b------c
"""
COURT_GRAPH = {
    "a": "bcdg",
    "b": "ace",
    "c": "abfi",
    "d": "aefg",
    "e": "bdf",
    "f": "cdei",
    "g": "adhi",
    "h": "gi",
    "i": "cfgh",
}


def get_court_coordinates(): 
    coords = np.array((
        (0, 0, 0), (10, 0, 0), (20, 0, 0),
        (0, 15, 0), (10, 15, 0), (20, 15, 0),
        (0, 29, 0), (10, 29, 0), (20, 29, 0),
    )) * 0.3048  # ft to meters
    return {node: coord for node, coord in zip(COURT_GRAPH, coords)}


COURT_COORDS = get_court_coordinates()


def get_court_corners():
    """Return all possible corners, where a corner is a sequence of nodes
    q r s where q shares an edge with r shares an edge with s but q and s do not.
    """
    corner_names = []
    for q in COURT_GRAPH:
        for r in COURT_GRAPH[q]:
            for s in COURT_GRAPH[r]:
                if (
                    q != r and
                    r != s and
                    s != q and
                    s not in COURT_GRAPH[q]
                ):
                    corner_names.append(q + r + s)

    name2coord = get_court_coordinates()

    corner_coords = []
    for name1, name2, name3 in corner_names:
        corner_coords.append([name2coord[name1], name2coord[name2], name2coord[name3]])

    return np.array(corner_coords)


COURT_CORNERS = get_court_corners()


def get_dense_court_points(pt_per_meter: float):
    """Return an Nx3 list of linearly-distributed points along the court lines."""
    edges = ("ac", "df", "ag", "ci", "be")
    pts = []
    for node1, node2 in edges:
        X1 = COURT_COORDS[node1]
        X2 = COURT_COORDS[node2]
        v = X2 - X1
        v_norm = np.linalg.norm(v)

        n = int(pt_per_meter * v_norm)
        for i in range(n):
            pts.append(X1 + i * v / v_norm / pt_per_meter)
    
    return np.array(pts)


DENSE_COURT_POINTS = get_dense_court_points(50)


if __name__ == "__main__":
    print(get_court_corners().shape)
    print(get_court_coordinates())

    import matplotlib.pyplot as plt
    plt.scatter(x=DENSE_COURT_POINTS[:, 0], y=DENSE_COURT_POINTS[:, 1])
    plt.scatter(x=COURT_CORNERS[:, :, 0], y=COURT_CORNERS[:, :, 1])
    plt.show()
