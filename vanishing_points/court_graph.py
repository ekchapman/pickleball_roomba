import numpy as np

"""The graph of keypoints on one side of a pickleball court.

A node "a" through "i" describes a graph as shown below, and an edge
exists between two nodes if a line can be drawn between them.

    o------o------o
    |      |      |
    |      |      |
    o------o------o
    |             |
    g------h------i
    |             |
    d------e------f
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


def get_court_corners():
    """Return all possible corners, where a corner is a sequence of nodes
    q r s where q shares an edge with r shares an edge with s but q and s do not.
    """
    corners = []
    for q in COURT_GRAPH:
        for r in COURT_GRAPH[q]:
            for s in COURT_GRAPH[r]:
                if (
                    q != r and
                    r != s and
                    s != q and
                    s not in COURT_GRAPH[q]
                ):
                    corners.append(q + r + s)

    return corners


COURT_CORNERS = get_court_corners()


def get_court_coordinates():
    coords = np.array((
        (0, 0), (10, 0), (20, 0),
        (0, 15), (10, 15), (20, 15),
        (0, 29), (10, 29), (20, 29),
    )) * 0.3048  # ft to meters
    return {node: coord for node, coord in zip(COURT_GRAPH, coords)}


COURT_COORDS = get_court_coordinates()


if __name__ == "__main__":
    print(len(get_court_corners()))
    print(get_court_coordinates())
