import numpy as np
from sklearn.cluster import AgglomerativeClustering
from stack_overflow import closestDistanceBetweenLines


def cluster_segments(
        lines: np.ndarray,
        dist_weight: float = 1/50,
        angular_dist_weight: float = 40,
) -> np.ndarray:
    """Merge similar line segments together.

    Uses agglomerative clustering with single linkage to cluster similar line
    segments. See:
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html

    Agglomerative clustering uses similarity scores to group similar line
    segments together. The score is the weighted sum of the following:
        dist: Minimum distance between the segments.
        theta: Angle between the segments.

    Each class has a corresponding "average" line which roughly represents the
    cluster.

    Args:
        lines: (N, 4) An array of N line segments, where each nth segment is
            defined by endpoints at lines[n, :2] and lines[n, 2:].
        dist_weight: The weight with which to consider distance between
            segments when computing similarity.
        angular_dist_weight: The weight with which to consider angular distance
            between segments when computing similarity.

    Returns:
        labels: Which class each line belongs to
        average_lines: For each class of line, the "average" line.
        average_lines_labels: The label for each line in average_lines.

    """
    N = len(lines)
    vecs = np.array([lines[:, 0] - lines[:, 2], lines[:, 1] - lines[:, 3]]).T

    dists = np.empty((N, N))
    dists[:] = np.nan
    np.fill_diagonal(dists, 0)
    angular_dists = np.empty((N, N))
    angular_dists[:] = np.nan
    np.fill_diagonal(angular_dists, 0)

    for i in range(N):
        line_i = lines[i]
        vec_i = vecs[i]

        for j in range(i):
            line_j = lines[j]
            vec_j = vecs[j]

            closest_dist = closestDistanceBetweenLines(
                a0=np.pad(line_i[:2], (0, 1)),
                a1=np.pad(line_i[2:], (0, 1)),
                b0=np.pad(line_j[:2], (0, 1)),
                b1=np.pad(line_j[2:], (0, 1)),
                clampAll=True,
            )[2]
            dists[i, j] = closest_dist
            dists[j, i] = closest_dist

            theta = np.arccos(
                np.clip(
                    np.dot(vec_i, vec_j) / (
                        np.linalg.norm(vec_i) * np.linalg.norm(vec_j)),
                    - 1, 1,
                ))
            theta = np.min([theta, abs(np.pi - theta), abs(2 * np.pi - theta)])

            angular_dists[i, j] = theta
            angular_dists[j, i] = theta

    weighted_dists = dist_weight * dists + angular_dist_weight * angular_dists

    labels = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="single",
        distance_threshold=1,
    ).fit(weighted_dists).labels_
    labels = np.array(labels)

    avg_lines = []
    avg_lines_labels = sorted(np.unique(labels))
    for label in avg_lines_labels:
        avg_lines.append(average_segments(lines[labels == label]))
    avg_lines = np.array(avg_lines)

    return dict(
        labels=labels,
        average_lines=avg_lines,
        average_lines_labels=avg_lines_labels
    )


def average_segments(lines: np.ndarray) -> np.ndarray:
    """Given an array of line segments, find the representative segment that
    covers roughly the same distance and is the average angle.
    """
    vecs = np.array([(x2 - x1, y2 - y1) for (x1, y1, x2, y2) in lines])
    summed_vec = np.array([0, 0], dtype=float)
    for vec in vecs:
        if np.linalg.norm(summed_vec + vec) > np.linalg.norm(summed_vec - vec):
            summed_vec += vec
        else:
            summed_vec -= vec

    if np.all(np.isclose(summed_vec, 0)):
        return np.array([0, 0, 0, 0])

    # angle of line with respect to x-axis
    theta = np.arctan(summed_vec[1] / summed_vec[0])
    # Roration matrix to rotate p ccw around origin:
    # p' = R @ p
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    Rinv = np.linalg.inv(R)

    # rotate all points cw about origin s.t. they're (mostly) in line w the x-
    # axis
    pts_1 = lines[:, :2]
    pts_2 = lines[:, 2:]

    pts_1_horiz = (Rinv @ pts_1.T).T
    pts_2_horiz = (Rinv @ pts_2.T).T

    y_avg = np.mean([pts_1_horiz[:, 1], pts_2_horiz[:, 1]])
    x_max = np.max([pts_1_horiz[:, 0], pts_2_horiz[:, 0]])
    x_min = np.min([pts_1_horiz[:, 0], pts_2_horiz[:, 0]])

    # end-points of average line
    p1_horiz = np.array([x_max, y_avg])
    p2_horiz = np.array([x_min, y_avg])

    p1 = R @ p1_horiz
    p2 = R @ p2_horiz

    return np.array([p1[0], p1[1], p2[0], p2[1]])
    