import numpy as np


def get_focal_length(p1: np.ndarray, p2: np.ndarray, X1: np.ndarray, X2: np.ndarray, im_width: int, im_height: int) -> float:
    """Given two pixels and their corresponding real-world 3D coordinates, compute the focal length.

    Relies on the following assumptions:
        1. fx = fy, cx = width / 2, cy = height / 2, no shear.
        2. X1 and X2 are equal in two of three dimensions.
            Without loss of generality, we'll call X1 - X2 = [dx, 0, 0].T
        3. Negligible non-linear effects e.g. lens distortion.
    
    The derivation follows from the homogeneous camera projection equation. See:
    https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html
    
    p = K*[R|t]*X
    where p = [u, v, 1].T and X = [dx, 0, 0, 1],
        K is the intrinsic matrix, R is the rotation matrix.

    Subtracting two points gives us:
    p1 - p2 = K*R*[dx, 0, 0, 1].T
    [du, dv, 1].T = K*[r11, r21, r31].T * dx
    K_inv * [du, dv, 1].T / dx = [r11, r21, r31].T

    Expanding the matrix multiplication gives us a closed form solution for the
    first column of R, [r11, r21, r31].T

        r11 = (du - cx) / (f * dx)
        r21 = (dv - cy) / (f * dy)
        r31 = 1/dx
    where du = u1 - u2, dv = v1 - v2

    Because R is a rotation matrix, we know:
    norm([r11, r22, r33]) = 1

    which gives us system of equations we can use to solve for f:

    f = sqrt(
        (((du - cx) / dx)**2 + ((dv - cy) / dx)**2) / (1 - (1/dx)**2)
    )
    """
    du, dv = np.abs(p1 - p2)
    assert sum(np.isclose(X1 - X2, 0)) == 2
    dx = np.max(np.abs(X1 - X2)) 

    cx = im_width / 2
    cy = im_height / 2

    return np.sqrt(
        (((du - cx) / dx)**2 + ((dv - cy) / dx)**2) / (1 - (1/dx)**2)
    )


if __name__ == "__main__":

    print(get_focal_length(
        p1=np.array((474, 583)),
        p2=np.array((695, 320)),
        X1=np.array((0, 0, 0)),
        X2=np.array((4.572, 0, 0)),
        im_width=912,
        im_height=600)
    )

    print(get_focal_length(
        p1=np.array((209, 276)),
        p2=np.array((695, 320)),
        X1=np.array((0, 0, 0)),
        X2=np.array((6.096, 0, 0)),
        im_width=912,
        im_height=600)
    )