import numpy as np
import cv2 as cv


# TODO: THIS DOESNT SEEM TO WORK :((((
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
    du, dv = p1 - p2
    assert sum(np.isclose(X1 - X2, 0)) == 2
    dx = X1 - X2
    dx = max(dx.min(), dx.max(), key=abs)

    # du, dv = np.abs(p1 - p2)
    # dx = np.abs(dx)

    cx = im_width / 2
    cy = im_height / 2

    return np.sqrt(
        (((du - cx) / dx)**2 + ((dv - cy) / dx)**2) / (1 - (1/dx)**2)
    )


if __name__ == "__main__":

    print(get_focal_length(
        p1=np.array((474, 583)),
        p2=np.array((695, 320)),
        X1=np.array((20, 0, 0)) * 0.3048,
        X2=np.array((20, 15, 0)) * 0.3048,
        im_width=912,
        im_height=600)
    )

    print(get_focal_length(
        p1=np.array((209, 276)),
        p2=np.array((695, 320)),
        X1=np.array((0, 15, 0)) * 0.3048,
        X2=np.array((20, 15, 0)) * 0.3048,
        im_width=912,
        im_height=600)
    )

    print(get_focal_length(
        p2=np.array((209, 276)),
        p1=np.array((695, 320)),
        X2=np.array((0, 15, 0)) * 0.3048,
        X1=np.array((20, 15, 0)) * 0.3048,
        im_width=912,
        im_height=600)
    )

    print(get_focal_length(
        p2=np.array((408, 293)),
        p1=np.array((209, 276)),
        X2=np.array((10, 15, 0)) * 0.3048,
        X1=np.array((0, 15, 0)) * 0.3048,
        im_width=912,
        im_height=600)
    )

    print()
    print()

    # https://stackoverflow.com/questions/73340550/how-does-opencv-projectpoints-perform-transformations-before-projecting
    def rtvec_to_matrix(rvec, tvec):
        """Convert rotation vector and translation vector to 4x4 matrix"""
        rvec = np.asarray(rvec)
        tvec = np.asarray(tvec)

        T = np.eye(4)
        R, jac = cv.Rodrigues(rvec)
        T[:3, :3] = R
        T[:3, 3] = tvec.squeeze()
        return T

    f = 727.27738659
    cx = 455.5
    cy = 299.5
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1],
    ])
    Kinv = np.array([
        [1/f, 0, -cx / f],
        [0, 1/f, -cy / f],
        [0, 0, 1],
    ])

    rvec = np.array([[1.74340779, 0.42996179, -0.34798124]]).T
    tvec = np.array([[-5.32239545, 0.4787943, 5.55458855]]).T

    Rt44 = rtvec_to_matrix(rvec, tvec)
    Rt = Rt44[:3]

    # R = np.array(
    #     [
    #         [8.85187248e-01, 4.65234770e-01, -3.80118672e-04],
    #         [9.73487830e-02, -1.86021238e-01, -9.77711263e-01],
    #         [-4.54935985e-01, 8.65420538e-01, -2.09953666e-01]
    #     ]
    # )

    # Rt = np.hstack((R, t))
    
    p1=np.array((209, 276, 1))
    p2=np.array((695, 320, 1))
    X1=np.array((0, 15, 0, 1/0.3048)) * 0.3048
    X2=np.array((20, 15, 0, 1/0.3048)) * 0.3048

    # Check fwd projection:
    p1_ = K @ Rt @ X1; p1_ /= p1_[-1]
    print(p1 - p1_, 'should be small')

    p2_ = K @ Rt @ X2; p2_ /= p2_[-1]
    print(p2 - p2_, 'should be small')


    dp = p1 - p2; dp[-1] = 1
    dX = X1 - X2; dX[-1] = 1

    # dp_ = K @ Rt @ dX; dp_ /= dp_[-1]
    dp_ = K @ Rt @ X1 - K @ Rt @ X2; dp_ /= dp_[-1]
    print(dp_)

    import ipdb; ipdb.set_trace()

    # du, dv = p1 - p2
    # dx = X1 - X2
    # dx = max(dx.min(), dx.max(), key=abs)

    # r1 = Kinv @ np.array([du, dv, 1]).T / dx

    # print(r1)

    # print(K @ r1 @ np.array([dx, 0, 0]))