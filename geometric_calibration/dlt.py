import numpy as np
from scipy.spatial.transform import Rotation


def Normalization(nd, x):
    """
    Normalization of coordinates (centroid to the origin and mean distance of
    sqrt(2 or 3).

    Args:
        nd (int): number of dimensions, typically 3
        x (numpy.array): the data to be normalized (directions at different
            columns and points at rows)

    Returns:
        numpy.array, numpy.array: the transformation matrix (translation plus
            scaling), the transformed data
    """
    x = np.asarray(x)
    m, s = np.mean(x, 0), np.std(x)
    if nd == 2:
        Tr = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    else:
        Tr = np.array(
            [[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]]
        )

    Tr = np.linalg.inv(Tr)
    x = np.dot(Tr, np.concatenate((x.T, np.ones((1, x.shape[0])))))
    x = x[0:nd, :].T

    return Tr, x


def DLTcalib(nd, xyz, uv, uv_ref=None):
    """
    Camera calibration by DLT using known object points and their
    corresponding image points.
    The coordinates (x,y,z and u,v) are given as columns and the different
    points as rows.
    There must be at least 6 calibration points for the 3D DLT.

    Args:
        nd (int): dimensions of the object space, typically 3
        xyz (numpy.array): coordinates in the object 3D space
        uv (numpy.array): coordinates in the image 2D space
        uv_ref (numpy.array, optional): [description]. Defaults to None.

    Raises:
        ValueError: Dimension not supported
        ValueError: xyz and uv have different number of points
        ValueError: Wrong dimension for coordinates
        ValueError: Insufficient number of points

    Returns:
        numpy.array, float: array of 11 parameters of the calibration matrix,
            followed by error of the DLT (mean residual of the DLT
            transformation in units of camera coordinates).
    """
    if nd != 3:
        raise ValueError(f"{nd}D DLT unsupported.")

    # Converting all variables to numpy array
    xyz = np.asarray(xyz)
    uv = np.asarray(uv)

    n = xyz.shape[0]

    # Validating the parameters:
    if uv.shape[0] != n:
        raise ValueError(
            f"Object ({n} points) and image ({uv.shape[0]} points) have different number of points."
        )

    if xyz.shape[1] != 3:
        raise ValueError(
            f"Incorrect number of coordinates ({xyz.shape[1]}) for {nd}D DLT (it should be {nd})."
        )

    if n < 6:
        raise ValueError(
            f"{nd}D DLT requires at least {2*nd} calibration points. Only {n} points were entered."
        )

    # Normalize the data to improve the DLT quality (DLT is dependent of the
    # coordinates system).
    # This is relevant when there is a considerable perspective distortion.
    # Normalization: mean position at origin and mean distance equals to 1 at
    # each direction.
    Txyz, xyzn = Normalization(nd, xyz)
    Tuv, uvn = Normalization(2, uv)

    A = []

    for i in range(n):
        x, y, z = xyzn[i, 0], xyzn[i, 1], xyzn[i, 2]
        u, v = uvn[i, 0], uvn[i, 1]

        A.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
        A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])

    # Convert A to array
    A = np.asarray(A)

    # Find the 11 parameters:
    U, S, V = np.linalg.svd(A)

    # The parameters are in the last line of Vh and normalize them
    L = V[-1, :] / V[-1, -1]

    # Camera projection matrix
    H = L.reshape(3, nd + 1)

    # Denormalization
    # pinv: Moore-Penrose pseudo-inverse of a matrix, generalized inverse of a matrix using its SVD
    H = np.dot(np.dot(np.linalg.pinv(Tuv), H), Txyz)
    H = H / H[-1, -1]
    L = H.flatten()

    # Mean error of the DLT (mean residual of the DLT transformation in units of camera coordinates):
    uv2 = np.dot(H, np.concatenate((xyz.T, np.ones((1, xyz.shape[0])))))
    uv2 = uv2 / uv2[2, :]

    # Mean distance:
    err = np.sqrt(np.mean(np.sum((uv2[0:2, :].T - uv) ** 2, 1)))
    if uv_ref is not None:
        err_ref_init = np.sqrt(np.mean(np.sum((uv - uv_ref) ** 2, 1)))
        err_ref_final = np.sqrt(
            np.mean(np.sum((uv2[0:2, :].T - uv_ref) ** 2, 1))
        )
        return L, err, err_ref_init, err_ref_final
    else:
        return L, err


def decompose_camera_matrix(L, image_size, pixel_spacing):
    # Computing the camera position
    PosMat = np.array(
        [[L[0], L[1], L[2]], [L[4], L[5], L[6]], [L[8], L[9], L[10]]]
    )

    PosY = np.array([[-L[3]], [-L[7]], [-1.0]])

    # Source in fixed frame
    X0 = np.dot(np.linalg.inv(PosMat), PosY)

    LL = np.sqrt(PosMat[2, 0] ** 2 + PosMat[2, 1] ** 2 + PosMat[2, 2] ** 2)
    LL = -1 / LL
    sid = -LL

    xp = (L[0] * L[8] + L[1] * L[9] + L[2] * L[10]) * (LL ** 2)
    yp = (L[4] * L[8] + L[5] * L[9] + L[6] * L[10]) * (LL ** 2)

    fx = np.sqrt((L[0] ** 2 + L[1] ** 2 + L[2] ** 2) * LL ** 2 - xp ** 2)
    fy = np.sqrt((L[4] ** 2 + L[5] ** 2 + L[6] ** 2) * LL ** 2 - yp ** 2)
    sdd = ((fx * 0.388) + (fy * 0.388)) / 2

    K = np.array([[fx, 0, xp], [0, fy, yp], [0, 0, 1]])
    invK = np.linalg.inv(K)

    R = invK @ PosMat
    R = R / np.linalg.norm(R, 2)

    euler = Rotation.from_matrix(R).as_euler("zxy")
    oa = euler[1]  # Rotation around X
    ga = euler[2]  # Rotation around Y
    ia = euler[0]  # Rotation around Z

    SourceOffset = np.dot(R, X0)
    sx = float(SourceOffset[0])
    sy = float(SourceOffset[1])

    px = sx + (image_size[0] / 2 * pixel_spacing[0]) - xp * pixel_spacing[0]
    py = sy + (image_size[1] / 2 * pixel_spacing[1]) - yp * pixel_spacing[1]

    # Return parameters list
    parameters = {
        "sid": sid,
        "sdd": sdd,
        "oa": -oa,
        "ga": -ga,
        "ia": -ia,
        "px": px,
        "py": py,
        "sx": sx,
        "sy": sy,
    }

    return parameters


def extract_param_from_matrix_Rit(camera_matrix):
    # print("Matrice iniziale")
    # print(camera_matrix)
    A = camera_matrix[:3, :3]
    # print("A")
    # print(A)

    p = camera_matrix[:, 3]
    # print("p")
    # print(p)

    # Compute determinant of A
    d = np.linalg.det(A)
    d = -1 * d / np.abs(d)
    # print("d")
    # print(d)

    # Extract intrinsic parameters u0, v0 and f (f is chosen to be positive at
    # that point). The extraction of u0 and v0 is independant of KR-decomp.
    u0 = (
        (camera_matrix[0, 0] * camera_matrix[2, 0])
        + (camera_matrix[0, 1] * camera_matrix[2, 1])
        + (camera_matrix[0, 2] * camera_matrix[2, 2])
    )
    v0 = (
        (camera_matrix[1, 0] * camera_matrix[2, 0])
        + (camera_matrix[1, 1] * camera_matrix[2, 1])
        + (camera_matrix[1, 2] * camera_matrix[2, 2])
    )
    aU = np.sqrt(
        camera_matrix[0, 0] * camera_matrix[0, 0]
        + camera_matrix[0, 1] * camera_matrix[0, 1]
        + camera_matrix[0, 2] * camera_matrix[0, 2]
        - u0 ** 2
    )
    aV = np.sqrt(
        camera_matrix[1, 0] * camera_matrix[1, 0]
        + camera_matrix[1, 1] * camera_matrix[1, 1]
        + camera_matrix[1, 2] * camera_matrix[1, 2]
        - v0 ** 2
    )
    sdd = 0.5 * (aU + aV)

    # print(f"u0: {u0}")
    # print(f"v0: {v0}")
    # print(f"aU: {aU}")
    # print(f"aV: {aV}")
    # print(f"sdd: {sdd}")

    # Def matrix K so that detK = det P[:,:3]
    K = np.zeros([3, 3])
    K[0, 0] = sdd
    K[1, 1] = sdd
    K[2, 2] = -1.0
    K[0, 2] = -1.0 * u0
    K[1, 2] = -1.0 * v0
    K *= d

    # print("K")
    # print(K)

    # Compute R (since det K = det P[:,:3], detR = 1 is enforced)
    invK = np.linalg.inv(K)
    rot = invK * A
    # print("R")
    # print(rot)
    # R = np.matmul(invK, A)

    # Declare a 3D euler transform in order to properly extract angles
    euler = R.from_matrix(rot).as_euler("zxy")  # ZXY order

    # Extract angle using parent method without orthogonality check
    oa = euler[1]
    ga = euler[2]
    ia = euler[0]

    # print(f"Out of Plane Angle: {oa}")
    # print(f"In Plane Angle: {ia}")
    # print(f"Gantry Angle: {ga}")

    # verify that extracted ZXY angles result in the *desired* matrix:
    # (at some angle constellations we may run into numerical troubles,
    # therefore, verify angles and try to fix instabilities)
    if VerifyAngles(oa, ga, ia, rot) is False:
        status, oa, ga, ia = FixAngles(rot)
        if status is False:
            raise ValueError("Failed to extract parameters")

    # Coordinates of source in oriented coord sys:
    # (sx,sy,sid) = RS = R(-A^{-1}P[:,3]) = -K^{-1}P[:,3]
    # invA = np.linalg.inv(A)
    v = np.matmul(invK, p)
    v *= -1.0
    sx = v[0]
    sy = v[1]
    sid = v[2]

    # print(f"sx: {sx}")
    # print(f"sy: {sy}")
    # print(f"sid: {sid}")

    # Return parameters list
    parameters = [
        sid,
        sdd,
        -1.0 * oa,
        -1.0 * ga,
        -1.0 * ia,
        sx - u0,
        sy - v0,
        sx,
        sy,
        u0,
        v0,
        aU,
        aV,
    ]

    return parameters


def VerifyAngles(
    outOfPlaneAngleRAD, gantryAngleRAD, inPlaneAngleRAD, referenceMatrix
):
    EPSILON = 1e-5
    # Check if parameters are Nan. Fails if they are.
    if (
        np.isnan(outOfPlaneAngleRAD)
        or np.isnan(gantryAngleRAD)
        or np.isnan(inPlaneAngleRAD)
    ):
        return False

    euler = R.from_euler(
        "zxy", [inPlaneAngleRAD, outOfPlaneAngleRAD, gantryAngleRAD]
    )  # ZXY order
    m = euler.as_matrix()  # resultant matrix

    # check whether matrices match
    if (np.greater(np.abs(referenceMatrix - m), EPSILON).any()) is True:
        return False

    return True


def FixAngles(rm):
    print("Trying to fix angles...")
    EPSILON = 1e-6

    if np.abs(np.abs(rm[2][1]) - 1.0) > EPSILON:
        # @see Slabaugh, GG, "Computing Euler angles from a rotation matrix"
        # but their convention is XYZ where we use the YXZ convention

        # first trial:
        oa = np.asin(rm[2][1])
        coa = np.cos(oa)
        ga = np.atan2(-rm[2][0] / coa, rm[2][2] / coa)
        ia = np.atan2(-rm[0][1] / coa, rm[1][1] / coa)
        if VerifyAngles(oa, ga, ia, rm):
            return True, oa, ga, ia

        # second trial:
        oa = np.pi - np.asin(rm[2][1])
        coa = np.cos(oa)
        ga = np.atan2(-rm[2][0] / coa, rm[2][2] / coa)
        ia = np.atan2(-rm[0][1] / coa, rm[1][1] / coa)
        if VerifyAngles(oa, ga, ia, rm):
            return True, oa, ga, ia

    else:
        # Gimbal lock, one angle in {ia,oa} has to be set randomly
        ia = 0.0
        if rm[2][1] < 0.0:
            oa = -np.pi / 2
            ga = np.atan2(rm[0][2], rm[0][0])
            if VerifyAngles(oa, ga, ia, rm):
                return True, oa, ga, ia
        else:
            oa = np.pi / 2
            ga = np.atan2(rm[0][2], rm[0][0])
            if VerifyAngles(oa, ga, ia, rm):
                return True, oa, ga, ia

    return False
