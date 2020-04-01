import numpy as np
from geometric_calibration.utils import angle2rotm, deg2rad


def read_bbs_ref_file(filename):
    # Load reference 3D BBs
    # spiral distribution visible on x-y plane
    # [X,Y,Z] coordinates of Brandis phantom reference BB
    bbs = np.loadtxt(filename, delimiter=" ")

    # spiral distribution visible on x-z plane
    # rotation along x about 90
    T_map = angle2rotm(deg2rad(90), deg2rad(0), deg2rad(0))
    bbs = np.append(bbs, np.ones((bbs.shape[0], 1)), axis=1)  # homogeneous
    bbs = np.matmul(T_map, bbs.T).T
    bbs = bbs[:, 0:3]  # back to not homogeneous

    return bbs


def read_img_label_file(filename):
    # read image labels
    with open(filename, "r") as f:
        f.readline()  # Skip first row (header)

        file = f.readlines()

        proj_file = []  # last part of the projection file path
        angles = []  # angles of rotation of each image
        for line in file:
            proj_file.append(line.split(" ")[0].split("\\")[-1])
            angles.append(float(line.split(" ")[1]))

    return proj_file, angles


def read_projection(filename, dim):
    image = np.fromfile(filename, dtype="uint16", sep="")
    image = np.reshape(image, newshape=[dim[1], dim[0]]).T
    return image
