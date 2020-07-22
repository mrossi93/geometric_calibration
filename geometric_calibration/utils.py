"""Utilities module."""
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

matplotlib.rcParams["toolbar"] = "None"


class DraggablePoints:
    """Draggable points on matplotlibe figure.

    Returns:
        DraggablePoints -- DraggablePoints object
    """

    def __init__(self, artists, tolerance=10):
        for artist in artists:
            artist.set_picker(tolerance)

        self.artists = artists
        self.final_coord = None

        # assume all artists are in the same figure, otherwise selection
        # is meaningless
        self.fig = self.artists[0].figure
        self.ax = self.artists[0].axes

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_pressed)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

        self.currently_dragging = False
        self.offset = np.zeros((1, 2))
        self.x0 = 0
        self.y0 = 0

        plt.title(
            "Drag&Drop red points on the image.\nPress Enter to continue"
        )  # noqa: E501
        plt.show()

    def on_press(self, event):
        """Event Handler for mouse button pression.

        Arguments:
            event -- Event that triggers the method
        """
        # is the press over some artist
        isonartist = False
        for artist in self.artists:
            if artist.contains(event)[0]:
                isonartist = artist
        self.x0 = event.xdata
        self.y0 = event.ydata
        if isonartist:
            # start dragging
            self.currently_dragging = True
            artist_center = np.array([a.center for a in self.artists])
            event_center = np.array([event.xdata, event.ydata])
            self.offset = artist_center - event_center

    def on_release(self, event):
        """Event Handler for mouse button release.

        Arguments:
            event -- Event that triggers the method
        """
        if self.currently_dragging:
            self.currently_dragging = False

    def on_motion(self, event):
        """Event Handler for mouse movement during dragging of points.

        Arguments:
            event -- Event that triggers the method
        """

        if self.currently_dragging:
            newcenters = np.array([event.xdata, event.ydata]) + self.offset
            for i, artist in enumerate(self.artists):
                artist.center = newcenters[i]
            self.fig.canvas.draw_idle()

    def on_key_pressed(self, event):
        """Event Handler for "enter" key pression.

        Arguments:
            event -- Event that triggers the method
        """
        if not self.currently_dragging and event.key == "enter":
            plt.close()

    def on_close(self, event):
        """Event Handler for closure of figure.

        Arguments:
            event -- Event that triggers the method
        """
        self.final_coord = self.get_coord()

    def get_coord(self):
        """Obtain current coordinates of points.

        :return: An array nx2 containing coordinates for every point [x,y]
        :rtype: numpy.array
        """
        return np.array([a.center for a in self.artists])


def drag_and_drop_bbs(projection_path, bbs_projected, grayscale_range):
    """Drag&Drop Routines for bbs position's correction.

    :param projection_path: Path to the projection .raw file
    :type projection_path: str
    :param bbs_projected: Array nx2 with bbs yet projected
    :type bbs_projected: numpy.array
    :param grayscale_range: Grayscale range for current projection
    :type grayscale_range: list
    :return: Array nx2 containing the updated coordinates for bbs
    :rtype: numpy.array
    """
    # Overlay reference bbs with projection
    fig = plt.figure(num="Drag&Drop")

    ax = fig.add_subplot(111)

    # Reference image in background (must stay in position always)
    ax.imshow(
        projection_path,
        cmap="gray",
        vmin=grayscale_range[0],
        vmax=grayscale_range[1],
    )

    # Drag&Drop
    pts = []
    for x, y in zip(bbs_projected[:, 0], bbs_projected[:, 1]):
        point = patches.Circle((x, y), fc="r")
        pts.append(point)
        ax.add_patch(point)

    r2d_corrected = DraggablePoints(pts)

    return r2d_corrected.final_coord


def search_bbs_centroids(img, ref_2d, search_area, dim_img, grayscale_range):
    """Search bbs based on projection.

    Starting from the updated coordinates, define a search area around them
    and identify the bbs as black pixels inside these areas (brandis are used
    as probes). Search for the bbs in the image (basically very low intensity
    surrounding by higher intensity pixel. Centroids coordinates are the mean
    pixels that have an intensity that is lower than the lowest nominal
    intensity plus a tollerance.

    :param img: Array containing the loaded .raw file
    :type img: numpy.array
    :param ref_2d: Array nx2 containing the coordinates for bbs projected on
     img
    :type ref_2d: numpy.array
    :param search_area: Size of the region in which to search for centroids.
     Actual dimension of the area is a square with dimension (2*search_area,
     2*search_area)
    :type search_area: int
    :param dim_img: Dimension of img
    :type dim_img: list
    :param grayscale_range: Grayscale range for current projection
    :type grayscale_range: list
    :raises Exception: if the function does not find any centroid, an
     exception is thrown
    :return: Array nx2 containing coordinates for every centroids found [x,y]
    :rtype: numpy.array
    """
    bbs_centroid = []
    for curr_point in ref_2d:  # for each bbs
        ind_row = round(curr_point[0])
        ind_col = round(curr_point[1])

        # define the field of research
        min_row = int(max([0, ind_row - search_area]))
        min_col = int(max([0, ind_col - search_area]))
        max_row = int(min([ind_row + search_area, dim_img[1]]))
        max_col = int(min([ind_col + search_area, dim_img[0]]))

        # define a mask on the original image to underline field of research
        sub_img = img[min_col:max_col, min_row:max_row]

        sub_img = adjust_image(
            sub_img, grayscale_range
        )  # rescale grey level of the sub-image

        # Search for the bbs in the image (basically very low intensity
        # surrounding by higher intensity pixel. ii,jj are the coordinates of
        # the pixels that have an intensity that is lower than the lowest
        # nominal intensity plus a tollerance.
        ii, jj = np.where(
            sub_img
            < grayscale_range[0]
            + 0.1 * (grayscale_range[1] - grayscale_range[0])
        )  # based on intensity

        if len(ii) == 0:
            bbs_centroid.append([np.nan, np.nan])
            continue

        if (max(ii) - min(ii) < search_area) and (
            max(jj) - min(jj) < search_area
        ):  # based on coordinates
            bbs_centroid.append(
                [min_row + np.mean(jj) - 1, min_col + np.mean(ii) - 1]
            )  # position of the centroid of the bbs
        else:
            bbs_centroid.append(
                [np.nan, np.nan]
            )  # if out of the searching area

    if len(bbs_centroid) == 0:
        raise Exception(
            "Error! Try to better overlap reference with projection"
        )

    bbs_centroid = np.array(bbs_centroid)

    return bbs_centroid


def deg2rad(angle_deg):
    """Convert angles from degrees to radians.

    :param angle_deg: Angle to convert
    :type angle_deg: int or float
    :return: Angle converted in radians
    :rtype: float
    """
    return (np.pi / 180) * angle_deg


def angle2rotm(rot_x, rot_y, rot_z):
    """Generate a rototranslator (only rotation) starting from Euler angles

    NB: Convention is 'XYZ'

    :param rot_x: Rotation along x
    :type rot_x: int or float
    :param rot_y: Rotation along y
    :type rot_y: int or float
    :param rot_z: Rotation along z
    :type rot_z: int or float
    :return: 4x4 Rototranslation matrix in homogeneous form
    :rtype: numpy.array
    """
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(rot_x).item(), -np.sin(rot_x).item()],
            [0.0, np.sin(rot_x).item(), np.cos(rot_x).item()],
        ]
    )

    Ry = np.array(
        [
            [np.cos(rot_y).item(), 0.0, np.sin(rot_y).item()],
            [0.0, 1.0, 0.0],
            [-np.sin(rot_y).item(), 0.0, np.cos(rot_y).item()],
        ]
    )

    Rz = np.array(
        [
            [np.cos(rot_z).item(), -np.sin(rot_z).item(), 0.0],
            [np.sin(rot_z).item(), np.cos(rot_z).item(), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    R = np.matmul(Rz, np.matmul(Ry, Rx))
    trans = R
    trans = np.append(trans, np.zeros((3, 1)), axis=1)
    trans = np.append(trans, np.zeros((1, 4)), axis=0)
    trans[3, 3] = 1.0
    return trans


def project_camera_matrix(r3d, image_center, camera_matrix, resolution_scale=1):
    """Project 3D data starting from camera matrix based on intrinsic and
    extrinsic parameters

    :param r3d: Array nx3 containing 3d coordinates of points [x,y,z]
    :type r3d: numpy.array
    :param image_center: Center of the image
    :type image_center: list
    :param camera_matrix: Projection matrix obtained combining extrinsic
     and intrinsic parameters
    :type camera_matrix: numpy.array
    :param resolution_scale: resolution factor, when mode is "cbct" this 
    parameter equals to 1, in 2D mode is 2 (because resolution is doubled),
     defaults to 1
    :type resolution_scale: int, optional
    :return: Array nx2 containing 2D coordinates of points projected on
     image plane [x,y]
    :rtype: numpy.array
    """

    r3d = np.append(r3d, np.ones((r3d.shape[0], 1)), axis=1)  # homogeneous

    r3d = np.matmul(camera_matrix, r3d.T).T  # apply proj_matrix and project
    r3d[:, 0] = (
        np.divide(r3d[:, 0], r3d[:, 2]) * resolution_scale + image_center[0]
    )  # offset
    r3d[:, 1] = (
        np.divide(r3d[:, 1], r3d[:, 2]) * resolution_scale + image_center[1]
    )  # offset
    r2d = r3d[:, :2]

    return r2d


def create_camera_matrix(panel_orientation, sid, sad, pixel_size, isocenter):
    """Generate projection matrix starting from extrinsic and intrinsic
    parameters.

    :param panel_orientation: Array nx3 containing rotations of the image's
     plane [rot_x, rot_y, rot_z]
    :type panel_orientation: numpy.array
    :param sid: SID distance
    :type sid: float
    :param sad: SAD distance
    :type sad: float
    :param pixel_size: Pixel Dimensions in mm
    :type pixel_size: list
    :param isocenter: Coordinates of isocenter
    :type isocenter: numpy.array
    :return: 3x4 Camera Matrix
    :rtype: numpy.array
    """
    # extrinsic parameters
    extrinsic = angle2rotm(
        panel_orientation[0], panel_orientation[1], panel_orientation[2],
    )
    extrinsic[:3, :3] = extrinsic[:3, :3].T

    # add isocenter projection in extrinsic matrix
    extrinsic[:3, 3] = np.dot(extrinsic[:3, :3], isocenter)
    extrinsic[2, 3] = extrinsic[2, 3] + sad  # add sad

    # intrinsic parameters
    intrinsic = np.zeros((3, 4))
    intrinsic[0, 0] = sid / pixel_size[1]
    intrinsic[1, 1] = sid / pixel_size[0]
    intrinsic[2, 2] = 1

    # total matrix
    T = np.matmul(intrinsic, extrinsic)

    return T


def get_grayscale_range(img):
    """New grayscale range for .raw images, since original values are too
    bright. New range is computed between min of image and one order of
    magnitude less than original image. Worst case scenario [0, 6553.5]
    (since im is loaded as uint16)

    :param img: Array containing the loaded .raw image
    :type img: numpy.array
    :return: Grayscale range for current projection
    :rtype: list
    """
    # image range - lowest and highest gray-level intensity for projection
    grayscale_range = [np.amin(img), np.amax(img) / 10]
    return grayscale_range


def adjust_image(img, new_grayscale_range):
    """Translate image data to the appropriate lower bound of the default data
    range. This translation is needed to show properly .raw images.

    :param img: Array containing the loaded .raw image
    :type img: numpy.array
    :param new_grayscale_range: Grayscale range to apply to image
    :type new_grayscale_range: list
    :return: Image corrected with new grayscale range
    :rtype: numpy.array
    """
    # get current image range
    curr_range = [np.amin(img), np.amax(img)]

    if curr_range[0] == curr_range[1]:
        return img

    # translate to "zero out" the data
    new_img = img - curr_range[0]

    # apply a linear stretch of the data such that the selected data range
    # spans the entire default data range
    scale_factor = (new_grayscale_range[1] - new_grayscale_range[0]) / (
        curr_range[1] - curr_range[0]
    )
    new_img = new_img * scale_factor

    new_img = new_img + new_grayscale_range[0]

    # clip all data that falls outside the default range
    new_img[new_img < new_grayscale_range[0]] = new_grayscale_range[0]
    new_img[new_img > new_grayscale_range[1]] = new_grayscale_range[1]

    return new_img
