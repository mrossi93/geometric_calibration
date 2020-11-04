"""Utilities module."""
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.exposure import histogram
from skimage.filters import threshold_otsu
from skimage.measure import regionprops

from scipy.spatial.transform import Rotation as R

matplotlib.rcParams["toolbar"] = "None"


class DraggablePoints:
    """Draggable points on matplotlibe figure.

    Returns:
        DraggablePoints -- DraggablePoints object
    """

    def __init__(self, artists, tolerance=15):
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
        self.fig.canvas.mpl_connect("key_release_event", self.on_key_released)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

        self.currently_dragging = False
        self.fine_mode = False
        self.selected_artist_index = None
        self.offset = np.zeros((1, 2))
        self.x0 = 0
        self.y0 = 0

        plt.title("Drag&Drop red points on the image.\nPress Enter to continue")
        plt.show()

    def on_press(self, event):
        """Event Handler for mouse button pression.

        Arguments:
            event -- Event that triggers the method
        """
        # is the press over some artist
        isonartist = False

        check_index = 0
        for artist in self.artists:
            if artist.contains(event)[0]:
                isonartist = artist
                # Remeber the index of artist selected, it will be needed in
                # case of fine tuning
                self.selected_artist_index = check_index
            check_index += 1

        self.x0 = event.xdata
        self.y0 = event.ydata

        if isonartist and not self.fine_mode:
            # start dragging the entire group
            self.currently_dragging = True
            artist_center = np.array([a.center for a in self.artists])
            event_center = np.array([event.xdata, event.ydata])
            self.offset = artist_center - event_center
        elif isonartist and self.fine_mode:
            # start dragging only selected element
            self.currently_dragging = True
            artist_center = np.array([isonartist.center])
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

        if self.currently_dragging and not self.fine_mode:
            # Update the entire group
            try:
                newcenters = np.array([event.xdata, event.ydata]) + self.offset
                for i, artist in enumerate(self.artists):
                    artist.center = newcenters[i]
            except Exception:
                pass
            self.fig.canvas.draw_idle()
        elif self.currently_dragging and self.fine_mode:
            # Update only selected artist
            newcenter = np.array([event.xdata, event.ydata]) + self.offset
            self.artists[self.selected_artist_index].center = newcenter[0][:]
            self.fig.canvas.draw_idle()

    def on_key_pressed(self, event):
        """Event Handler for "enter" key pression.

        Arguments:
            event -- Event that triggers the method
        """
        if not self.currently_dragging and event.key == "enter":
            plt.close()
        if not self.currently_dragging and event.key == "control":
            self.fine_mode = True

    def on_key_released(self, event):
        if self.fine_mode:
            self.fine_mode = False

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


def drag_and_drop_bbs(projection, bbs_projected, image_center, grayscale_range):
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
        projection,
        cmap="gray",
        vmin=grayscale_range[0],
        vmax=grayscale_range[1],
    )

    # Drag&Drop
    pts = []
    for x, y in zip(bbs_projected[:, 0], bbs_projected[:, 1]):
        point = patches.Circle((x, y), fc="r", alpha=0.5)
        pts.append(point)
        ax.add_patch(point)

    # Append also an invisible point to update also image_center coordinates
    point = patches.Circle(
        (image_center[0], image_center[1]), fc="r", alpha=0.5
    )
    pts.append(point)

    r2d_corrected = DraggablePoints(pts)

    # remember that last coordinate is image_center
    return r2d_corrected.final_coord[:-1], r2d_corrected.final_coord[-1]


def search_bbs_centroids(
    img, ref_2d, search_area, image_size, grayscale_range, debug_level=0
):
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

        # if bbs is not even inside the image, skip it
        if (
            (ind_row < 0)
            or (ind_col < 0)
            or (ind_row > image_size[0])
            or (ind_col > image_size[1])
        ):
            bbs_centroid.append([np.nan, np.nan])
            continue

        # define the field of research
        min_row = int(max([0, ind_row - search_area]))
        min_col = int(max([0, ind_col - search_area]))
        max_row = int(min([ind_row + search_area, image_size[0]]))
        max_col = int(min([ind_col + search_area, image_size[1]]))

        # define a mask on the original image to underline field of research
        sub_img = img[min_col:max_col, min_row:max_row]

        # rescale grey level of the sub-image
        sub_img = adjust_image(sub_img, grayscale_range)

        # Binarize sub_image to extract the bbs
        try:
            thresh = threshold_otsu(sub_img)
        except Exception:
            # we are finished in a total white window, otsu thresholding needs
            # at least two level of gray to work
            bbs_centroid.append([np.nan, np.nan])
            continue
        binary = sub_img > thresh

        # Count how many pixel belong to background vs foreground
        blob = np.count_nonzero(binary == 0)
        background = np.count_nonzero(binary)

        # if background is too dark, discard the bbs
        if blob / background > 0.25:
            bbs_centroid.append([np.nan, np.nan])
            continue

        # Extract centroid from thresholded sub_image
        labeled_foreground = (sub_img < thresh).astype(int)
        properties = regionprops(labeled_foreground, sub_img)
        centroid = properties[0].centroid

        # Append centroid to bbs list
        bbs_centroid.append([min_row + centroid[1], min_col + centroid[0]])

        def on_key_pressed(event):
            if event.key == "enter":
                plt.close()

        if debug_level == 2:
            # Show every centroid found
            hist, hist_centers = histogram(sub_img, nbins=100)

            fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
            fig.canvas.mpl_connect("key_press_event", on_key_pressed)

            ax = axes.ravel()
            ax[0] = plt.subplot(1, 3, 1)
            ax[1] = plt.subplot(1, 3, 2)
            ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

            ax[0].imshow(
                sub_img,
                cmap="gray",
                vmin=grayscale_range[0],
                vmax=grayscale_range[1],
            )
            ax[0].set_title("Original")

            ax[1].plot(hist_centers, hist, lw=2)
            ax[1].set_title("Histogram")
            ax[1].axvline(thresh, color="r")

            ax[2].imshow(binary, cmap=plt.cm.gray)
            ax[2].scatter(centroid[1], centroid[0], marker="*", c="g")
            ax[2].set_title("Thresholded")

            plt.show()

    bbs_centroid = np.array(bbs_centroid)

    if debug_level >= 1:
        # Show final position for found cetroids
        bbs_dbg = bbs_centroid[~np.isnan(bbs_centroid).any(axis=1)]
        print(bbs_dbg)

        fig = plt.figure()
        fig.canvas.mpl_connect("key_press_event", on_key_pressed)

        plt.imshow(
            img, cmap="gray", vmin=grayscale_range[0], vmax=grayscale_range[1],
        )
        plt.scatter(bbs_dbg[:, 0], bbs_dbg[:, 1], marker="x", c="c", alpha=0.5)
        plt.scatter(ref_2d[:, 0], ref_2d[:, 1], marker="x", c="r", alpha=0.5)
        # plt.grid(True, color="r")
        plt.show()

    return bbs_centroid


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

    rot = np.matmul(Rz, np.matmul(Ry, Rx))
    trans = rot
    trans = np.append(trans, np.zeros((3, 1)), axis=1)
    trans = np.append(trans, np.zeros((1, 4)), axis=0)
    trans[3, 3] = 1.0
    return trans


def project_camera_matrix(coord_3d, camera_matrix, image_center):
    """Project 3D data starting from camera matrix based on intrinsic and
    extrinsic parameters

    :param r3d: Array nx3 containing 3d coordinates of points [x,y,z]
    :type r3d: numpy.array
    :param image_center: Center of the image
    :type image_center: list
    :param camera_matrix: Projection matrix obtained combining extrinsic
     and intrinsic parameters
    :type camera_matrix: numpy.array
    :return: Array nx2 containing 2D coordinates of points projected on
     image plane [x,y]
    :rtype: numpy.array
    """
    # homogeneous
    coord_3d = np.append(coord_3d, np.ones((coord_3d.shape[0], 1)), axis=1)

    # Apply proj_matrix and project
    coord_3d = np.matmul(camera_matrix, coord_3d.T).T
    coord_3d[:, 0] = np.divide(coord_3d[:, 0], coord_3d[:, 2]) + image_center[0]
    coord_3d[:, 1] = np.divide(coord_3d[:, 1], coord_3d[:, 2]) + image_center[1]
    coord_2d = coord_3d[:, :2]

    return coord_2d


def create_camera_matrix(panel_orientation, sdd, sid, pixel_spacing, isocenter):
    """Generate projection matrix starting from extrinsic and intrinsic
    parameters.

    :param panel_orientation: Array nx3 containing rotations of the image's
     plane [rot_x, rot_y, rot_z]
    :type panel_orientation: numpy.array
    :param sdd: SDD distance
    :type sdd: float
    :param sid: SID distance
    :type sid: float
    :param pixel_size: Pixel Dimensions in mm
    :type pixel_size: list
    :param isocenter: Coordinates of isocenter
    :type isocenter: numpy.array
    :return: 3x4 Camera Matrix
    :rtype: numpy.array
    """
    # References:
    # - http://ksimek.github.io/2012/08/14/decompose/
    # - http://ksimek.github.io/2012/08/22/extrinsic

    # extrinsic parameters (in homogeneous form)
    extrinsic = np.zeros([4, 4])
    extrinsic[3, 3] = 1

    rot = R.from_euler("zxy", panel_orientation).as_matrix().T
    C = np.array(isocenter)
    transl = -np.matmul(rot, C)

    extrinsic[:3, :3] = rot[:]
    extrinsic[:3, 3] = transl
    extrinsic[2, 3] = extrinsic[2, 3] + sid  # add sid

    # intrinsic parameters
    intrinsic = np.zeros((3, 4))
    intrinsic[0, 0] = sdd / pixel_spacing[0]
    intrinsic[1, 1] = sdd / pixel_spacing[1]
    intrinsic[2, 2] = 1

    # total camera matrix
    camera_matrix = np.matmul(intrinsic, extrinsic)

    return camera_matrix


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
