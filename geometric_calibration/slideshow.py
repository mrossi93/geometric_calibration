"""
Modules for slideshow utility of CLI.
"""
import numpy as np
import matplotlib.pyplot as plt
from geometric_calibration.reader import (
    read_projection_hnc,
    read_projection_raw,
)
from geometric_calibration.utils import get_grayscale_range
from geometric_calibration.geometric_calibration import (
    create_camera_matrix,
    project_camera_matrix,
)


class IndexTracker(object):
    """
    Main handler for slideshow
    """

    def __init__(self, ax, X, angles, bbs_2d, grayscale_range):
        """
        Initialize an instance of IndexTracker class.

        Args:
            ax (matplotlib Figure): Figure object in which slideshow is
                presented
            X (list): list of images. Each image is a numpy array
            angles (list): List of gantry angles for each projection
            bbs_2d (list): List of 2D coordinates of BBs for each projection
            grayscale_range (list): Grayscale value found with
                :py:meth:`geometric_calibration.utils.get_grayscale_range`
                function
        """
        self.ax = ax
        self.ax.set_title(
            "Use mouse scroll wheel to navigate between projections.\nPress Enter to close"
        )

        self.X = X
        self.bbs = bbs_2d
        self.angles = angles
        self.slices, rows, cols = X.shape
        self.ind = 0

        self.im = self.ax.imshow(
            self.X[self.ind, :, :],
            cmap="gray",
            vmin=grayscale_range[0],
            vmax=grayscale_range[1],
        )
        self.im_ref = self.ax.scatter(
            self.bbs[self.ind, :, 0],
            self.bbs[self.ind, :, 1],
            marker="x",
            c="r",
            s=10,
            alpha=0.5,
        )
        self.update()

    def onscroll(self, event):
        """
        Event Handler for mouse scroll.

        Args:
            event (event): Event that triggers the method.
        """

        if event.button == "up":
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        """
        Update visualized image on screen.
        """
        self.im.set_data(self.X[self.ind, :, :])
        self.im_ref.remove()
        self.im_ref = self.ax.scatter(
            self.bbs[self.ind, :, 0],
            self.bbs[self.ind, :, 1],
            marker="x",
            c="r",
            s=10,
            alpha=0.5,
        )
        self.ax.set_xlabel(
            f"Proj {self.ind+1} - Angle: {self.angles[self.ind]}"
        )
        self.im.axes.figure.canvas.draw()


def slideshow(calibration_results, bbs_3d, mode):
    """
    Create a slideshow of some projections with found BBs superimposed on each
    image as red circles.

    Args:
        calibration_results (dict): Dictionary containing the results of a
            complete calibration.
        bbs_3d (numpy.array): Array with BBs coodinates in world coordinate system
        mode (str): Calibration modality, it can be "cbct or "2d"
    """
    if mode == "cbct":
        img_dim = [768, 1024]
        pixel_size = [0.388, 0.388]
    elif mode == "2d":
        img_dim = [1536, 2048]
        pixel_size = [0.194, 0.194]

    # Load projections
    projections = []
    bbs_2d = []
    for k in range(len(calibration_results["proj_path"])):
        if ".raw" in calibration_results["proj_path"][k]:
            current_img = read_projection_raw(
                calibration_results["proj_path"][k], img_dim
            )
        elif ".hnc" in calibration_results["proj_path"][k]:
            current_img = read_projection_hnc(
                calibration_results["proj_path"][k], img_dim
            )
        projections.append(current_img)

        proj_matrix = create_camera_matrix(
            detector_orientation=calibration_results["detector_orientation"][k],
            sdd=calibration_results["sdd"][k],
            sid=calibration_results["sid"][k],
            pixel_spacing=pixel_size,
            isocenter=calibration_results["isocenter"][k],
            proj_offset=calibration_results["proj_offset"][k],
            source_offset=calibration_results["source_offset"][k],
            image_size=img_dim,
        )
        # projected coordinates of brandis on panel plane
        curr_bbs_2d = project_camera_matrix(
            coord_3d=bbs_3d, camera_matrix=proj_matrix, image_size=img_dim
        )
        bbs_2d.append(curr_bbs_2d)

    projections = np.array(projections)
    bbs_2d = np.array(bbs_2d)

    grayscale_range = get_grayscale_range(projections)

    gantry_angles = calibration_results["gantry_angles"]

    fig = plt.figure(num="Slideshow")
    ax = fig.add_subplot(111)

    tracker = IndexTracker(
        ax, projections, gantry_angles, bbs_2d, grayscale_range
    )

    def on_key_pressed(event):
        if event.key == "enter":
            plt.close()

    fig.canvas.mpl_connect("key_press_event", on_key_pressed)
    fig.canvas.mpl_connect("scroll_event", tracker.onscroll)

    plt.show()
