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
    def __init__(self, ax, X, angles, bbs_2d, grayscale_range):
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
            marker=".",
            c="r",
            s=30,
            alpha=0.5,
        )
        self.update()

    def onscroll(self, event):
        if event.button == "up":
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind, :, :])
        self.im_ref.remove()
        self.im_ref = self.ax.scatter(
            self.bbs[self.ind, :, 0],
            self.bbs[self.ind, :, 1],
            marker=".",
            c="r",
            s=30,
            alpha=0.5,
        )
        self.ax.set_xlabel(
            f"Proj {self.ind+1} - Angle: {self.angles[self.ind]}"
        )
        self.im.axes.figure.canvas.draw()


def slideshow(calibration_results, bbs_3d, mode):
    if mode == "cbct":
        img_dim = [1024, 768]
        pixel_size = [0.388, 0.388]
    elif mode == "2d":
        img_dim = [2048, 1536]
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

        T = create_camera_matrix(
            calibration_results["panel_orientation"][k],
            calibration_results["sdd"][k],
            calibration_results["sid"][k],
            pixel_size,
            calibration_results["isocenter"][k],
        )
        # projected coordinates of brandis on panel plane
        curr_bbs_2d = project_camera_matrix(
            bbs_3d, calibration_results["img_center"][k], T
        )
        bbs_2d.append(curr_bbs_2d)

    projections = np.array(projections)
    bbs_2d = np.array(bbs_2d)

    grayscale_range = get_grayscale_range(projections)

    angles = calibration_results["proj_angles"]

    fig = plt.figure(num="Slideshow")
    ax = fig.add_subplot(111)

    tracker = IndexTracker(ax, projections, angles, bbs_2d, grayscale_range)

    def on_key_pressed(event):
        if event.key == "enter":
            plt.close()

    fig.canvas.mpl_connect("key_press_event", on_key_pressed)
    fig.canvas.mpl_connect("scroll_event", tracker.onscroll)

    plt.show()
