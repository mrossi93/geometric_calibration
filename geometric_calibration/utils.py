"""
    This module contains some utility function for image manipulation.
"""
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.feature import canny
from skimage import color
from skimage.util import img_as_ubyte
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.draw import disk, circle_perimeter, ellipse, ellipse_perimeter
from skimage.measure import EllipseModel, CircleModel, ransac, label
from skimage.color import label2rgb

from scipy.spatial.transform import Rotation as R

# matplotlib.rcParams["toolbar"] = "None"


class DraggablePoints:
    """
    Draggable points on a matplotlib figure.

    Returns:
        DraggablePoints: DraggablePoints object
    """

    def __init__(self, artists, tolerance=15):
        """
        Initialize an instance of DraggablePoints object and superimpose it on
        the current matplotlib Figure.

        Args:
            artists (list): list of matplotlib Circles with coordinates (x,y)
                in pixel coordinates.
            tolerance (int, optional): Tolerance for mouse selection when
                dragging on screen. Defaults to 15.
        """
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

        plt.title(
            "Drag&Drop red points on the image.\nPress CTRL to move a single point.\nPress Enter to continue"
        )
        plt.show()

    def on_press(self, event):
        """
        Event Handler for mouse button pression.

        Args:
            event (event): Event that triggers the method.
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
        """
        Event Handler for mouse button release.

        Args:
            event (event): Event that triggers the method
        """
        if self.currently_dragging:
            self.currently_dragging = False

    def on_motion(self, event):
        """
        Event Handler for mouse movement during dragging of points.

        Args:
            event (event): Event that triggers the method
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
        """
        Event Handler for "Enter" key pression.

        Args:
            event (event): Event that triggers the method
        """
        if not self.currently_dragging and event.key == "enter":
            plt.close()
        if not self.currently_dragging and event.key == "control":
            self.fine_mode = True

    def on_key_released(self, event):
        """
        Event Handler for any key released.

        Args:
            event (event): Event that triggers the method
        """
        if self.fine_mode:
            self.fine_mode = False

    def on_close(self, event):
        """
        Event Handler for closure of figure.

        Args:
            event (event): Event that triggers the method
        """
        self.final_coord = self.get_coord()

    def get_coord(self):
        """
        Obtain current coordinates (x,y) of points.

        Returns:
            numpy.array: An array Nx2 containing coordinates for N point (x,y).
        """
        return np.array([a.center for a in self.artists])


def drag_and_drop_bbs(projection, bbs_projected):
    """
    Drag&Drop Routines for bbs position's correction.

    Args:
        projection (str): Path to the projection (.raw of .hnc) file
        bbs_projected (numpy.array): Array Nx2 with N BBs yet projected on
            image plane

    Returns:
        numpy.array: Array Nx2 containing the updated coordinates for N BBs
    """
    # Overlay reference bbs with projection
    fig = plt.figure(num="Drag&Drop")

    ax = fig.add_subplot(111)

    # Reference image in background (must stay in position always)
    ax.imshow(projection, cmap="gray")

    # Drag&Drop
    pts = []
    for x, y in zip(bbs_projected[:, 0], bbs_projected[:, 1]):
        point = patches.Circle((x, y), fc="r", alpha=0.5)
        pts.append(point)
        ax.add_patch(point)

    r2d_corrected = DraggablePoints(pts)

    return r2d_corrected.final_coord


def search_bbs_centroids(
    img, ref_2d, search_area, image_size, mode, debug_level=0
):
    """
    Search bbs based on projection.

    Starting from the updated coordinates, define a search area around them
    and identify the BBs centroid as the center of a circle or an ellipse
    (based on mode argument). This function automatically set as (np.nan,
    np.nan) the coordinates of BBs outside image space, too dark or too close
    to another BBs.

    Args:
        img (numpy.array): Array containing the loaded .raw or .hnc file
        ref_2d (numpy.array): Nx2 array containing the coordinates for BBs
            projected on img
        search_area (int): Size of the region in which to search for centroids.
            Actual dimension of the area is a square with dimension (2*
            search_area,2*search_area)
        image_size (list): Dimension of img
        mode (str): Centroid search modality. It can be "circle" or "ellipse".
            Ellipse is slower but provide better results in general.
        debug_level (int, optional): Level for debug messages, 0 means no
            debug messages, 1 light debug and 2 hard debug. Defaults to 0.

    Returns:
        numpy.array: Nx2 array containing coordinates for every centroids found
        (x,y)
    """

    def on_key_pressed(event):
        if event.key == "enter":
            plt.close()

    # TODO valutare se duplicate_tol può diventare un paramentro della funzione
    duplicate_tol = 10  # pixel tolerance to consider two BBs too close

    for i in range(len(ref_2d)):
        if np.isnan(ref_2d[i][0]):
            continue

        for j in range(i + 1, len(ref_2d)):
            if np.isnan(ref_2d[j][0]):
                continue
            if (np.abs(ref_2d[i][0] - ref_2d[j][0]) < duplicate_tol) and (
                np.abs(ref_2d[i][1] - ref_2d[j][1]) < duplicate_tol
            ):
                ref_2d[i][:] = [np.nan, np.nan]
                ref_2d[j][:] = [np.nan, np.nan]
                if debug_level >= 1:
                    print("Reference point discarded: too close to another BB")

    bbs_centroid = []
    for curr_point in ref_2d:  # for each bbs
        if np.isnan(curr_point[0]):
            bbs_centroid.append([np.nan, np.nan])
            continue

        ind_col = round(curr_point[0])
        ind_row = round(curr_point[1])

        # if bbs is not even inside the image, skip it
        if (
            (ind_row < 0)
            or (ind_col < 0)
            or (ind_row > image_size[1])
            or (ind_col > image_size[0])
        ):
            bbs_centroid.append([np.nan, np.nan])
            if debug_level >= 1:
                print("Out of image")
            continue

        # define the field of research
        min_col = int(max([0, ind_col - search_area]))
        min_row = int(max([0, ind_row - search_area]))

        max_col = int(min([ind_col + search_area, image_size[0]]))
        max_row = int(min([ind_row + search_area, image_size[1]]))

        # define a mask on the original image to underline field of research
        sub_img = img[min_row:max_row, min_col:max_col]

        # Contrast stretching
        # TODO Valutare se rimuovere, in alcuni casi da problemi
        # p1, p2 = np.percentile(sub_img, (1, 60))
        # sub_img = exposure.rescale_intensity(sub_img, in_range=(p1, p2))

        # TODO valore migliore?
        edges = canny(sub_img, sigma=1.5)  # 1.5 per ellisse
        # edges = canny(sub_img, sigma=2) # per cerchio

        # label image regions
        label_image, num_label = label(edges, return_num=True)

        coords = []
        for i in range(1, num_label + 1):
            temp_edges = np.zeros(sub_img.shape)
            temp_edges[label_image == i] = 1

            temp_coords = np.column_stack(np.nonzero(temp_edges))
            if len(temp_coords) != 0:
                coords.append(temp_coords)
            else:
                pass

        if len(coords) == 0:
            bbs_centroid.append([np.nan, np.nan])
            if debug_level >= 1:
                print("No edges identified")
            continue

        if mode == "ellipse":
            model, inliers = ransac(
                coords,
                EllipseModel,
                min_samples=3,
                residual_threshold=1,
                max_trials=100,
            )

            if model is None:
                if debug_level >= 1:
                    print("BB discarded: no ellipse found")
                cx = 0.0
                cy = 0.0
                a = 0.0
                b = 0.0
                t = 0.0
                bbs_centroid.append([np.nan, np.nan])
            else:
                cx, cy, a, b, t = [x for x in model.params]

                if a > b:
                    eccentricity = np.sqrt(1 - (b / a) ** 2)
                else:
                    # When orientation is not in [-pi:pi] the function
                    # hough_ellipse returns a and b swapped.
                    eccentricity = np.sqrt(1 - (a / b) ** 2)

                if eccentricity > 0.5:
                    if debug_level >= 1:
                        print("BB discarded: eccentricity is too high")
                    cx = 0.0
                    cy = 0.0
                    a = 0.0
                    b = 0.0
                    t = 0.0
                    bbs_centroid.append([np.nan, np.nan])
                else:
                    bbs_centroid.append([min_col + cx, min_row + cy])
                    if debug_level >= 2:
                        print("---------")
                        print(f"Cx: {cx}, Cy: {cy}")
                        print(f"a: {a}, b: {b}")
                        print(f"Orientation: {t}")
                        print(f"Eccentricity: {eccentricity}")
                        print("---------")

            edges = color.gray2rgb(img_as_ubyte(edges))
            sub_img_cont = np.ones(sub_img.shape)
            sub_img_cont = color.gray2rgb(sub_img_cont)

            # fill ellipse
            rr, cc = ellipse(cx, cy, a, b, sub_img.shape, rotation=t)
            sub_img_cont[rr, cc, :] = (1, 1, 0)
        elif mode == "circle":
            cx_list = []
            cy_list = []
            radius_list = []
            for coord in coords:
                if len(coord) <= 3:
                    pass
                else:
                    model = None
                    model, inliers = ransac(
                        coord,
                        CircleModel,
                        min_samples=3,
                        residual_threshold=1,
                        max_trials=100,
                    )

                if model is None:
                    pass
                else:
                    cx_list.append(model.params[0])
                    cy_list.append(model.params[1])
                    radius_list.append(model.params[2])

            if len(cx_list) == 0:
                if debug_level >= 1:
                    print("BB discarded: no ellipse found")
                cx = 0.0
                cy = 0.0
                radius = 0.0
                bbs_centroid.append([np.nan, np.nan])
            else:
                best_radius = np.argmin(radius_list)
                cx = cx_list[best_radius]
                cy = cy_list[best_radius]
                radius = radius_list[best_radius]

                if radius > 6:
                    if debug_level >= 1:
                        print("BBs discarded. Radius too big")
                    cx = 0.0
                    cy = 0.0
                    radius = 0.0
                    bbs_centroid.append([np.nan, np.nan])
                else:
                    # cx, cy, radius = [x for x in model.params]
                    if debug_level >= 2:
                        print("---------")
                        print(f"Cx: {cx}, Cy: {cy}")
                        print(f"Radius: {radius}")
                        print("---------")
                    bbs_centroid.append([min_col + cy, min_row + cx])

            edges = color.gray2rgb(img_as_ubyte(edges))

            sub_img_cont = np.ones(sub_img.shape)
            sub_img_cont = color.gray2rgb(sub_img_cont)

            # fill circle
            rr, cc = disk((cx, cy), radius, shape=sub_img.shape)
            sub_img_cont[rr, cc, :] = (1, 1, 0)

        if debug_level == 2:
            fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
            fig.canvas.mpl_connect("key_press_event", on_key_pressed)

            ax = axes.ravel()
            ax[0] = plt.subplot(1, 3, 1)
            ax[1] = plt.subplot(1, 3, 2)
            ax[2] = plt.subplot(1, 3, 3)

            ax[0].imshow(sub_img, cmap="gray")
            ax[0].set_title("Original")

            ax[1].set_title("Edges")
            ax[1].imshow(edges)
            ax[1].scatter(cy, cx, c="r", s=2)

            ax[2].set_title("Centroid Found")
            ax[2].imshow(sub_img, cmap="gray")
            ax[2].imshow(sub_img_cont, alpha=0.5)
            ax[2].scatter(cy, cx, c="r", s=2)

            plt.show()

    bbs_centroid = np.array(bbs_centroid)

    for i in range(len(bbs_centroid)):
        if np.isnan(bbs_centroid[i][0]):
            continue

        for j in range(i + 1, len(bbs_centroid)):
            if np.isnan(bbs_centroid[j][0]):
                continue
            if (
                np.abs(bbs_centroid[i][0] - bbs_centroid[j][0]) < duplicate_tol
            ) and (
                np.abs(bbs_centroid[i][1] - bbs_centroid[j][1]) < duplicate_tol
            ):
                bbs_centroid[i][:] = [np.nan, np.nan]
                bbs_centroid[j][:] = [np.nan, np.nan]
                if debug_level >= 1:
                    print("BB discarded: too close to another BB")

    if debug_level >= 1:
        # Show final position for found cetroids
        bbs_dbg = bbs_centroid[~np.isnan(bbs_centroid).any(axis=1)]
        if debug_level >= 2:
            print(f"Centroid found: {bbs_dbg.shape[0]}")
            print("Centroid positions:")
            print(bbs_dbg)

        fig, ax = plt.subplots()
        fig.canvas.mpl_connect("key_press_event", on_key_pressed)

        ax.imshow(img, cmap="gray")
        ax.scatter(
            bbs_dbg[:, 0], bbs_dbg[:, 1], marker="x", c="g"
        )  # , alpha=0.5)
        ax.scatter(ref_2d[:, 0], ref_2d[:, 1], marker="x", c="r", alpha=0.5)
        # plt.grid(True, color="r")
        plt.show()

    return bbs_centroid


def search_bbs_centroids_hough(
    img, ref_2d, search_area, image_size, mode, debug_level=0
):
    """
    Search bbs based on projection.

    Starting from the updated coordinates, define a search area around them
    and identify the BBs centroid as the center of a circle or an ellipse
    (based on mode argument). This function automatically set as (np.nan,
    np.nan) the coordinates of BBs outside image space, too dark or too close
    to another BBs.

    Args:
        img (numpy.array): Array containing the loaded .raw or .hnc file
        ref_2d (numpy.array): Nx2 array containing the coordinates for BBs
            projected on img
        search_area (int): Size of the region in which to search for centroids.
            Actual dimension of the area is a square with dimension (2*
            search_area,2*search_area)
        image_size (list): Dimension of img
        mode (str): Centroid search modality. It can be "circle" or "ellipse".
            Ellipse is slower but provide better results in general.
        debug_level (int, optional): Level for debug messages, 0 means no
            debug messages, 1 light debug and 2 hard debug. Defaults to 0.

    Returns:
        numpy.array: Nx2 array containing coordinates for every centroids found
        (x,y)
    """

    def on_key_pressed(event):
        if event.key == "enter":
            plt.close()

    bbs_centroid = []
    for curr_point in ref_2d:  # for each bbs
        ind_col = round(curr_point[0])
        ind_row = round(curr_point[1])

        # if bbs is not even inside the image, skip it
        if (
            (ind_row < 0)
            or (ind_col < 0)
            or (ind_row > image_size[1])
            or (ind_col > image_size[0])
        ):
            bbs_centroid.append([np.nan, np.nan])
            if debug_level >= 1:
                print("Out of image")
            continue

        # define the field of research
        min_col = int(max([0, ind_col - search_area]))
        min_row = int(max([0, ind_row - search_area]))

        max_col = int(min([ind_col + search_area, image_size[0]]))
        max_row = int(min([ind_row + search_area, image_size[1]]))

        # define a mask on the original image to underline field of research
        sub_img = img[min_row:max_row, min_col:max_col]

        # Contrast stretching
        # TODO Valutare se rimuovere, in alcuni casi da problemi
        # p1, p2 = np.percentile(sub_img, (1, 60))
        # sub_img = exposure.rescale_intensity(sub_img, in_range=(p1, p2))

        # TODO valore migliore?
        edges = canny(sub_img, sigma=1.5)  # per ellisse
        # edges = canny(sub_img, sigma=2) # per cerchio

        if mode == "ellipse":
            # Perform a Hough Transform to find ellipses in image.
            # The accuracy corresponds to the bin size of a major axis.
            # The value is chosen in order to get a single high accumulator.
            # The threshold eliminates low accumulators

            # Working version
            # result = hough_ellipse(image=edges,threshold=4, accuracy=0.1, min_size=1, max_size=10)

            result = hough_ellipse(
                image=edges,
                threshold=4,
                accuracy=0.1,
                min_size=1,
                max_size=5,  # aiuta ad evitare che due bbs molto vicine vengano scambiate per un ellisse enorme
            )

            if result.shape[0] != 0:
                result.sort(order="accumulator")

                # Estimated parameters for the ellipse
                best = list(result[-1])
                accums, cy, cx, a, b, orientation = [x for x in best]

                if a > b:
                    eccentricity = np.sqrt(1 - (b / a) ** 2)
                else:
                    # When orientation is not in [-pi:pi] the function
                    # hough_ellipse returns a and b swapped.
                    eccentricity = np.sqrt(1 - (a / b) ** 2)

                if debug_level >= 2:
                    print("---------")
                    print(f"Accum: {accums}")
                    print(f"Cx: {cx}, Cy: {cy}")
                    print(f"a: {a}, b: {b}")
                    print(f"Orientation: {orientation}")
                    print(f"Eccentricity: {eccentricity}")
                    print("---------")
            else:
                if debug_level >= 1:
                    print("BB discarded: no ellipse found")
                cx = 0
                cy = 0
                a = 0
                b = 0
                orientation = 0
                eccentricity = -1
                bbs_centroid.append([np.nan, np.nan])

            if eccentricity != -1:
                if eccentricity > 0.5:
                    if debug_level >= 1:
                        print("BB discarded: eccentricity is too high")
                    cx = 0
                    cy = 0
                    a = 0
                    b = 0
                    orientation = 0
                    bbs_centroid.append([np.nan, np.nan])
                else:
                    bbs_centroid.append([min_col + cx, min_row + cy])

            edges = color.gray2rgb(img_as_ubyte(edges))

            sub_img_cont = np.ones(sub_img.shape)
            sub_img_cont = color.gray2rgb(sub_img_cont)

            # Draw the ellipse on the original image
            ellipse_y, ellipse_x = ellipse_perimeter(
                int(round(cy)),
                int(round(cx)),
                int(round(a)),
                int(round(b)),
                orientation,
            )
            # Remove countour pixel outside of the image
            # ellipse_y = ellipse_y.reshape((len(ellipse_y, 1)))
            # ellipse_x = ellipse_x.reshape((len(ellipse_x, 1)))
            # ellipse_contours = np.concatenate(ellipse_y, ellipse_x, axis=1)

            ###sub_img_cont[ellipse_y, ellipse_x] = (1, 0, 0)
            # sub_img_cont[int(round(cy)), int(round(cx))] = (1, 0, 0)
            # edges[int(round(cy)), int(round(cx))] = (250, 0, 0)
            ###edges[ellipse_y, ellipse_x] = (250, 0, 0)
        elif mode == "circle":
            # Radii to be detected
            hough_radii = range(2, 10)
            hough_res = hough_circle(
                edges, hough_radii, normalize=True, full_output=False
            )

            # Select the most prominent circle
            accums, cx, cy, radii = hough_circle_peaks(
                hough_res,
                hough_radii,
                min_xdistance=0,
                min_ydistance=0,
                num_peaks=1,
                total_num_peaks=1,
            )
            if debug_level >= 2:
                print("---------")
                print(f"Accum: {accums}")
                print(f"Cx: {cx}, Cy: {cy}")
                print(f"Radius: {radii}")
                print("---------")

            if len(accums) == 0:
                if debug_level >= 1:
                    print("BB discarded: no circle found")
                cx = np.array([0])
                cy = np.array([0])
                radii = np.array([0])
                bbs_centroid.append([np.nan, np.nan])
            elif accums[0] < 0.30:
                if debug_level >= 1:
                    print("BB discarded: image is not sufficiently clear")
                cx[0] = 0
                cy[0] = 0
                radii[0] = 0
                bbs_centroid.append([np.nan, np.nan])
            else:
                bbs_centroid.append(
                    [float(min_col + cx[0]), float(min_row + cy[0])]
                )

            edges = color.gray2rgb(img_as_ubyte(edges))

            sub_img_cont = np.ones(sub_img.shape)
            sub_img_cont = color.gray2rgb(sub_img_cont)

            for center_y, center_x, radius in zip(cy, cx, radii):
                circy, circx = circle_perimeter(
                    center_y, center_x, radius, shape=sub_img.shape
                )
                sub_img_cont[circy, circx] = (1, 0, 0)
                sub_img_cont[cy, cx] = (1, 0, 0)
                edges[cy, cx] = (250, 0, 0)
                edges[circy, circx] = (250, 0, 0)

        if debug_level == 2:
            fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
            fig.canvas.mpl_connect("key_press_event", on_key_pressed)

            ax = axes.ravel()
            ax[0] = plt.subplot(1, 3, 1)
            ax[1] = plt.subplot(1, 3, 2)
            ax[2] = plt.subplot(1, 3, 3)

            ax[0].imshow(sub_img, cmap="gray")
            ax[0].set_title("Original")

            ax[1].set_title("Edges")
            ax[1].imshow(edges)
            ax[1].scatter(cx, cy, c="r", s=2)

            ax[2].set_title("Centroid Found")
            ax[2].imshow(sub_img, cmap="gray")
            # ax[2].imshow(sub_img_cont, alpha=0.5)
            ax[2].scatter(cx, cy, c="r", s=2)

            plt.show()

    bbs_centroid = np.array(bbs_centroid)
    # TODO valutare se duplicate_tol può diventare un paramentro della funzione
    duplicate_tol = 5  # pixel tolerance to consider two BBs too close

    for i in range(len(bbs_centroid)):
        if np.isnan(bbs_centroid[i][0]):
            continue

        for j in range(i + 1, len(bbs_centroid)):
            if np.isnan(bbs_centroid[j][0]):
                continue
            if (
                np.abs(bbs_centroid[i][0] - bbs_centroid[j][0]) < duplicate_tol
            ) and (
                np.abs(bbs_centroid[i][1] - bbs_centroid[j][1]) < duplicate_tol
            ):
                bbs_centroid[i][:] = [np.nan, np.nan]
                bbs_centroid[j][:] = [np.nan, np.nan]
                if debug_level >= 1:
                    print("BB discarded: too close to another BB")

    if debug_level >= 1:
        # Show final position for found cetroids
        bbs_dbg = bbs_centroid[~np.isnan(bbs_centroid).any(axis=1)]
        if debug_level >= 2:
            print(f"Centroid found: {bbs_dbg.shape[0]}")
            print("Centroid positions:")
            print(bbs_dbg)

        fig, ax = plt.subplots()
        fig.canvas.mpl_connect("key_press_event", on_key_pressed)

        ax.imshow(img, cmap="gray")
        ax.scatter(
            bbs_dbg[:, 0], bbs_dbg[:, 1], marker="x", c="g"
        )  # , alpha=0.5)
        ax.scatter(ref_2d[:, 0], ref_2d[:, 1], marker="x", c="r", alpha=0.5)
        # plt.grid(True, color="r")
        plt.show()

    return bbs_centroid


def project_camera_matrix(coord_3d, camera_matrix, image_size):
    """
    Project 3D data (x,y,z) in world coordinate system to 2D (u,v) coordinate
    system using camera matrix computed with
    :py:meth:`geometric_calibration.utils.create_camera_matrix` function.

    Args:
        coord_3d (numpy.array): Nx3 array containing 3D coordinates of points
            (x,y,z) in world coordinate system.
        camera_matrix (numpy.array): 3x4 projection matrix obtained combining
            both extrinsic and intrinsic parameters.
        image_size (list): Dimension of the image

    Returns:
        numpy.array: Nx2 array containing 2D coordinates of points (u,v)
        projected on image plane (u,v)
    """
    # Apply proj_matrix and project
    coord_2d = np.dot(
        camera_matrix,
        np.concatenate((coord_3d.T, np.ones((1, coord_3d.shape[0])))),
    )
    coord_2d = coord_2d / coord_2d[2, :]
    coord_2d = coord_2d[:2, :].T

    coord_2d = coord_2d + np.array(image_size) / 2

    return coord_2d


def create_camera_matrix(
    detector_orientation,
    sdd,
    sid,
    pixel_spacing,
    isocenter,
    proj_offset,
    source_offset,
    image_size,
):
    """
    Generate projection matrix starting from extrinsic and intrinsic
    parameters (according to the rules of creation of a projection matrix).

    Args:
        detector_orientation (numpy.array): Nx3 array containing rotations of
            the image's plane [rot_x, rot_y, rot_z]
        sdd (float): Source to Detector distance
        sid (float): Source to Isocenter distance
        pixel_spacing (list): Pixel dimension in mm
        isocenter (numpy.array): Coordinates of isocenter
        proj_offset (list): Detector offset, expressed as [offset_x, offset_y]
        source_offset (list): Source offset, expressed as [offset_x, offset_y]
        image_size (list): Dimension of image

    Returns:
        numpy.array: 3x4 camera matrix
    """
    # extrinsic parameters (in homogeneous form)
    extrinsic = np.identity(4)
    extrinsic[:3, :3] = R.from_euler("zxy", detector_orientation).as_matrix().T

    # add isocenter projection in extrinsic matrix
    extrinsic[:3, 3] = np.matmul(extrinsic[:3, :3], isocenter)

    # add source offset to the equations
    extrinsic[0, 3] = extrinsic[0, 3] - source_offset[0]
    extrinsic[1, 3] = extrinsic[1, 3] - source_offset[1]
    extrinsic[2, 3] = extrinsic[2, 3] + sid

    # intrinsic parameters
    intrinsic = np.zeros([3, 4])
    intrinsic[0, 0] = sdd / pixel_spacing[0]
    intrinsic[1, 1] = sdd / pixel_spacing[1]
    intrinsic[2, 2] = 1
    # add projection offset to the equations
    intrinsic[0, 2] = (source_offset[0] - proj_offset[0]) / pixel_spacing[0]
    intrinsic[1, 2] = (source_offset[1] - proj_offset[1]) / pixel_spacing[1]

    # camera matrix
    camera_matrix = np.matmul(intrinsic, extrinsic)

    return camera_matrix


def get_grayscale_range(img):
    """
    New grayscale range for .raw or .hnc images, since original values are too
    bright. New range is computed between min of image and one order of
    magnitude less than original image. Worst case scenario [0, 6553.5]
    (since image is loaded as uint16).

    Args:
        img (numpy.array): Array containing the loaded .raw or .hnc image

    Returns:
        list: Grayscale range for current projection
    """
    # image range - lowest and highest gray-level intensity for projection
    # grayscale_range = [np.amin(img), np.amax(img) / 8]
    grayscale_range = [0, 10000]
    return grayscale_range
