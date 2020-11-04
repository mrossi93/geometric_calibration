"""Main module."""

import os
import logging
import sys
from datetime import datetime
import configparser
import numpy as np
from scipy.optimize import least_squares
import click
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from geometric_calibration.reader import (
    read_img_label_file,
    read_projection_hnc,
    read_projection_raw,
)

from geometric_calibration.utils import (
    get_grayscale_range,
    create_camera_matrix,
    project_camera_matrix,
    drag_and_drop_bbs,
    search_bbs_centroids,
)


def calibrate_cbct(
    projection_dir,
    bbs_3d,
    sid,
    sdd,
    center_offset=0,
    drag_every=0,
    debug_level=0,
):
    """Main CBCT Calibration routines.

    :param projection_dir: path to directory containing .raw files
    :type projection_dir: str
    :param bbs_3d: array containing 3D coordinates of BBs
    :type bbs_3d: numpy.array
    :param sid: nominal source to isocenter (A) distance
    :type sid: float
    :param sdd: nominal source to image distance
    :type sdd: float
    :param center_offset: panel offset for half fan mode, defaults to 0
    :type center_offset: float, optional
    :return: dictionary with calibration results
    :rtype: dict
    """
    # RCS: room coordinate system
    # A: isocenter

    # Read image labels
    logging.info("Reading imgLabels file...")
    try:
        labels_file_path = os.path.join(projection_dir, "imgLabels.txt")
        proj_file, gantry_angles = read_img_label_file(labels_file_path)
        proj_num_init = len(proj_file)
    except Exception:
        logging.error(
            """File imgLabels.txt not found in current directory. Please check the path in configuration file."""
        )
        sys.exit(1)
    logging.info("File imgLabels read.")

    logging.info("Check files consistency...")
    # Check if files in imgLabels actually exist.
    files_to_remove = []
    angles_to_remove = []
    for f, gantry_angle in zip(proj_file, gantry_angles):
        if not (os.path.exists(os.path.join(projection_dir, f))):
            logging.warning(f"File {f} not found. Skip it.")
            files_to_remove.append(f)
            angles_to_remove.append(gantry_angle)

    # Remove files that don't exist
    for f, a in zip(files_to_remove, angles_to_remove):
        proj_file.remove(f)
        gantry_angles.remove(a)

    # Check if image is actually acquired, avoid to process all-black images
    files_to_remove = []
    angles_to_remove = []

    for f, gantry_angle in zip(proj_file, gantry_angles):
        if ".raw" in f:
            img = read_projection_raw(
                os.path.join(projection_dir, f), [768, 1024]
            )
        elif ".hnc" in f:
            img = read_projection_hnc(
                os.path.join(projection_dir, f), [768, 1024]
            )

        if np.count_nonzero(img) / (768 * 1024) <= 0.9:
            logging.warning(f"File {f} is empty. Skip it.")
            files_to_remove.append(f)
            angles_to_remove.append(gantry_angle)

    # Remove all-black files
    for f, a in zip(files_to_remove, angles_to_remove):
        proj_file.remove(f)
        gantry_angles.remove(a)

    logging.info(
        f"Files concistency checked. Calibration will be performed on {len(proj_file)}/{proj_num_init} files."
    )

    # Initialize output dictionary
    results = {
        "proj_path": [],
        "gantry_angles": [],
        "detector_orientation": [],
        "sdd": [],
        "sid": [],
        "isocenter": [],
        "source": [],
        "panel": [],
        "image_center": [],
        "detector_rot_matrix": [],
        "panel_offset": [],
        "err_init": [],
        "err_final": [],
    }

    logging.info("Calibrating the system. Please Wait...")

    # Calibrate views
    with click.progressbar(
        iterable=range(len(gantry_angles[:10])), fill_char="=", empty_char=" ",
    ) as prog_bar:
        for k in prog_bar:
            # path of the current image
            proj_path = os.path.join(projection_dir, proj_file[k])

            # For first projection, we have only the nominal values, so we need
            # to manually correct bbs position with drag&drop
            if k == 0:
                # Calibrate first view with drag and drop procedure
                view_results = calibrate_projection(
                    projection_file=proj_path,
                    bbs_3d=bbs_3d,
                    sid=sid,
                    sdd=sdd,
                    gantry_angle=gantry_angles[k],
                    gantry_angle_offset=90,  # 90 for simulation room, 0 for room 2
                    center_offset=center_offset,
                    image_size=[768, 1024],
                    pixel_spacing=[0.388, 0.388],
                    search_area=7,
                    image_center=None,
                    drag_and_drop=True,
                    debug_level=debug_level,
                )
            # For every other projection, we can simpy use the results of
            # previous view as a starting point, avoiding drag&drop
            else:
                # initialize geometry (based on previous optimization)
                gantry_angle_offset = gantry_angles[k] - gantry_angles[k - 1]
                image_center = view_results["image_center"]

                if k % drag_every != 0:
                    # Calibrate other views without drag and drop procedure
                    view_results = calibrate_projection(
                        projection_file=proj_path,
                        bbs_3d=bbs_3d,
                        sid=sid,
                        sdd=sdd,
                        gantry_angle=gantry_angles[k - 1],
                        gantry_angle_offset=90
                        + gantry_angle_offset,  # 90 sim room, 0 room 2
                        image_size=[768, 1024],
                        pixel_spacing=[0.388, 0.388],
                        search_area=7,
                        image_center=image_center,
                        drag_and_drop=False,
                    )
                else:
                    # if "drag-every" parameter is setted, calibration for the
                    # current view will be performed after drag&drop
                    view_results = calibrate_projection(
                        projection_file=proj_path,
                        bbs_3d=bbs_3d,
                        sid=sid,
                        sdd=sdd,
                        gantry_angle=gantry_angles[k - 1],
                        gantry_angle_offset=90
                        + gantry_angle_offset,  # 90 sim room, 0 room 2
                        image_size=[768, 1024],
                        pixel_spacing=[0.388, 0.388],
                        search_area=7,
                        image_center=image_center,
                        drag_and_drop=True,
                        debug_level=debug_level,
                    )

            # Update output results dictionary
            results["proj_path"].append(proj_path)
            results["gantry_angles"].append(view_results["gantry_angle"])
            results["detector_orientation"].append(
                view_results["detector_orientation"]
            )
            results["sdd"].append(view_results["sdd"])
            results["sid"].append(view_results["sid"])
            results["isocenter"].append(view_results["isocenter"])
            results["source"].append(view_results["source"])
            results["panel"].append(view_results["panel"])
            results["image_center"].append(view_results["image_center"])
            results["detector_rot_matrix"].append(
                view_results["detector_rot_matrix"]
            )
            results["panel_offset"].append(view_results["panel_offset"])
            results["err_init"].append(view_results["err_init"])
            results["err_final"].append(view_results["err_final"])

    return results


def calibrate_2d(projection_dir, bbs_3d, sid, sdd, debug_level=0):
    """Main 2D Calibration routines.

    :param projection_dir: path to directory containing .raw files
    :type projection_dir: str
    :param bbs_3d: array containing 3D coordinates of BBs
    :type bbs_3d: numpy.array
    :param sid: nominal source to isocenter (A) distance
    :type sid: float
    :param sdd: nominal source to image distance
    :type sdd: float
    :return: dictionary with calibration results
    :rtype: dict
    """
    # RCS: room coordinate system
    # A: isocenter

    # Find projection files in the current folder
    proj_file = []
    gantry_angles = []

    for f in os.listdir(projection_dir):
        if ("AP" or "RL") and (".raw" or ".hnc") in f:
            proj_file.append(f)
            if "AP" in f:
                gantry_angles.append(0)
            elif "RL" in f:
                gantry_angles.append(90)

    if len(proj_file) == 0:
        logging.error(
            """AP and RL projection not found.
Please check input_path parameter in configuration file."""
        )
        sys.exit(1)

    # Initialize output dictionary
    results = {
        "proj_path": [],
        "gantry_angles": [],
        "detector_orientation": [],
        "sdd": [],
        "sid": [],
        "isocenter": [],
        "source": [],
        "panel": [],
        "image_center": [],
        "detector_rot_matrix": [],
        "panel_offset": [],
        "err_init": [],
        "err_final": [],
    }

    logging.info("Calibrating the system. Please Wait...")
    # Calibrate views
    with click.progressbar(
        iterable=range(len(gantry_angles)), fill_char="=", empty_char=" ",
    ) as prog_bar:
        for k in prog_bar:
            # path of the current image
            proj_path = os.path.join(projection_dir, proj_file[k])

            # Calibrate views with drag and drop procedure
            view_results = calibrate_projection(
                projection_file=proj_path,
                bbs_3d=bbs_3d,
                sid=sid,
                sdd=sdd,
                gantry_angle=gantry_angles[k],
                gantry_angle_offset=0,
                image_size=[1536, 2048],
                pixel_spacing=[0.194, 0.194],
                search_area=14,
                drag_and_drop=True,
                debug_level=debug_level,
            )

            # Update output dictionary
            results["proj_path"].append(proj_path)
            results["gantry_angles"].append(view_results["gantry_angle"])
            results["detector_orientation"].append(
                view_results["detector_orientation"]
            )
            results["sdd"].append(view_results["sdd"])
            results["sid"].append(view_results["sid"])
            results["isocenter"].append(view_results["isocenter"])
            results["source"].append(view_results["source"])
            results["panel"].append(view_results["panel"])
            results["image_center"].append(view_results["image_center"])
            results["detector_rot_matrix"].append(
                view_results["detector_rot_matrix"]
            )
            results["panel_offset"].append(view_results["panel_offset"])
            results["err_init"].append(view_results["err_init"])
            results["err_final"].append(view_results["err_final"])

    return results


def calibrate_projection(
    projection_file,
    bbs_3d,
    sid,
    sdd,
    gantry_angle,
    gantry_angle_offset=0,
    center_offset=0,
    image_size=[768, 1024],
    pixel_spacing=[0.388, 0.388],
    search_area=7,
    image_center=None,
    drag_and_drop=True,
    debug_level=0,
):
    """Calibration of a single projection.

    :param projection_file: path to file
    :type projection_file: str
    :param bbs_3d: 3D coordinates of phantom's reference points
    :type bbs_3d: numpy.array
    :param sid: nominal source to isocenter (A) distance
    :type sid: float
    :param sdd: nominal source to image distance
    :type sdd: float
    :param angle: gantry angle for current projection
    :type angle: float
    :param angle_offset: angle offset for panel, defaults to 0
    :type angle_offset: int, optional
    :param center_offset: panel shift for half fan reconstruction, defaults to None
    :type center_offset: float, optional
    :param img_dim: image dimensions in pixels, defaults to [768, 1024]
    :type img_dim: list, optional
    :param pixel_size: pixel dimensions in mm, defaults to [0.388, 0.388]
    :type pixel_size: list, optional
    :param search_area: dimension of reference point's searching area, defaults
     to 7
    :type search_area: int, optional
    :param resolution_factor: resolution factor, when mode is "cbct" this
     parameter equals to 1, in 2D mode is 2 (because resolution is doubled),
     defaults to 1
    :type resolution_factor: int, optional
        :param image_center: [description], defaults to None
    :type image_center: [type], optional
    :param image_center: center of image, defaults to None
    :type image_center: list, optional
    :param drag_and_drop: whether or not perform Drag&Drop correction routines,
     typically set to True for first projection. Defaults to True
    :type drag_and_drop: bool, optional
    :raises Exception: if less than 5 BBs centroids are recognized, optimizer
     automatically fails since calibration can't be consider reliable
    :return: dictionary with calibration results for current projection
    :rtype: dict
    """

    results = {}

    # center_offset is the shift of the panel in half fan mode
    if image_center is None:  # in case image_center is not declared
        image_center = [image_size[0] / 2, image_size[1] / 2 + center_offset]

    isocenter = [0, 0, 0]

    # panel orientation (from panel to brandis reference - rotation along y)
    # Notation "zxy" to avoid gimbal lock on y=90
    # out_of_plane_angle is rotation around x
    # in_plane_angle is rotation around z
    out_of_plane_angle = 0
    in_plane_angle = 0
    detector_orientation = np.deg2rad(
        np.array(
            [
                in_plane_angle,
                out_of_plane_angle,
                (gantry_angle + gantry_angle_offset),
            ]
        )
    )

    # Load projection
    if ".raw" in projection_file:
        img = read_projection_raw(projection_file, image_size)
    elif ".hnc" in projection_file:
        img = read_projection_hnc(projection_file, image_size)

    # Project points starting from extrinsic and intrinsic parameters
    # generate proj_matrix (extrinsic and intrinsic parameters)
    proj_matrix = create_camera_matrix(
        detector_orientation, sdd, sid, pixel_spacing, isocenter
    )
    # projected coordinates of brandis on panel plane
    # 2d coordinates of reference points
    bbs_2d = project_camera_matrix(bbs_3d, proj_matrix, image_center)

    grayscale_range = get_grayscale_range(img)

    if drag_and_drop is True:
        # Overlay reference bbs with projection
        bbs_2d_corrected = drag_and_drop_bbs(
            projection=img,
            bbs_projected=bbs_2d,
            grayscale_range=grayscale_range,
        )

    # Starting from the updated coordinates, define a search area around them
    # and identify the bbs as black pixels inside these areas (brandis are used
    # as probes)
    if drag_and_drop is True:
        bbs_centroid = search_bbs_centroids(
            img=img,
            ref_2d=bbs_2d_corrected,
            search_area=search_area,
            image_size=image_size,
            grayscale_range=grayscale_range,
            debug_level=debug_level,
        )
    else:
        bbs_centroid = search_bbs_centroids(
            img=img,
            ref_2d=bbs_2d,
            search_area=search_area,
            image_size=image_size,
            grayscale_range=grayscale_range,
        )

    # Calibration - non linear data fitting optimization problem
    good_bbs_index = np.where(~np.isnan(bbs_centroid[:, 0]))[0]

    # Estimated BBs - consider only good bbs
    bbs_estim_init = bbs_centroid[~np.isnan(bbs_centroid).any(axis=1)]
    # Real Brandis BBs
    bbs_real_init = bbs_3d[good_bbs_index, :]

    # Initialization of parameters
    parameters = np.append(detector_orientation, image_center).tolist()
    parameters.append(sdd)
    parameters.append(sid)

    # Boundaries
    angle_limit = 0.1  # rad
    distance_limit = 3  # mm
    center_limit = 10  # pixel
    low_bound = [
        -angle_limit,
        -angle_limit,
        detector_orientation[2] - angle_limit,
        image_center[0] - center_limit,
        image_center[1] - center_limit,
        sdd - distance_limit,
        sid - distance_limit,
    ]
    up_bound = [
        angle_limit,
        angle_limit,
        detector_orientation[2] + angle_limit,
        image_center[0] + center_limit,
        image_center[1] + center_limit,
        sdd + distance_limit,
        sid + distance_limit,
    ]

    if good_bbs_index.shape[0] >= 5:  # at least 5 BBs
        solution = least_squares(
            fun=calibration_cost_function,
            x0=parameters,
            args=(bbs_real_init, bbs_estim_init, pixel_spacing, isocenter),
            method="trf",
            bounds=(low_bound, up_bound),
            # verbose=2,
        )

        # Solution found
        solution = solution.x

        # New detector orientation ("ZXY" convention)
        detector_orientation_new = np.array(solution[:3])

        # New center of image
        image_center_new = np.array(solution[3:5])

        sdd_new = solution[5]
        sid_new = solution[6]

        isocenter_new = isocenter

    else:
        logging.error(
            f"""Cannot properly process projection at angle {gantry_angle}.
Please acquire again calibration phantom and then retry."""
        )
        sys.exit(1)
        # raise Exception(
        #    f"""Cannot properly process projection at angle {angle}. Please Retry.
        # Tip: Try to better overlap reference with projection"""
        # )

    # Project points based on calibration - use new panel orientation,
    # tube and panel position
    proj_matrix_new = create_camera_matrix(
        detector_orientation_new, sdd_new, sid_new, pixel_spacing, isocenter_new
    )

    bbs_estim_final = project_camera_matrix(
        bbs_3d, proj_matrix_new, image_center_new
    )

    # calculate improvement: estimated - projected
    err_init = bbs_estim_init - bbs_2d[good_bbs_index, :]
    err_final = bbs_estim_init - bbs_estim_final[good_bbs_index, :]

    err_init = np.mean(abs(err_init))
    err_final = np.mean(abs(err_final))

    # calculate new source/panel position
    R_new = R.from_euler("zxy", detector_orientation_new).as_matrix()

    # Referred to isocenter
    R_new = np.matmul(
        R_new.T, R.from_euler("zxy", [-90, 0, 0], degrees=True).as_matrix()
    )

    # source position (X-ray tube)
    source_new = np.matmul(R_new, np.array([[0], [0], [sid_new]]))

    # panel position (center of panel)
    panel_center_new = np.matmul(
        R_new, np.array([[0], [0], [sid_new - sdd_new]])
    )

    # TODO: questo va rivisto
    dim_array = np.array([image_size[0] / 2, image_size[1] / 2, 0], ndmin=2)
    pixel_array = np.array([pixel_spacing[0], pixel_spacing[1], 1], ndmin=2)

    # offset in local coordinate system
    panel_offset_new = dim_array * pixel_array

    # referred to isocenter
    panel_offset_new = panel_center_new - np.matmul(R_new, panel_offset_new.T)

    # update with new value
    results["gantry_angle"] = gantry_angle
    results["detector_orientation"] = detector_orientation_new
    results["sdd"] = sdd_new
    results["sid"] = sid_new
    results["isocenter"] = isocenter_new
    results["source"] = source_new.flatten()
    results["panel"] = panel_center_new.flatten()
    results["image_center"] = image_center_new
    results["detector_rot_matrix"] = R_new
    results["panel_offset"] = panel_offset_new.flatten()
    results["err_init"] = err_init
    results["err_final"] = err_final

    return results


def calibration_cost_function(param, bbs_3d, bbs_2d, pixel_size, isocenter):
    """Cost Function for calibration optimizers.

    :param param: parameters to be optimized
    :type param: list
    :param bbs_3d: 3D coordinates of reference BBs
    :type bbs_3d: numpy.array
    :param pixel_size: pixel dimensions in mm
    :type pixel_size: list
    :param bbs_2d: 2D coordinates of BBs projected on the current image
    :type bbs_2d: numpy.array
    :param isocenter: coordinates of isocenter
    :type isocenter: numpy.array
    :return: cost function value to be minimized
    :rtype: float
    """
    # unknown
    detector_orientation = np.array(param[:3])
    image_center = np.array(param[3:5])
    sdd = np.array(param[5])
    sid = np.array(param[6])

    proj_matrix = create_camera_matrix(
        detector_orientation, sdd, sid, pixel_size, isocenter
    )
    r2d = project_camera_matrix(bbs_3d, proj_matrix, image_center)

    delta = r2d - bbs_2d  # Error

    diff = np.sqrt(np.square(delta[:, 0]) + np.square(delta[:, 1]))

    return diff


def plot_calibration_results(calib_results):
    """Plot source/panel position after calibration.

    :param calib_results: dictionary containing results of a calibration
    :type calib_results: dict
    """
    source_pos = np.array(calib_results["source"])
    panel_pos = np.array(calib_results["panel"])
    # panel_pos = np.array(calib_results["panel_offset"])
    isocenter = np.array(calib_results["isocenter"])

    def on_key_pressed(event):
        if event.key == "enter":
            plt.close()

    # Plot panel and source positions (trajectory)
    fig = plt.figure(num="Source/Panel Position")
    fig.canvas.mpl_connect("key_press_event", on_key_pressed)

    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        source_pos[:, 0],
        source_pos[:, 1],
        source_pos[:, 2],
        marker=".",
        c="g",
        label="Source Position",
    )

    ax.scatter(
        panel_pos[:, 0],
        panel_pos[:, 1],
        panel_pos[:, 2],
        marker=".",
        c="r",
        label="Panel Position",
    )

    ax.scatter(
        isocenter[0, 0],
        isocenter[0, 1],
        isocenter[0, 2],
        marker=".",
        c="b",
        label="Isocenter Position",
    )

    plt.title("Panel/Source position after calibration\nPress Enter to close")
    ax.set_xlabel("X Label [mm]")
    ax.set_ylabel("Y Label [mm]")
    ax.set_zlabel("Z Label [mm]")
    fig.legend(loc="lower right")
    plt.show()


def save_lut_new_style(path, calib_results):
    """Save LUT file for a calibration.

    :param path: path to .raw file directory, where LUT will be saved
    :type path: str
    :param calib_results: dictionary containing results for a calibration
    :type calib_results: dict
    """
    gantry_angles = calib_results["gantry_angles"]
    detector_orientation = calib_results["detector_orientation"]
    image_center = calib_results["image_center"]
    sdd = calib_results["sdd"]
    sid = calib_results["sid"]

    clock = datetime.now()

    filename = "CBCT_LUT_{}_{:02}_{:02}-{:02}_{:02}_{:02}.txt".format(
        clock.year,
        clock.month,
        clock.day,
        clock.hour,
        clock.minute,
        clock.second,
    )
    output_file = os.path.join(path, filename)

    with open(output_file, "w") as res_file:
        res_file.write("#Look Up Table for CBCT reconstruction\n")
        res_file.write(
            "#Angle (deg) | Panel Orientation (rad) [X  Y  Z] | Image_center (pixel) X Y | SDD (mm) | SID (mm)\n"
        )
        res_file.write(
            "#Date:{}_{}_{}_Time:{}_{}_{}.{}\n".format(
                clock.year,
                clock.month,
                clock.day,
                clock.hour,
                clock.minute,
                clock.second,
                clock.microsecond,
            )
        )
        res_file.write("#\n")
        res_file.write("# --> END OF HEADER. FIXED SIZE: 5 lines. \n")

        for k in range(len(gantry_angles)):
            res_file.write(
                "{:6.12f} {:6.12f} {:6.12f} {:6.12f} {:6.12f} {:6.12f} {:6.12f} {:6.12f}\n".format(
                    gantry_angles[k],
                    detector_orientation[k][1],
                    detector_orientation[k][2],
                    detector_orientation[k][0],
                    image_center[k][0],
                    image_center[k][1],
                    sdd[k],
                    sid[k],
                )
            )

        res_file.write(
            r"# END OF FILE. REQUIRED TO ENSURE '\n' at the end of last calibration line. NO MORE LINES AFTER THIS!!!"
        )


def save_lut_classic_style(path, calib_results):
    """Save LUT file for a calibration.

    :param path: path to .raw file directory, where LUT will be saved
    :type path: str
    :param calib_results: dictionary containing results for a calibration
    :type calib_results: dict
    """
    gantry_angles = calib_results["gantry_angles"]
    detector_orientation = calib_results["detector_orientation"]
    image_center = calib_results["image_center"]

    clock = datetime.now()

    filename = "CBCT_LUT_{}_{:02}_{:02}-{:02}_{:02}_{:02}.txt".format(
        clock.year,
        clock.month,
        clock.day,
        clock.hour,
        clock.minute,
        clock.second,
    )
    output_file = os.path.join(path, filename)

    with open(output_file, "w") as res_file:
        res_file.write("#Look Up Table for CBCT reconstruction\n")
        res_file.write(
            "#Angle (deg) | Panel Orientation (rad) [X  Y  Z] | Image_center (pixel) X Y\n"
        )
        res_file.write(
            "#Date:{}_{}_{}_Time:{}_{}_{}.{}\n".format(
                clock.year,
                clock.month,
                clock.day,
                clock.hour,
                clock.minute,
                clock.second,
                clock.microsecond,
            )
        )
        res_file.write("#\n")
        res_file.write("# --> END OF HEADER. FIXED SIZE: 5 lines. \n")

        for k in range(len(gantry_angles)):
            res_file.write(
                "{:6.12f} {:6.12f} {:6.12f} {:6.12f} {:6.12f} {:6.12f}\n".format(
                    gantry_angles[k],
                    detector_orientation[k][1],
                    detector_orientation[k][2],
                    detector_orientation[k][0],
                    image_center[k][0],
                    image_center[k][1],
                )
            )

        res_file.write(
            r"# END OF FILE. REQUIRED TO ENSURE '\n' at the end of last calibration line. NO MORE LINES AFTER THIS!!!"
        )


def save_lut_planar(path, calib_results):
    output_file = os.path.join(path, "geometryCalibration.ini")

    proj_files = calib_results["proj_path"]
    for index in range(len(proj_files)):
        if "AP" in os.path.basename(proj_files[index]):
            ap_index = index
        elif "RL" in os.path.basename(proj_files[index]):
            rl_index = index

    config = configparser.ConfigParser()
    config["GENERAL_R2"] = {}
    config["AP_COUCH"] = {}
    config["RL_COUCH"] = {}

    general = config["GENERAL_R2"]
    general["pixSpace"] = "0.194"
    general["dimH"] = "2048"
    general["dimL"] = "1536"

    ap_couch = config["AP_COUCH"]
    ap_couch["sp_0"] = str(calib_results["source"][ap_index][0])
    ap_couch["sp_1"] = str(calib_results["source"][ap_index][1])
    ap_couch["sp_2"] = str(calib_results["source"][ap_index][2])
    ap_couch["A_0"] = str(calib_results["detector_rot_matrix"][ap_index][0][0])
    ap_couch["A_1"] = str(calib_results["detector_rot_matrix"][ap_index][1][0])
    ap_couch["A_2"] = str(calib_results["detector_rot_matrix"][ap_index][2][0])
    ap_couch["A_3"] = str(calib_results["detector_rot_matrix"][ap_index][0][1])
    ap_couch["A_4"] = str(calib_results["detector_rot_matrix"][ap_index][1][1])
    ap_couch["A_5"] = str(calib_results["detector_rot_matrix"][ap_index][2][1])
    ap_couch["A_6"] = str(calib_results["detector_rot_matrix"][ap_index][0][2])
    ap_couch["A_7"] = str(calib_results["detector_rot_matrix"][ap_index][1][2])
    ap_couch["A_8"] = str(calib_results["detector_rot_matrix"][ap_index][2][2])
    ap_couch["A_off_0"] = str(calib_results["panel_offset"][ap_index][0])
    ap_couch["A_off_1"] = str(calib_results["panel_offset"][ap_index][1])
    ap_couch["A_off_2"] = str(calib_results["panel_offset"][ap_index][2])

    rl_couch = config["RL_COUCH"]
    rl_couch["sp_0"] = str(calib_results["source"][rl_index][0])
    rl_couch["sp_1"] = str(calib_results["source"][rl_index][1])
    rl_couch["sp_2"] = str(calib_results["source"][rl_index][2])
    rl_couch["A_0"] = str(calib_results["detector_rot_matrix"][rl_index][0][0])
    rl_couch["A_1"] = str(calib_results["detector_rot_matrix"][rl_index][1][0])
    rl_couch["A_2"] = str(calib_results["detector_rot_matrix"][rl_index][2][0])
    rl_couch["A_3"] = str(calib_results["detector_rot_matrix"][rl_index][0][1])
    rl_couch["A_4"] = str(calib_results["detector_rot_matrix"][rl_index][1][1])
    rl_couch["A_5"] = str(calib_results["detector_rot_matrix"][rl_index][2][1])
    rl_couch["A_6"] = str(calib_results["detector_rot_matrix"][rl_index][0][2])
    rl_couch["A_7"] = str(calib_results["detector_rot_matrix"][rl_index][1][2])
    rl_couch["A_8"] = str(calib_results["detector_rot_matrix"][rl_index][2][2])
    rl_couch["A_off_0"] = str(calib_results["panel_offset"][rl_index][0])
    rl_couch["A_off_1"] = str(calib_results["panel_offset"][rl_index][1])
    rl_couch["A_off_2"] = str(calib_results["panel_offset"][rl_index][2])

    with open(output_file, "w") as configfile:
        config.write(configfile)
