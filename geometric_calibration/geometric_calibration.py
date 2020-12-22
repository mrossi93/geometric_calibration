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

from geometric_calibration.dlt import DLTcalib, decompose_camera_matrix


def calibrate_cbct(
    projection_dir,
    bbs_3d,
    sid,
    sdd,
    proj_offset=[0, 0],
    source_offset=[0, 0],
    # center_offset=0,
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
    gantry_offset = 90  # 90 for room Sim, 0 for room 2

    # Read image labels
    logging.info("Reading imgLabels file...")
    try:
        labels_file_path = os.path.join(projection_dir, "imgLabels.txt")
        proj_files, gantry_angles = read_img_label_file(labels_file_path)
        proj_num_init = len(proj_files)
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
    for f, gantry_angle in zip(proj_files, gantry_angles):
        if not (os.path.exists(os.path.join(projection_dir, f))):
            logging.warning(f"File {f} not found. Skip it.")
            files_to_remove.append(f)
            angles_to_remove.append(gantry_angle)

    # Remove files that don't exist
    for f, a in zip(files_to_remove, angles_to_remove):
        proj_files.remove(f)
        gantry_angles.remove(a)

    # Check if image is actually acquired, avoid to process all-black images
    files_to_remove = []
    angles_to_remove = []

    for f, gantry_angle in zip(proj_files, gantry_angles):
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
        proj_files.remove(f)
        gantry_angles.remove(a)

    logging.info(
        f"Files concistency checked. Calibration will be performed on {len(proj_files)}/{proj_num_init} files."
    )

    # Initialize output dictionary
    results = initialize_results()

    logging.info("Calibrating the system. Please Wait...")

    # Calibrate views
    with click.progressbar(
        iterable=range(len(gantry_angles[:10])), fill_char="=", empty_char=" ",
    ) as prog_bar:
        for k in prog_bar:
            # path of the current image
            proj_path = os.path.join(projection_dir, proj_files[k])

            # For first projection, we have only the nominal values, so we need
            # to manually correct bbs position with drag&drop
            if k == 0:
                # Calibrate first view with drag and drop procedure
                proj_results = calibrate_projection(
                    projection_file=proj_path,
                    bbs_3d=bbs_3d,
                    sid=sid,
                    sdd=sdd,
                    gantry_angle=gantry_angles[k],
                    gantry_angle_offset=gantry_offset,
                    proj_offset=[0, 0],
                    source_offset=[0, 0],
                    isocenter=[0, 0, 0],
                    image_size=[768, 1024],
                    pixel_spacing=[0.388, 0.388],
                    search_area=15,
                    drag_and_drop=True,
                    debug_level=debug_level,
                )
            # For every other projection, we can simpy use the results of
            # previous view as a starting point, avoiding drag&drop
            else:
                # initialize geometry (based on previous optimization)
                proj_offset = proj_results["proj_offset"]
                source_offset = proj_results["source_offset"]
                # image_center = view_results["image_center"]

                if k % drag_every != 0:
                    # Calibrate other views without drag and drop procedure
                    proj_results = calibrate_projection(
                        projection_file=proj_path,
                        bbs_3d=bbs_3d,
                        sid=sid,
                        sdd=sdd,
                        gantry_angle=gantry_angles[k],
                        gantry_angle_offset=gantry_offset,
                        proj_offset=proj_offset,
                        source_offset=source_offset,
                        isocenter=[0, 0, 0],
                        image_size=[768, 1024],
                        pixel_spacing=[0.388, 0.388],
                        search_area=15,
                        drag_and_drop=False,
                        debug_level=debug_level,
                    )
                else:
                    # if "drag-every" parameter is setted, calibration for the
                    # current view will be performed after drag&drop
                    proj_results = calibrate_projection(
                        projection_file=proj_path,
                        bbs_3d=bbs_3d,
                        sid=sid,
                        sdd=sdd,
                        gantry_angle=gantry_angles[k],
                        gantry_angle_offset=gantry_offset,
                        proj_offset=proj_offset,
                        source_offset=source_offset,
                        isocenter=[0, 0, 0],
                        image_size=[768, 1024],
                        pixel_spacing=[0.388, 0.388],
                        search_area=15,
                        drag_and_drop=True,
                        debug_level=debug_level,
                    )

            # Update output results dictionary
            results = update_results(
                global_res=results, proj_res=proj_results, proj_path=proj_path
            )

    return results


def calibrate_2d(
    projection_dir,
    bbs_3d,
    sid,
    sdd,
    proj_offset=[0, 0],
    source_offset=[0, 0],
    debug_level=0,
):
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
        if (".raw" in f) or (".hnc" in f):
            if "AP" in f:
                proj_file.append(f)
                gantry_angles.append(0)
            if "RL" in f:
                proj_file.append(f)
                gantry_angles.append(90)

    if len(proj_file) == 0:
        logging.error(
            """AP and RL projection not found.\nPlease check input_path parameter in configuration file."""
        )
        sys.exit(1)

    # Initialize output dictionary
    results = initialize_results()

    logging.info("Calibrating the system. Please Wait...")
    # Calibrate views
    with click.progressbar(
        iterable=range(len(gantry_angles)), fill_char="=", empty_char=" ",
    ) as prog_bar:
        for k in prog_bar:
            # path of the current image
            proj_path = os.path.join(projection_dir, proj_file[k])

            # Calibrate views with drag and drop procedure
            proj_results = calibrate_projection(
                projection_file=proj_path,
                bbs_3d=bbs_3d,
                sid=sid,
                sdd=sdd,
                gantry_angle=gantry_angles[k],
                gantry_angle_offset=0,
                proj_offset=proj_offset,
                source_offset=source_offset,
                image_size=[1536, 2048],
                pixel_spacing=[0.194, 0.194],
                search_area=14,
                drag_and_drop=True,
                debug_level=debug_level,
            )

            # Update output dictionary
            results = update_results(results, proj_results, proj_path)

    return results


def calibrate_projection(
    projection_file,
    bbs_3d,
    sid,
    sdd,
    gantry_angle,
    gantry_angle_offset=0,
    proj_offset=[0, 0],
    source_offset=[0, 0],
    isocenter=[0, 0, 0],
    image_size=[768, 1024],
    pixel_spacing=[0.388, 0.388],
    search_area=7,
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
    ##if image_center is None:  # in case image_center is not declared
    ##    image_center = [image_size[0] / 2, image_size[1] / 2 + center_offset]

    # detector orientation (from panel to brandis reference - rotation along y)
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
        detector_orientation=detector_orientation,
        sdd=sdd,
        sid=sid,
        pixel_spacing=pixel_spacing,
        isocenter=isocenter,
        proj_offset=proj_offset,  # [10, 20],  #
        source_offset=source_offset,  # [30, 40],  #
        image_size=image_size,
    )

    # projected coordinates of brandis on panel plane
    # 2d coordinates of reference points
    bbs_2d = project_camera_matrix(bbs_3d, proj_matrix, image_size)

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
        # bbs_centroid, dlt_weights = search_bbs_centroids(
        bbs_centroid = search_bbs_centroids(
            img=img,
            ref_2d=bbs_2d_corrected,
            search_area=search_area,
            image_size=image_size,
            grayscale_range=grayscale_range,
            debug_level=debug_level,
        )
    else:
        # bbs_centroid, dlt_weights = search_bbs_centroids(
        bbs_centroid = search_bbs_centroids(
            img=img,
            ref_2d=bbs_2d,
            search_area=search_area,
            image_size=image_size,
            grayscale_range=grayscale_range,
            debug_level=debug_level,
        )

    # Calibration - non linear data fitting optimization problem
    good_bbs_index = np.where(~np.isnan(bbs_centroid[:, 0]))[0]

    # Estimated BBs - consider only good bbs
    bbs_2d_init = bbs_centroid[good_bbs_index, :]

    # Real Brandis BBs (skip BBs 3D that don't have a corresponding BBs 2D)
    bbs_3d = bbs_3d[good_bbs_index, :]

    # Compute DLT to find projection matrix
    dlt_camera_matrix, dlt_err = DLTcalib(
        nd=3, xyz=bbs_3d, uv=bbs_2d_init, weights=None, uv_ref=None
    )
    """
    dlt_camera_matrix, dlt_err, dlt_err_gt_init, dlt_err_gt_final = DLTcalib(
        nd=3, xyz=bbs_3d, uv=bbs_2d_init, weights=dlt_weights, uv_ref=bbs_2d
    )
    """

    # dlt_camera_matrix, dlt_err = DLTcalib(3, bbs_3d, bbs_2d)  # Just for tests

    parameters = decompose_camera_matrix(
        dlt_camera_matrix, image_size, pixel_spacing
    )

    if debug_level > 0:
        print("\n..........DLT..........")
        print(f"Error: {dlt_err}\n")

        print(f"sid:       {parameters[0]:>15.3f}")
        print(f"sdd:       {parameters[1]:>15.3f}")
        print(
            f"AngleX:    {parameters[2]:>15.3f} rad -> {np.rad2deg(parameters[2]):>8.3f} deg"
        )
        print(
            f"AngleY:    {parameters[3]:>15.3f} rad -> {np.rad2deg(parameters[3]):>8.3f} deg"
        )
        print(
            f"AngleZ:    {parameters[4]:>15.3f} rad -> {np.rad2deg(parameters[4]):>8.3f} deg"
        )
        print(f"P offset X:{parameters[5]:>15.3f}")
        print(f"P offset Y:{parameters[6]:>15.3f}")
        print(f"S offset X:{parameters[7]:>15.3f}")
        print(f"S offset Y:{parameters[8]:>15.3f}")
        print(".......................")

    refine = True
    if refine is True:
        ###### LM refining ######
        refine_parameters = []
        # refine_parameters.append(parameters[0])  # sid
        # refine_parameters.append(parameters[1])  # sdd
        refine_parameters.append(parameters[2])  # oa
        # refine_parameters.append(parameters[3])  # ga
        refine_parameters.append(parameters[4])  # ia
        refine_parameters.append(parameters[5])  # px
        refine_parameters.append(parameters[6])  # py
        refine_parameters.append(parameters[7])  # sx
        refine_parameters.append(parameters[8])  # sy

        # print("xxxxxxxxxxxxxx")
        # print(refine_parameters)
        # print("xxxxxxxxxxxxxx")

        # Boundaries
        # TODO tolerance limits set by user
        angle_limit = np.deg2rad(0.2)  # rad
        gantry_angle_limit = np.deg2rad(0.2)  # rad
        distance_limit = 3  # mm
        offset_limit = 30  # mm

        low_bound = []
        # low_bound.append(parameters[0] - distance_limit)  # sid
        # low_bound.append(parameters[1] - distance_limit)  # sdd
        low_bound.append(parameters[2] - angle_limit)  # oa
        # low_bound.append(parameters[3] - angle_limit)  # ga
        low_bound.append(parameters[4] - gantry_angle_limit)  # ia
        low_bound.append(parameters[5] - offset_limit)  # px
        low_bound.append(parameters[6] - offset_limit)  # py
        low_bound.append(parameters[7] - offset_limit)  # sx
        low_bound.append(parameters[8] - offset_limit)  # sy

        up_bound = []
        # up_bound.append(parameters[0] + distance_limit)  # sid
        # up_bound.append(parameters[1] + distance_limit)  # sdd
        up_bound.append(parameters[2] + angle_limit)  # oa
        # up_bound.append(parameters[3] + angle_limit)  # ga
        up_bound.append(parameters[4] + gantry_angle_limit)  # ia
        up_bound.append(parameters[5] + offset_limit)  # px
        up_bound.append(parameters[6] + offset_limit)  # py
        up_bound.append(parameters[7] + offset_limit)  # sx
        up_bound.append(parameters[8] + offset_limit)  # sy

        if good_bbs_index.shape[0] >= len(parameters):  # at least 9 BBs
            solution = least_squares(
                fun=calibration_cost_function,
                x0=refine_parameters,
                args=(
                    bbs_3d,
                    bbs_2d_init,
                    pixel_spacing,
                    isocenter,
                    image_size,
                    sid,
                    sdd,
                    detector_orientation[2],
                ),
                method="trf",
                # method="lm",
                bounds=(low_bound, up_bound),
                # verbose=1,
            )
        else:
            logging.error(
                f"""Cannot properly process projection at angle {gantry_angle}.\nPlease acquire again calibration phantom and then retry."""
            )
            sys.exit(1)

        # Solution found
        solution = solution.x
        """
        # [sid, sdd, oa, ga, ia, px, py, sx, sy]
        parameters = []
        parameters.append(solution[0])
        parameters.append(solution[1])
        parameters.append(solution[2])
        parameters.append(solution[3])
        parameters.append(solution[4])
        parameters.append(solution[5])
        parameters.append(solution[6])
        parameters.append(solution[7])
        parameters.append(solution[8])
        
        # [oa, ga, ia, px, py, sx, sy]
        parameters = []
        parameters.append(sid)
        parameters.append(sdd)
        parameters.append(solution[0])
        parameters.append(solution[1])
        parameters.append(solution[2])
        parameters.append(solution[3])
        parameters.append(solution[4])
        parameters.append(solution[5])
        parameters.append(solution[6])
        """
        # [oa, ia, px, py, sx, sy]
        parameters = []
        parameters.append(sid)
        parameters.append(sdd)
        parameters.append(solution[0])
        parameters.append(detector_orientation[2])
        parameters.append(solution[1])
        parameters.append(solution[2])
        parameters.append(solution[3])
        parameters.append(solution[4])
        parameters.append(solution[5])

    # Extract explicitly new parameters from solution
    # Remember: solution has this structure
    # [sid, sdd, oa, ga, ia, px, py, sx, sy]
    sid_new = parameters[0]
    sdd_new = parameters[1]

    # Remember: "ZXY" convention
    detector_orientation_new = np.array(
        [parameters[4], parameters[2], parameters[3]]
    )
    proj_offset_new = np.array([parameters[5], parameters[6]])
    source_offset_new = np.array([parameters[7], parameters[8]])

    # New center of image
    image_center_new = (source_offset_new - proj_offset_new) / np.array(
        pixel_spacing
    ) + np.array(image_size) / 2
    isocenter_new = isocenter

    #### Just to check reprojection ####
    # Project points based on calibration - use new detector orientation,
    # tube and panel position
    proj_matrix_new = create_camera_matrix(
        detector_orientation=detector_orientation_new,
        sdd=sdd_new,
        sid=sid_new,
        pixel_spacing=pixel_spacing,
        isocenter=isocenter,
        proj_offset=proj_offset_new,
        source_offset=source_offset_new,
        image_size=image_size,
    )

    bbs_2d_final = project_camera_matrix(
        coord_3d=bbs_3d, camera_matrix=proj_matrix_new, image_size=image_size
    )

    ls_err = np.mean(np.sqrt(np.sum((bbs_2d_init - bbs_2d_final) ** 2, 1)))

    if (refine is True) and (debug_level > 0):
        print("..........LS...........")
        print(f"Error: {ls_err}\n")
        print(f"sid:       {parameters[0]:>15.3f}")
        print(f"sdd:       {parameters[1]:>15.3f}")
        print(
            f"AngleX:    {parameters[2]:>15.3f} rad -> {np.rad2deg(parameters[2]):>8.3f} deg"
        )
        print(
            f"AngleY:    {parameters[3]:>15.3f} rad -> {np.rad2deg(parameters[3]):>8.3f} deg"
        )
        print(
            f"AngleZ:    {parameters[4]:>15.3f} rad -> {np.rad2deg(parameters[4]):>8.3f} deg"
        )
        print(f"P offset X:{parameters[5]:>15.3f}")
        print(f"P offset Y:{parameters[6]:>15.3f}")
        print(f"S offset X:{parameters[7]:>15.3f}")
        print(f"S offset Y:{parameters[8]:>15.3f}")
        print(".......................")

    if debug_level >= 1:

        def on_key_pressed(event):
            if event.key == "enter":
                plt.close()

        fig = plt.figure(num="Results")
        fig.canvas.mpl_connect("key_press_event", on_key_pressed)
        ax = fig.add_subplot(111)

        # Reference image in background (must stay in position always)
        ax.imshow(
            img, cmap="gray", vmin=grayscale_range[0], vmax=grayscale_range[1],
        )

        # Final Position
        ax.scatter(
            bbs_2d_final[:, 0],
            bbs_2d_final[:, 1],
            s=10,
            facecolors="none",
            edgecolors="g",
        )

        # Found Centroid
        ax.scatter(
            bbs_2d_init[:, 0],
            bbs_2d_init[:, 1],
            s=10,
            alpha=0.5,
            facecolors="r",
            edgecolors="r",
        )

        # Starting position: for first proj it can be very distant
        # For the remaining projections it's the position computed
        # using previous projection's parameters
        ax.scatter(
            bbs_2d[:, 0], bbs_2d[:, 1], s=10, facecolors="none", edgecolors="y",
        )

        plt.show()

    if refine is False:
        error = dlt_err  # _gt_final
    else:
        error = ls_err

    # calculate new source/panel position
    R_new_base = R.from_euler("zxy", detector_orientation_new).as_matrix()

    # Referred to isocenter
    R_new = np.matmul(
        R_new_base.T, R.from_euler("zxy", [-90, 0, 0], degrees=True).as_matrix()
    )

    # source position (X-ray tube)
    ##source_new = np.matmul(R_new, np.array([[0], [0], [sid_new]]))
    source_new = np.matmul(
        R_new_base,
        np.array([[source_offset_new[0]], [source_offset_new[1]], [-sid_new]]),
    )

    # panel position (center of panel)
    # image_center_new_mm = image_center_new * pixel_spacing
    ##image_center_new_mm = image_center_new * pixel_spacing

    # panel_center_new = np.matmul(
    #    R_new, np.array([0], [0], [sid_new - sdd_new]])
    # )

    # panel_new = np.matmul(
    #    R_new,
    #    np.array(
    #        [
    #            [image_center_new_mm[0]],
    #            [image_center_new_mm[1]],
    #            [sid_new - sdd_new],
    #        ]
    #    ),
    # )

    panel_new = np.matmul(
        R_new_base,
        np.array(
            [
                [source_offset_new[0] - proj_offset_new[0]],
                [source_offset_new[1] - proj_offset_new[1]],
                [sdd_new - sid_new],
            ]
        ),
    )

    # TODO: questo va rivisto
    # dim_array = np.array([image_size[0] / 2, image_size[1] / 2, 0], ndmin=2)
    # pixel_array = np.array([pixel_spacing[0], pixel_spacing[1], 1], ndmin=2)

    # offset in local coordinate system
    # panel_offset_new = dim_array * pixel_array

    # referred to isocenter
    # panel_offset_new = panel_center_new - np.matmul(R_new, panel_offset_new.T)

    # update with new value
    results["gantry_angle"] = gantry_angle
    results["detector_orientation"] = detector_orientation_new
    results["sdd"] = sdd_new
    results["sid"] = sid_new
    results["proj_offset"] = proj_offset_new
    results["source_offset"] = source_offset_new
    results["isocenter"] = isocenter_new
    results["source_3d"] = source_new.flatten()
    results["panel_3d"] = panel_new.flatten()  # panel_center_new.flatten()
    results["image_center"] = image_center_new
    results["detector_rot_matrix"] = R_new_base
    results["panel_offset"] = panel_new.flatten()  # _offset_new.flatten()
    results["error"] = error

    return results


def calibration_cost_function(
    param,
    bbs_3d,
    bbs_2d,
    pixel_spacing,
    isocenter,
    image_size,
    sid,
    sdd,
    gantry_angle,
):
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
    """
    ### Ottimizza tutto
    # parameters = [sid, sdd, oa, ga, ia, px, py, sx, sy]
    sid = np.array(param[0])
    sdd = np.array(param[1])
    detector_orientation = np.array([param[4], param[2], param[3]])
    proj_offset = np.array([param[5], param[6]])
    source_offset = np.array([param[7], param[8]])
    
    # Tutto tranne sid, sdd
    # parameters = [oa, ga, ia, px, py, sx, sy]
    detector_orientation = np.array([param[2], param[0], param[1]])
    proj_offset = np.array([param[3], param[4]])
    source_offset = np.array([param[5], param[6]])
    """
    # Tutto tranne sid, sdd, gantry angle
    # parameters = [oa, ia, px, py, sx, sy]
    detector_orientation = np.array([param[1], param[0], gantry_angle])
    proj_offset = np.array([param[2], param[3]])
    source_offset = np.array([param[4], param[5]])

    proj_matrix = create_camera_matrix(
        detector_orientation,
        sdd,
        sid,
        pixel_spacing,
        isocenter,
        proj_offset,
        source_offset,
        image_size,
    )
    # Reprojected 2D points
    r2d = project_camera_matrix(bbs_3d, proj_matrix, image_size)

    # delta = r2d - bbs_2d  # Error
    # diff = np.sqrt(np.square(delta[:, 0]) + np.square(delta[:, 1]))

    # Mean distance:
    # for a, b in zip(bbs_2d, r2d):
    #    print(f"Valori: {a} - {b}")
    # print(f"Diff: {bbs_2d - r2d}")
    # print(f"Squared: {(bbs_2d - r2d)**2}")
    # print(f"SOS: {np.sum((bbs_2d - r2d) ** 2, 1)}")
    # print(f"SQRT: {np.sqrt(np.sum((bbs_2d - r2d) ** 2, 1))}")
    # print(f"Mean: {np.mean(np.sqrt(np.sum((bbs_2d - r2d) ** 2, 1)))}")

    # err = np.mean(np.sqrt(np.sum((bbs_2d - r2d) ** 2, 1)))
    err = np.sqrt(np.sum((bbs_2d - r2d) ** 2, 1))
    # print(f"Err: {err}")
    # print(f"Cost: {(err).sum()}")
    # input()
    # err = np.sqrt(np.sum((bbs_2d - r2d) ** 2, 1))

    return err


def initialize_results():
    results = {
        "proj_path": [],
        "gantry_angles": [],
        "detector_orientation": [],
        "sdd": [],
        "sid": [],
        "proj_offset": [],
        "source_offset": [],
        "isocenter": [],
        "source_3d": [],
        "panel_3d": [],
        "image_center": [],
        "detector_rot_matrix": [],
        "panel_offset": [],
        "error": [],
    }

    return results


def update_results(global_res, proj_res, proj_path):
    global_res["proj_path"].append(proj_path)
    global_res["gantry_angles"].append(proj_res["gantry_angle"])
    global_res["detector_orientation"].append(proj_res["detector_orientation"])
    global_res["sdd"].append(proj_res["sdd"])
    global_res["sid"].append(proj_res["sid"])
    global_res["proj_offset"].append(proj_res["proj_offset"])
    global_res["source_offset"].append(proj_res["source_offset"])
    global_res["isocenter"].append(proj_res["isocenter"])
    global_res["source_3d"].append(proj_res["source_3d"])
    global_res["panel_3d"].append(proj_res["panel_3d"])
    global_res["image_center"].append(proj_res["image_center"])
    global_res["detector_rot_matrix"].append(proj_res["detector_rot_matrix"])
    # global_res["panel_offset"].append(proj_res["panel_offset"])
    global_res["error"].append(proj_res["error"])

    return global_res


def plot_calibration_results(calib_results):
    """Plot source/panel position after calibration.

    :param calib_results: dictionary containing results of a calibration
    :type calib_results: dict
    """
    angles = np.array(calib_results["gantry_angles"])
    rot = np.array(calib_results["detector_rot_matrix"])
    source_pos = np.array(calib_results["source_3d"])
    panel_pos = np.array(calib_results["panel_3d"])

    print(f"Angle: {angles[0]}")
    print(f"Rot matrix:")
    print(rot[0])
    print(f"Source: {source_pos[0,:]}")
    print(f"Panel: {panel_pos[0,:]}")

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
        source_pos[0, 0],
        source_pos[0, 1],
        source_pos[0, 2],
        marker="x",
        c="g",
        label="Source Position",
    )

    ax.scatter(
        panel_pos[0, 0],
        panel_pos[0, 1],
        panel_pos[0, 2],
        marker="x",
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

    for source, panel in zip(source_pos, panel_pos):
        ax.plot(
            [source[0], panel[0]],
            [source[1], panel[1]],
            [source[2], panel[2]],
            c="k",
            linewidth=0.1,
        )

    plt.title("Panel/Source position after calibration\nPress Enter to close")
    ax.set_xlabel("X Label [mm]")
    ax.set_ylabel("Y Label [mm]")
    ax.set_zlabel("Z Label [mm]")
    fig.legend(loc="lower right")
    plt.show()


def plot_calibration_errors(calib_results):
    errors = np.array(calib_results["error"])

    def on_key_pressed(event):
        if event.key == "enter":
            plt.close()

    # Plot panel and source positions (trajectory)
    fig = plt.figure(num="Mean Reprojection Errors")
    fig.canvas.mpl_connect("key_press_event", on_key_pressed)

    ax = fig.add_subplot(111)

    ax.scatter(
        range(len(errors)),
        errors[:],
        marker="."
        # label="Source Position",
    )

    plt.hlines(y=1, xmin=0, xmax=len(errors))

    plt.show()


def save_lut(path, calib_results, style="complete"):
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
    proj_offset = calib_results["proj_offset"]
    source_offset = calib_results["source_offset"]

    if style == "classic":
        num_columns = 6
    elif style == "new":
        num_columns = 8
    elif style == "complete":
        num_columns = 10

    clock = datetime.now()

    filename = "CBCT_LUT{}_{}_{:02}_{:02}-{:02}_{:02}.txt".format(
        num_columns,
        clock.year,
        clock.month,
        clock.day,
        clock.hour,
        clock.minute,
    )
    output_file = os.path.join(path, filename)

    with open(output_file, "w") as res_file:
        res_file.write("#Look Up Table for CBCT reconstruction\n")

        if num_columns == 6:
            res_file.write(
                "#Angle (deg) | Panel Orientation(rad) [X  Y  Z] | Image_center(pixel) X Y\n"
            )
        elif num_columns == 8:
            res_file.write(
                "#Angle (deg) | Panel Orientation(rad) [X  Y  Z] | Image_center(pixel) X Y | SDD(mm) | SID(mm)\n"
            )
        elif num_columns == 10:
            res_file.write(
                "#Angle (deg) | Panel Orientation(rad) [X  Y  Z] | Projection Offset (mm) X Y | SDD(mm) | SID(mm) | Source Offset (mm) X Y\n"
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
            if num_columns == 6:
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
            elif num_columns == 8:
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
            elif num_columns == 10:
                res_file.write(
                    "{:6.12f} {:6.12f} {:6.12f} {:6.12f} {:6.12f} {:6.12f} {:6.12f} {:6.12f} {:6.12f} {:6.12f}\n".format(
                        gantry_angles[k],
                        detector_orientation[k][1],
                        detector_orientation[k][2],
                        detector_orientation[k][0],
                        proj_offset[k][0],
                        proj_offset[k][1],
                        sdd[k],
                        sid[k],
                        source_offset[k][0],
                        source_offset[k][1],
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
