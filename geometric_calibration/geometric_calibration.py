"""Main module."""

import os
from datetime import datetime
import numpy as np
from scipy.optimize import least_squares
import click
import matplotlib.pyplot as plt

from geometric_calibration.reader import (
    read_img_label_file,
    read_projection_hnc,
    read_projection_raw,
)

from geometric_calibration.utils import (
    deg2rad,
    angle2rotm,
    get_grayscale_range,
    create_camera_matrix,
    project_camera_matrix,
    drag_and_drop_bbs,
    search_bbs_centroids,
)


def calibrate_cbct(projection_dir, bbs_3d, sad, sid):
    """Main CBCT Calibration routines.

    :param projection_dir: path to directory containing .raw files
    :type projection_dir: str
    :param bbs_3d: array containing 3D coordinates of BBs
    :type bbs_3d: numpy.array
    :param sad: nominal source to isocenter (A) distance
    :type sad: float
    :param sid: nominal source to image distance
    :type sid: float
    :return: dictionary with calibration results
    :rtype: dict
    """
    # RCS: room coordinate system
    # A: isocenter

    # Read image labels
    labels_file_path = os.path.join(projection_dir, "imgLabels.txt")
    proj_file, angles = read_img_label_file(labels_file_path)

    # Initialize output dictionary
    results = {
        "proj_angles": [],
        "panel_orientation": [],
        "sid": [],
        "sad": [],
        "isocenter": [],
        "source": [],
        "panel": [],
        "img_center": [],
        "err_init": [],
        "err_final": [],
    }

    # Calibrate views
    with click.progressbar(
        iterable=range(len(angles)), fill_char="=", empty_char=" "
    ) as prog_bar:
        for k in prog_bar:
            proj_path = os.path.join(
                projection_dir, proj_file[k]
            )  # path of the current image

            if k == 0:  # no indications other than nominal values
                # Calibrate first view with drag and drop procedure
                view_results = calibrate_projection(
                    proj_path,
                    bbs_3d,
                    sad=sad,
                    sid=sid,
                    angle=angles[k],
                    angle_offset=0,
                    img_dim=[1024, 768],
                    pixel_size=[0.388, 0.388],
                    search_area=7,
                    drag_and_drop=True,
                )
            else:  # if not first iteration
                # initialize geometry (based on previous optimization)
                angle_offset = angles[k] - angles[k - 1]
                image_center = view_results["img_center"]

                # Calibrate other views without drag and drop procedure
                view_results = calibrate_projection(
                    proj_path,
                    bbs_3d,
                    sad=sad,
                    sid=sid,
                    angle=angles[k - 1],
                    angle_offset=angle_offset,
                    img_dim=[1024, 768],
                    pixel_size=[0.388, 0.388],
                    search_area=7,
                    image_center=image_center,
                    drag_and_drop=False,
                )

            # Update output dictionary
            results["proj_angles"].append(view_results["proj_angle"])
            results["panel_orientation"].append(
                view_results["panel_orientation"]
            )
            results["sid"].append(view_results["sid"])
            results["sad"].append(view_results["sad"])
            results["isocenter"].append(view_results["isocenter"])
            results["source"].append(view_results["source"])
            results["panel"].append(view_results["panel"])
            results["img_center"].append(view_results["img_center"])
            results["err_init"].append(view_results["err_init"])
            results["err_final"].append(view_results["err_final"])

    return results


def calibrate_2d(projection_dir, bbs_3d, sad, sid):
    """Main 2D Calibration routines.

    :param projection_dir: path to directory containing .raw files
    :type projection_dir: str
    :param bbs_3d: array containing 3D coordinates of BBs
    :type bbs_3d: numpy.array
    :param sad: nominal source to isocenter (A) distance
    :type sad: float
    :param sid: nominal source to image distance
    :type sid: float
    :return: dictionary with calibration results
    :rtype: dict
    """
    # RCS: room coordinate system
    # A: isocenter

    # Find projection files in the current folder
    proj_file = []
    angles = []

    for f in os.listdir(projection_dir):
        if ("AP" or "RL") and (".raw" or ".hnc") in f:
            proj_file.append(f)
            if "AP" in f:
                angles.append(0)
            elif "RL" in f:
                angles.append(90)

    # Initialize output dictionary
    results = {
        "proj_angles": [],
        "panel_orientation": [],
        "sid": [],
        "sad": [],
        "isocenter": [],
        "source": [],
        "panel": [],
        "img_center": [],
        "err_init": [],
        "err_final": [],
    }

    # Calibrate views
    with click.progressbar(
        iterable=range(len(angles)), fill_char="=", empty_char=" ",
    ) as prog_bar:
        for k in prog_bar:
            proj_path = os.path.join(
                projection_dir, proj_file[k]
            )  # path of the current image

            # Calibrate views with drag and drop procedure
            view_results = calibrate_projection(
                proj_path,
                bbs_3d,
                sad=sad,
                sid=sid,
                angle=angles[k],
                angle_offset=0,
                img_dim=[2048, 1536],
                pixel_size=[0.388, 0.388],
                search_area=14,
                resolution_factor=2,
                drag_and_drop=True,
            )

            # Update output dictionary
            results["proj_angles"].append(view_results["proj_angle"])
            results["panel_orientation"].append(
                view_results["panel_orientation"]
            )
            results["sid"].append(view_results["sid"])
            results["sad"].append(view_results["sad"])
            results["isocenter"].append(view_results["isocenter"])
            results["source"].append(view_results["source"])
            results["panel"].append(view_results["panel"])
            results["img_center"].append(view_results["img_center"])
            results["err_init"].append(view_results["err_init"])
            results["err_final"].append(view_results["err_final"])

    return results


def calibrate_projection(
    projection_file,
    bbs_3d,
    sad,
    sid,
    angle,
    angle_offset=0,
    img_dim=[1024, 768],
    pixel_size=[0.388, 0.388],
    search_area=7,
    resolution_factor=1,
    image_center=None,
    drag_and_drop=True,
):
    """Calibration of a single projection.

    :param projection_file: path to file
    :type projection_file: str
    :param bbs_3d: 3D coordinates of phantom's reference points
    :type bbs_3d: numpy.array
    :param sad: nominal source to isocenter (A) distance
    :type sad: float
    :param sid: nominal source to image distance
    :type sid: float
    :param angle: gantry angle for current projection
    :type angle: float
    :param angle_offset: angle offset for panel, defaults to 0
    :type angle_offset: int, optional
    :param img_dim: image dimensions in pixels, defaults to [1024, 768]
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

    if image_center is None:  # in case image_center is not declared
        image_center = [img_dim[1] / 2, img_dim[0] / 2]
    isocenter = [0, 0, 0]

    # panel orientation (from panel to brandis reference - rotation along y)
    panel_orientation = np.array([0, deg2rad(angle), 0]) + np.array(
        [0, deg2rad(angle_offset), 0]
    )

    # Load projection
    if ".raw" in projection_file:
        img = read_projection_raw(projection_file, img_dim)
    elif ".hnc" in projection_file:
        img = read_projection_hnc(projection_file, img_dim)

    # Project points starting from extrinsic and intrinsic parameters
    # generate proj_matrix (extrinsic and intrinsic parameters)
    T = create_camera_matrix(panel_orientation, sid, sad, pixel_size, isocenter)
    # projected coordinates of brandis on panel plane
    r2d = project_camera_matrix(
        bbs_3d, image_center, T, resolution_factor
    )  # 2d coordinates of reference points

    grayscale_range = get_grayscale_range(img)

    if drag_and_drop is True:
        # Overlay reference bbs with projection
        r2d_corrected = drag_and_drop_bbs(
            projection_path=img,
            bbs_projected=r2d,
            grayscale_range=grayscale_range,
        )

    # Starting from the updated coordinates, define a search area around them
    # and identify the bbs as black pixels inside these areas (brandis are used
    # as probes)
    if drag_and_drop is True:
        bbs_centroid = search_bbs_centroids(
            img=img,
            ref_2d=r2d_corrected,
            search_area=search_area,
            dim_img=img_dim,
            grayscale_range=grayscale_range,
        )
    else:
        bbs_centroid = search_bbs_centroids(
            img=img,
            ref_2d=r2d,
            search_area=search_area,
            dim_img=img_dim,
            grayscale_range=grayscale_range,
        )

    # Calibration - non linear data fitting optimization problem
    index = np.where(~np.isnan(bbs_centroid[:, 0]))[0]

    # Estimated BBs
    bbs_estim_init = bbs_centroid[
        ~np.isnan(bbs_centroid).any(axis=1)
    ]  # not consider if out of searching area

    # Real Brandis BBs
    bbs_real_init = bbs_3d[index, :]

    # x0
    parameters = np.append(panel_orientation, image_center).tolist()
    parameters.append(sid)
    parameters.append(sad)

    # Boundaries
    angle_limit = 0.05
    sid_sad_limit = 1
    low_bound = [
        -angle_limit,
        -np.pi,
        -angle_limit,
        0,
        0,
        sid - sid_sad_limit,
        sad - sid_sad_limit,
    ]
    up_bound = [
        angle_limit,
        np.pi,
        angle_limit,
        img_dim[1],
        img_dim[0],
        sid + sid_sad_limit,
        sad + sid_sad_limit,
    ]

    if index.shape[0] > 5:  # at least 5 BBs
        sol = least_squares(
            fun=calibration_cost_function,
            x0=parameters,
            args=(bbs_real_init, pixel_size, bbs_estim_init, isocenter,),
            method="trf",
            bounds=(low_bound, up_bound)
            # verbose=2,
        )

        sol = sol.x  # Solution found

        panel_orientation_new = np.array(sol[:3])  # New panel orientation
        image_center_new = np.array(sol[3:5])  # New center of image
        sid_new = sol[5]
        sad_new = sol[6]
        isocenter_new = isocenter
    else:
        raise Exception("Cannot properly process last projection. Please Retry")

    # project based on calibration - use new panel orientation,
    # tube and panel position
    T = create_camera_matrix(
        panel_orientation_new, sid_new, sad_new, pixel_size, isocenter_new
    )  # projected coordinates of brandis on panel plane

    bbs_estim_final = project_camera_matrix(
        bbs_3d, image_center_new, T
    )  # projected BBs (considering unknown)

    # calculate improvement
    err_init = bbs_estim_init - r2d[index, :]  # estimated - projected
    err_final = bbs_estim_init - bbs_estim_final[index, :]

    err_init = np.mean(abs(err_init))
    err_final = np.mean(abs(err_final))

    # calculate new source/panel position
    T_new = angle2rotm(
        panel_orientation_new[0],
        panel_orientation_new[1],
        panel_orientation_new[2],
    )
    R_new = T_new[:3, :3]

    source_new = (isocenter_new + (R_new * np.array([0, 0, sad_new])))[:, 2]
    panel_new = (isocenter_new + (R_new * np.array([0, 0, sad_new - sid_new])))[
        :, 2
    ]

    # update with new value
    results["proj_angle"] = angle
    results["panel_orientation"] = panel_orientation_new
    results["sid"] = sid_new
    results["sad"] = sad_new
    results["isocenter"] = isocenter_new
    results["source"] = source_new
    results["panel"] = panel_new
    results["img_center"] = image_center_new
    results["err_init"] = err_init
    results["err_final"] = err_final

    return results


def calibration_cost_function(param, bbs_3d, pixel_size, bbs_2d, isocenter):
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
    panel_orientation = np.array(param[:3])
    img_center = np.array(param[3:5])
    sid = np.array(param[5])
    sad = np.array(param[6])

    T = create_camera_matrix(
        panel_orientation, sid, sad, pixel_size, isocenter
    )  # projected coordinates of brandis on panel plane
    r2d = project_camera_matrix(
        bbs_3d, img_center, T
    )  # projected bbs (considering unknown)

    delta = r2d - bbs_2d  # Error

    diff = np.square(delta[:, 0]) + np.square(
        delta[:, 1]
    )  # consider both directions

    return diff


def plot_calibration_results(calib_results):
    """Plot source/panel position after calibration.

    :param calib_results: dictionary containing results of a calibration
    :type calib_results: dict
    """
    source_pos = np.array(calib_results["source"])
    panel_pos = np.array(calib_results["panel"])
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


def save_lut(path, calib_results, mode):
    """Save LUT file for a calibration.

    :param path: path to .raw file directory, where LUT will be saved
    :type path: str
    :param calib_results: dictionary containing results for a calibration
    :type calib_results: dict
    :param calib_results: acquisition modality for calibration
    :type calib_results: string
    """
    angles = calib_results["proj_angles"]
    panel_orientation = calib_results["panel_orientation"]
    image_center = calib_results["img_center"]
    sid = calib_results["sid"]
    sad = calib_results["sad"]

    clock = datetime.now()

    if mode == "cbct":
        filename = "CBCT_LUT_{}_{:02}_{:02}-{:02}_{:02}.txt".format(
            clock.year, clock.month, clock.day, clock.hour, clock.minute,
        )
    elif mode == "2d":
        filename = "2D_LUT_{}_{:02}_{:02}-{:02}_{:02}.txt".format(
            clock.year, clock.month, clock.day, clock.hour, clock.minute,
        )
    output_file = os.path.join(path, filename)

    with open(output_file, "w") as res_file:
        res_file.write(f"#Look Up Table for {mode.upper()} reconstruction\n")
        res_file.write(
            "#Angle (deg) | Panel Orientation(rad) [X  Y  Z] | Image_center(pixel) X Y | SID(mm) | SAD(mm)\n"
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

        for k in range(len(angles)):
            res_file.write(
                "{:6.12f} {:6.12f} {:6.12f} {:6.12f} {:6.12f} {:6.12f} {:6.12f} {:6.12f}\n".format(
                    angles[k],
                    panel_orientation[k][0],
                    panel_orientation[k][1],
                    panel_orientation[k][2],
                    image_center[k][0],
                    image_center[k][1],
                    sid[k],
                    sad[k],
                )
            )

        res_file.write(
            r"# END OF FILE. REQUIRED TO ENSURE '\n' at the end of last calibration line. NO MORE LINES AFTER THIS!!!"
        )
