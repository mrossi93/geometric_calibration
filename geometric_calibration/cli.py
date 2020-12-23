"""Console script for geometric_calibration."""
import sys
import os
import logging
import click
import click_config_file
from geometric_calibration.reader import read_bbs_ref_file
from geometric_calibration.geometric_calibration import (
    calibrate_cbct,
    calibrate_2d,
    save_lut,
    save_lut_planar,
    plot_calibration_results,
    plot_calibration_errors,
    plot_offset_variability,
)
from geometric_calibration.slideshow import slideshow

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

if getattr(sys, "frozen", False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    APPLICATION_PATH = sys._MEIPASS
else:
    APPLICATION_PATH = os.path.dirname(os.path.abspath(__file__))


REF_BBS_DEFAULT_PATH = os.path.join(
    APPLICATION_PATH, "app_data", "ref_brandis.txt"
)


def save_cli(path, results, mode):
    found_old = False
    old_files = []

    for file in os.listdir(path):
        if "LUT" in file:
            found_old = True
            old_files.append(file)

    if found_old is True:
        click.echo("\nFound existing LUT for this phantom:")
        for file in old_files:
            click.echo("{}".format(file))
        # !!! NB: per il momento le vecchie lut vengono sempre cancellate di default
        # Ricordarsi di impostare default=False in production
        if click.confirm(
            "Do you want to delete it?", default=True, show_default=True
        ):
            for file in old_files:
                os.remove(os.path.join(path, file))
            logging.info("Old LUT deleted")

    # If mode is "cbct" then ask for lut style, classic mode is default
    if mode == "cbct":
        lut_style = click.prompt(
            """\nChoose a style:
                    c\tClassic Style (6 columns)
                    n\tNew Style (8 columns)
                    t\tComplete (10 columns)
                    *\tSave every modality
                    """,
            prompt_suffix="\rYour choice: ",
            type=str,
            default="*",  # Save every style for now
        )
        if lut_style == "c":
            save_lut(path, results, "classic")
            style_string = "classic"
        elif lut_style == "n":
            save_lut(path, results, "new")
            style_string = "new"
        elif lut_style == "t":
            save_lut(path, results, "complete")
            style_string = "complete"
        elif lut_style == "*":
            save_lut(path, results, "classic")
            save_lut(path, results, "new")
            save_lut(path, results, "complete")
            style_string = "every"
        else:
            print("Invalid style")
            return
        logging.info(f"New LUT saved with {style_string} style.")
    elif mode == "2d":
        save_lut_planar(path, results)
        logging.info("Calibration file saved.")

    return


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--mode", "-m", help="Acquisition modality: 'cbct' or '2d'", default="cbct"
)
@click.option(
    "--input_path",
    "-i",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="""Path .raw projection. It can be a folder or a single file depending on
     'mode' parameter""",
    default=os.getcwd(),
)
@click.option(
    "--sid",
    type=click.FLOAT,
    help="Nominal source to isocenter distance",
    default=1172.2,
)
@click.option(
    "--sdd",
    type=click.FLOAT,
    help="Nominal source to image distance",
    default=1672.2,
)
@click.option(
    "--drag_every",
    type=click.INT,
    help="Manually reposition the reference points every N projections",
    default=1000,
)
@click.option(
    "--debug_level",
    type=click.INT,
    help="Just for debug purposes. It can be 0, 1 or 2",
    default=0,
)
@click.option(
    "--ref",
    "-r",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Reference File for calibration phantom",
    default=REF_BBS_DEFAULT_PATH,
)
@click_config_file.configuration_option()
def main(mode, input_path, sid, sdd, drag_every, debug_level, ref):
    """Console script for geometric_calibration.

    Author: Matteo Rossi"""

    # set up logging to file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%d-%m-%y %H:%M:%S",
        filename=os.path.join(input_path, "calibration_log.txt"),
        # filemode="a",
    )
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter("[%(levelname)-8s] %(message)s")
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger("").addHandler(console)

    # Read reference file
    bbs = read_bbs_ref_file(ref)

    logging.info(
        f"""Starting new calibration with the following parameters:
    Mode: {mode}
    Input Path: {input_path}
    SID: {sid}
    SDD: {sdd}"""
    )

    # Just to avoid division by zero, in case user wrongly set this parameter
    if drag_every == 0:
        drag_every = 1000

    if debug_level not in [0, 1, 2]:
        logging.warning(
            f"Debug Level {debug_level} doesn't exist. Calibration will be performed with debug level 0 as default."
        )
        debug_level = 0
    elif debug_level in [1, 2]:
        logging.info(f"Debug mode activated with level {debug_level}")

    if mode == "cbct":
        calibration_results = calibrate_cbct(
            projection_dir=input_path,
            bbs_3d=bbs,
            sid=sid,
            sdd=sdd,
            proj_offset=[0, 0],
            source_offset=[0, 0],
            drag_every=drag_every,
            debug_level=debug_level,
        )
    elif mode == "2d":
        calibration_results = calibrate_2d(
            projection_dir=input_path,
            bbs_3d=bbs,
            sid=sid,
            sdd=sdd,
            proj_offset=[0, 0],
            source_offset=[0, 0],
            debug_level=debug_level,
        )
    else:
        logging.critical(
            f"Mode {mode} not recognized. Please choose between '2d' or 'cbct'."
        )
        return 0

    logging.info("Calibration ended successfully.")

    opt = "s"
    save_flag = False
    while opt in ["s", "p", "l", "c"]:
        user_choice = click.prompt(
            """\nChoose an option:
                s\tSave
                p\tPlot
                l\tSlideshow
                e\tErrors
                v\tVariability
                c\tClose
                """,
            prompt_suffix="\rYour choice: ",
            type=str,
        )
        if user_choice == "s":
            save_cli(input_path, calibration_results, mode)
            save_flag = True
        elif user_choice == "p":
            plot_calibration_results(calibration_results)
        elif user_choice == "l":
            slideshow(calibration_results, bbs, mode)
        elif user_choice == "e":
            plot_calibration_errors(calibration_results)
        elif user_choice == "v":
            plot_offset_variability(calibration_results)
        elif user_choice == "c":
            """
            if save_flag is False:
                if click.confirm(
                    "New LUT not saved. Do you want to save it?",
                    default=True,
                    show_default=True,
                ):
                    save_cli(input_path, calibration_results, mode)
            """
            break
        else:
            click.echo("Command '{}' not recognized.".format(user_choice))
    return 0
