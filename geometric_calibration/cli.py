"""Console script for geometric_calibration."""
import sys
import os
import click
import click_config_file
from geometric_calibration.reader import read_bbs_ref_file
from geometric_calibration.geometric_calibration import (
    calibrate_cbct,
    calibrate_2d,
    save_lut,
    plot_calibration_results,
)

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

if getattr(sys, "frozen", False):
    # If the application is run as a bundle, the PyInstaller bootloader
    # extends the sys module by a flag frozen=True and sets the app
    # path into variable _MEIPASS'.
    APPLICATION_PATH = sys._MEIPASS
else:
    APPLICATION_PATH = os.path.dirname(os.path.abspath(__file__))


REF_BBS_DEFAULT_PATH = os.path.join(APPLICATION_PATH, "app_data")
REF_BBS_DEFAULT_PATH = os.path.join(REF_BBS_DEFAULT_PATH, "ref_brandis.txt")


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
        if click.confirm(
            "Do you want to delete it?", default=False, show_default=True
        ):
            for file in old_files:
                os.remove(os.path.join(path, file))
            click.echo("---\tOld LUT deleted\t---")

    save_lut(path, results, mode)
    click.echo("---\tNew LUT saved\t---")
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
    "--sad",
    type=click.FLOAT,
    help="Nominal source to isocenter distance",
    default=1172.2,
)
@click.option(
    "--sid",
    type=click.FLOAT,
    help="Nominal source to image distance",
    default=1672.2,
)
@click.option(
    "--ref",
    "-r",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Reference File for calibration phantom",
    default=REF_BBS_DEFAULT_PATH,
)
@click_config_file.configuration_option()
def main(mode, input_path, sad, sid, ref):
    """Console script for geometric_calibration.

    Author: Matteo Rossi"""

    bbs = read_bbs_ref_file(ref)

    click.echo("Calibration Parameters:")
    click.echo("Mode: '{}'".format(mode))
    click.echo("Input Path: '{}'".format(input_path))
    click.echo("SAD: '{}'".format(sad))
    click.echo("SID: '{}'".format(sid))
    click.echo("")

    if mode == "cbct":
        click.echo("Calibrating the system. Please Wait...")
        # sad = 1115 + 57.2  # source to isocenter (A) distance
        # sid = sad + 500  # source to image distance
        calibration_results = calibrate_cbct(input_path, bbs, sad, sid)
    elif mode == "2d":
        click.echo("Calibrating the system. Please Wait...")
        calibration_results = calibrate_2d(input_path, bbs, sad, sid)
    else:
        click.echo("Mode '{}' not recognized.".format(mode))
        return 0

    opt = "s"
    save_flag = False
    while opt in ["s", "p", "c"]:
        user_choice = click.prompt(
            """\nChoose an option:
                s\tSave
                p\tPlot
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
        elif user_choice == "c":
            if save_flag is False:
                if click.confirm(
                    "New LUT not saved. Do you want to save it?",
                    default=True,
                    show_default=True,
                ):
                    save_cli(input_path, calibration_results, mode)
            break
        else:
            click.echo("Command '{}' not recognized.".format(user_choice))
    return 0
