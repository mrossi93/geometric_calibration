"""Console script for geometric_calibration."""
import sys
import os
import click
from geometric_calibration.reader import read_bbs_ref_file
from geometric_calibration.geometric_calibration import (
    calibrate,
    save_lut,
    plot_calibration_results,
)


def save_routine(path, results):
    found_old = False
    old_files = []

    for file in os.listdir(path):
        if "CBCT" in file:
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

    save_lut(path, results)
    click.echo("---\tNew LUT saved\t---")
    return


@click.command()
@click.option("--ref", "-r", help="Reference File for calibration phantom")
@click.option("--dir", "-d", help="Path to directory with .raw projection")
def main(ref, dir):
    """Console script for geometric_calibration."""

    if ref is None:
        click.echo(
            click.style(
                "Error: please indicate the path to Reference File", fg="red",
            )
        )
        return 1
    elif dir is None:
        click.echo(
            click.style(
                "Error: please indicate the path to directory with .raw projection",
                fg="red",
            )
        )
        return 1
    else:
        bbs = read_bbs_ref_file(ref)
        click.echo("Calibrating the system. Please Wait...")
        calibration_results = calibrate(dir, bbs)

        # TODO Stampare errori o simili

        # TODO Inserire un prompt per decidere se salvare i risultati

        opt = "s"
        save_flag = False
        while opt in ["s", "p", "c"]:
            val = click.prompt(
                """\nChoose an option:
            s\tSave
            p\tPlot
            c\tClose
            """,
                prompt_suffix="\rYour choice: ",
                type=str,
            )

            if val == "s":
                save_routine(dir, calibration_results)
                save_flag = True
            elif val == "p":
                plot_calibration_results(calibration_results)
            elif val == "c":
                if save_flag is False:
                    if click.confirm(
                        "New LUT not saved. Do you want to save it?",
                        default=True,
                        show_default=True,
                    ):
                        save_routine(dir, calibration_results)
                break
            else:
                click.echo("Command '{}' not recognized.".format(val))

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
