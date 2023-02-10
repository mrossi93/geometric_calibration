"""Entry point file for pyinstaller.

To build the .exe run the following command:

.. code-block:: python

    build_exe.bat

"""

import sys

from geometric_calibration.cli import main

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
