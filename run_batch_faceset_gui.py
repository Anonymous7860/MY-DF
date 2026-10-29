"""Launcher for the (compiled) batch extraction and sorting tool.

The tool itself ships as a compiled extension module
(batch_faceset_gui.pyd) and therefore cannot be run as a plain script, so this
stub imports it and calls its entry point. The bundled python uses a ._pth
file, so neither the script directory nor PYTHONPATH is added automatically
and the DeepFaceLab folder has to be put on sys.path here.

The window also re-launches this same stub with --worker to run an extraction
in a separate process, so every command-line argument is passed straight
through to the tool's own parser via sys.argv.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from batch_faceset_gui import main

if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    raise SystemExit(main())
