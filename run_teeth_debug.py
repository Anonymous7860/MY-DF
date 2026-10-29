"""Launcher for the (compiled) teeth-mask debug tool.

The tool lives in the samplelib package as a compiled extension module
(samplelib/debug_teeth_masks.pyd) and cannot be run as a plain script. This
stub puts the DeepFaceLab folder on sys.path (the bundled python uses a ._pth
file, so neither the script dir nor PYTHONPATH is added automatically), then
calls the tool's main() with the command-line arguments. Usage:

    python run_teeth_debug.py "<workspace>\\data_xxx\\aligned"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from samplelib.debug_teeth_masks import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
