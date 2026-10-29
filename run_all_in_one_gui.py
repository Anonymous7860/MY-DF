"""Launcher for the (compiled) All-in-One GUI.

The GUI itself ships as a compiled extension module
(deepfacelab_all_in_one_gui.pyd) and therefore cannot be run as a plain
script. This tiny stub puts the DeepFaceLab folder on sys.path (the bundled
python uses a ._pth file, so neither the script dir nor PYTHONPATH is added
automatically), imports the GUI module, and calls its entry point. All
command-line arguments are read from sys.argv by the GUI's own parser.

Before importing anything it also finishes any update that could not replace
a file while it was in use. Windows locks a loaded .pyd, so an update that
touches the GUI's own module (or a module a running trainer has loaded)
leaves the new copy next to the old one as "<name>.new". Nothing is loaded
yet at this point, so the swap succeeds here.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def finish_pending_updates(root: str) -> None:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for name in filenames:
            path = os.path.join(dirpath, name)
            if name.endswith(".old"):
                # left behind when an update renamed a locked file aside
                try:
                    os.remove(path)
                except OSError:
                    pass
            elif name.endswith(".new"):
                target = path[:-4]
                try:
                    os.replace(path, target)
                except OSError:
                    # still in use (training or merge running); retried next launch
                    pass


finish_pending_updates(HERE)

from deepfacelab_all_in_one_gui import main

if __name__ == "__main__":
    raise SystemExit(main())
