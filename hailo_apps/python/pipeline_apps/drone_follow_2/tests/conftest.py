"""Path setup so tests work whether hailo-apps is pip-installed or not."""

import os
import sys

# hailo-apps root is 5 directories up from this conftest.py
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
if _root not in sys.path:
    sys.path.insert(0, _root)

# drone_follow app dir (parent of tests/) so `from drone_control import ...` resolves
_app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)
