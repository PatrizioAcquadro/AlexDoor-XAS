#!/usr/bin/env python
"""Run the Purdue operational scene/control/capture gate."""

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("verify_purdue_runtime.py")), run_name="__main__")
