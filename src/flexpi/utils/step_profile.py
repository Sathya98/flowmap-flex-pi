"""Moved to flowmap_core.step_profile; this path is an alias of that module."""
import sys

import flowmap_core.step_profile as _module

sys.modules[__name__] = _module
