"""Moved to flowmap_core.ema; this path is an alias of that module."""
import sys

import flowmap_core.ema as _module

sys.modules[__name__] = _module
