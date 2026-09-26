"""Moved to flowmap_core.attention; this path is an alias of that module."""
import sys

import flowmap_core.attention as _module

sys.modules[__name__] = _module
