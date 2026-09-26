"""Moved to flowmap_core.flowmap_self; this path is an alias of that module."""
import sys

import flowmap_core.flowmap_self as _module

sys.modules[__name__] = _module
