"""Moved to flowmap_core.checkpoint; this path is an alias of that module."""
import sys

import flowmap_core.checkpoint as _module

sys.modules[__name__] = _module
