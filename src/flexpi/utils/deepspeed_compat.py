"""Moved to flowmap_core.deepspeed_compat; this path is an alias of that module."""
import sys

import flowmap_core.deepspeed_compat as _module

sys.modules[__name__] = _module
