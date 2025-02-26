# ========================================================================
# 'pyvocals' - A Python tool for vocal turn-taking feature extraction
# ========================================================================
# Author: Natasha Yamane <natasha.yamane@gmail.com>
# Version: 0.1
#
# Example usage:
#   import pyvocals
#   result = pyvocals.extract_features(...)
# ========================================================================

__version__ = '0.1'

# Import the entire `pyvocals` module
from .pyvocals import *

# Allow wildcard imports
__all__ = ['pyvocals']