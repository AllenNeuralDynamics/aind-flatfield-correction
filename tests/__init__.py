"""Unit tests for aind-flatfield-correction.

The modules under test log progress at INFO and warn loudly about
suspicious flatfields.  Logging is disabled for the suite so that the
expected warnings do not read as test failures.
"""

import logging

logging.disable(logging.CRITICAL)
