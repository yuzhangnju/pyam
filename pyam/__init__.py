import logging

from pyam.core import *
from pyam.utils import *
from pyam.statistics import *
from pyam.timeseries import *
from pyam.read_ixmp import *
from pyam.logging import *
from pyam.run_control import *
from pyam.iiasa import read_iiasa  # noqa: F401

# in Jupyter notebooks: disable autoscroll, activate warnings
try:
    get_ipython().run_cell_magic(u'javascript', u'',
                                 u'IPython.OutputArea.prototype._should_scroll = function(lines) { return false; }')
    warnings.simplefilter('default')

    def custom_formatwarning(msg, category, filename, lineno, line=''):
        # ignore everything except the message
        return str(msg) + '\n'
    warnings.formatwarning = custom_formatwarning
except Exception:
    pass

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

logging.getLogger(__name__).addHandler(logging.NullHandler())
