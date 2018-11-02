import numpy as np
import pytest
import pandas as pd
from numpy import testing as npt
import re

from pyam import OpenSCMDataFrame


def test_init_openscm_df_with_float_cols(test_pd_df):
    _test_df_openscm = test_pd_df.rename(columns={2005: 2005.5, 2010: 2010.})
    obs = OpenSCMDataFrame(_test_df_openscm)
    # see here for explanation of numpy data type hierarchy
    # https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html
    assert obs.data.year.dtype <= np.float
    npt.assert_array_equal(obs.data.year.unique(), np.array([2005.5, 2010. ]))


def test_check_aggregate_no_method_openscm(test_df_openscm):
    error_msg = re.escape(
        "'OpenSCMDataFrame' object has no attribute 'check_aggregate'"
    )
    with pytest.raises(AttributeError, match=error_msg):
        test_df_openscm.check_aggregate()
