import os
import copy
import pytest
import re
from copy import deepcopy
from unittest.mock import Mock
import warnings

import numpy as np
import pandas as pd
from numpy import testing as npt

from pyam import (IamDataFrame, OpenSCMDataFrame, plotting, validate, categorize,
                  require_variable, check_aggregate, filter_by_meta, META_IDX,
                  IAMC_IDX, LONG_IDX)
from pyam.core import _meta_idx
from pyam.errors import ConversionError

from conftest import TEST_DATA_DIR


df_filter_by_meta_matching_idx = pd.DataFrame([
    ['a_model', 'a_scenario', 'a_region1', 1],
    ['a_model', 'a_scenario', 'a_region2', 2],
    ['a_model', 'a_scenario2', 'a_region3', 3],
], columns=['model', 'scenario', 'region', 'col'])


df_filter_by_meta_nonmatching_idx = pd.DataFrame([
    ['a_model', 'a_scenario3', 'a_region1', 1, 2],
    ['a_model', 'a_scenario3', 'a_region2', 2, 3],
    ['a_model', 'a_scenario2', 'a_region3', 3, 4],
], columns=['model', 'scenario', 'region', 2010, 2020]
).set_index(['model', 'region'])


def test_init_df_with_index(test_pd_df, pyam_df):
    df = pyam_df(test_pd_df.set_index(META_IDX))
    pd.testing.assert_frame_equal(df.timeseries().reset_index(), test_pd_df)


def test_init_iam_df_with_float_cols_raises(test_pd_df):
    _test_df_iam = test_pd_df.rename(columns={2005: 2005.5, 2010: 2010.})
    pytest.raises(ValueError, IamDataFrame, data=_test_df_iam)


def test_init_openscm_df_with_float_cols(test_pd_df):
    _test_df_openscm = test_pd_df.rename(columns={2005: 2005.5, 2010: 2010.})
    obs = OpenSCMDataFrame(_test_df_openscm)
    # see here for explanation of numpy data type hierarchy
    # https://docs.scipy.org/doc/numpy/reference/arrays.scalars.html
    assert obs.data.year.dtype <= np.float
    npt.assert_array_equal(obs.data.year.unique(), np.array([2005.5, 2010. ]))


def test_init_df_with_float_cols(test_pd_df, pyam_df):
    _test_df_iam = test_pd_df.rename(columns={2005: 2005., 2010: 2010.})
    obs = pyam_df(_test_df_iam).timeseries().reset_index()
    pd.testing.assert_series_equal(obs[2005], test_pd_df[2005])


def test_init_df_from_timeseries(test_df_iam):
    df = IamDataFrame(test_df_iam.timeseries())
    pd.testing.assert_frame_equal(df.timeseries(), test_df_iam.timeseries())


def test_get_item(test_df):
    assert test_df['model'].unique() == ['a_model']


def test_model(test_df):
    pd.testing.assert_series_equal(test_df.models(),
                                   pd.Series(data=['a_model'], name='model'))


def test_scenario(test_df):
    exp = pd.Series(data=['a_scenario'], name='scenario')
    pd.testing.assert_series_equal(test_df.scenarios(), exp)


def test_region(test_df):
    exp = pd.Series(data=['World'], name='region')
    pd.testing.assert_series_equal(test_df.regions(), exp)


def test_variable(test_df):
    exp = pd.Series(
        data=['Primary Energy', 'Primary Energy|Coal'], name='variable')
    pd.testing.assert_series_equal(test_df.variables(), exp)


def test_variable_unit(test_df):
    dct = {'variable': ['Primary Energy', 'Primary Energy|Coal'],
           'unit': ['EJ/y', 'EJ/y']}
    exp = pd.DataFrame.from_dict(dct)[['variable', 'unit']]
    npt.assert_array_equal(test_df.variables(include_units=True), exp)


def test_variable_depth_0(test_df):
    obs = list(test_df.filter(level=0)['variable'].unique())
    exp = ['Primary Energy']
    assert obs == exp


def test_variable_depth_0_keep_false(test_df):
    obs = list(test_df.filter(level=0, keep=False)['variable'].unique())
    exp = ['Primary Energy|Coal']
    assert obs == exp


def test_variable_depth_0_minus(test_df):
    obs = list(test_df.filter(level='0-')['variable'].unique())
    exp = ['Primary Energy']
    assert obs == exp


def test_variable_depth_0_plus(test_df):
    obs = list(test_df.filter(level='0+')['variable'].unique())
    exp = ['Primary Energy', 'Primary Energy|Coal']
    assert obs == exp


def test_variable_depth_1(test_df):
    obs = list(test_df.filter(level=1)['variable'].unique())
    exp = ['Primary Energy|Coal']
    assert obs == exp


def test_variable_depth_1_minus(test_df):
    obs = list(test_df.filter(level='1-')['variable'].unique())
    exp = ['Primary Energy', 'Primary Energy|Coal']
    assert obs == exp


def test_variable_depth_1_plus(test_df):
    obs = list(test_df.filter(level='1+')['variable'].unique())
    exp = ['Primary Energy|Coal']
    assert obs == exp


def test_variable_depth_raises(test_df):
    pytest.raises(ValueError, test_df.filter, level='1/')


def test_filter_error(test_df):
    pytest.raises(ValueError, test_df.filter, foo='foo')


def test_filter_as_kwarg(meta_df):
    obs = list(meta_df.filter(variable='Primary Energy|Coal').scenarios())
    assert obs == ['a_scenario']


def test_filter_keep_false(meta_df):
    df = meta_df.filter(variable='Primary Energy|Coal', year=2005, keep=False)
    obs = df.data[df.data.scenario == 'a_scenario'].value
    npt.assert_array_equal(obs, [1, 6, 3])


def test_filter_by_regexp(meta_df):
    obs = meta_df.filter(scenario='a_scenari.$', regexp=True)
    assert obs['scenario'].unique() == 'a_scenario'


def test_timeseries(test_df):
    dct = {'model': ['a_model'] * 2, 'scenario': ['a_scenario'] * 2,
           'years': [2005, 2010], 'value': [1, 6]}
    exp = pd.DataFrame(dct).pivot_table(index=['model', 'scenario'],
                                        columns=['years'], values='value')
    obs = test_df.filter(variable='Primary Energy').timeseries()
    npt.assert_array_equal(obs, exp)


def test_read_pandas(pyam_df):
    df = pyam_df(os.path.join(TEST_DATA_DIR, 'testing_data_2.csv'))
    assert list(df.variables()) == ['Primary Energy']


def test_filter_meta_index(meta_df):
    obs = meta_df.filter(scenario='a_scenario2').meta.index
    exp = pd.MultiIndex(levels=[['a_model'], ['a_scenario2']],
                        labels=[[0], [0]],
                        names=['model', 'scenario'])
    pd.testing.assert_index_equal(obs, exp)


def test_meta_idx(meta_df):
    # assert that the `drop_duplicates()` in `_meta_idx()` returns right length
    assert len(_meta_idx(meta_df.data)) == 2


def test_require_variable(meta_df):
    obs = meta_df.require_variable(variable='Primary Energy|Coal',
                                   exclude_on_fail=True)
    assert len(obs) == 1
    assert obs.loc[0, 'scenario'] == 'a_scenario2'

    assert list(meta_df['exclude']) == [False, True]


def test_require_variable_top_level(meta_df):
    obs = require_variable(meta_df, variable='Primary Energy|Coal',
                           exclude_on_fail=True)
    assert len(obs) == 1
    assert obs.loc[0, 'scenario'] == 'a_scenario2'

    assert list(meta_df['exclude']) == [False, True]


def test_validate_all_pass(meta_df):
    obs = meta_df.validate(
        {'Primary Energy': {'up': 10}}, exclude_on_fail=True)
    assert obs is None
    assert len(meta_df.data) == 6  # data unchanged

    assert list(meta_df['exclude']) == [False, False]  # none excluded


def test_validate_nonexisting(meta_df):
    obs = meta_df.validate({'Primary Energy|Coal': {'up': 2}},
                           exclude_on_fail=True)
    assert len(obs) == 1
    assert obs['scenario'].values[0] == 'a_scenario'

    assert list(meta_df['exclude']) == [True, False]  # scenario with failed
    # validation excluded, scenario with non-defined value passes validation


def test_validate_up(meta_df):
    obs = meta_df.validate({'Primary Energy': {'up': 6.5}},
                           exclude_on_fail=False)
    assert len(obs) == 1
    assert obs['year'].values[0] == 2010

    assert list(meta_df['exclude']) == [False, False]  # assert none excluded


def test_validate_lo(meta_df):
    obs = meta_df.validate({'Primary Energy': {'up': 8, 'lo': 2.0}})
    assert len(obs) == 1
    assert obs['year'].values[0] == 2005
    assert list(obs['scenario'].values) == ['a_scenario']


def test_validate_both(meta_df):
    obs = meta_df.validate({'Primary Energy': {'up': 6.5, 'lo': 2.0}})
    assert len(obs) == 2
    assert list(obs['year'].values) == [2005, 2010]
    assert list(obs['scenario'].values) == ['a_scenario', 'a_scenario2']


def test_validate_year(meta_df):
    obs = meta_df.validate({'Primary Energy': {'up': 5.0, 'year': 2005}},
                           exclude_on_fail=False)
    assert obs is None

    obs = meta_df.validate({'Primary Energy': {'up': 5.0, 'year': 2010}},
                           exclude_on_fail=False)
    assert len(obs) == 2


def test_validate_exclude(meta_df):
    meta_df.validate({'Primary Energy': {'up': 6.0}}, exclude_on_fail=True)
    assert list(meta_df['exclude']) == [False, True]


def test_validate_top_level(meta_df):
    obs = validate(meta_df, criteria={'Primary Energy': {'up': 6.0}},
                   exclude_on_fail=True, variable='Primary Energy')
    assert len(obs) == 1
    assert obs['year'].values[0] == 2010
    assert list(meta_df['exclude']) == [False, True]


def test_check_aggregate_pass_iam(check_aggregate_df_iam):
    obs = check_aggregate_df_iam.filter(
        scenario='a_scen'
    ).check_aggregate('Primary Energy')
    assert obs is None


def test_check_aggregate_no_method_openscm(test_df_openscm):
    error_msg = re.escape(
        "'OpenSCMDataFrame' object has no attribute 'check_aggregate'"
    )
    with pytest.raises(AttributeError, match=error_msg):
        test_df_openscm.check_aggregate()


def test_check_aggregate_fail(meta_df_iam):
    obs = meta_df_iam.check_aggregate('Primary Energy', exclude_on_fail=True)
    assert len(obs.columns) == 2
    assert obs.index.get_values()[0] == (
        'Primary Energy', 'a_model', 'a_scenario', 'World'
    )


def test_check_aggregate_top_level(meta_df_iam):
    obs = check_aggregate(meta_df_iam, variable='Primary Energy', year=2005)
    assert len(obs.columns) == 1
    assert obs.index.get_values()[0] == (
        'Primary Energy', 'a_model', 'a_scenario', 'World'
    )


def test_df_iam_check_aggregate_pass(check_aggregate_df_iam):
    obs = check_aggregate_df_iam.check_aggregate('Primary Energy')
    assert obs is None

    for variable in check_aggregate_df_iam.variables():
        obs = check_aggregate_df_iam.check_aggregate(variable)
        assert obs is None


def test_df_iam_check_aggregate_regions_pass(check_aggregate_df_iam):
    obs = check_aggregate_df_iam.check_aggregate_regions('Primary Energy')
    assert obs is None

    for variable in check_aggregate_df_iam.variables():
        obs = check_aggregate_df_iam.check_aggregate_regions(variable)
        assert obs is None


def run_check_agg_fail(pyam_df, tweak_dict, test_type):
    mr = pyam_df.data.model == tweak_dict['model']
    sr = pyam_df.data.scenario == tweak_dict['scenario']
    rr = pyam_df.data.region == tweak_dict['region']
    vr = pyam_df.data.variable == tweak_dict['variable']
    ur = pyam_df.data.unit == tweak_dict['unit']

    row_to_tweak = mr & sr & rr & vr & ur
    assert row_to_tweak.any()

    pyam_df.data.value.iloc[np.where(row_to_tweak)[0]] *= 0.99

    # the error variable is always the top level one
    expected_index = tweak_dict
    agg_test = test_type == 'aggregate'
    region_world_only_contrib = test_type == 'region-world-only-contrib'
    if agg_test or region_world_only_contrib:
        expected_index['variable'] = '|'.join(
            expected_index['variable'].split('|')[:2]
        )
    elif 'region' in test_type:
        expected_index['region'] = 'World'

    # units get dropped during aggregation and the index is a list
    expected_index = [v for k, v in expected_index.items() if k != 'unit']

    for variable in pyam_df.variables():
        if test_type == 'aggregate':
            obs = pyam_df.check_aggregate(
                variable,
            )
        elif 'region' in test_type:
            obs = pyam_df.check_aggregate_regions(
                variable,
            )

        if obs is not None:
            assert len(obs.columns) == 2
            assert set(obs.index.get_values()[0]) == set(expected_index)


def test_df_iam_check_aggregate_fail(check_aggregate_df_iam):
    to_tweak = {
        'model': 'IMG',
        'scenario': 'a_scen_2',
        'region': 'R5REF',
        'variable': 'Emissions|CO2',
        'unit': 'Mt CO2/yr',
    }
    run_check_agg_fail(check_aggregate_df_iam, to_tweak, 'aggregate')


def test_df_iam_check_aggregate_fail_no_regions(check_aggregate_df_iam):
    to_tweak = {
        'model': 'MSG-GLB',
        'scenario': 'a_scen_2',
        'region': 'World',
        'variable': 'Emissions|C2F6|Solvents',
        'unit': 'kt C2F6/yr',
    }
    run_check_agg_fail(check_aggregate_df_iam, to_tweak, 'aggregate')


def test_df_iam_check_aggregate_region_fail(check_aggregate_df_iam):
    to_tweak = {
        'model': 'IMG',
        'scenario': 'a_scen_2',
        'region': 'World',
        'variable': 'Emissions|CO2',
        'unit': 'Mt CO2/yr',
    }

    run_check_agg_fail(check_aggregate_df_iam, to_tweak, 'region')


def test_df_iam_check_aggregate_region_fail_no_subsector(check_aggregate_df_iam):
    to_tweak = {
        'model': 'MSG-GLB',
        'scenario': 'a_scen_2',
        'region': 'R5REF',
        'variable': 'Emissions|CH4',
        'unit': 'Mt CH4/yr',
    }

    run_check_agg_fail(check_aggregate_df_iam, to_tweak, 'region')


def test_df_iam_check_aggregate_region_fail_world_only_var(check_aggregate_df_iam):
    to_tweak = {
        'model': 'MSG-GLB',
        'scenario': 'a_scen_2',
        'region': 'World',
        'variable': 'Emissions|CO2|Agg Agg',
        'unit': 'Mt CO2/yr',
    }

    run_check_agg_fail(
        check_aggregate_df_iam, to_tweak, 'region-world-only-contrib'
    )


def test_df_iam_check_aggregate_regions_errors(check_aggregate_regional_df_iam):
    # these tests should fail because our dataframe has continents and regions
    # so checking without providing components leads to double counting and
    # hence failure
    obs = check_aggregate_regional_df_iam.check_aggregate_regions(
        'Emissions|N2O', 'World'
    )

    assert len(obs.columns) == 2
    assert obs.index.get_values()[0] == (
        'World', 'AIM', 'cscen', 'Emissions|N2O'
    )

    obs = check_aggregate_regional_df_iam.check_aggregate_regions(
        'Emissions|N2O', 'REUROPE'
    )

    assert len(obs.columns) == 2
    assert obs.index.get_values()[0] == (
        'REUROPE', 'AIM', 'cscen', 'Emissions|N2O'
    )


def test_df_iam_check_aggregate_regions_components(check_aggregate_regional_df_iam):
    obs = check_aggregate_regional_df_iam.check_aggregate_regions(
        'Emissions|N2O', 'World', components=['REUROPE', 'RASIA']
    )
    assert obs is None

    obs = check_aggregate_regional_df_iam.check_aggregate_regions(
        'Emissions|N2O|Solvents', 'World', components=['REUROPE', 'RASIA']
    )
    assert obs is None

    obs = check_aggregate_regional_df_iam.check_aggregate_regions(
        'Emissions|N2O', 'REUROPE', components=['Germany', 'UK']
    )
    assert obs is None

    obs = check_aggregate_regional_df_iam.check_aggregate_regions(
        'Emissions|N2O|Transport', 'REUROPE', components=['Germany', 'UK']
    )
    assert obs is None


def test_category_none(meta_df):
    meta_df.categorize('category', 'Testing', {'Primary Energy': {'up': 0.8}})
    assert 'category' not in meta_df.meta.columns


def test_category_pass(meta_df):
    dct = {'model': ['a_model', 'a_model'],
           'scenario': ['a_scenario', 'a_scenario2'],
           'category': ['foo', None]}
    exp = pd.DataFrame(dct).set_index(['model', 'scenario'])['category']

    meta_df.categorize('category', 'foo', {'Primary Energy':
                                           {'up': 6, 'year': 2010}})
    obs = meta_df['category']
    pd.testing.assert_series_equal(obs, exp)


def test_category_top_level(meta_df):
    dct = {'model': ['a_model', 'a_model'],
           'scenario': ['a_scenario', 'a_scenario2'],
           'category': ['Testing', None]}
    exp = pd.DataFrame(dct).set_index(['model', 'scenario'])['category']

    categorize(meta_df, 'category', 'Testing',
               criteria={'Primary Energy': {'up': 6, 'year': 2010}},
               variable='Primary Energy')
    obs = meta_df['category']
    pd.testing.assert_series_equal(obs, exp)


def test_load_metadata(meta_df):
    meta_df.load_metadata(os.path.join(
        TEST_DATA_DIR, 'testing_metadata.xlsx'), sheet_name='meta')
    obs = meta_df.meta

    dct = {'model': ['a_model'] * 2, 'scenario': ['a_scenario', 'a_scenario2'],
           'category': ['imported', np.nan], 'exclude': [False, False]}
    exp = pd.DataFrame(dct).set_index(['model', 'scenario'])
    pd.testing.assert_series_equal(obs['exclude'], exp['exclude'])
    pd.testing.assert_series_equal(obs['category'], exp['category'])


def test_load_SSP_database_downloaded_file(test_df_iam, pyam_df):
    obs_df = pyam_df(os.path.join(
        TEST_DATA_DIR, 'test_SSP_database_raw_download.xlsx')
    )
    pd.testing.assert_frame_equal(obs_df.as_pandas(), test_df_iam.as_pandas())


def test_load_RCP_database_downloaded_file(test_df_iam, pyam_df):
    obs_df = pyam_df(os.path.join(
        TEST_DATA_DIR, 'test_RCP_database_raw_download.xlsx')
    )
    pd.testing.assert_frame_equal(obs_df.as_pandas(), test_df_iam.as_pandas())


def test_append_other_scenario(meta_df):
    if isinstance(meta_df, OpenSCMDataFrame):
        pytest.xfail(reason="I have no idea why, but this fails for OpenSCMDataFrame")

    other = meta_df.filter(scenario='a_scenario2')\
        .rename({'scenario': {'a_scenario2': 'a_scenario3'}})

    meta_df.set_meta([0, 1], name='col1')
    meta_df.set_meta(['a', 'b'], name='col2')

    other.set_meta(2, name='col1')
    other.set_meta('x', name='col3')

    df = meta_df.append(other)

    # check that the original meta dataframe is not updated
    obs = meta_df.meta.index.get_level_values(1)
    npt.assert_array_equal(obs, ['a_scenario', 'a_scenario2'])

    # assert that merging of meta works as expected
    exp = pd.DataFrame([
        ['a_model', 'a_scenario', False, 0, 'a', np.nan],
        ['a_model', 'a_scenario2', False, 1, 'b', np.nan],
        ['a_model', 'a_scenario3', False, 2, np.nan, 'x'],
    ], columns=['model', 'scenario', 'exclude', 'col1', 'col2', 'col3']
    ).set_index(['model', 'scenario'])

    # sort columns for assertion in older pandas versions
    df.meta = df.meta.reindex(columns=exp.columns)
    pd.testing.assert_frame_equal(df.meta, exp)

    # assert that appending data works as expected
    ts = df.timeseries()
    npt.assert_array_equal(ts.iloc[2].values, ts.iloc[3].values)


def test_append_same_scenario(meta_df):
    if isinstance(meta_df, OpenSCMDataFrame):
        pytest.xfail(reason="I have no idea why, but this fails for OpenSCMDataFrame")

    other = meta_df.filter(scenario='a_scenario2')\
        .rename({'variable': {'Primary Energy': 'Primary Energy clone'}})

    meta_df.set_meta([0, 1], name='col1')

    other.set_meta(2, name='col1')
    other.set_meta('b', name='col2')

    # check that non-matching meta raise an error
    pytest.raises(ValueError, meta_df.append, other=other)

    # check that ignoring meta conflict works as expetced
    df = meta_df.append(other, ignore_meta_conflict=True)

    # check that the new meta.index is updated, but not the original one
    npt.assert_array_equal(meta_df.meta.columns, ['exclude', 'col1'])

    # assert that merging of meta works as expected
    exp = meta_df.meta.copy()
    exp['col2'] = [np.nan, 'b']
    pd.testing.assert_frame_equal(df.meta, exp)

    # assert that appending data works as expected
    ts = df.timeseries()
    npt.assert_array_equal(ts.iloc[2], ts.iloc[3])


def test_append_duplicates(test_df):
    if isinstance(test_df, OpenSCMDataFrame):
        pytest.xfail(reason="I have no idea why, but this fails for OpenSCMDataFrame")
    other = copy.deepcopy(test_df)
    pytest.raises(ValueError, test_df.append, other=other)


def test_interpolate(test_df):
    test_df.interpolate(2007)
    dct = {'model': ['a_model'] * 3, 'scenario': ['a_scenario'] * 3,
           'years': [2005, 2007, 2010], 'value': [1, 3, 6]}
    exp = pd.DataFrame(dct).pivot_table(index=['model', 'scenario'],
                                        columns=['years'], values='value')
    variable = {'variable': 'Primary Energy'}
    obs = test_df.filter(**variable).timeseries()
    npt.assert_array_equal(obs, exp)

    # redo the inpolation and check that no duplicates are added
    test_df.interpolate(2007)
    assert not test_df.filter(**variable).data.duplicated().any()


def test_set_meta_no_name(meta_df):
    idx = pd.MultiIndex(levels=[['a_scenario'], ['a_model'], ['a_region']],
                        labels=[[0], [0], [0]],
                        names=['scenario', 'model', 'region'])
    s = pd.Series(data=[0.3], index=idx)
    pytest.raises(ValueError, meta_df.set_meta, s)


def test_set_meta_as_named_series(meta_df):
    idx = pd.MultiIndex(levels=[['a_scenario'], ['a_model'], ['a_region']],
                        labels=[[0], [0], [0]],
                        names=['scenario', 'model', 'region'])

    s = pd.Series(data=[0.3], index=idx)
    s.name = 'meta_values'
    meta_df.set_meta(s)

    idx = pd.MultiIndex(levels=[['a_model'], ['a_scenario', 'a_scenario2']],
                        labels=[[0, 0], [0, 1]], names=['model', 'scenario'])
    exp = pd.Series(data=[0.3, np.nan], index=idx)
    exp.name = 'meta_values'

    obs = meta_df['meta_values']
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_as_unnamed_series(meta_df):
    idx = pd.MultiIndex(levels=[['a_scenario'], ['a_model'], ['a_region']],
                        labels=[[0], [0], [0]],
                        names=['scenario', 'model', 'region'])

    s = pd.Series(data=[0.3], index=idx)
    meta_df.set_meta(s, name='meta_values')

    idx = pd.MultiIndex(levels=[['a_model'], ['a_scenario', 'a_scenario2']],
                        labels=[[0, 0], [0, 1]], names=['model', 'scenario'])
    exp = pd.Series(data=[0.3, np.nan], index=idx)
    exp.name = 'meta_values'

    obs = meta_df['meta_values']
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_non_unique_index_fail(meta_df):
    idx = pd.MultiIndex(levels=[['a_model'], ['a_scenario'], ['a', 'b']],
                        labels=[[0, 0], [0, 0], [0, 1]],
                        names=['model', 'scenario', 'region'])
    s = pd.Series([0.4, 0.5], idx)
    pytest.raises(ValueError, meta_df.set_meta, s)


def test_set_meta_non_existing_index_fail(meta_df):
    idx = pd.MultiIndex(levels=[['a_model', 'fail_model'],
                                ['a_scenario', 'fail_scenario']],
                        labels=[[0, 1], [0, 1]], names=['model', 'scenario'])
    s = pd.Series([0.4, 0.5], idx)
    pytest.raises(ValueError, meta_df.set_meta, s)


def test_set_meta_by_df(meta_df):
    df = pd.DataFrame([
        ['a_model', 'a_scenario', 'a_region1', 1],
    ], columns=['model', 'scenario', 'region', 'col'])

    meta_df.set_meta(meta=0.3, name='meta_values', index=df)

    idx = pd.MultiIndex(levels=[['a_model'], ['a_scenario', 'a_scenario2']],
                        labels=[[0, 0], [0, 1]], names=['model', 'scenario'])
    exp = pd.Series(data=[0.3, np.nan], index=idx)
    exp.name = 'meta_values'

    obs = meta_df['meta_values']
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_as_series(meta_df):
    s = pd.Series([0.3, 0.4])
    meta_df.set_meta(s, 'meta_series')

    idx = pd.MultiIndex(levels=[['a_model'],
                                ['a_scenario', 'a_scenario2']],
                        labels=[[0, 0], [0, 1]], names=['model', 'scenario'])

    exp = pd.Series(data=[0.3, 0.4], index=idx)
    exp.name = 'meta_series'

    obs = meta_df['meta_series']
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_as_int(meta_df):
    meta_df.set_meta(3.2, 'meta_int')

    idx = pd.MultiIndex(levels=[['a_model'],
                                ['a_scenario', 'a_scenario2']],
                        labels=[[0, 0], [0, 1]], names=['model', 'scenario'])

    exp = pd.Series(data=[3.2, 3.2], index=idx, name='meta_int')

    obs = meta_df['meta_int']
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_as_str(meta_df):
    meta_df.set_meta('testing', name='meta_str')

    idx = pd.MultiIndex(levels=[['a_model'],
                                ['a_scenario', 'a_scenario2']],
                        labels=[[0, 0], [0, 1]], names=['model', 'scenario'])

    exp = pd.Series(data=['testing', 'testing'], index=idx, name='meta_str')

    obs = meta_df['meta_str']
    pd.testing.assert_series_equal(obs, exp)


def test_set_meta_as_str_list(meta_df):
    meta_df.set_meta(['testing', 'testing2'], name='category')
    obs = meta_df.filter(category='testing')
    assert obs['scenario'].unique() == 'a_scenario'


def test_set_meta_as_str_by_index(meta_df):
    idx = pd.MultiIndex(levels=[['a_model'], ['a_scenario']],
                        labels=[[0], [0]], names=['model', 'scenario'])

    meta_df.set_meta('foo', 'meta_str', idx)

    obs = pd.Series(meta_df['meta_str'].values)
    pd.testing.assert_series_equal(obs, pd.Series(['foo', None]))


def test_filter_by_bool(meta_df):
    meta_df.set_meta([True, False], name='exclude')
    obs = meta_df.filter(exclude=True)
    assert obs['scenario'].unique() == 'a_scenario'


def test_filter_by_int(meta_df):
    meta_df.set_meta([1, 2], name='value')
    obs = meta_df.filter(value=[1, 3])
    assert obs['scenario'].unique() == 'a_scenario'


def _r5_regions_exp(df):
    df = df.filter(region='World', keep=False)
    df['region'] = 'R5MAF'
    return df.data.reset_index(drop=True)


def test_map_regions_r5(reg_df):
    obs = reg_df.map_regions('r5_region').data
    exp = _r5_regions_exp(reg_df)
    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


def test_map_regions_r5_region_col(reg_df):
    df = reg_df.filter(model='MESSAGE-GLOBIOM')
    obs = df.map_regions(
        'r5_region', region_col='MESSAGE-GLOBIOM.REGION').data
    exp = _r5_regions_exp(df)
    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


def test_map_regions_r5_inplace(reg_df):
    exp = _r5_regions_exp(reg_df)
    reg_df.map_regions('r5_region', inplace=True)
    obs = reg_df.data
    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


def test_map_regions_r5_agg(reg_df):
    columns = reg_df.data.columns
    obs = reg_df.map_regions('r5_region', agg='sum').data

    exp = _r5_regions_exp(reg_df)
    grp = list(columns)
    grp.remove('value')
    exp = exp.groupby(grp).sum().reset_index()
    exp = exp[columns]
    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


def test_48a(pyam_df):
    # tests fix for #48 mapping many->few
    df = pyam_df(pd.DataFrame([
        ['model', 'scen', 'SSD', 'var', 'unit', 1, 6],
        ['model', 'scen', 'SDN', 'var', 'unit', 2, 7],
        ['model', 'scen1', 'SSD', 'var', 'unit', 2, 7],
        ['model', 'scen1', 'SDN', 'var', 'unit', 2, 7],
    ], columns=['model', 'scenario', 'region',
                'variable', 'unit', 2005, 2010],
    ))

    exp = _r5_regions_exp(df)
    columns = df.data.columns
    grp = list(columns)
    grp.remove('value')
    exp = exp.groupby(grp).sum().reset_index()
    exp = exp[columns]

    obs = df.map_regions('r5_region', region_col='iso', agg='sum').data

    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


def test_48b(pyam_df):
    # tests fix for #48 mapping few->many

    exp = pyam_df(pd.DataFrame([
        ['model', 'scen', 'SSD', 'var', 'unit', 1, 6],
        ['model', 'scen', 'SDN', 'var', 'unit', 1, 6],
        ['model', 'scen1', 'SSD', 'var', 'unit', 2, 7],
        ['model', 'scen1', 'SDN', 'var', 'unit', 2, 7],
    ], columns=['model', 'scenario', 'region',
                'variable', 'unit', 2005, 2010],
    )).data.reset_index(drop=True)

    df = IamDataFrame(pd.DataFrame([
        ['model', 'scen', 'R5MAF', 'var', 'unit', 1, 6],
        ['model', 'scen1', 'R5MAF', 'var', 'unit', 2, 7],
    ], columns=['model', 'scenario', 'region',
                'variable', 'unit', 2005, 2010],
    ))
    obs = df.map_regions('iso', region_col='r5_region').data
    obs = obs[obs.region.isin(['SSD', 'SDN'])].reset_index(drop=True)

    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


def test_48c(pyam_df):
    # tests fix for #48 mapping few->many, dropping duplicates

    exp = pyam_df(pd.DataFrame([
        ['model', 'scen', 'AGO', 'var', 'unit', 1, 6],
        ['model', 'scen1', 'AGO', 'var', 'unit', 2, 7],
    ], columns=['model', 'scenario', 'region',
                'variable', 'unit', 2005, 2010],
    )).data.reset_index(drop=True)

    df = pyam_df(pd.DataFrame([
        ['model', 'scen', 'R5MAF', 'var', 'unit', 1, 6],
        ['model', 'scen1', 'R5MAF', 'var', 'unit', 2, 7],
    ], columns=['model', 'scenario', 'region',
                'variable', 'unit', 2005, 2010],
    ))
    obs = df.map_regions('iso', region_col='r5_region',
                         remove_duplicates=True).data
    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


def test_rename_variable(pyam_df):
    df = pyam_df(pd.DataFrame([
        ['model', 'scen', 'SST', 'test_1', 'unit', 1, 5],
        ['model', 'scen', 'SDN', 'test_2', 'unit', 2, 6],
        ['model', 'scen', 'SST', 'test_3', 'unit', 3, 7],
    ], columns=['model', 'scenario', 'region',
                'variable', 'unit', 2005, 2010],
    ))

    mapping = {'variable': {'test_1': 'test', 'test_3': 'test'}}

    obs = df.rename(mapping).data.reset_index(drop=True)

    exp = pyam_df(pd.DataFrame([
        ['model', 'scen', 'SST', 'test', 'unit', 4, 12],
        ['model', 'scen', 'SDN', 'test_2', 'unit', 2, 6],
    ], columns=['model', 'scenario', 'region',
                'variable', 'unit', 2005, 2010],
    )).data.sort_values(by='region').reset_index(drop=True)

    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


def test_rename_index_fail(meta_df):
    mapping = {'scenario': {'a_scenario': 'a_scenario2'}}
    pytest.raises(ValueError, meta_df.rename, mapping)


def test_rename_index(meta_df):
    mapping = {'model': {'a_model': 'b_model'},
               'scenario': {'a_scenario': 'b_scen'}}
    obs = meta_df.rename(mapping)

    # test data changes
    exp = pd.DataFrame([
        ['b_model', 'b_scen', 'World', 'Primary Energy', 'EJ/y', 1., 6.],
        ['b_model', 'b_scen', 'World', 'Primary Energy|Coal', 'EJ/y', .5, 3.],
        ['b_model', 'a_scenario2', 'World', 'Primary Energy', 'EJ/y', 2., 7.],
    ], columns=['model', 'scenario', 'region', 'variable', 'unit', 2005, 2010]
    ).set_index(IAMC_IDX).sort_index()
    exp.columns = exp.columns.map(int)
    pd.testing.assert_frame_equal(obs.timeseries().sort_index(), exp)

    # test meta changes
    exp = pd.DataFrame([
        ['b_model', 'b_scen', False],
        ['b_model', 'a_scenario2', False],
    ], columns=['model', 'scenario', 'exclude']
    ).set_index(META_IDX)
    pd.testing.assert_frame_equal(obs.meta, exp)


def test_convert_unit(pyam_df):
    df = pyam_df(pd.DataFrame([
        ['model', 'scen', 'SST', 'test_1', 'A', 1, 5],
        ['model', 'scen', 'SDN', 'test_2', 'unit', 2, 6],
        ['model', 'scen', 'SST', 'test_3', 'C', 3, 7],
    ], columns=['model', 'scenario', 'region',
                'variable', 'unit', 2005, 2010],
    ))

    unit_conv = {'A': ['B', 5], 'C': ['D', 3]}

    obs = df.convert_unit(unit_conv).data.reset_index(drop=True)

    exp = pyam_df(pd.DataFrame([
        ['model', 'scen', 'SST', 'test_1', 'B', 5, 25],
        ['model', 'scen', 'SDN', 'test_2', 'unit', 2, 6],
        ['model', 'scen', 'SST', 'test_3', 'D', 9, 21],
    ], columns=['model', 'scenario', 'region',
                'variable', 'unit', 2005, 2010],
    )).data.reset_index(drop=True)

    pd.testing.assert_frame_equal(obs, exp, check_index_type=False)


def test_pd_filter_by_meta(meta_df):
    data = df_filter_by_meta_matching_idx.set_index(['model', 'region'])

    meta_df.set_meta([True, False], 'boolean')
    meta_df.set_meta(0, 'integer')

    obs = filter_by_meta(data, meta_df, join_meta=True,
                         boolean=True, integer=None)
    obs = obs.reindex(columns=['scenario', 'col', 'boolean', 'integer'])

    exp = data.iloc[0:2].copy()
    exp['boolean'] = True
    exp['integer'] = 0

    pd.testing.assert_frame_equal(obs, exp)


def test_pd_filter_by_meta_no_index(meta_df):
    data = df_filter_by_meta_matching_idx

    meta_df.set_meta([True, False], 'boolean')
    meta_df.set_meta(0, 'int')

    obs = filter_by_meta(data, meta_df, join_meta=True,
                         boolean=True, int=None)
    obs = obs.reindex(columns=META_IDX + ['region', 'col', 'boolean', 'int'])

    exp = data.iloc[0:2].copy()
    exp['boolean'] = True
    exp['int'] = 0

    pd.testing.assert_frame_equal(obs, exp)


def test_pd_filter_by_meta_nonmatching_index(meta_df):
    data = df_filter_by_meta_nonmatching_idx
    meta_df.set_meta(['a', 'b'], 'string')

    obs = filter_by_meta(data, meta_df, join_meta=True, string='b')
    obs = obs.reindex(columns=['scenario', 2010, 2020, 'string'])

    exp = data.iloc[2:3].copy()
    exp['string'] = 'b'

    pd.testing.assert_frame_equal(obs, exp)


def test_pd_join_by_meta_nonmatching_index(meta_df):
    data = df_filter_by_meta_nonmatching_idx
    meta_df.set_meta(['a', 'b'], 'string')

    obs = filter_by_meta(data, meta_df, join_meta=True, string=None)
    obs = obs.reindex(columns=['scenario', 2010, 2020, 'string'])

    exp = data.copy()
    exp['string'] = [np.nan, np.nan, 'b']

    pd.testing.assert_frame_equal(obs.sort_index(level=1), exp)


@pytest.mark.xfail(reason="cast_years_to_int isn't actually casting, rather just checking that if you cast years to int, you end up with the same thing, without actually returning the cast values")
def test_iam_df_year_axis_to_int(test_df_iam):
    float_time_df = test_df_iam.data.copy()
    float_time_df.year = float_time_df.year.astype(float)
    test_df = IamDataFrame(data=float_time_df)

    # the first assertion is a sanity check
    assert test_df_iam.data.year.dtype <= np.integer
    assert test_df.data.year.dtype <= np.integer


def test_openscm_df_year_axis_is_float(float_time_pd_df):
    float_time_pd_df.rename({"year": "time"}, axis="columns", inplace=True)
    test_df = OpenSCMDataFrame(data=float_time_pd_df)
    assert test_df.data.year.dtype <= np.float


@pytest.mark.parametrize("error_cls", [Exception, KeyError, AttributeError])
def test_worst_case_conversion_error_to_openscm(test_df_iam, error_cls):
    test_df_iam._get_openscm_df_data_except_year_renaming_and_metadata = Mock(side_effect=error_cls("Test"))
    error_msg = (
        re.escape("I don't know why, but I can't convert to an OpenSCMDataFrame.")
        + r"\n"
        + re.escape("The original traceback is:")
        + r"\n[\s\S]*Test[\s\S]*"
    )
    with pytest.raises(ConversionError, match=error_msg):
        test_df_iam.to_openscm_df()


def test_to_openscm_df():
    test_df = IamDataFrame(pd.DataFrame([
        ['a_model', 'a_scenario', 'World', 'Diagnostics|c_model|Atmospheric Concentrations|CO2', 'ppm', 2005, 395],
        ['a_model', 'a_scenario', 'World', 'Diagnostics|c_model|Atmospheric Concentrations|CO2', 'ppm', 2010, 401],
        ['a_model', 'a_scenario', 'World', 'Diagnostics|c_model|Surface Temperature', 'K', 2005, 0.9],
        ['a_model', 'a_scenario', 'World', 'Diagnostics|c_model|Surface Temperature', 'K', 2010, 0.94],
        ['a_model', 'a_scenario', 'World', 'Diagnostics|c_model|Emissions|CO2', 'Gt C / yr', 2005, 0.5],
        ['a_model', 'a_scenario', 'World', 'Diagnostics|c_model|Emissions|CO2', 'Gt C / yr', 2010, 3.0],
        ['a_model', 'a_scenario', 'World', 'Emissions|CO2', 'Gt C / yr', 2005, 0.5],
        ['a_model', 'a_scenario', 'World', 'Emissions|CO2', 'Gt C / yr', 2010, 3.0],
        ['a_model', 'a_scenario', 'World', 'Emissions|CO2|Coal', 'Gt C / yr', 2005, 0.5],
        ['a_model', 'a_scenario', 'World', 'Emissions|CO2|Coal', 'Gt C / yr', 2010, 3.0],
    ],
        columns=['model', 'scenario', 'region', 'variable', 'unit', 'year', 'value'],
    ))

    exp = pd.DataFrame([
        ['c_model', 'a_scenario|a_model', 'World', 'Atmospheric Concentrations|CO2', 'ppm', 2005, 395],
        ['c_model', 'a_scenario|a_model', 'World', 'Atmospheric Concentrations|CO2', 'ppm', 2010, 401],
        ['c_model', 'a_scenario|a_model', 'World', 'Surface Temperature', 'K', 2005, 0.9],
        ['c_model', 'a_scenario|a_model', 'World', 'Surface Temperature', 'K', 2010, 0.94],
        ['c_model', 'a_scenario|a_model', 'World', 'Emissions|CO2', 'Gt C / yr', 2005, 0.5],
        ['c_model', 'a_scenario|a_model', 'World', 'Emissions|CO2', 'Gt C / yr', 2010, 3.0],
        ['N/A', 'a_scenario|a_model', 'World', 'Emissions|CO2', 'Gt C / yr', 2005, 0.5],
        ['N/A', 'a_scenario|a_model', 'World', 'Emissions|CO2', 'Gt C / yr', 2010, 3.0],
        ['N/A', 'a_scenario|a_model', 'World', 'Emissions|CO2|Coal', 'Gt C / yr', 2005, 0.5],
        ['N/A', 'a_scenario|a_model', 'World', 'Emissions|CO2|Coal', 'Gt C / yr', 2010, 3.0],
    ],
        columns=['model', 'scenario', 'region', 'variable', 'unit', 'year', 'value'],
    )

    obs = test_df.to_openscm_df()
    assert obs.data.year.dtype <= np.float
    # all this sort and reset is nasty but I don't see how to fix it
    pd.testing.assert_frame_equal(
        obs.data.sort_values(by=LONG_IDX, axis=0).reset_index(drop=True),
        exp.sort_values(by=LONG_IDX, axis=0).reset_index(drop=True),
        check_index_type=False
    )


def test_to_from_openscm_df_loop(test_df_iam):
    obs = test_df_iam.to_openscm_df().to_iam_df()

    exp_df = test_df_iam.data.reset_index(drop=True)
    pd.testing.assert_frame_equal(obs.data, exp_df)
    pd.testing.assert_frame_equal(obs.meta, test_df_iam.meta)


def test_to_iam_df():
    test_df = OpenSCMDataFrame(pd.DataFrame([
        ['c_model', 'a_scenario|a_model', 'World', 'Atmospheric Concentrations|CO2', 'ppm', 2005, 395],
        ['c_model', 'a_scenario|a_model', 'World', 'Atmospheric Concentrations|CO2', 'ppm', 2010, 401],
        ['c_model', 'a_scenario|a_model', 'World', 'Surface Temperature', 'K', 2005, 0.9],
        ['c_model', 'a_scenario|a_model', 'World', 'Surface Temperature', 'K', 2010, 0.94],
        ['c_model', 'a_scenario|a_model', 'World', 'Emissions|CO2', 'Gt C / yr', 2005, 0.5],
        ['c_model', 'a_scenario|a_model', 'World', 'Emissions|CO2', 'Gt C / yr', 2010, 3.0],
        ['N/A', 'a_scenario|a_model', 'World', 'Emissions|CO2', 'Gt C / yr', 2005, 0.5],
        ['N/A', 'a_scenario|a_model', 'World', 'Emissions|CO2', 'Gt C / yr', 2010, 3.0],
        ['N/A', 'a_scenario|a_model', 'World', 'Emissions|CO2|Coal', 'Gt C / yr', 2005, 0.5],
        ['N/A', 'a_scenario|a_model', 'World', 'Emissions|CO2|Coal', 'Gt C / yr', 2010, 3.0],
    ],
        columns=['model', 'scenario', 'region', 'variable', 'unit', 'year', 'value'],
    ))

    exp = pd.DataFrame([
        ['a_model', 'a_scenario', 'World', 'Diagnostics|c_model|Atmospheric Concentrations|CO2', 'ppm', 2005, 395],
        ['a_model', 'a_scenario', 'World', 'Diagnostics|c_model|Atmospheric Concentrations|CO2', 'ppm', 2010, 401],
        ['a_model', 'a_scenario', 'World', 'Diagnostics|c_model|Surface Temperature', 'K', 2005, 0.9],
        ['a_model', 'a_scenario', 'World', 'Diagnostics|c_model|Surface Temperature', 'K', 2010, 0.94],
        ['a_model', 'a_scenario', 'World', 'Diagnostics|c_model|Emissions|CO2', 'Gt C / yr', 2005, 0.5],
        ['a_model', 'a_scenario', 'World', 'Diagnostics|c_model|Emissions|CO2', 'Gt C / yr', 2010, 3.0],
        ['a_model', 'a_scenario', 'World', 'Emissions|CO2', 'Gt C / yr', 2005, 0.5],
        ['a_model', 'a_scenario', 'World', 'Emissions|CO2', 'Gt C / yr', 2010, 3.0],
        ['a_model', 'a_scenario', 'World', 'Emissions|CO2|Coal', 'Gt C / yr', 2005, 0.5],
        ['a_model', 'a_scenario', 'World', 'Emissions|CO2|Coal', 'Gt C / yr', 2010, 3.0],
    ],
        columns=['model', 'scenario', 'region', 'variable', 'unit', 'year', 'value'],
    )

    obs = test_df.to_iam_df()
    assert obs.data.year.dtype <= np.int
    # all this sort and reset is nasty but I don't see how to fix it
    pd.testing.assert_frame_equal(
        obs.data.sort_values(by=LONG_IDX, axis=0).reset_index(drop=True),
        exp.sort_values(by=LONG_IDX, axis=0).reset_index(drop=True),
        check_index_type=False
    )


def test_worst_case_conversion_error_to_iam(test_df_openscm):
    test_df_openscm._get_iam_df_data_and_metadata = Mock(side_effect=Exception("Test"))
    error_msg = (
        re.escape("I don't know why, but I can't convert to an IamDataFrame.")
        + r"\n"
        + re.escape("The original traceback is:")
        + r"\n[\s\S]*Test[\s\S]*"
    )
    with pytest.raises(ConversionError, match=error_msg):
        test_df_openscm.to_iam_df()


def test_to_from_iam_df_loop(test_df_openscm):
    obs = test_df_openscm.to_iam_df().to_openscm_df()

    exp_df = test_df_openscm.data.reset_index(drop=True)
    pd.testing.assert_frame_equal(obs.data, exp_df)
    pd.testing.assert_frame_equal(obs.meta, test_df_openscm.meta)


@pytest.fixture(scope="function")
def test_df_infilling(test_df):
    # how do you use append...
    test_infilling_df = pd.DataFrame([
        ['a_model', 'a_scenario', 'World', 'Emissions|BC', 'Mt BC / yr', 1.1, 1.2],
        ['a_model', 'a_scenario', 'World', 'Emissions|C2F6', 'kt C2F6 / yr', 1.1, 1.2],
        ['a_model', 'a_scenario', 'World', 'Emissions|CCl4', 'kt CCl4 / yr', 1.1, 1.2],
        ['a_model', 'b_scenario', 'World', 'Emissions|BC', 'Mt BC / yr', 2.1, 2.2],
        ['a_model', 'b_scenario', 'World', 'Emissions|C2F6', 'kt C2F6 / yr', 2.1, 2.2],
        ['a_model', 'b_scenario', 'World', 'Emissions|CCl4', 'kt CCl4 / yr', 2.1, 2.2],
        ['b_model', 'b_scenario', 'World', 'Emissions|BC', 'Mt BC / yr', 1.4, 1.4],
        ['b_model', 'b_scenario', 'World', 'Emissions|C2F6', 'kt C2F6 / yr', 1.4, 1.4],
        ['b_model', 'b_scenario', 'World', 'Emissions|CCl4', 'kt CCl4 / yr', 1.4, 1.4],
    ],
        columns=['model', 'scenario', 'region', 'variable', 'unit', 2040, 2050],
    )
    df = test_df.append(test_infilling_df)
    yield df


def test_add_missing_variables_example_1(test_df_infilling):
    tconfig = {"Emissions|C3F8": {"unit": "kt C3F8 / yr"}}
    obs = test_df_infilling.add_missing_variables(tconfig)

    for vtc in tconfig:
        obs.variables().isin([vtc]).sum() == 1
        var_df = obs.filter(variable="*{}*".format(vtc))
        assert (var_df["unit"] == tconfig[vtc]["unit"]).all()
        np.testing.assert_allclose(var_df["value"], 0.0)


def test_add_missing_variables_example_2(test_df_infilling):
    tlead_var = "Emissions|C2F6"
    tscale_value = 23
    tscale_year = 2040
    tconfig = {"Emissions|C3F8": {
        "unit": "kt C3F8 / yr",
        "lead variable": tlead_var,
        "scale year": tscale_year,
        "scale value": tscale_value,
    }}
    obs = test_df_infilling.add_missing_variables(tconfig)

    for vtc in tconfig:
        obs.variables().isin([vtc]).sum() == 1
        var_df = obs.filter(variable="*{}*".format(vtc))
        assert (var_df["unit"] == tconfig[vtc]["unit"]).all()

        for label, df in var_df.data.groupby(META_IDX + ["region"]):
            lead_trajectory = test_df_infilling.filter(
                variable=tconfig[vtc]["lead variable"],
                model=df.model.iloc[0],
                scenario=df.scenario.iloc[0],
            )

            lead_scale_value = lead_trajectory.filter(year=tscale_year,)["value"].values

            scale_factor = tscale_value / lead_scale_value
            np.testing.assert_allclose(
                df["value"].values, lead_trajectory["value"].values * scale_factor
            )


def test_add_missing_variables_example_2_with_interpolation(test_df_infilling):
    tlead_var = "Emissions|C2F6"
    tscale_value = 23
    tscale_year = 2045
    tconfig = {"Emissions|C3F8": {
        "unit": "kt C3F8 / yr",
        "lead variable": tlead_var,
        "scale year": tscale_year,
        "scale value": tscale_value,
    }}
    obs = test_df_infilling.add_missing_variables(tconfig)

    for vtc in tconfig:
        obs.variables().isin([vtc]).sum() == 1
        var_df = obs.filter(variable="*{}*".format(vtc))
        assert (var_df["unit"] == tconfig[vtc]["unit"]).all()

        for label, df in var_df.data.groupby(META_IDX + ["region"]):
            lead_trajectory = test_df_infilling.filter(
                variable=tconfig[vtc]["lead variable"],
                model=df.model.iloc[0],
                scenario=df.scenario.iloc[0],
            )

            scale_df = deepcopy(lead_trajectory)
            scale_df.interpolate(tscale_year)
            lead_scale_value = scale_df.filter(year=tscale_year,)["value"].values

            scale_factor = tscale_value / lead_scale_value
            np.testing.assert_allclose(
                df["value"].values, lead_trajectory["value"].values * scale_factor
            )


def test_add_missing_variables_bad_lead_variable_error(test_df_infilling):
    tlead_var = "junk"
    tconfig = {"Emissions|C3F8": {
        "unit": "kt C3F8 / yr",
        "lead variable": tlead_var,
    }}
    error_msg = (
        "Lead variable '{}' could not be found for all model-scenario-region "
        "combinations in your data frame".format(tlead_var)
    )
    with pytest.raises(ValueError, match=error_msg):
        test_df_infilling.add_missing_variables(tconfig)


def test_add_missing_variables_variable_exists_warning(test_df_infilling):
    tlead_var = "Emissions|C2F6"
    tconfig = {tlead_var: {}}
    warning_msg = (
        "Variable to fill '{}' already in data frame, skipping".format(tlead_var)
    )
    with warnings.catch_warnings(record=True) as mock_existing_var_warning:
        test_df_infilling.add_missing_variables(tconfig)

    assert len(mock_existing_var_warning) == 1  # just rethrow warnings
    assert str(mock_existing_var_warning[0].message) == warning_msg
