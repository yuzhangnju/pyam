import numpy as np
import pytest
import pandas as pd

from pyam import IamDataFrame, check_aggregate


def test_init_df_from_timeseries(test_df_iam):
    df = IamDataFrame(test_df_iam.timeseries())
    pd.testing.assert_frame_equal(df.timeseries(), test_df_iam.timeseries())


def test_init_iam_df_with_float_cols_raises(test_pd_df):
    _test_df_iam = test_pd_df.rename(columns={2005: 2005.5, 2010: 2010.})
    pytest.raises(ValueError, IamDataFrame, data=_test_df_iam)


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


def test_check_aggregate_pass_iam(check_aggregate_df_iam):
    obs = check_aggregate_df_iam.filter(
        scenario='a_scen'
    ).check_aggregate('Primary Energy')
    assert obs is None


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


def test_df_iam_check_aggregate_regions_pass(check_aggregate_df_iam):
    obs = check_aggregate_df_iam.check_aggregate_regions('Primary Energy')
    assert obs is None

    for variable in check_aggregate_df_iam.variables():
        obs = check_aggregate_df_iam.check_aggregate_regions(variable)
        assert obs is None
