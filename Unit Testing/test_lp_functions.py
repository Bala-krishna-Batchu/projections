import sys
import pandas as pd
import numpy as np

sys.path.append('..')
import CPMO_lp_functions as lpf
import CPMO_parameters as p

test_dataframe = pd.DataFrame({
    'CLIENT': p.TIERED_PRICE_CLIENT[0],
    'REGION': p.TIERED_PRICE_CLIENT[0],
    'MEASUREMENT': (['M30']*4 + ['R30']*4)*2,
    'GPI_NDC': ['1'.zfill(14), '2'.zfill(14), '3'.zfill(14), '4'.zfill(14)]*4,
    'MAC_PRICE_UNIT_ADJ': [1., 1.5, 2., 8.]*4,
    'MAC1026_UNIT_PRICE': [1.25, 1.25, 1.25, 1.25]*4,
    'AVG_AWP': [10, 10, 10, 100]*4,
    'PRICE_MUTABLE': 1,
    'UC_UNIT': [3.50, 3.75, 1., 15.]*4,
    'PRICE_REIMB_CLAIM': [10., 30., 100., 300.]*4,
    'CHAIN_GROUP': ['MAIL']*4+['CVS']*4+['MAIL']*4+['WMT']*4,
    'CLAIMS': 100,
    'CLAIMS_PROJ_EOY': 100,
    'QTY': [1000, 3000, 10000, 30000]*4,
    'QTY_PROJ_EOY': [1000, 3000, 10000, 30000] * 4,
    'GPI_CHANGE_EXCEPT': False,
    'CLIENT_MAX_PRICE': 1000,
    'CLIENT_MIN_PRICE': 0,
    'PRICE_TIER': '0',
    'BREAKOUT_AWP_MAX': 0,
    'PRICE_CHANGED_UC': False,
    'MAC_PRICE_UPPER_LIMIT_UC': 1.49,
    'RAISED_PRICE_UC': False,
    'GOODRX_UPPER_LIMIT': 1000,
})

def test_priceBoundParameters():
    # Test that the price bound parameters have been set in increasing/decreasing order
    upper_bound = list(p.PRICE_BOUNDS_DF['upper_bound'])
    max_percent_increase = list(p.PRICE_BOUNDS_DF['max_percent_increase'])
    max_dollar_increase = list(p.PRICE_BOUNDS_DF['max_dollar_increase'])
    assert sorted(upper_bound) == upper_bound and sorted(max_dollar_increase) == max_dollar_increase \
           and sorted(max_percent_increase, reverse = True) == max_percent_increase, "Doublecheck p.PRICE_BOUNDS input"
    # Comparing lpf.priceBound() results to before the change:
    results = [lpf.priceBound(row, 0.8, 1.2, True, 12) for _, row in test_dataframe.iterrows()]
    assert results == [(0.19999999999999996, 2.0), (0.29999999999999993, 1.5), (0.3999999999999999, 1.0), (1.5999999999999996, 8.8),
     (1.25, 2.0), (1.25, 1.5), (1.25, 1.2501), (1.5999999999999996, 8.8), (0.19999999999999996, 2.0),
     (0.29999999999999993, 1.5), (0.3999999999999999, 1.0), (1.5999999999999996, 8.8), (1.25, 2.0), (1.25, 1.5),
     (1.25, 1.2501), (1.5999999999999996, 8.8)], "Output does not match previous logic"



def test_priceBound():
    assert p.TIERED_PRICE_LIM, "test_priceBound only works if p.TIERED_PRICE_LIM=True"
    results = [lpf.priceBound(row, 0.8, 1.2, True, 12) for _, row in test_dataframe.iterrows()]
    # Bounds below: 80% drop, 50% increase; 80% drop, 50% increase; 80% drop, U&C upper limit; 80% drop, 30% increase;
    # MAC1026 floor, 50% increase; MAC1026 floor, 50% increase; MAC1026 floor, MAC1026 floor + 0.0001; 80% drop, 30% increase
    expected_results = [(0.2, 1.5), (0.3, 2.25), (0.4, 1.2501), (1.6, 10.40),
                        (1.25, 1.5), (1.25, 2.25), (1.25, 1.2501), (1.60, 10.40)]*2
    assert np.allclose(results, expected_results), "Not all price bounds were as expected for tiered case"

    # test_df_nontiered = test_dataframe.copy()
    # test_df_nontiered['CLIENT'] = 'Dummy Client'
    # results = [lpf.priceBound(row, 0.8, 1.2, True, 12) for _, row in test_df_nontiered.iterrows()]
    # # Upper limits are now min(uc_unit, 1.2*mac_price_unit_adj
    # expected_results = [(0.2, 1.2), (0.3, 1.8), (0.4, 1.0), (1.6, 9.6),
    #                     (1.25, 1.2501), (1.25, 1.8), (1.25, 1.2501), (1.60, 9.6)]*2
    # assert np.allclose(results, expected_results), "Not all price bounds were as expected for non-tiered case"

    if 'GOODRX_OPT' in dir(p) and p.GOODRX_OPT:
        test_df_goodrx = test_dataframe.copy()
        test_df_goodrx['GOODRX_UPPER_LIMIT'] = [0.8, 1.7, 20.0, 20.0]*4
        results = [lpf.priceBound(row, 0.8, 1.2, True, 12) for _, row in test_df_goodrx.iterrows()]
        # Upper limits are now min(uc_unit, 1.2*mac_price_unit_adj, goodrx_price).
        # Note that for the first item, the "goodrx price" actually gets moved to the mac1026 floor.
        # That's because we don't have goodrx prices for mail, so mail only has a goodrx price to match retail...
        # ...and retail always DOES have mac1026 applied.
        expected_results = [(0.2, 1.25), (0.3, 1.7), (0.4, 1.0), (1.6, 10.40),
                            (1.25, 1.2501), (1.25, 1.7), (1.25, 1.2501), (1.60, 10.40)] * 2
        assert np.allclose(results, expected_results), "Not all price bounds were as expected for goodrx case"

    # if 'UNC_OPT' in dir(p) and p.UNC_OPT:
    #     test_df_uc = test_dataframe.copy()
    #     test_df_uc['PRICE_CHANGED_UC'] = True
    #     test_df_uc['RAISED_PRICE_UC'] = [False, True]*8
    #     test_df_uc['MAC_PRICE_UPPER_LIMIT_UC'] = 1.30
    #     results = [lpf.priceBound(row, 0.8, 1.2, True, 12) for _, row in test_df_uc.iterrows()]
    #     expected_results = [(0.2, 1.3), (1.50, 1.50), (0.4, 1.3), (8., 8.),
    #                         (1.25, 1.3), (1.50, 1.50), (1.25, 1.3), (8., 8.)] * 2
    #     assert np.allclose(results, expected_results), "Not all price bounds were as expected for U&C optimization"

    if ('UNC_OPT' in dir(p) and p.UNC_OPT
            and 'GOODRX_OPT' in dir(p) and p.GOODRX_OPT
            and 'UNC_OVERRIDE_GOODRX' in dir(p) and p.UNC_OVERRIDE_GOODRX is False):
        test_df_uc_goodrx = test_dataframe.copy()
        test_df_uc_goodrx['GOODRX_UPPER_LIMIT'] = [0.8, 1.7, 20.0, 7.0]*4
        test_df_uc_goodrx['PRICE_CHANGED_UC'] = True
        test_df_uc_goodrx['RAISED_PRICE_UC'] = [False, True]*8
        test_df_uc_goodrx['MAC_PRICE_UPPER_LIMIT_UC'] = 1.30
        results = [lpf.priceBound(row, 0.8, 1.2, True, 12) for _, row in test_df_uc_goodrx.iterrows()]
        # The first and fitth items have goodrx < unc upper limit, so they get a goodrx upper limit
        # the second and sixth are U&C price raises that are less than the goodrx upper limit, so they get the u&c price raise
        # the third and seventh have a u&c price drop that is less than the goodrx upper limit, so they get u&c
        # the final item has a goodrx upper limit lower than the u&c price, so u&c gets superceded.
        expected_results = [(0.2, 1.25), (1.50, 1.50), (0.4, 1.3), (1.6, 7.),
                            (1.25, 1.2501), (1.50, 1.50), (1.25, 1.3), (1.6, 7.)] * 2
        for(r, er) in zip(results, expected_results):
            print(r,er)
        assert np.allclose(results, expected_results), "Not all price bounds were as expected for GoodRX + U&C optimization with GoodRX taking precedence"

    if ('UNC_OPT' in dir(p) and p.UNC_OPT
            and 'GOODRX_OPT' in dir(p) and p.GOODRX_OPT
            and 'UNC_OVERRIDE_GOODRX' in dir(p) and p.UNC_OVERRIDE_GOODRX):
        test_df_uc_goodrx = test_dataframe.copy()
        test_df_uc_goodrx['GOODRX_UPPER_LIMIT'] = [0.8, 1.7, 20.0, 7.0]*4
        test_df_uc_goodrx['PRICE_CHANGED_UC'] = True
        test_df_uc_goodrx['RAISED_PRICE_UC'] = [False, True]*8
        test_df_uc_goodrx['MAC_PRICE_UPPER_LIMIT_UC'] = 1.30
        results = [lpf.priceBound(row, 0.8, 1.2, True, 12) for _, row in test_df_uc_goodrx.iterrows()]
        # The first and fitth items have goodrx < unc upper limit, so they get a goodrx upper limit
        # the second and sixth are U&C price raises that are less than the goodrx upper limit, so they get the u&c price raise
        # the third and seventh have a u&c price drop that is less than the goodrx upper limit, so they get u&c
        # the final item has a goodrx upper limit lower than the u&c price, and u&c overrides, so goodrx gets superceded.
        expected_results = [(0.2, 1.25), (1.50, 1.50), (0.4, 1.3), (8., 8.),
                            (1.25, 1.2501), (1.50, 1.50), (1.25, 1.3), (8., 8.)] * 2
        for(r, er) in zip(results, expected_results):
            print(r,er)
        assert np.allclose(results, expected_results), "Not all price bounds were as expected for GoodRX + U&C optimization with U&C taking precedence"


def test_generatePriceBounds():
    pass

def test_lb_ub():
    pass

def test_current_price_conflict():
    pass

def test_createPriceDecisionVarByClientBreakoutMeasureRegionChain():
    pass

def test_generatePricingDecisionVariables():
    pass

def test_generateLambdaDecisionVariables():
    pass

def test_generateLambdaDecisionVariables_ebit():
    pass

def test_generateCost_new():
    pass

def test_generate_constraint_prices():
    pass

def test_generate_constraint_pharm():
    pass

def test_generate_constraint_pharm_new():
    pass

def test_generate_constraint_client():
    pass

def test_generateGuaranteeConstraintEbit():
    pass

def test_pharmacy_type():
    pass

def test_pharmacy_type_new():
    pass

def test_determine_effective_price():
    pass

