# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 14:51:07 2021

@author: C172222
"""
import sys
import pandas as pd
import numpy as np
import datetime as dt
import pytest

sys.path.append('..')
import CPMO_lp_functions as lpf
import CPMO_parameters as p



# Parameters:
         
entry = [1]*12
ing_cost = [0 for i in entry]
ing_cost[1] = 29
ing_cost[2] = 310
gen_launch_dic = {'CLIENT' : ['4588' for i in entry],
                'BREAKOUT': ['4588_R30' for i in entry],
                'REGION' : ['4588' for i in entry],
                'MEASUREMENT' : ['R30' for i in entry],
                'CHAIN_GROUP' : ['ACH' for i in entry],
                'MONTH' : [i+1 for i in list(range(12))],
                'QTY' : [999 for i in entry],
                'ING_COST' : ing_cost,
                'FULLAWP' : [100 for i in entry]}
gen_launch_df = pd.DataFrame(data=gen_launch_dic)

pref_pharm_list_dic = {'CLIENT' : '4588',
                'BREAKOUT': '4588_R30',
                'REGION' : '4588',
                'PREF_PHARM' : 'None',
                'PSAO_GUARANTEE' : 'No',
                'CLIENT_NAME' : 'STATE OF GEORGIA',
                'PREF_PHARMS' : ['None']}
pref_pharm_list_df = pd.DataFrame(data=pref_pharm_list_dic)


def test_same_month():
    p.LAST_DATA = dt.datetime.strptime('2/1/2020', '%m/%d/%Y')
    p.GO_LIVE = dt.datetime.strptime('2/10/2020', '%m/%d/%Y')
    p.FULL_YEAR = False
    ytd, lag, eoy = lpf.gen_launch_df_generator_ytd_lag_eoy(gen_launch_df, pref_pharm_list_df)
    assert ytd['ING_COST'][0] == 1.0
    assert lag['ING_COST'][0] == 8.0
    assert eoy['ING_COST'][0] == 330.0
    
def test_small_lag():
    p.LAST_DATA = dt.datetime.strptime('2/1/2020', '%m/%d/%Y')
    p.GO_LIVE = dt.datetime.strptime('2/2/2020', '%m/%d/%Y')
    p.FULL_YEAR = False
    ytd, lag, eoy = lpf.gen_launch_df_generator_ytd_lag_eoy(gen_launch_df, pref_pharm_list_df)
    assert ytd['ING_COST'][0] == 1.0
    assert lag['ING_COST'][0] == 0.0
    assert eoy['ING_COST'][0] == 338.0
    
    
def test_diff_month_start():
    p.LAST_DATA = dt.datetime.strptime('2/15/2020', '%m/%d/%Y')
    p.GO_LIVE = dt.datetime.strptime('3/1/2020', '%m/%d/%Y')
    p.FULL_YEAR = False
    ytd, lag, eoy = lpf.gen_launch_df_generator_ytd_lag_eoy(gen_launch_df, pref_pharm_list_df)
    assert ytd['ING_COST'][0] == 15.0
    assert (lag['ING_COST'][0] - 14.0) < 1e-6
    assert eoy['ING_COST'][0] == 310.0
    
def test_diff_month_mid():
    p.LAST_DATA = dt.datetime.strptime('2/15/2020', '%m/%d/%Y')
    p.GO_LIVE = dt.datetime.strptime('3/15/2020', '%m/%d/%Y')
    p.FULL_YEAR = False
    ytd, lag, eoy = lpf.gen_launch_df_generator_ytd_lag_eoy(gen_launch_df, pref_pharm_list_df)
    assert ytd['ING_COST'][0] == 15.0
    assert lag['ING_COST'][0] == 154.0
    assert eoy['ING_COST'][0] == 170.0
    
def test_same_date():
    p.LAST_DATA = dt.datetime.strptime('2/1/2020', '%m/%d/%Y')
    p.GO_LIVE = dt.datetime.strptime('2/1/2020', '%m/%d/%Y')
    p.FULL_YEAR = False
    with pytest.raises(AssertionError):
        ytd, lag, eoy = lpf.gen_launch_df_generator_ytd_lag_eoy(gen_launch_df, pref_pharm_list_df)
    
def test_switch_date_order():
    p.LAST_DATA = dt.datetime.strptime('2/10/2020', '%m/%d/%Y')
    p.GO_LIVE = dt.datetime.strptime('2/1/2020', '%m/%d/%Y')
    p.FULL_YEAR = False
    with pytest.raises(AssertionError):
        ytd, lag, eoy = lpf.gen_launch_df_generator_ytd_lag_eoy(gen_launch_df, pref_pharm_list_df)
    
    
def test_diff_years():
    p.LAST_DATA = dt.datetime.strptime('2/10/2020', '%m/%d/%Y')
    p.GO_LIVE = dt.datetime.strptime('2/1/2021', '%m/%d/%Y')
    p.FULL_YEAR = False
    with pytest.raises(AssertionError):
        ytd, lag, eoy = lpf.gen_launch_df_generator_ytd_lag_eoy(gen_launch_df, pref_pharm_list_df)   
        
        
        
lp_vol_mv_agg_df = pd.DataFrame.from_dict({'CLIENT' : ['1']*8 + ['2']*8 + ['3']*8 + ['1', '2', '3', '3'],
                                 'GPI': ['1','2','3','4','1','2','3','4','1','2','3','4','1','2','3','4','1','2','3','4','1','2','3','4','5','5','5','5'],
                                 'MAC_LIST': ['1','1','1','1','2','2','2','2','1','1','1','1','2','2','2','2','1','1','1','1','2','2','2','2','2','2','2','1'],
                                 'PRICE_MUTABLE': [1]*28,
                                 'CURRENT_MAC_PRICE': [0]*28,
                                 'RAND':['X']*28},
                                orient='columns')
lp_vol_mv_agg_df.PRICE_MUTABLE = lp_vol_mv_agg_df.PRICE_MUTABLE.astype(np.float64)

mac_price_override = pd.DataFrame.from_dict({'CLIENT':      ['ALL','1',  '1', '1', '1', '2', '2', 'ALL'],
                                       'GPI':               ['1',  '2',  '3', '3', '4', '3', '4', '5'],
                                       'VCML_ID':           ['ALL','ALL','1', '2', '1', '1', '1', '2'],
                                       'PRICE_OVRD_AMOUNT': [1,     2,    3,   4,   5,   3,   6,   7]},
                                            orient='columns')


lp_vol_mv_agg_df_out = pd.DataFrame.from_dict({'CLIENT' : ['1']*8 + ['2']*8 + ['3']*8 + ['1', '2', '3', '3'],
                                 'GPI': ['1','2','3','4','1','2','3','4','1','2','3','4','1','2','3','4','1','2','3','4','1','2','3','4','5','5','5','5'],
                                 'MAC_LIST': ['1','1','1','1','2','2','2','2','1','1','1','1','2','2','2','2','1','1','1','1','2','2','2','2','2','2','2','1'],
                                 'PRICE_MUTABLE':     [0,0,0,0,0,0,0,1,0,1,0,0,0,1,1,1,0,1,1,1,0,1,1,1,0,0,0,1],
                                 'CURRENT_MAC_PRICE': [1,2,3,5,1,2,4,0,1,0,3,6,1,0,0,0,1,0,0,0,1,0,0,0,7,7,7,0],
                                 'RAND':['X']*28},
                                orient='columns')
lp_vol_mv_agg_df_out[['CURRENT_MAC_PRICE', 'PRICE_MUTABLE']] = lp_vol_mv_agg_df_out[['CURRENT_MAC_PRICE', 'PRICE_MUTABLE']].astype(np.float64)


def test_price_overrider():
    '''
    Test all the different logic at the client, vcml and gpi level
    '''
    out = lpf.price_overrider_function(mac_price_override, lp_vol_mv_agg_df)
        
    assert out.equals(lp_vol_mv_agg_df_out)
    
    