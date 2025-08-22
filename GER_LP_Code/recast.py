# -*- coding: utf-8 -*-
"""
"""
import CPMO_parameters as p
import datetime as dt
import pandas as pd
import numpy as np

from CPMO_lp_functions import *
from CPMO_shared_functions import *


DAILY_TOTALS_FILE = '20200217_2020WellCareMedD_thruJan.csv'
data_file = DAILY_TOTALS_FILE

if True:
    # lp_data_output_df = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + 'LAXGRAgg.csv'))
    lp_data_output_df = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + 'TieredAgg.csv'))


gpi_vol_awp_df = pd.read_csv(p.FILE_INPUT_PATH + data_file)
gpi_vol_awp_df = gpi_vol_awp_df.dropna(how='all') ## drop null rows, which came through bad csv format
na_rows = gpi_vol_awp_df.loc[gpi_vol_awp_df.CLIENT.isna()]
assert len(na_rows) < 5 #No more than 10 rows should ever be blank due to bad input read
gpi_vol_awp_df = gpi_vol_awp_df.drop(index = na_rows.index)


#rename and drop some of the columns
gpi_vol_awp_df.rename(columns={'NDC11': 'NDC',
                               'AWP': 'FULLAWP_ADJ',
                               'CLAIM_DATE': 'DOF',
                               'SPEND': 'PRICE_REIMB',
                               'UCAMT_UNIT': 'uc_unit',
                               'PCT25_UCAMT_UNIT': 'uc_unit25',
                               'PREFERRED': 'Pharmacy_Type'}, inplace=True)
    
gpi_vol_awp_df.drop(columns=['CUSTOMER_ID', 'MAILIND', 'GENIND', 'REC_CURR_IND', 'REC_ADD_USER',
                             'REC_ADD_TS', 'REC_CHG_USER', 'REC_CHG_USER', 'UNIQUE_ROW_ID'], 
                            inplace=True, errors='ignore')

gpi_vol_awp_df = standardize_df(gpi_vol_awp_df)
gpi_vol_awp_df.loc[gpi_vol_awp_df.BREAKOUT=='MAIL', 'Pharmacy_Type'] = 'Non_Preferred'
gpi_vol_awp_df.loc[gpi_vol_awp_df.Pharmacy_Type=='Non-preferred', 'Pharmacy_Type'] = 'Non_Preferred'

new_prices = lp_data_output_df.loc[lp_data_output_df.Price_Mutable==1, ['BREAKOUT', 'REGION', 'CHAIN_GROUP', 'MEASUREMENT', 
                                                                        'GPI', 'NDC', 'New_Price']]

new_prices_gpi = new_prices.loc[new_prices.NDC == '***********'].copy(deep=True)
new_prices_gpi.rename(columns={'New_Price': 'GPI_PRICE'}, inplace=True)
new_prices_ndc = new_prices.loc[new_prices.NDC != '***********']
new_prices_ndc.rename(columns={'New_Price': 'NDC_PRICE'}, inplace=True)
assert (len(new_prices_gpi) + len(new_prices_ndc)) == len(new_prices)

lp_vol_macprice_df = pd.merge(gpi_vol_awp_df, new_prices_ndc[['BREAKOUT', 'REGION', 'CHAIN_GROUP', 'MEASUREMENT', 'NDC', 'NDC_PRICE']], 
                              how ='left', on = ['BREAKOUT', 'REGION', 'CHAIN_GROUP', 'MEASUREMENT', 'NDC'])
lp_vol_macprice_df = pd.merge(lp_vol_macprice_df, new_prices_gpi[['BREAKOUT', 'REGION', 'CHAIN_GROUP', 'MEASUREMENT', 'GPI', 'GPI_PRICE']], 
                              how ='left', on = ['BREAKOUT', 'REGION', 'CHAIN_GROUP', 'MEASUREMENT', 'GPI'])
assert (gpi_vol_awp_df.FULLAWP_ADJ.sum() - lp_vol_macprice_df.FULLAWP_ADJ.sum()) < 0.0001
assert len(lp_vol_macprice_df) == len(gpi_vol_awp_df)


lp_vol_macprice_df['avg_awp'] = lp_vol_macprice_df.FULLAWP_ADJ/ lp_vol_macprice_df.QTY
lp_vol_macprice_df['NEW_MAC_PRICE'] = lp_vol_macprice_df.apply(lambda df: df.NDC_PRICE if np.isfinite(df.NDC_PRICE) else df.GPI_PRICE, axis=1)
lp_vol_macprice_df['Capped_new_price'] = determine_effective_price(lp_vol_macprice_df,
                                                                   old_price='NEW_MAC_PRICE',
                                                                   uc_unit='uc_unit25',
                                                                   capped_only=True)
lp_vol_macprice_df['Capped_new_price'].fillna(0) 
lp_vol_macprice_df['Capped_Reimb'] = lp_vol_macprice_df['Capped_new_price'] * lp_vol_macprice_df.QTY
lp_vol_macprice_df['Eff_capped_reimb'] = lp_vol_macprice_df.apply(lambda df: df['Capped_Reimb'] if df['Capped_Reimb'] > 0 else df['PRICE_REIMB'], axis=1)
assert(len(lp_vol_macprice_df.loc[(lp_vol_macprice_df.Eff_capped_reimb.isna())]) == 0) 

effective_new_perf_last_month_dict = calculatePerformance2(lp_vol_macprice_df, client_guarantees, pharmacy_guarantees,
                                               client_list, p.CAPPED_PHARMACY_LIST, oc_eoy_pharm_perf, gen_launch_dummy, pharmacy_approx,
                                               days=31, reimb_column='Eff_capped_reimb', AWP_column='FULLAWP_ADJ')

old_perf_last_month = calculatePerformance2(gpi_vol_awp_df, client_guarantees, pharmacy_guarantees,
                                               client_list, p.CAPPED_PHARMACY_LIST, oc_eoy_pharm_perf, gen_launch_dummy, pharmacy_approx,
                                               days=31, reimb_column='PRICE_REIMB', AWP_column='FULLAWP_ADJ')
