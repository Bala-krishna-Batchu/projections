"""
CLIENT_PHARMACY_MAC_OPTIMIZATION

This is the main script for client pharmacy MAC optimization.
This will take utilization projections and create MAC pricing to satisfy the
guarantees.
"""
from typing import NamedTuple
from kfp.components import InputPath, OutputPath
import sys

# import CPMO_parameters as p
# import datetime as dt
# import pandas as pd
# import numpy as np
# import calendar
# import copy
# import time
# import logging
# import pulp

# import CPMO_lp_functions as lpf
# import CPMO_shared_functions as sf
# import CPMO_plan_liability as pl

# cli arguments and parameters
def arguments():
    import argparse as ap
    import json
    import jinja2 as jj2
    from collections import namedtuple

    parser = ap.ArgumentParser()
    parser.add_argument(
        '--custom-args-json',
        help=('JSON file URL, for file supplying custom values. '
              'for example: '
              '{"client_name": "client1", "timestamp": "2020-12-29", ...}'),
        required=False,
        default=None
    )
    parser.add_argument(
        '-t', '--template',
        help='Path to parameters template.',
        required=False,
        default=None
    )
    parser.add_argument(
        '-p', '--parameter-file',
        help=('Path to hard coded parameters file.'
              'This option will override the --template option if it is specified.'),
        required=False,
        default=None
    )
    parser.add_argument(
        '--loglevel', choices=['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL', 'CRITICAL'], 
        default='INFO',
        help='Logging is currently not supported in QA.py; however, this is included for future compatability.'
    )
    args = parser.parse_args()    
    # setup parameters
    outpt = namedtuple('Output', ['params', 'loglevel'])
    # if hard coded parameters are supplied, read and return as string
    if not args.parameter_file and not args.template:
        raise Exception('Either --parameter-file or --template CLI arg must be supplied.')
    if args.parameter_file:
        params = open(args.parameter_file, 'r').read()
        return outpt(params=params, loglevel=args.loglevel)
    # if template path is supplied (and not hard coded parameters), supply custom parameters (if any)
    # and return parameter string
    elif args.template:
        _arguments = dict(json.load(open(args.custom_args_json)))
        template = jj2.Template(open(args.template).read())
        return outpt(params=template.render(**_arguments), loglevel=args.loglevel)

def opt_preprocessing(
    # required inputs
    m:int,
    params_file_in: str,
    unc_flag: bool, 
    # file placeholders (for kubeflow components)
    eoy_days_out: OutputPath('pickle'),
    proj_days_out: OutputPath('pickle'),
    lp_vol_mv_agg_df_out: OutputPath('pickle'),
    mac_list_df_out: OutputPath('pickle'),
    chain_region_mac_mapping_out: OutputPath('pickle'),
    lp_vol_mv_agg_df_actual_out: OutputPath('pickle'),
    client_list_out: OutputPath('pickle'),
    breakout_df_out: OutputPath('pickle'),
    client_guarantees_out: OutputPath('pickle'),
    pharmacy_guarantees_out: OutputPath('pickle'),
    pref_pharm_list_out: OutputPath('pickle'),
    # non_capped_pharmacy_list_out: OutputPath('pickle'),
    # agreement_pharmacy_list_out: OutputPath('pickle'),
    oc_pharm_surplus_out: OutputPath('pickle'),
    other_client_pharm_lageoy_out: OutputPath('pickle'),
    oc_eoy_pharm_perf_out: OutputPath('pickle'),
    generic_launch_df_out: OutputPath('pickle'),
    oc_pharm_dummy_out: OutputPath('pickle'),
    dummy_perf_dict_out: OutputPath('pickle'),
    perf_dict_col_out: OutputPath('pickle'),
    gen_launch_eoy_dict_out: OutputPath('pickle'),        
    gen_launch_lageoy_dict_out: OutputPath('pickle'),
    ytd_perf_pharm_actuals_dict_out: OutputPath('pickle'),
    performance_dict_out: OutputPath('pickle'),
    act_performance_dict_out: OutputPath('pickle'),
    # total_pharm_list_out: OutputPath('pickle'),
    lp_data_df_out: OutputPath('pickle'),
    price_lambdas_out: OutputPath('pickle'),
    # brand-surplus files
    brand_surplus_ytd_out: OutputPath('pickle'),
    brand_surplus_lag_out: OutputPath('pickle'),
    brand_surplus_eoy_out: OutputPath('pickle'),
    # specialty surplus files
    specialty_surplus_ytd_out: OutputPath('pickle'),
    specialty_surplus_lag_out: OutputPath('pickle'),
    specialty_surplus_eoy_out: OutputPath('pickle'),
    # dispensing feei surplus files
    disp_fee_surplus_ytd_out: OutputPath('pickle'),
    disp_fee_surplus_lag_out: OutputPath('pickle'),
    disp_fee_surplus_eoy_out: OutputPath('pickle'),
    loglevel: str = 'INFO'
    # kube_run: bool = True,
) -> NamedTuple('Output', [('lag_price_col', str), ('pharm_lag_price_col', str)]):

    import sys
    import os
    sys.path.append('/')
    import copy
    import logging
    import pickle
    import datetime as dt
    import numpy as np
    import pandas as pd
    import pulp
    import util_funcs as uf
    import BQ
    from dateutil import relativedelta

    uf.write_params(params_file_in)
    import CPMO_parameters as p
    from qa_checks import qa_dataframe
    from CPMO_shared_functions import standardize_df, is_column_unique, df_to_dict, calculatePerformance,determine_effective_price, add_virtual_r90, add_target_ingcost, read_tru_mac_list_prices
    from CPMO_lp_functions import (
        pharmacy_type_new, generate_price_bounds,
        price_overrider_function, gen_launch_df_generator_ytd_lag_eoy
    )
    import CPMO_shared_functions as sf
    from CPMO_shared_functions import update_run_status
    try:
        out_path = os.path.join(p.FILE_LOG_PATH, 'ClientPharmacyMacOptimization.log')
        logger = uf.log_setup(log_file_path=out_path, loglevel=loglevel)

        month = p.LP_RUN[m]

        logger.info('*******STARTING MONTH %d*******', month)

        contract_date_df = pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.CONTRACT_DATE_FILE))
        contract_date_df['CONTRACT_EFF_DT'] = pd.to_datetime(contract_date_df['CONTRACT_EFF_DT'])
        eoc = dt.datetime.strptime(contract_date_df['CONTRACT_EXPRN_DT'][0], '%Y-%m-%d')
        one_day = dt.timedelta(1)
        if p.FULL_YEAR:
            lag_days = 0
            # foy = dt.datetime.strptime('1/1/' + str(p.LAST_DATA.year + 1), '%m/%d/%Y')
            contract_length = relativedelta.relativedelta(eoc, contract_date_df['CONTRACT_EFF_DT'][0]).months + 1
            eoy = eoc + relativedelta.relativedelta(months = contract_length)
        else:
            lag_days = (p.GO_LIVE - p.LAST_DATA).days - 1
            # foy = dt.datetime.strptime('1/1/' + str(p.LAST_DATA.year), '%m/%d/%Y')
            eoy = eoc

        eoy_days = (eoy - p.GO_LIVE).days + 1
        proj_days = lag_days + eoy_days
        ytd_days = (p.LAST_DATA - dt.datetime.strptime(p.DATA_START_DAY, '%Y-%m-%d')).days

        lp_vol_mv_agg_df = pd.DataFrame()
        gpi_vol_awp_agg_YTD = pd.DataFrame()

        logger.info("Loading stored aggregate data")
        """ 
        if p.WRITE_TO_BQ and (unc_flag == True or p.UNC_ADJUST == False):
            lp_vol_mv_agg_df = uf.read_BQ_data(
                BQ.lp_data, project_id = p.BQ_OUTPUT_PROJECT_ID, dataset_id = p.BQ_OUTPUT_DATASET, table_id = 'lp_data', 
                client = ', '.join(sorted(p.CUSTOMER_ID)), period = p.TIMESTAMP, output = True)
            lp_vol_mv_agg_df = lp_vol_mv_agg_df.rename(columns = {"num1026_NDC_PRICE":"1026_NDC_PRICE", "num1026_GPI_PRICE": "1026_GPI_PRICE"})
        else:
            lp_vol_mv_agg_df = pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, 'lp_data_' + p.DATA_ID + '.csv'), dtype = p.VARIABLE_TYPE_DIC)
        """
        lp_vol_mv_agg_df = pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, 'lp_data_' + p.DATA_ID + '.csv'), dtype = p.VARIABLE_TYPE_DIC)
        lp_vol_mv_agg_df = standardize_df(lp_vol_mv_agg_df)
                
        remove_from_nan_check = ['BEG_Q_PRICE', 'BEG_M_PRICE'] if p.TRUECOST_CLIENT or p.UCL_CLIENT else []
        nanlist = list(lp_vol_mv_agg_df.columns.drop(['PHARMACY_RATE','VCML_PHARMACY_RATE','VENDOR_PRICE','PHARMACY_GER',
                                                     'INTERCEPT_REASON', 'MAC_DECREASE', 'MAC_INCREASE','VCML_LOW','VCML_HIGH',
                                                     'MIN_PRICE','MAX_PRICE', 'MIN_DISP_FEE_UNIT','MAX_DISP_FEE_UNIT',
                                                     'CS_PHARM_GRTE_TYPE', 'CS_PHARMACY_RATE', 'CS_TARGET_PFEE', 
                                                     'CS_PHARM_GRTE_TYPE2','CS_PHARMACY_RATE2', 'CS_TARGET_PFEE2', 
                                                     'CS_PHARM_GRTE_TYPE3', 'CS_PHARMACY_RATE3', 'CS_TARGET_PFEE3', 
                                                     'CS_PHARM_TARGET_IC_UNIT', 'CS_PHARM_TARGET_PRICE','CS_PHARM_TARGET_DISP_FEE_CLAIM',
                                                      # new columns
                                                      'NADAC','WAC','BNCHMK','PHRM_GRTE_TYPE','PHARM_TARG_INGCOST_ADJ','LM_PHARM_TARG_INGCOST_ADJ',
                                                     'PHARM_TARG_INGCOST_ADJ_ZBD','CS_LM_PHARM_TARG_INGCOST_ADJ', 'CS_LM_PHARM_TARG_INGCOST_ADJ_ZBD',
                                                     'PHARM_TARG_INGCOST_ADJ_PROJ_LAG', 'PHARM_TARG_INGCOST_ADJ_PROJ_EOY','AVG_QTY_CLM'] 
                                                     + remove_from_nan_check, errors = 'ignore'))
         
        qa_dataframe(lp_vol_mv_agg_df, nanlist = nanlist,dataset = 'lp_vol_mv_agg_df_AT_{}'.format(os.path.basename(__file__)))

        if p.UNC_OPT:
            # TODO: SUPPORT THIS TABLE WITH BQ READ
            lp_vol_mv_agg_df_actual = pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, 'lp_data_actual_' + p.DATA_ID + '.csv'), dtype = p.VARIABLE_TYPE_DIC)
            lp_vol_mv_agg_df_actual = standardize_df(lp_vol_mv_agg_df_actual)
            
            nanlist = list(lp_vol_mv_agg_df_actual.columns.drop(['PHARMACY_RATE','PHRM_GRTE_TYPE','PHARM_TARG_INGCOST_ADJ','LM_PHARM_TARG_INGCOST_ADJ',\
                                                    'PHARM_TARG_INGCOST_ADJ_ZBD','CS_LM_PHARM_TARG_INGCOST_ADJ','CS_LM_PHARM_TARG_INGCOST_ADJ_ZBD',
                                                          'PHARM_TARG_INGCOST_ADJ_PROJ_LAG', 'PHARM_TARG_INGCOST_ADJ_PROJ_EOY'], errors = 'ignore'))
            qa_dataframe(lp_vol_mv_agg_df_actual, nanlist = nanlist, dataset = 'lp_vol_mv_agg_df_actual_AT_{}'.format(os.path.basename(__file__)))
        else:
            lp_vol_mv_agg_df_actual = pd.DataFrame()

        if p.CLIENT_NAME_TABLEAU.startswith(('WTW','AON')):
            drug_mac_hist = pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.DRUG_MAC_HIST_FILE), dtype = p.VARIABLE_TYPE_DIC)
            drug_mac_hist = standardize_df(drug_mac_hist)
            lp_vol_mv_agg_df = pd.merge(lp_vol_mv_agg_df, drug_mac_hist[['GPI','BG_FLAG','MAC_LIST', 'BEG_Q_PRICE', 'BEG_M_PRICE']], how='left', on=['GPI', 'BG_FLAG', 'MAC_LIST'])

            #For WTW clients, we use quarterly beginning price as beginning period price 
            if p.CLIENT_NAME_TABLEAU.startswith('WTW'):
                lp_vol_mv_agg_df['BEG_PERIOD_PRICE'] = lp_vol_mv_agg_df['BEG_Q_PRICE'].fillna(lp_vol_mv_agg_df['CURRENT_MAC_PRICE'])

                if p.UNC_OPT:
                    lp_vol_mv_agg_df_actual = pd.merge(lp_vol_mv_agg_df_actual, drug_mac_hist[['GPI','BG_FLAG', 'MAC_LIST', 'BEG_Q_PRICE']], how='left', on=['GPI', 'BG_FLAG', 'MAC_LIST'])
                    lp_vol_mv_agg_df_actual['BEG_PERIOD_PRICE'] = lp_vol_mv_agg_df_actual['BEG_Q_PRICE'].fillna(lp_vol_mv_agg_df_actual['CURRENT_MAC_PRICE'])

            #For AON clients, we use monthly beginning price as beginning period price        
            if p.CLIENT_NAME_TABLEAU.startswith('AON'):
                lp_vol_mv_agg_df['BEG_PERIOD_PRICE'] = lp_vol_mv_agg_df['BEG_M_PRICE'].fillna(lp_vol_mv_agg_df['CURRENT_MAC_PRICE'])

                if p.UNC_OPT:
                    lp_vol_mv_agg_df_actual = pd.merge(lp_vol_mv_agg_df_actual, drug_mac_hist[['GPI','BG_FLAG', 'MAC_LIST', 'BEG_M_PRICE']], how='left', on=['GPI', 'BG_FLAG', 'MAC_LIST'])
                    lp_vol_mv_agg_df_actual['BEG_PERIOD_PRICE'] = lp_vol_mv_agg_df_actual['BEG_M_PRICE'].fillna(lp_vol_mv_agg_df_actual['CURRENT_MAC_PRICE'])


        mac_list_df = pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, 'mac_lists_' + p.DATA_ID + '.csv'), dtype = p.VARIABLE_TYPE_DIC)
        mac_list_df = standardize_df(mac_list_df)
        qa_dataframe(mac_list_df, dataset = 'mac_list_df_AT_{}'.format(os.path.basename(__file__)))
        chain_region_mac_mapping = pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, 'mac_mapping_' + p.DATA_ID + '.csv'), dtype = p.VARIABLE_TYPE_DIC)
        chain_region_mac_mapping = standardize_df(chain_region_mac_mapping)

        if 'CHAIN_SUBGROUP' not in chain_region_mac_mapping:
            logger.info("Using MAC Mapping file without new chain_subgroup column")
            chain_region_mac_mapping['CHAIN_SUBGROUP'] = chain_region_mac_mapping['CHAIN_GROUP']

        qa_dataframe(chain_region_mac_mapping, dataset = 'chain_region_mac_mapping_AT_{}'.format(os.path.basename(__file__)))

        if p.READ_IN_NEW_MACS:
            logger.info('--------------------')
            logger.info("Adding on new MACs")
            lp_vol_mv_agg_df = lp_vol_mv_agg_df.drop(columns=['CURRENT_MAC_PRICE', 'EFF_UNIT_PRICE',
                                                              'EFF_CAPPED_PRICE', 'MAC_PRICE_UNIT_ADJ'])
            if p.UNC_OPT:
                lp_vol_mv_agg_df_actual = lp_vol_mv_agg_df_actual.drop(columns=['CURRENT_MAC_PRICE', 'EFF_UNIT_PRICE', 'EFF_CAPPED_PRICE', 'MAC_PRICE_UNIT_ADJ'])

            if p.READ_FROM_BQ:
                mac_list_df = uf.read_BQ_data(BQ.mac_list, 
                                              project_id = p.BQ_INPUT_PROJECT_ID, 
                                              dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP, 
                                              table_id = 'mac_list', #+ p.WS_SUFFIX, --- We do not need the welcome season table for the mac_list table as we need the most updated drug costs & WS tables do not get updated often
                                              customer = ', '.join(sorted(p.CUSTOMER_ID)), 
                                              mac = True,
                                              vcml_ref_table_id='vcml_reference')
            else:
                mac_list_df = standardize_df(pd.read_csv(os.path.join(p.FILE_INPUT_PATH, p.NEW_MAC_FILE), dtype = p.VARIABLE_TYPE_DIC))
                
            if p.TRUECOST_CLIENT or p.UCL_CLIENT:
                assert p.READ_FROM_BQ == True, "Use p.READ_FROM_BQ=True to read table"
                mac_list_df = read_tru_mac_list_prices()
                mac_list_df = mac_list_df[mac_list_df['PRICE'] > 0]

            qa_dataframe(mac_list_df, dataset = 'NEW_MAC_FILE_AT_{}'.format(os.path.basename(__file__)))
            assert len(mac_list_df.loc[mac_list_df['GPI'].str.len() == 14, 'GPI']) == len(mac_list_df.GPI), "len(mac_list_df.loc[mac_list_df['GPI'].str.len() == 14, 'GPI']) == len(mac_list_df.GPI)"
            assert len(mac_list_df.drop_duplicates(subset=['GPI', 'MAC', 'NDC','BG_FLAG'])) == len(mac_list_df.GPI), "len(mac_list_df.drop_duplicates(subset=['GPI', 'MAC', 'NDC','BG_FLAG'])) == len(mac_list_df.GPI)"

            if 'MAC_LIST'  not in mac_list_df.columns:
                mac_list_df['MAC_LIST'] = mac_list_df.MAC.str[3:] # Remove 'MAC' from MAC list name
                mac_list_df['MAC_LIST'] = mac_list_df['MAC_LIST'].astype(np.int64)

            mac_list_df = mac_list_df.loc[mac_list_df.PRICE != 0]

            mac_list_gpi = mac_list_df.loc[mac_list_df.NDC == '***********'].copy(deep=True)
            mac_list_gpi.rename(columns={'PRICE': 'GPI_PRICE'}, inplace=True)
            mac_list_ndc = mac_list_df.loc[mac_list_df.NDC != '***********']
            mac_list_ndc.rename(columns={'PRICE': 'NDC_PRICE'}, inplace=True)
            assert (len(mac_list_gpi) + len(mac_list_ndc)) == len(mac_list_df), "(len(mac_list_gpi) + len(mac_list_ndc)) == len(mac_list_df)"
            
            selected_cols_ndc = ['MAC_LIST', 'MAC', 'NDC', 'BG_FLAG','NDC_PRICE','IS_MAC']
            selected_cols_gpi = ['MAC_LIST', 'MAC', 'GPI', 'BG_FLAG', 'GPI_PRICE','IS_MAC']
            
            lp_vol_macprice_df = pd.merge(lp_vol_mv_agg_df,
                                            mac_list_ndc[selected_cols_ndc],
                                            how ='left', on = ['NDC','BG_FLAG', 'MAC_LIST'])
            lp_vol_macprice_df = pd.merge(lp_vol_macprice_df,
                                            mac_list_gpi[selected_cols_gpi],
                                            how ='left', on = ['GPI','BG_FLAG', 'MAC_LIST'],
                                            suffixes=('_NDC', '_GPI'))
            
            lp_vol_macprice_df['MAC'] = np.where(lp_vol_macprice_df['MAC_NDC'].isna(), lp_vol_macprice_df['MAC_GPI'], lp_vol_macprice_df['MAC_NDC'])
            lp_vol_macprice_df.drop(columns=['MAC_NDC', 'MAC_GPI'], inplace=True)
            assert (lp_vol_mv_agg_df.FULLAWP_ADJ.sum() - lp_vol_macprice_df.FULLAWP_ADJ.sum()) < 0.0001, "(lp_vol_mv_agg_df.FULLAWP_ADJ.sum() - lp_vol_macprice_df.FULLAWP_ADJ.sum()) < 0.0001"

            lp_vol_macprice_df['CURRENT_MAC_PRICE'] = lp_vol_macprice_df['NDC_PRICE'].where(np.isfinite(lp_vol_macprice_df.NDC_PRICE), lp_vol_macprice_df.GPI_PRICE )
            lp_vol_macprice_df['EFF_UNIT_PRICE'] = determine_effective_price(lp_vol_macprice_df,
                                                                             old_price='CURRENT_MAC_PRICE')
            lp_vol_macprice_df['EFF_UNIT_PRICE'].where(lp_vol_macprice_df['EFF_UNIT_PRICE'] > 0, lp_vol_macprice_df['PRICE_REIMB_UNIT'], inplace = True)
            lp_vol_macprice_df['EFF_CAPPED_PRICE'] = determine_effective_price(lp_vol_macprice_df,
                                                                               old_price='CURRENT_MAC_PRICE',
                                                                               uc_unit='UC_UNIT25',
                                                                               capped_only=True)
            lp_vol_macprice_df['EFF_CAPPED_PRICE'].where(lp_vol_macprice_df['EFF_CAPPED_PRICE'] > 0, lp_vol_macprice_df['PRICE_REIMB_UNIT'], inplace = True)

            lp_vol_macprice_df['MAC_PRICE_UNIT_ADJ'] = lp_vol_macprice_df['CURRENT_MAC_PRICE'].where(lp_vol_macprice_df['CURRENT_MAC_PRICE']>0, lp_vol_macprice_df['PRICE_REIMB_UNIT'])

            lp_vol_mv_agg_df = lp_vol_macprice_df.fillna(0)
            if p.UNC_OPT:
                lp_vol_macprice_df_actual = pd.merge(lp_vol_mv_agg_df_actual, mac_list_ndc[selected_cols_ndc],
                                                how='left', on=['NDC', 'BG_FLAG', 'MAC_LIST'])
                lp_vol_macprice_df_actual = pd.merge(lp_vol_macprice_df_actual, mac_list_gpi[selected_cols_gpi],
                                                how='left', on=['GPI', 'BG_FLAG', 'MAC_LIST'])
                assert (lp_vol_mv_agg_df_actual.FULLAWP_ADJ.sum() - lp_vol_macprice_df_actual.FULLAWP_ADJ.sum()) < 0.0001, "(lp_vol_mv_agg_df_actual.FULLAWP_ADJ.sum() - lp_vol_macprice_df_actual.FULLAWP_ADJ.sum()) < 0.0001"

                lp_vol_macprice_df_actual['CURRENT_MAC_PRICE'] = lp_vol_macprice_df_actual.apply(
                    lambda df: df.NDC_PRICE if np.isfinite(df.NDC_PRICE) else df.GPI_PRICE, axis=1)
                lp_vol_macprice_df_actual['EFF_UNIT_PRICE'] = determine_effective_price(lp_vol_macprice_df_actual,
                                                                                        old_price='CURRENT_MAC_PRICE')
                lp_vol_macprice_df_actual['EFF_UNIT_PRICE'] = lp_vol_macprice_df_actual.apply(
                    lambda df: df['EFF_UNIT_PRICE'] if df['EFF_UNIT_PRICE'] > 0 else df['PRICE_REIMB_UNIT'], axis=1)
                lp_vol_macprice_df_actual['EFF_CAPPED_PRICE'] = determine_effective_price(lp_vol_macprice_df_actual,
                                                                                          old_price='CURRENT_MAC_PRICE',
                                                                                          uc_unit='UC_UNIT25',
                                                                                          capped_only=True)
                lp_vol_macprice_df_actual['EFF_CAPPED_PRICE'] = lp_vol_macprice_df_actual.apply(
                    lambda df: df['EFF_CAPPED_PRICE'] if df['EFF_CAPPED_PRICE'] > 0 else df['PRICE_REIMB_UNIT'], axis=1)

                lp_vol_macprice_df_actual['MAC_PRICE_UNIT_ADJ'] = lp_vol_macprice_df_actual.apply(
                    lambda df: df['CURRENT_MAC_PRICE'] if df['CURRENT_MAC_PRICE'] > 0 else df['PRICE_REIMB_UNIT'],
                    axis=1)
                lp_vol_mv_agg_df_actual = lp_vol_macprice_df_actual.fillna(0)
            else:
                lp_vol_mv_agg_df_actual = pd.DataFrame()
            logger.info('--------------------')

        # Reading cross contract or market check data in here in case NO_MAIL is true
        if p.CROSS_CONTRACT_PROJ:
            current_contract_data = standardize_df(pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.CURRENT_CONTRACT_DATA_FILE), dtype = p.VARIABLE_TYPE_DIC))
        
        if p.MARKET_CHECK:
            premc_data = standardize_df(pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.PRE_MC_DATA_FILE), dtype = p.VARIABLE_TYPE_DIC))
            postmc_data = standardize_df(pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.POST_MC_DATA_FILE), dtype = p.VARIABLE_TYPE_DIC))
            client_guarantees_premc = standardize_df(pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.CLIENT_GUARANTEE_PREMC_FILE), dtype = p.VARIABLE_TYPE_DIC))
            qa_dataframe(client_guarantees_premc, dataset = 'CLIENT_GUARANTEE_PREMC_FILE_AT_{}'.format(os.path.basename(__file__)))

        if p.NO_MAIL:
            lp_vol_mv_agg_df = lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df.MEASUREMENT != 'M30']
            if p.UNC_OPT:
                lp_vol_mv_agg_df_actual = lp_vol_mv_agg_df_actual.loc[lp_vol_mv_agg_df_actual.MEASUREMENT != 'M30']
            if p.CROSS_CONTRACT_PROJ:
                current_contract_data = current_contract_data.loc[current_contract_data.MEASUREMENT != 'M30']
            if p.MARKET_CHECK:
                premc_data = premc_data.loc[premc_data.MEASUREMENT != 'M30']
                postmc_data = postmc_data.loc[postmc_data.MEASUREMENT != 'M30']
                
        client_list = lp_vol_mv_agg_df[['CLIENT']].drop_duplicates().values[:,0]

        breakout_df = lp_vol_mv_agg_df[['CLIENT','BREAKOUT']].drop_duplicates()
        breakout_df['Combined'] = breakout_df['CLIENT'] + '_' + breakout_df['BREAKOUT']

        subchain_df = lp_vol_mv_agg_df.groupby(['CLIENT', 'CHAIN_GROUP', 'PHARMACY_PERF_NAME'], as_index=False)['FULLAWP_ADJ_PROJ_EOY'].sum()
        chain_df = lp_vol_mv_agg_df.groupby(['CLIENT', 'CHAIN_GROUP'], as_index=False)['FULLAWP_ADJ_PROJ_EOY'].sum()
        subchain_df = subchain_df.merge(chain_df, on=['CLIENT', 'CHAIN_GROUP'], suffixes=('_SUBGROUP', '_GROUP'), how='inner')
        subchain_df['MULTIPLIER'] = 0
        subchain_df.loc[subchain_df.FULLAWP_ADJ_PROJ_EOY_GROUP>0, 'MULTIPLIER'] = (subchain_df.loc[subchain_df.FULLAWP_ADJ_PROJ_EOY_GROUP>0, 'FULLAWP_ADJ_PROJ_EOY_SUBGROUP']
                                                                                   /subchain_df.loc[subchain_df.FULLAWP_ADJ_PROJ_EOY_GROUP>0, 'FULLAWP_ADJ_PROJ_EOY_GROUP'])
        subchain_df.drop(columns=['FULLAWP_ADJ_PROJ_EOY_SUBGROUP', 'FULLAWP_ADJ_PROJ_EOY_GROUP'], inplace=True)

        ## Read in guarantees
        client_guarantees = pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.CLIENT_GUARANTEE_FILE), dtype = p.VARIABLE_TYPE_DIC)
        client_guarantees = standardize_df(client_guarantees)
        
        if p.CLIENT_TYPE == 'MEDD' and p.FULL_YEAR and not p.EGWP:
            client_guarantees = standardize_df(pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.CLIENT_GUARANTEE_FILE), dtype = p.VARIABLE_TYPE_DIC))
            
        qa_dataframe(client_guarantees, dataset = 'CLIENT_GUARANTEE_FILE_AT_{}'.format(os.path.basename(__file__)))

        if not p.TRUECOST_CLIENT:
            if p.FULL_YEAR:
                client_guarantees['RATE'] += p.CLIENT_TARGET_BUFFER
            client_guarantees.loc[client_guarantees.BREAKOUT.str.upper().str.contains('M'),'RATE'] += p.MAIL_TARGET_BUFFER
            client_guarantees.loc[client_guarantees.BREAKOUT.str.upper().str.contains('R', regex=True),'RATE'] += p.RETAIL_TARGET_BUFFER
        else:
            # modifiying guarantees for nonAWP-based contracts by re-calculating Target Ingredient Cost
            agg_lp_vol_mv_agg_df = lp_vol_mv_agg_df.copy()\
                                                    .rename(columns={'CHAIN_GROUP':'PHARMACY'})\
                                                    .groupby(['CLIENT','REGION','BREAKOUT','MEASUREMENT','BG_FLAG','PHARMACY_TYPE'])\
                                                    .agg({'TARG_INGCOST_ADJ':'sum'
                                                        ,'TARG_INGCOST_ADJ_PROJ_LAG':'sum'
                                                        ,'TARG_INGCOST_ADJ_PROJ_EOY':'sum'
                                                        ,'FULLAWP_ADJ':'sum'
                                                        ,'FULLAWP_ADJ_PROJ_LAG':'sum'
                                                        ,'FULLAWP_ADJ_PROJ_EOY':'sum'})\
                                                    .reset_index()
            nonawp_client_guarantees = client_guarantees.copy()
            nonawp_client_guarantees = nonawp_client_guarantees.merge(agg_lp_vol_mv_agg_df,how='left', 
                                                                      on=['CLIENT','REGION','BREAKOUT','MEASUREMENT','BG_FLAG','PHARMACY_TYPE'])
            # Filling 0s in AWP and TIC for pharmacies with no claims
            nonawp_client_guarantees = nonawp_client_guarantees.fillna(0)
            nonawp_client_guarantees['AGG_AWP'] = (nonawp_client_guarantees['FULLAWP_ADJ']+
                                             nonawp_client_guarantees['FULLAWP_ADJ_PROJ_LAG']+
                                             nonawp_client_guarantees['FULLAWP_ADJ_PROJ_EOY'])
            nonawp_client_guarantees['AGG_TIC'] = (nonawp_client_guarantees['TARG_INGCOST_ADJ']+
                                                         nonawp_client_guarantees['TARG_INGCOST_ADJ_PROJ_LAG']+
                                                         nonawp_client_guarantees['TARG_INGCOST_ADJ_PROJ_EOY'])
            nonawp_client_guarantees['BENCHMARK_PRC'] = (nonawp_client_guarantees['AGG_TIC'])/(1 - nonawp_client_guarantees['RATE'])
            nonawp_client_guarantees['BUFFER_SURPLUS'] = np.nan
            if p.FULL_YEAR:
                nonawp_client_guarantees.loc[nonawp_client_guarantees.BREAKOUT.str.upper().str.contains('M'),'BUFFER_SURPLUS'] = (p.CLIENT_TARGET_BUFFER + p.MAIL_TARGET_BUFFER) * nonawp_client_guarantees['AGG_AWP']
                nonawp_client_guarantees.loc[nonawp_client_guarantees.BREAKOUT.str.upper().str.contains('R', regex=True),'BUFFER_SURPLUS'] = (p.CLIENT_TARGET_BUFFER + p.RETAIL_TARGET_BUFFER) * nonawp_client_guarantees['AGG_AWP']
            else:
                nonawp_client_guarantees.loc[nonawp_client_guarantees.BREAKOUT.str.upper().str.contains('M'),'BUFFER_SURPLUS'] = p.MAIL_TARGET_BUFFER * nonawp_client_guarantees['AGG_AWP']
                nonawp_client_guarantees.loc[nonawp_client_guarantees.BREAKOUT.str.upper().str.contains('R', regex=True),'BUFFER_SURPLUS'] = p.RETAIL_TARGET_BUFFER * nonawp_client_guarantees['AGG_AWP']
            
            nonawp_client_guarantees['NEW_TIC'] = nonawp_client_guarantees['AGG_TIC'] - nonawp_client_guarantees['BUFFER_SURPLUS']
            # initializing new rate column
            nonawp_client_guarantees['NEW_RATE'] = (1 - (nonawp_client_guarantees['NEW_TIC'] / nonawp_client_guarantees['BENCHMARK_PRC'])).round(4)
            nonawp_client_guarantees.loc[nonawp_client_guarantees['AGG_TIC'] == 0,'NEW_RATE'] = nonawp_client_guarantees['RATE']
            client_guarantees_temp = nonawp_client_guarantees[list(client_guarantees.columns)+['NEW_RATE']].copy()
            assert len(client_guarantees_temp[client_guarantees_temp.NEW_RATE > 1.0])==0,"GERs cannot be greater than 1.00. Modify CLIENT_TARGET_BUFFER, MAIL_TARGET_BUFFER, or RETAIL_TARGET_BUFFER to a lesser value"
            client_guarantees = client_guarantees.merge(client_guarantees_temp,how='left',on=list(client_guarantees.columns))

            client_guarantees['RATE'] = client_guarantees['NEW_RATE']
            del client_guarantees['NEW_RATE']

        pharmacy_guarantees = pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH,p.PHARM_GUARANTEE_FILE), dtype = p.VARIABLE_TYPE_DIC)
        pharmacy_guarantees = standardize_df(pharmacy_guarantees)
        qa_dataframe(pharmacy_guarantees, dataset = 'PHARM_GUARANTEE_FILE_AT_{}'.format(os.path.basename(__file__)))
        if p.FULL_YEAR:
            # modifiying guarantees for AWP-based contracts directly using the buffers
            awp_pharmacy_guarantees = pharmacy_guarantees.loc[pharmacy_guarantees['PHRM_GRTE_TYPE']=='AWP'].copy()
            awp_pharmacy_guarantees['PHARM_NEW_RATE'] = awp_pharmacy_guarantees['RATE']
            # to-do: use different buffers for generics and brands once we separate them out
            awp_pharmacy_guarantees.loc[(awp_pharmacy_guarantees['PHARMACY'].isin(p.BIG_CAPPED_PHARMACY_LIST['GNRC'])) & (awp_pharmacy_guarantees['BG_FLAG']=='G'), 
                                        'PHARM_NEW_RATE'] += p.PHARMACY_TARGET_BUFFER
            awp_pharmacy_guarantees.loc[(awp_pharmacy_guarantees['PHARMACY'].isin(p.BIG_CAPPED_PHARMACY_LIST['BRND'])) & (awp_pharmacy_guarantees['BG_FLAG']=='B'), 
                                        'PHARM_NEW_RATE'] += p.PHARMACY_TARGET_BUFFER
            awp_pharmacy_guarantees.loc[(~awp_pharmacy_guarantees['PHARMACY'].isin(p.BIG_CAPPED_PHARMACY_LIST['GNRC'])) & (awp_pharmacy_guarantees['BG_FLAG']=='G'), 
                                        'PHARM_NEW_RATE'] += p.CLIENT_TARGET_BUFFER
            awp_pharmacy_guarantees.loc[(~awp_pharmacy_guarantees['PHARMACY'].isin(p.BIG_CAPPED_PHARMACY_LIST['BRND'])) & (awp_pharmacy_guarantees['BG_FLAG']=='B'), 
                                        'PHARM_NEW_RATE'] += p.CLIENT_TARGET_BUFFER

            # modifiying guarantees for nonAWP-based contracts by re-calculating Target Ingredient Cost
            agg_lp_vol_mv_agg_df = lp_vol_mv_agg_df.copy()\
                                                    .rename(columns={'CHAIN_GROUP':'PHARMACY'})\
                                                    .groupby(['CLIENT','REGION','BREAKOUT','MEASUREMENT','BG_FLAG','PHARMACY'])\
                                                    .agg({'PHARM_TARG_INGCOST_ADJ':'sum'
                                                        ,'PHARM_TARG_INGCOST_ADJ_PROJ_LAG':'sum'
                                                        ,'PHARM_TARG_INGCOST_ADJ_PROJ_EOY':'sum'
                                                        ,'PHARM_FULLAWP_ADJ':'sum'
                                                        ,'PHARM_FULLAWP_ADJ_PROJ_LAG':'sum'
                                                        ,'PHARM_FULLAWP_ADJ_PROJ_EOY':'sum'})\
                                                    .reset_index()
            nonawp_pharmacy_guarantees = pharmacy_guarantees.loc[pharmacy_guarantees['PHRM_GRTE_TYPE']!='AWP'].copy()
            nonawp_pharmacy_guarantees = nonawp_pharmacy_guarantees.merge(agg_lp_vol_mv_agg_df,how='left'
                                                                                    ,on=['CLIENT','REGION','BREAKOUT','MEASUREMENT','BG_FLAG','PHARMACY'])
            # Filling 0s in AWP and TIC for pharmacies with no claims
            nonawp_pharmacy_guarantees = nonawp_pharmacy_guarantees.fillna(0)
            nonawp_pharmacy_guarantees['PHARM_AGG_AWP'] = (nonawp_pharmacy_guarantees['PHARM_FULLAWP_ADJ']+
                                             nonawp_pharmacy_guarantees['PHARM_FULLAWP_ADJ_PROJ_LAG']+
                                             nonawp_pharmacy_guarantees['PHARM_FULLAWP_ADJ_PROJ_EOY'])
            nonawp_pharmacy_guarantees['PHARM_AGG_TIC'] = (nonawp_pharmacy_guarantees['PHARM_TARG_INGCOST_ADJ']+
                                                         nonawp_pharmacy_guarantees['PHARM_TARG_INGCOST_ADJ_PROJ_LAG']+
                                                         nonawp_pharmacy_guarantees['PHARM_TARG_INGCOST_ADJ_PROJ_EOY'])
            nonawp_pharmacy_guarantees['PHARM_BENCHMARK_PRC'] = (nonawp_pharmacy_guarantees['PHARM_AGG_TIC'])/(1 - nonawp_pharmacy_guarantees['RATE'])
            nonawp_pharmacy_guarantees['PHARM_BUFFER_SURPLUS'] = np.nan
            nonawp_pharmacy_guarantees.loc[(nonawp_pharmacy_guarantees['PHARMACY'].isin(p.BIG_CAPPED_PHARMACY_LIST['GNRC'])) & (nonawp_pharmacy_guarantees['BG_FLAG']=='G'), 
                                           'PHARM_BUFFER_SURPLUS'] = p.PHARMACY_TARGET_BUFFER*nonawp_pharmacy_guarantees['PHARM_AGG_AWP']
            nonawp_pharmacy_guarantees.loc[(nonawp_pharmacy_guarantees['PHARMACY'].isin(p.BIG_CAPPED_PHARMACY_LIST['BRND'])) & (nonawp_pharmacy_guarantees['BG_FLAG']=='B'), 
                                           'PHARM_BUFFER_SURPLUS'] = p.PHARMACY_TARGET_BUFFER*nonawp_pharmacy_guarantees['PHARM_AGG_AWP']
            nonawp_pharmacy_guarantees.loc[~nonawp_pharmacy_guarantees['PHARMACY'].isin(p.BIG_CAPPED_PHARMACY_LIST['GNRC']) & (nonawp_pharmacy_guarantees['BG_FLAG']=='G'), 
                                           'PHARM_BUFFER_SURPLUS'] = p.CLIENT_TARGET_BUFFER*nonawp_pharmacy_guarantees['PHARM_AGG_AWP']
            nonawp_pharmacy_guarantees.loc[~nonawp_pharmacy_guarantees['PHARMACY'].isin(p.BIG_CAPPED_PHARMACY_LIST['BRND']) & (nonawp_pharmacy_guarantees['BG_FLAG']=='B'), 
                                           'PHARM_BUFFER_SURPLUS'] = p.CLIENT_TARGET_BUFFER*nonawp_pharmacy_guarantees['PHARM_AGG_AWP']
            nonawp_pharmacy_guarantees['PHARM_NEW_TIC'] = nonawp_pharmacy_guarantees['PHARM_AGG_TIC'] - nonawp_pharmacy_guarantees['PHARM_BUFFER_SURPLUS']
            # initializing new rate column
            nonawp_pharmacy_guarantees['PHARM_NEW_RATE'] = (1 - (nonawp_pharmacy_guarantees['PHARM_NEW_TIC'] / nonawp_pharmacy_guarantees['PHARM_BENCHMARK_PRC'])).round(4)
            nonawp_pharmacy_guarantees.loc[nonawp_pharmacy_guarantees['PHARM_AGG_TIC'] == 0,'PHARM_NEW_RATE'] = nonawp_pharmacy_guarantees['RATE']
            pharmacy_guarantees_temp = pd.concat([awp_pharmacy_guarantees,nonawp_pharmacy_guarantees[list(pharmacy_guarantees.columns)+['PHARM_NEW_RATE']]],ignore_index = True)
            assert set(pharmacy_guarantees.loc[pharmacy_guarantees.RATE.isna(),'PHARMACY']) == set(pharmacy_guarantees_temp.loc[pharmacy_guarantees_temp.PHARM_NEW_RATE.isna(),'PHARMACY']),"Pharmacy without guarantee got modified. Check code."
            assert len(pharmacy_guarantees_temp[pharmacy_guarantees_temp.PHARM_NEW_RATE > 1.0])==0,"GERs cannot be greater than 1.00. Modify PHARMACY_TARGET_BUFFER or CLIENT_TARGET_BUFFER to a lesser value"
            pharmacy_guarantees = pharmacy_guarantees.merge(pharmacy_guarantees_temp,how='left',on=list(pharmacy_guarantees.columns))

            pharmacy_guarantees['RATE'] = pharmacy_guarantees['PHARM_NEW_RATE']
            del pharmacy_guarantees['PHARM_NEW_RATE']
            lp_vol_mv_agg_df_temp = pd.merge(lp_vol_mv_agg_df
                            , pharmacy_guarantees[['CLIENT','REGION','BREAKOUT','MEASUREMENT','BG_FLAG','PHARMACY','PHARMACY_SUB','RATE']]
                            , how ='left', 
                            left_on = ['CLIENT', 'BREAKOUT','REGION','MEASUREMENT','BG_FLAG','CHAIN_GROUP', 'CHAIN_SUBGROUP'],
                            right_on = ['CLIENT', 'BREAKOUT','REGION','MEASUREMENT','BG_FLAG','PHARMACY', 'PHARMACY_SUB'])
            assert len(lp_vol_mv_agg_df) == len(lp_vol_mv_agg_df_temp), "len(lp_vol_mv_agg_df) == len(lp_vol_mv_agg_df_temp)"
            lp_vol_mv_agg_df_temp['PHARMACY_RATE'] = lp_vol_mv_agg_df_temp['RATE'].fillna(0)
            lp_vol_mv_agg_df_temp = lp_vol_mv_agg_df_temp.drop(columns=['PHARMACY', 'PHARMACY_SUB', 'RATE'])
            assert all(lp_vol_mv_agg_df_temp.columns==lp_vol_mv_agg_df.columns), "lp_vol_mv_agg_df_temp.columns==lp_vol_mv_agg_df.columns"
            lp_vol_mv_agg_df_temp = add_target_ingcost(lp_vol_mv_agg_df_temp, client_guarantees, client_rate_col = 'RATE')
            lp_vol_mv_agg_df = lp_vol_mv_agg_df_temp.copy()
            
            if p.UNC_OPT:
                # updating UNC df with the same rates as calculated above
                lp_vol_mv_agg_df_temp = pd.merge(lp_vol_mv_agg_df_actual
                                        , pharmacy_guarantees[['CLIENT','REGION','BREAKOUT','MEASUREMENT','BG_FLAG','PHARMACY','PHARMACY_SUB','RATE']]
                                        , how ='left', 
                                        left_on = ['CLIENT', 'BREAKOUT','REGION','MEASUREMENT','BG_FLAG','CHAIN_GROUP', 'CHAIN_SUBGROUP'],
                                        right_on = ['CLIENT', 'BREAKOUT','REGION','MEASUREMENT','BG_FLAG','PHARMACY', 'PHARMACY_SUB'])
                assert len(lp_vol_mv_agg_df_actual) == len(lp_vol_mv_agg_df_temp), "len(lp_vol_mv_agg_df_actual) == len(lp_vol_mv_agg_df_temp)"
                lp_vol_mv_agg_df_temp['PHARMACY_RATE'] = lp_vol_mv_agg_df_temp['RATE'].fillna(0)
                lp_vol_mv_agg_df_temp = lp_vol_mv_agg_df_temp.drop(columns=['PHARMACY', 'PHARMACY_SUB', 'RATE'])
                assert all(lp_vol_mv_agg_df_temp.columns==lp_vol_mv_agg_df_actual.columns), "lp_vol_mv_agg_df_temp.columns==lp_vol_mv_agg_df_actual.columns"
                lp_vol_mv_agg_df_temp = add_target_ingcost(lp_vol_mv_agg_df_temp, client_guarantees, client_rate_col = 'RATE')
                lp_vol_mv_agg_df_actual = lp_vol_mv_agg_df_temp.copy()
        else:
            lp_vol_mv_agg_df = add_target_ingcost(lp_vol_mv_agg_df, client_guarantees, client_rate_col = 'RATE')
            if p.UNC_OPT:
                lp_vol_mv_agg_df_actual = add_target_ingcost(lp_vol_mv_agg_df_actual, client_guarantees, client_rate_col = 'RATE')
                
        client_guarantees.to_csv(p.FILE_DYNAMIC_INPUT_PATH + 'Buffer_' + p.CLIENT_GUARANTEE_FILE, index = False)
        pharmacy_guarantees.to_csv(p.FILE_DYNAMIC_INPUT_PATH + 'Buffer_' + p.PHARM_GUARANTEE_FILE, index=False)
            
        pref_pharm_list = pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.PREFERRED_PHARM_FILE), dtype = p.VARIABLE_TYPE_DIC)
        pref_pharm_list = standardize_df(pref_pharm_list)
        qa_dataframe(pref_pharm_list, dataset = 'PREFERRED_PHARM_FILE_AT_{}'.format(os.path.basename(__file__)))
        pref_pharm_list['PREF_PHARMS'] = pref_pharm_list.PREF_PHARM.apply(lambda x: x.split(','))

        oc_pharm_surplus = standardize_df(pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.OC_PHARM_PERF_FILE), dtype = p.VARIABLE_TYPE_DIC))
        qa_dataframe(oc_pharm_surplus, dataset = 'OC_PHARM_PERF_FILE_AT_{}'.format(os.path.basename(__file__)))
        oc_pharm_surplus['LAG_SURPLUS'] = oc_pharm_surplus.SURPLUS * (lag_days/proj_days)
        oc_pharm_surplus['EOY_SURPLUS'] = oc_pharm_surplus.SURPLUS * (eoy_days/proj_days)
        assert ((oc_pharm_surplus['LAG_SURPLUS'].sum() +
                 oc_pharm_surplus['EOY_SURPLUS'].sum()) - oc_pharm_surplus.SURPLUS.sum()) < .0001, "LAG_SURPLUS + EOY_SUPLUS - SURPLUS.sum() < 0.0001"
        other_client_pharm_lageoy = df_to_dict(oc_pharm_surplus, ['CHAIN_GROUP', 'SURPLUS'])
        other_client_pharm_lag = df_to_dict(oc_pharm_surplus, ['CHAIN_GROUP', 'LAG_SURPLUS'])
        oc_eoy_pharm_perf = df_to_dict(oc_pharm_surplus, ['CHAIN_GROUP', 'EOY_SURPLUS'])

        # Read in generic launches CAN WE DELETE?
        if False:# p.WRITE_TO_BQ and (unc_flag == True or p.UNC_ADJUST == False): #forthcomming implementation if we decide to go that way.
            generic_launch_df = uf.read_BQ_data(BQ.Gen_Launch,
                                                project_id = p.BQ_OUTPUT_PROJECT_ID,
                                                dataset_id = p.BQ_OUTPUT_DATASET,
                                                table_id = 'Gen_Launch' + p.WS_SUFFIX,
                                                run_id = p.AT_RUN_ID,
                                                client = ', '.join(sorted(p.CUSTOMER_ID)),
                                                period = p.TIMESTAMP,
                                                output = True)
            generic_launch_df = standardize_df(generic_launch_df)
        else:

            generic_launch_df = standardize_df(pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.GENERIC_LAUNCH_FILE), dtype = p.VARIABLE_TYPE_DIC))
        qa_dataframe(generic_launch_df, dataset = 'GENERIC_LAUNCH_FILE_AT_{}'.format(os.path.basename(__file__)))
        gen_launch_ytd, gen_launch_lag, gen_launch_eoy = gen_launch_df_generator_ytd_lag_eoy(generic_launch_df, pref_pharm_list)

        oc_pharm_dummy = dict()
        for pharm in list(set(p.AGREEMENT_PHARMACY_LIST['GNRC']+p.AGREEMENT_PHARMACY_LIST['BRND'])):
            oc_pharm_dummy[pharm] = 0

        # Creating a zeroed out dummy dictionary. Used as a dummy dict for generic launch, brand surplus, specialty surplus, and dispensing fee perf dictionaries
        dummy_perf_dict = {}
        for perf_name in subchain_df['PHARMACY_PERF_NAME'].unique():
            dummy_perf_dict[perf_name] = 0
        for breakout in breakout_df['Combined'].tolist():
            dummy_perf_dict[breakout] = 0

        perf_dict_col = ['ENTITY', 'PERFORMANCE']

        gen_launch_ytd_dict = calculatePerformance(gen_launch_ytd, client_guarantees, pharmacy_guarantees, client_list,
                                                    p.AGREEMENT_PHARMACY_LIST, oc_pharm_dummy, dummy_perf_dict, dummy_perf_dict, dummy_perf_dict, dummy_perf_dict,
                                                    client_reimb_column='ING_COST', pharm_reimb_column='ING_COST',
                                                    client_TARG_column='FULLAWP', pharm_TARG_column='FULLAWP', other=False, subchain_df=subchain_df)

        if p.FULL_YEAR:
            gen_launch_lag_dict = copy.deepcopy(dummy_perf_dict)
        else:
            gen_launch_lag_dict = calculatePerformance(gen_launch_lag, client_guarantees, pharmacy_guarantees, client_list,
                                                        p.AGREEMENT_PHARMACY_LIST, oc_pharm_dummy, dummy_perf_dict, dummy_perf_dict, dummy_perf_dict, dummy_perf_dict,
                                                        client_reimb_column='ING_COST', pharm_reimb_column='ING_COST',
                                                        client_TARG_column='FULLAWP', pharm_TARG_column='FULLAWP', other=False, subchain_df=subchain_df)

        if p.FULL_YEAR:
            gen_launch_eoy_dict = copy.deepcopy(dummy_perf_dict)
        else:
            gen_launch_eoy_dict = calculatePerformance(gen_launch_eoy, client_guarantees, pharmacy_guarantees, client_list,
                                                        p.AGREEMENT_PHARMACY_LIST, oc_pharm_dummy, dummy_perf_dict, dummy_perf_dict, dummy_perf_dict, dummy_perf_dict,
                                                        client_reimb_column='ING_COST', pharm_reimb_column='ING_COST',
                                                        client_TARG_column='FULLAWP', pharm_TARG_column='FULLAWP', other=False, subchain_df=subchain_df)

        # Read in Brand/Generic offset and create dictionary
        brand_generic_df = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.BRAND_SURPLUS_FILE)
        brand_generic_df = standardize_df(brand_generic_df)
        # create the performance dictionary
        # this uses an existing dictionary as a template
        brand_surplus_ytd_dict, brand_surplus_lag_dict, brand_surplus_eoy_dict = sf.brand_surplus_dict_generator_ytd_lag_eoy(brand_generic_df,
                                                                        p.LAST_DATA,
                                                                        gen_launch_eoy_dict, perf_dict_col)
        # End of Brand/Generic offset dictionary
        
        # Read in specialty offset data and create dictionaries
        specialty_offset_df = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.SPECIALTY_SURPLUS_FILE)
        specialty_offset_df = standardize_df(specialty_offset_df)
        specialty_surplus_ytd_dict, specialty_surplus_lag_dict, specialty_surplus_eoy_dict = sf.specialty_surplus_dict_generator_ytd_lag_eoy(specialty_offset_df, 
                                                                                                                                             contract_date_df,
                                                                                                                                             gen_launch_eoy_dict, 
                                                                                                                                             perf_dict_col)
        
        # Create dispensing fee offset dictionary
        disp_fee_surplus_ytd_dict, disp_fee_surplus_lag_dict, disp_fee_surplus_eoy_dict = sf.disp_fee_surplus_dict_generator_ytd_lag_eoy(lp_vol_mv_agg_df,
                                                                        gen_launch_eoy_dict, perf_dict_col)
        
        if isinstance(disp_fee_surplus_ytd_dict,pd.DataFrame):
            disp_offset = disp_fee_surplus_ytd_dict.rename(columns={'PERFORMANCE':'SURPLUS_YTD'})
            disp_offset = disp_offset.merge(disp_fee_surplus_lag_dict.rename(columns={'PERFORMANCE':'SURPLUS_LAG'}), on='ENTITY')
            disp_offset = disp_offset.merge(disp_fee_surplus_eoy_dict.rename(columns={'PERFORMANCE':'SURPLUS_EOY'}), on='ENTITY')
        else:
            disp_offset = pd.DataFrame(disp_fee_surplus_ytd_dict, index=['SURPLUS_YTD']).T
            disp_offset = disp_offset.merge(pd.DataFrame(disp_fee_surplus_lag_dict, index=['SURPLUS_LAG']).T, left_index=True, right_index=True)
            disp_offset = disp_offset.merge(pd.DataFrame(disp_fee_surplus_eoy_dict, index=['SURPLUS_EOY']).T, left_index=True, right_index=True)
            disp_offset.reset_index(inplace=True)
            disp_offset.rename(columns={'index':'ENTITY'}, inplace=True)
        disp_offset.to_csv(os.path.join(p.FILE_OUTPUT_PATH, 'Dispensing_fee_surplus_{}_{}.csv'.format(str(p.GO_LIVE.month), p.DATA_ID)), index=False)

        #Back out generic launches
        if p.READ_FROM_BQ:
                gpi_backout = uf.read_BQ_data(
                    BQ.gen_launch_backout,
                    project_id=p.BQ_INPUT_PROJECT_ID,
                    dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                    table_id="gen_launch_backout" + p.WS_SUFFIX,
                    #client = ', '.join(sorted(p.CUSTOMER_ID)
                )
        else:
            gpi_backout = standardize_df(pd.read_csv(os.path.join(p.FILE_INPUT_PATH, p.BACKOUT_GEN), dtype = p.VARIABLE_TYPE_DIC))
        gpi_backout = standardize_df(gpi_backout)
        qa_dataframe(gpi_backout, dataset = 'BACKOUT_GEN_AT_{}'.format(os.path.basename(__file__)))
        gpi_backout = pd.DataFrame(columns = ['GPI'])
        gpi_backout['BG_FLAG'] = 'G'
      
        lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df.set_index(['GPI', 'BG_FLAG']).index.isin(gpi_backout.set_index(['GPI', 'BG_FLAG']).index), 'QTY_PROJ_EOY'] = 0
        lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df.set_index(['GPI', 'BG_FLAG']).index.isin(gpi_backout.set_index(['GPI', 'BG_FLAG']).index), 'CLAIMS_PROJ_EOY'] = 0
        lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df.set_index(['GPI', 'BG_FLAG']).index.isin(gpi_backout.set_index(['GPI', 'BG_FLAG']).index), 'FULLAWP_ADJ_PROJ_EOY'] = 0
        lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df.set_index(['GPI', 'BG_FLAG']).index.isin(gpi_backout.set_index(['GPI', 'BG_FLAG']).index), 'TARG_INGCOST_ADJ_PROJ_EOY'] = 0
        lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df.set_index(['GPI', 'BG_FLAG']).index.isin(gpi_backout.set_index(['GPI', 'BG_FLAG']).index), 'QTY_PROJ_LAG'] = 0
        lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df.set_index(['GPI', 'BG_FLAG']).index.isin(gpi_backout.set_index(['GPI', 'BG_FLAG']).index), 'CLAIMS_PROJ_LAG'] = 0
        lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df.set_index(['GPI', 'BG_FLAG']).index.isin(gpi_backout.set_index(['GPI', 'BG_FLAG']).index), 'FULLAWP_ADJ_PROJ_LAG'] = 0
        lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df.set_index(['GPI', 'BG_FLAG']).index.isin(gpi_backout.set_index(['GPI', 'BG_FLAG']).index), 'TARG_INGCOST_ADJ_PROJ_LAG'] = 0

        if p.UNC_OPT:
            lp_vol_mv_agg_df_actual.loc[lp_vol_mv_agg_df_actual.set_index(['GPI', 'BG_FLAG']).index.isin(gpi_backout.set_index(['GPI', 'BG_FLAG']).index), 'QTY_PROJ_EOY'] = 0
            lp_vol_mv_agg_df_actual.loc[lp_vol_mv_agg_df_actual.set_index(['GPI', 'BG_FLAG']).index.isin(gpi_backout.set_index(['GPI', 'BG_FLAG']).index), 'CLAIMS_PROJ_EOY'] = 0
            lp_vol_mv_agg_df_actual.loc[lp_vol_mv_agg_df_actual.set_index(['GPI', 'BG_FLAG']).index.isin(gpi_backout.set_index(['GPI', 'BG_FLAG']).index), 'FULLAWP_ADJ_PROJ_EOY'] = 0
            lp_vol_mv_agg_df_actual.loc[lp_vol_mv_agg_df_actual.set_index(['GPI', 'BG_FLAG']).index.isin(gpi_backout.set_index(['GPI', 'BG_FLAG']).index), 'TARG_INGCOST_ADJ_PROJ_EOY'] = 0
            lp_vol_mv_agg_df_actual.loc[lp_vol_mv_agg_df_actual.set_index(['GPI', 'BG_FLAG']).index.isin(gpi_backout.set_index(['GPI', 'BG_FLAG']).index), 'QTY_PROJ_LAG'] = 0
            lp_vol_mv_agg_df_actual.loc[lp_vol_mv_agg_df_actual.set_index(['GPI', 'BG_FLAG']).index.isin(gpi_backout.set_index(['GPI', 'BG_FLAG']).index), 'CLAIMS_PROJ_LAG'] = 0
            lp_vol_mv_agg_df_actual.loc[lp_vol_mv_agg_df_actual.set_index(['GPI', 'BG_FLAG']).index.isin(gpi_backout.set_index(['GPI', 'BG_FLAG']).index), 'FULLAWP_ADJ_PROJ_LAG'] = 0
            lp_vol_mv_agg_df_actual.loc[lp_vol_mv_agg_df_actual.set_index(['GPI', 'BG_FLAG']).index.isin(gpi_backout.set_index(['GPI', 'BG_FLAG']).index), 'TARG_INGCOST_ADJ_PROJ_LAG'] = 0            
            

        for key in dummy_perf_dict:
            if key not in gen_launch_ytd_dict:
                gen_launch_ytd_dict[key] = 0
            if key not in gen_launch_lag_dict:
                gen_launch_lag_dict[key] = 0
            if key not in gen_launch_eoy_dict:
                gen_launch_eoy_dict[key] = 0

        gen_launch_lageoy_dict = dict()
        for key in gen_launch_eoy_dict:
            gen_launch_lageoy_dict[key] = gen_launch_eoy_dict[key] + gen_launch_lag_dict[key]

        if p.NDC_UPDATE and (month == p.LP_RUN[0]):
            lag_price_col = 'EFF_CAPPED_PRICE_OLD'
            pharm_lag_price_col = 'PHARM_EFF_CAPPED_PRICE_OLD'
        else:
            lag_price_col = 'EFF_CAPPED_PRICE'
            pharm_lag_price_col = 'PHARM_EFF_CAPPED_PRICE'
            lp_vol_mv_agg_df['OLD_MAC_PRICE'] = lp_vol_mv_agg_df['CURRENT_MAC_PRICE']

            if p.UNC_OPT:
                lp_vol_mv_agg_df_actual['OLD_MAC_PRICE'] = lp_vol_mv_agg_df_actual['CURRENT_MAC_PRICE']
                # for U&C price raises set OLD_MAC_PRICE back to original price
                lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df['RAISED_PRICE_UC'], 'OLD_MAC_PRICE'] = lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df['RAISED_PRICE_UC'], 'PRE_UC_MAC_PRICE']
        
        if p.INTERCEPTOR_OPT or p.COSTSAVER_CLIENT:
            #Simplified the equations for coding purpose. This equation is applicable pre and post costsaver implementation.
            #Actual equation reads as LAG_REIMB = MAC*QTY*(1-ZBD_FRAC) + MAC*QTY*ZBD_FRAC*KEEP_SEND + GRX*QTY*ZBD_FRAC*(1-KEEP_SEND)
            #For Pharmacy the lag quantity and awp is already modified and set equal to the qty/awp adjudicated at MAC price.
            
            lp_vol_mv_agg_df['LAG_REIMB'] = lp_vol_mv_agg_df.QTY_PROJ_LAG * (lp_vol_mv_agg_df[lag_price_col] - lp_vol_mv_agg_df.QTY_ZBD_FRAC * \
                                                                             lp_vol_mv_agg_df[[lag_price_col, 'VENDOR_PRICE']].min(axis=1) * (1 - lp_vol_mv_agg_df.CURRENT_KEEP_SEND)) 
             
        else:
            lp_vol_mv_agg_df['LAG_REIMB'] = lp_vol_mv_agg_df.QTY_PROJ_LAG * lp_vol_mv_agg_df[lag_price_col]
            
        lp_vol_mv_agg_df['PHARM_LAG_REIMB'] = lp_vol_mv_agg_df.PHARM_QTY_PROJ_LAG * lp_vol_mv_agg_df[pharm_lag_price_col]
        lp_vol_mv_agg_df['PRICE_REIMB_LAG'] = lp_vol_mv_agg_df.LAG_REIMB + lp_vol_mv_agg_df.PRICE_REIMB
        lp_vol_mv_agg_df['FULLAWP_ADJ_YTDLAG']= lp_vol_mv_agg_df.FULLAWP_ADJ_PROJ_LAG + lp_vol_mv_agg_df.FULLAWP_ADJ
        lp_vol_mv_agg_df['TARG_INGCOST_ADJ_YTDLAG']= lp_vol_mv_agg_df.TARG_INGCOST_ADJ_PROJ_LAG + lp_vol_mv_agg_df.TARG_INGCOST_ADJ
        lp_vol_mv_agg_df['PHARM_PRICE_REIMB_LAG'] = lp_vol_mv_agg_df.PHARM_LAG_REIMB + lp_vol_mv_agg_df.PHARM_PRICE_REIMB
        lp_vol_mv_agg_df['PHARM_FULLAWP_ADJ_YTDLAG']= lp_vol_mv_agg_df.PHARM_FULLAWP_ADJ_PROJ_LAG + lp_vol_mv_agg_df.PHARM_FULLAWP_ADJ
        lp_vol_mv_agg_df['PHARM_TARG_INGCOST_ADJ_YTDLAG']= lp_vol_mv_agg_df.PHARM_TARG_INGCOST_ADJ_PROJ_LAG + lp_vol_mv_agg_df.PHARM_TARG_INGCOST_ADJ

        ytd_df = lp_vol_mv_agg_df
        

        if p.UNC_OPT:
            lp_vol_mv_agg_df_actual['LAG_REIMB'] = lp_vol_mv_agg_df_actual.QTY_PROJ_LAG * lp_vol_mv_agg_df_actual[lag_price_col]
            lp_vol_mv_agg_df_actual['PRICE_REIMB_LAG'] = lp_vol_mv_agg_df_actual.LAG_REIMB + lp_vol_mv_agg_df_actual.PRICE_REIMB
            lp_vol_mv_agg_df_actual['FULLAWP_ADJ_YTDLAG'] = lp_vol_mv_agg_df_actual.FULLAWP_ADJ_PROJ_LAG + lp_vol_mv_agg_df_actual.FULLAWP_ADJ
            lp_vol_mv_agg_df_actual['TARG_INGCOST_ADJ_YTDLAG'] = lp_vol_mv_agg_df_actual.TARG_INGCOST_ADJ_PROJ_LAG + lp_vol_mv_agg_df_actual.TARG_INGCOST_ADJ
            lp_vol_mv_agg_df_actual['PHARM_LAG_REIMB'] = lp_vol_mv_agg_df_actual.PHARM_QTY_PROJ_LAG * lp_vol_mv_agg_df_actual[pharm_lag_price_col]
            lp_vol_mv_agg_df_actual['PHARM_PRICE_REIMB_LAG'] = lp_vol_mv_agg_df_actual.PHARM_LAG_REIMB + lp_vol_mv_agg_df_actual.PHARM_PRICE_REIMB
            lp_vol_mv_agg_df_actual['PHARM_FULLAWP_ADJ_YTDLAG'] = lp_vol_mv_agg_df_actual.PHARM_FULLAWP_ADJ_PROJ_LAG +\
            lp_vol_mv_agg_df_actual.PHARM_FULLAWP_ADJ
            lp_vol_mv_agg_df_actual['PHARM_TARG_INGCOST_ADJ_YTDLAG'] = lp_vol_mv_agg_df_actual.PHARM_TARG_INGCOST_ADJ_PROJ_LAG +\
            lp_vol_mv_agg_df_actual.PHARM_TARG_INGCOST_ADJ

        ytd_df = lp_vol_mv_agg_df
        if p.UNC_OPT:
            ytd_df = lp_vol_mv_agg_df_actual

        if p.MARKET_CHECK == False:
            ytd_perf_pharm_actuals_dict = calculatePerformance(ytd_df, client_guarantees, pharmacy_guarantees,
                                                               client_list, p.AGREEMENT_PHARMACY_LIST, oc_pharm_dummy,
                                                               gen_launch_ytd_dict, brand_surplus_ytd_dict, specialty_surplus_ytd_dict, disp_fee_surplus_ytd_dict,
                                                               client_reimb_column='PRICE_REIMB', client_TARG_column='TARG_INGCOST_ADJ',
                                                               pharm_reimb_column='PHARM_PRICE_REIMB', pharm_TARG_column='PHARM_TARG_INGCOST_ADJ')
            
        else:
            pre_mc_ytd_dict = calculatePerformance(premc_data, client_guarantees_premc, pharmacy_guarantees,
                                                   client_list, p.AGREEMENT_PHARMACY_LIST, oc_pharm_dummy,
                                                   dummy_perf_dict, dummy_perf_dict, dummy_perf_dict, dummy_perf_dict,
                                                   client_reimb_column='PRICE_REIMB', client_TARG_column='TARG_INGCOST_ADJ',
                                                   pharm_reimb_column='PHARM_PRICE_REIMB', pharm_TARG_column='PHARM_TARG_INGCOST_ADJ')
            
            post_mc_ytd_dict = calculatePerformance(postmc_data, client_guarantees, pharmacy_guarantees,
                                                    client_list, p.AGREEMENT_PHARMACY_LIST, oc_pharm_dummy,
                                                    dummy_perf_dict, dummy_perf_dict, dummy_perf_dict, dummy_perf_dict,
                                                    client_reimb_column='PRICE_REIMB', client_TARG_column='TARG_INGCOST_ADJ',
                                                    pharm_reimb_column='PHARM_PRICE_REIMB', pharm_TARG_column='PHARM_TARG_INGCOST_ADJ')
            
            ytd_perf_pharm_actuals_dict = {k: pre_mc_ytd_dict.get(k, 0) + post_mc_ytd_dict.get(k, 0) for k in set(pre_mc_ytd_dict) | set(post_mc_ytd_dict)}
            
        #Zero out ytd performance for full year pricing
        if p.FULL_YEAR:
            for key in ytd_perf_pharm_actuals_dict:
                ytd_perf_pharm_actuals_dict[key] = 0.0

        # Override YTD performance with current contract data only when using cross contract projections.
        # If a pharmacy with a VCML that has no utilization was added in Daily Input Read to the ytd_df
        # after the current contract data was saved off, or if a pharmacy has utilization in the prior period
        # but not in the current contract period, they need to be added back to the performance dictionary.

        if p.CROSS_CONTRACT_PROJ:
            dict_prior_ccp = ytd_perf_pharm_actuals_dict
            ytd_perf_pharm_actuals_dict = calculatePerformance(current_contract_data, client_guarantees, pharmacy_guarantees,
                                                               client_list, p.AGREEMENT_PHARMACY_LIST, oc_pharm_dummy,
                                                               gen_launch_ytd_dict, brand_surplus_ytd_dict, specialty_surplus_ytd_dict, disp_fee_surplus_ytd_dict,
                                                               client_reimb_column='PRICE_REIMB', client_TARG_column='TARG_INGCOST_ADJ',
                                                               pharm_reimb_column='PHARM_PRICE_REIMB', pharm_TARG_column='PHARM_TARG_INGCOST_ADJ')
            
            dict_diff = set(dict_prior_ccp.keys()) - set(ytd_perf_pharm_actuals_dict.keys())
            dict_diff = dict.fromkeys(dict_diff, 0) # Zero out values as these pharmacies have no YTD utilization
            ytd_perf_pharm_actuals_dict.update(dict_diff)

            del dict_prior_ccp, dict_diff
       
        if p.YTD_OVERRIDE:
            perf_override = pd.read_csv(p.FILE_INPUT_PATH + p.LAG_YTD_Override_File, dtype = p.VARIABLE_TYPE_DIC)
            perf_override = standardize_df(perf_override)
            qa_dataframe(perf_override, dataset = 'LAG_YTD_Override_File_AT_{}'.format(os.path.basename(__file__)))
            perf_override_dict = sf.df_to_dict(perf_override, ['BREAKOUT', 'SURPLUS'])

            for key in perf_override_dict:
                ytd_perf_pharm_actuals_dict[key] = perf_override_dict[key]

        lag_performance_dict = calculatePerformance(ytd_df, client_guarantees, pharmacy_guarantees,
                                       client_list, p.AGREEMENT_PHARMACY_LIST, other_client_pharm_lag, gen_launch_lag_dict,
                                       brand_surplus_lag_dict, specialty_surplus_lag_dict, disp_fee_surplus_lag_dict,
                                       client_reimb_column = 'LAG_REIMB', pharm_reimb_column = 'PHARM_LAG_REIMB',
                                       client_TARG_column = 'TARG_INGCOST_ADJ_PROJ_LAG', pharm_TARG_column = 'PHARM_TARG_INGCOST_ADJ_PROJ_LAG')
        
        if p.FULL_YEAR:
            for key in lag_performance_dict:
                lag_performance_dict[key] = 0.0

        #Zero out lag performance for full year pricing
        if p.FULL_YEAR:
            for key in lag_performance_dict:
                lag_performance_dict[key] = 0.0

        #This is used in simulation mode with YTD_OVERRIDE files
        if p.YTD_OVERRIDE:
            lag_performance_df = sf.dict_to_df(lag_performance_dict, ['BREAKOUT', 'SURPLUS'])
            lag_performance_df.to_csv(p.FILE_DYNAMIC_INPUT_PATH + 'lag_surplus_{}.csv'.format(p.DATA_ID))

        performance_dict = dict()
        for key in ytd_perf_pharm_actuals_dict:
            performance_dict[key] = ytd_perf_pharm_actuals_dict[key] + lag_performance_dict[key]

        if p.LAG_YTD_OVERRIDE:
            perf_override = pd.read_csv(os.path.join(p.FILE_INPUT_PATH, p.LAG_YTD_Override_File), dtype = p.VARIABLE_TYPE_DIC)
            perf_override = standardize_df(perf_override)
            qa_dataframe(perf_override, dataset = 'LAG_YTD_Override_File_AT_{}'.format(os.path.basename(__file__)))
            perf_override_dict = df_to_dict(perf_override, ['BREAKOUT', 'SURPLUS'])

            for key in perf_override_dict:
                performance_dict[key] = perf_override_dict[key]

        # Push YTD + Lag + EOY to next year because we want a full year of data--this makes the new implementation period a full year
        #For Costsaver - we already took care of this in DIR under correct_costsaver_projections()    
        if p.FULL_YEAR:
            if not (p.INTERCEPTOR_OPT or p.COSTSAVER_CLIENT):
                lp_vol_mv_agg_df['QTY_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df.QTY_PROJ_EOY.copy()
                lp_vol_mv_agg_df['QTY_PROJ_EOY'] = lp_vol_mv_agg_df.QTY + lp_vol_mv_agg_df.QTY_PROJ_LAG + lp_vol_mv_agg_df.QTY_PROJ_EOY

                lp_vol_mv_agg_df['CLAIMS_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df.CLAIMS_PROJ_EOY.copy()
                lp_vol_mv_agg_df['CLAIMS_PROJ_EOY'] = lp_vol_mv_agg_df.CLAIMS + lp_vol_mv_agg_df.CLAIMS_PROJ_LAG + lp_vol_mv_agg_df.CLAIMS_PROJ_EOY

                lp_vol_mv_agg_df['FULLAWP_ADJ_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df.FULLAWP_ADJ_PROJ_EOY.copy()
                lp_vol_mv_agg_df['FULLAWP_ADJ_PROJ_EOY'] = lp_vol_mv_agg_df.FULLAWP_ADJ + lp_vol_mv_agg_df.FULLAWP_ADJ_PROJ_LAG + lp_vol_mv_agg_df.FULLAWP_ADJ_PROJ_EOY
                
                lp_vol_mv_agg_df['TARG_INGCOST_ADJ_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df.TARG_INGCOST_ADJ_PROJ_EOY.copy()
                lp_vol_mv_agg_df['TARG_INGCOST_ADJ_PROJ_EOY'] = lp_vol_mv_agg_df.TARG_INGCOST_ADJ + \
                lp_vol_mv_agg_df.TARG_INGCOST_ADJ_PROJ_LAG + lp_vol_mv_agg_df.TARG_INGCOST_ADJ_PROJ_EOY

                lp_vol_mv_agg_df['PHARM_QTY_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df.PHARM_QTY_PROJ_EOY.copy()
                lp_vol_mv_agg_df['PHARM_QTY_PROJ_EOY'] = lp_vol_mv_agg_df.QTY + lp_vol_mv_agg_df.PHARM_QTY_PROJ_LAG + lp_vol_mv_agg_df.PHARM_QTY_PROJ_EOY

                lp_vol_mv_agg_df['PHARM_CLAIMS_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df.PHARM_CLAIMS_PROJ_EOY.copy()
                lp_vol_mv_agg_df['PHARM_CLAIMS_PROJ_EOY'] = lp_vol_mv_agg_df.PHARM_CLAIMS + lp_vol_mv_agg_df.PHARM_CLAIMS_PROJ_LAG + lp_vol_mv_agg_df.PHARM_CLAIMS_PROJ_EOY

                lp_vol_mv_agg_df['PHARM_FULLAWP_ADJ_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df.PHARM_FULLAWP_ADJ_PROJ_EOY.copy()
                lp_vol_mv_agg_df['PHARM_FULLAWP_ADJ_PROJ_EOY'] = lp_vol_mv_agg_df.PHARM_FULLAWP_ADJ + lp_vol_mv_agg_df.PHARM_FULLAWP_ADJ_PROJ_LAG +\
                lp_vol_mv_agg_df.PHARM_FULLAWP_ADJ_PROJ_EOY
                
                lp_vol_mv_agg_df['PHARM_TARG_INGCOST_ADJ_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df.PHARM_TARG_INGCOST_ADJ_PROJ_EOY.copy()
                lp_vol_mv_agg_df['PHARM_TARG_INGCOST_ADJ_PROJ_EOY'] = lp_vol_mv_agg_df.PHARM_TARG_INGCOST_ADJ + \
                lp_vol_mv_agg_df.PHARM_TARG_INGCOST_ADJ_PROJ_LAG + lp_vol_mv_agg_df.PHARM_TARG_INGCOST_ADJ_PROJ_EOY
            

            if p.INCLUDE_PLAN_LIABILITY:
                lp_vol_mv_agg_df['DAYSSUP_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df.DAYSSUP_PROJ_EOY.copy()
                lp_vol_mv_agg_df['DAYSSUP_PROJ_EOY'] = lp_vol_mv_agg_df.DAYSSUP + lp_vol_mv_agg_df.DAYSSUP_PROJ_LAG + lp_vol_mv_agg_df.DAYSSUP_PROJ_EOY

            if p.UNC_OPT:
                lp_vol_mv_agg_df_actual['QTY_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df_actual.QTY_PROJ_EOY.copy()
                lp_vol_mv_agg_df_actual['QTY_PROJ_EOY'] = lp_vol_mv_agg_df_actual.QTY + lp_vol_mv_agg_df_actual.QTY_PROJ_LAG + lp_vol_mv_agg_df_actual.QTY_PROJ_EOY

                lp_vol_mv_agg_df_actual['CLAIMS_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df_actual.CLAIMS_PROJ_EOY.copy()
                lp_vol_mv_agg_df_actual['CLAIMS_PROJ_EOY'] = lp_vol_mv_agg_df_actual.CLAIMS + lp_vol_mv_agg_df_actual.CLAIMS_PROJ_LAG + lp_vol_mv_agg_df_actual.CLAIMS_PROJ_EOY

                lp_vol_mv_agg_df_actual['FULLAWP_ADJ_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df_actual.FULLAWP_ADJ_PROJ_EOY.copy()
                lp_vol_mv_agg_df_actual['FULLAWP_ADJ_PROJ_EOY'] = lp_vol_mv_agg_df_actual.FULLAWP_ADJ + lp_vol_mv_agg_df_actual.FULLAWP_ADJ_PROJ_LAG + lp_vol_mv_agg_df_actual.FULLAWP_ADJ_PROJ_EOY
                
                lp_vol_mv_agg_df_actual['TARG_INGCOST_ADJ_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df_actual.TARG_INGCOST_ADJ_PROJ_EOY.copy()
                lp_vol_mv_agg_df_actual['TARG_INGCOST_ADJ_PROJ_EOY'] = lp_vol_mv_agg_df_actual.TARG_INGCOST_ADJ + \
                lp_vol_mv_agg_df_actual.TARG_INGCOST_ADJ_PROJ_LAG + lp_vol_mv_agg_df_actual.TARG_INGCOST_ADJ_PROJ_EOY

                lp_vol_mv_agg_df_actual['PHARM_QTY_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df_actual.QTY_PROJ_EOY.copy()
                lp_vol_mv_agg_df_actual['PHARM_QTY_PROJ_EOY'] = lp_vol_mv_agg_df_actual.QTY + lp_vol_mv_agg_df_actual.PHARM_QTY_PROJ_LAG + lp_vol_mv_agg_df_actual.PHARM_QTY_PROJ_EOY

                lp_vol_mv_agg_df_actual['PHARM_CLAIMS_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df_actual.PHARM_CLAIMS_PROJ_EOY.copy()
                lp_vol_mv_agg_df_actual['PHARM_CLAIMS_PROJ_EOY'] = lp_vol_mv_agg_df_actual.PHARM_CLAIMS + lp_vol_mv_agg_df_actual.PHARM_CLAIMS_PROJ_LAG + lp_vol_mv_agg_df_actual.CLAIMS_PROJ_EOY

                lp_vol_mv_agg_df_actual['PHARM_FULLAWP_ADJ_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df_actual.PHARM_FULLAWP_ADJ_PROJ_EOY.copy()
                lp_vol_mv_agg_df_actual['PHARM_FULLAWP_ADJ_PROJ_EOY'] = lp_vol_mv_agg_df_actual.PHARM_FULLAWP_ADJ + \
                lp_vol_mv_agg_df_actual.PHARM_FULLAWP_ADJ_PROJ_LAG + lp_vol_mv_agg_df_actual.PHARM_FULLAWP_ADJ_PROJ_EOY
                
                lp_vol_mv_agg_df_actual['PHARM_TARG_INGCOST_ADJ_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df_actual.PHARM_TARG_INGCOST_ADJ_PROJ_EOY.copy()
                lp_vol_mv_agg_df_actual['PHARM_TARG_INGCOST_ADJ_PROJ_EOY'] = lp_vol_mv_agg_df_actual.PHARM_TARG_INGCOST_ADJ + \
                lp_vol_mv_agg_df_actual.PHARM_TARG_INGCOST_ADJ_PROJ_LAG + lp_vol_mv_agg_df_actual.PHARM_TARG_INGCOST_ADJ_PROJ_EOY

        # Push prior contract utilization to EOY columns in order to use those data as our EOY projections.
        if p.CROSS_CONTRACT_PROJ:
            lp_vol_mv_agg_df['QTY_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df.QTY_PROJ_EOY.copy()
            lp_vol_mv_agg_df['QTY_PROJ_EOY'] = lp_vol_mv_agg_df.QTY_PRIOR

            lp_vol_mv_agg_df['CLAIMS_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df.CLAIMS_PROJ_EOY.copy()
            lp_vol_mv_agg_df['CLAIMS_PROJ_EOY'] = lp_vol_mv_agg_df.CLAIMS_PRIOR

            lp_vol_mv_agg_df['FULLAWP_ADJ_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df.FULLAWP_ADJ_PROJ_EOY.copy()
            lp_vol_mv_agg_df['FULLAWP_ADJ_PROJ_EOY'] = lp_vol_mv_agg_df.FULLAWP_ADJ_PRIOR
            
            lp_vol_mv_agg_df['TARG_INGCOST_ADJ_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df.TARG_INGCOST_ADJ_PROJ_EOY.copy()
            lp_vol_mv_agg_df['TARG_INGCOST_ADJ_PROJ_EOY'] = lp_vol_mv_agg_df.TARG_INGCOST_ADJ_PRIOR

            lp_vol_mv_agg_df['PHARM_QTY_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df.PHARM_QTY_PROJ_EOY.copy()
            lp_vol_mv_agg_df['PHARM_QTY_PROJ_EOY'] = lp_vol_mv_agg_df.PHARM_QTY_PRIOR

            lp_vol_mv_agg_df['PHARM_CLAIMS_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df.PHARM_CLAIMS_PROJ_EOY.copy()
            lp_vol_mv_agg_df['PHARM_CLAIMS_PROJ_EOY'] = lp_vol_mv_agg_df.PHARM_CLAIMS_PRIOR

            lp_vol_mv_agg_df['PHARM_FULLAWP_ADJ_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df.PHARM_FULLAWP_ADJ_PROJ_EOY.copy()
            lp_vol_mv_agg_df['PHARM_FULLAWP_ADJ_PROJ_EOY'] = lp_vol_mv_agg_df.PHARM_FULLAWP_ADJ_PRIOR
            
            lp_vol_mv_agg_df['PHARM_TARG_INGCOST_ADJ_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df.PHARM_TARG_INGCOST_ADJ_PROJ_EOY.copy()
            # calculate this target cost here itself
            lp_vol_mv_agg_df['PHARM_TARG_INGCOST_ADJ_PROJ_EOY'] = lp_vol_mv_agg_df.PHARM_TARG_INGCOST_ADJ_PRIOR

            if p.UNC_OPT:
                lp_vol_mv_agg_df_actual['QTY_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df_actual.QTY_PROJ_EOY.copy()
                lp_vol_mv_agg_df_actual['QTY_PROJ_EOY'] = lp_vol_mv_agg_df_actual.QTY_PRIOR

                lp_vol_mv_agg_df_actual['CLAIMS_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df_actual.CLAIMS_PROJ_EOY.copy()
                lp_vol_mv_agg_df_actual['CLAIMS_PROJ_EOY'] = lp_vol_mv_agg_df_actual.CLAIMS_PRIOR

                lp_vol_mv_agg_df_actual['FULLAWP_ADJ_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df_actual.FULLAWP_ADJ_PROJ_EOY.copy()
                lp_vol_mv_agg_df_actual['FULLAWP_ADJ_PROJ_EOY'] = lp_vol_mv_agg_df_actual.FULLAWP_ADJ_PRIOR
                
                lp_vol_mv_agg_df_actual['TARG_INGCOST_ADJ_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df_actual.TARG_INGCOST_ADJ_PROJ_EOY.copy()
                lp_vol_mv_agg_df_actual['TARG_INGCOST_ADJ_PROJ_EOY'] = lp_vol_mv_agg_df_actual.TARG_INGCOST_ADJ_PRIOR

                lp_vol_mv_agg_df_actual['PHARM_QTY_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df_actual.QTY_PROJ_EOY.copy()
                lp_vol_mv_agg_df_actual['PHARM_QTY_PROJ_EOY'] = lp_vol_mv_agg_df_actual.PHARM_QTY_PRIOR

                lp_vol_mv_agg_df_actual['PHARM_CLAIMS_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df_actual.PHARM_CLAIMS_PROJ_EOY.copy()
                lp_vol_mv_agg_df_actual['PHARM_CLAIMS_PROJ_EOY'] = lp_vol_mv_agg_df_actual.CLAIMS_PRIOR

                lp_vol_mv_agg_df_actual['PHARM_FULLAWP_ADJ_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df_actual.PHARM_FULLAWP_ADJ_PROJ_EOY.copy()
                lp_vol_mv_agg_df_actual['PHARM_FULLAWP_ADJ_PROJ_EOY'] = lp_vol_mv_agg_df_actual.PHARM_FULLAWP_ADJ_PRIOR
                
                lp_vol_mv_agg_df_actual['PHARM_TARG_INGCOST_ADJ_PROJ_EOY_ORIG'] = lp_vol_mv_agg_df_actual.PHARM_TARG_INGCOST_ADJ_PROJ_EOY.copy()
                # calculate this target cost here itself
                lp_vol_mv_agg_df_actual['PHARM_TARG_INGCOST_ADJ_PROJ_EOY'] = lp_vol_mv_agg_df_actual.PHARM_TARG_INGCOST_ADJ_PRIOR

        # Read in and apply the UC adjustment
        if unc_flag:
            act_performance_dict = copy.copy(performance_dict)
            unc_adjust = pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.UNC_ADJUSTMENT), dtype = p.VARIABLE_TYPE_DIC)
            for item in unc_adjust.BREAKOUT.unique():
                performance_dict[item] = performance_dict[item] + unc_adjust.loc[unc_adjust.BREAKOUT==item, 'DELTA'].values[0]
        else:
            act_performance_dict = performance_dict

        # Read in and implement client guard rails
        if p.CLIENT_GR:
            guard_rails_df = pd.read_csv(os.path.join(p.FILE_INPUT_PATH, p.WC_SUGGESTED_GUARDRAILS), dtype = p.VARIABLE_TYPE_DIC)
            guard_rails_df = standardize_df(guard_rails_df)

            qa_dataframe(guard_rails_df, dataset = 'WC_SUGGESTED_GUARDRAILS_AT_{}'.format(os.path.basename(__file__)))
            #guard_rails_df.TIER = guard_rails_df.TIER.astype(str)

            # if p.LIM_TIER:
            #     guard_rails_df = guard_rails_df.loc[guard_rails_df.TIER.isin(p.TIER_LIST)]

            # Remove this once we get a clean set of guard_rails
            guard_rails_agg = guard_rails_df.groupby(by=['REGION', 'CHAIN_GROUP', 'GPI', 'NDC', 'BG_FLAG'])
            guard_rails_clean = guard_rails_agg['MIN'].agg(min)
            guard_rails_clean = pd.concat([guard_rails_clean, guard_rails_agg['MAX'].agg(max)], axis=1).reset_index(drop = True)

            # Create the different guard rails for preferred and nonpreferred lists
            full_guard_rails = guard_rails_clean.rename(columns={'MIN': 'CLIENT_MIN_PRICE',
                                                                 'MAX': 'CLIENT_MAX_PRICE'})

            #merge the guardrails onto the full dataset
            lp_vol_mv_agg_df_temp = pd.merge(lp_vol_mv_agg_df,
                                             full_guard_rails, how='left',
                                             on=['REGION', 'CHAIN_GROUP', 'GPI', 'NDC', 'BG_FLAG'])
            assert len(lp_vol_mv_agg_df_temp) == len(lp_vol_mv_agg_df_temp), "len(lp_vol_mv_agg_df_temp) == len(lp_vol_mv_agg_df_temp)"
    #        lp_vol_mv_agg_df = lp_vol_mv_agg_df_temp

            #fill empty values
            lp_vol_mv_agg_df_temp['CLIENT_MIN_PRICE'] = lp_vol_mv_agg_df_temp['CLIENT_MIN_PRICE'].fillna(0.0000)
            lp_vol_mv_agg_df_temp['CLIENT_MAX_PRICE'] = lp_vol_mv_agg_df_temp['CLIENT_MAX_PRICE'].fillna(9999.9999)

            lp_vol_mv_agg_df = lp_vol_mv_agg_df_temp

        #if no client provided guiderails
        else:
            lp_vol_mv_agg_df['CLIENT_MIN_PRICE'] = 0.0000
            lp_vol_mv_agg_df['CLIENT_MAX_PRICE'] = 9999.9999
        
        if p.CLIENT_TIERS:
            client_tiers_df = standardize_df(pd.read_csv(os.path.join(p.FILE_INPUT_PATH, p.CLIENT_SUGGESTED_TIERS), dtype = p.VARIABLE_TYPE_DIC))
            if 'BG_FLAG' in client_tiers_df:
                client_tiers_df=client_tiers_df.drop(columns=['BG_FLAG']).drop_duplicates()
            qa_dataframe(client_tiers_df, dataset = 'CLIENT_SUGGESTED_TIERS_AT_{}'.format(os.path.basename(__file__)))
            client_tiers_df.TIER = client_tiers_df.TIER.astype(str)
            client_tiers_df.rename(columns={'TIER': 'PRICE_TIER'}, inplace=True)
            lp_vol_mv_agg_df_temp = pd.merge(lp_vol_mv_agg_df, client_tiers_df, how='left',
                                             on=['CLIENT', 'BREAKOUT', 'REGION', 'GPI'])
            assert len(lp_vol_mv_agg_df_temp) == len(lp_vol_mv_agg_df), "len(lp_vol_mv_agg_df_temp) == len(lp_vol_mv_agg_df)"
            lp_vol_mv_agg_df_temp['PRICE_TIER'] = lp_vol_mv_agg_df_temp['PRICE_TIER'].fillna("0")
            lp_vol_mv_agg_df = lp_vol_mv_agg_df_temp
        elif p.HANDLE_CONFLICT_GPI and p.CONFLICT_GPI_AS_TIERS:
            client_tiers_df = standardize_df(pd.read_csv(os.path.join(p.FILE_INPUT_PATH, p.CONFLICT_GPI_LIST_FILE), dtype = p.VARIABLE_TYPE_DIC))
            if 'BG_FLAG' in client_tiers_df:
                client_tiers_df=client_tiers_df.drop(columns=['BG_FLAG']).drop_duplicates()
            qa_dataframe(client_tiers_df, dataset = 'CONFLICT_GPI_LIST_FILE_AT_{}'.format(os.path.basename(__file__)))
            client_tiers_df[['CLIENT', 'REGION', 'GPI']].drop_duplicates(inplace=True)
            client_tiers_df['PRICE_TIER'] = 'CONFLICT'
            lp_vol_mv_agg_df_temp = pd.merge(lp_vol_mv_agg_df, client_tiers_df, how='left', on=['CLIENT', 'REGION', 'GPI'])
            assert len(lp_vol_mv_agg_df_temp) == len(lp_vol_mv_agg_df), "len(lp_vol_mv_agg_df_temp) == len(lp_vol_mv_agg_df)"
            lp_vol_mv_agg_df_temp['PRICE_TIER'] = lp_vol_mv_agg_df_temp['PRICE_TIER'].fillna("0")
            lp_vol_mv_agg_df = lp_vol_mv_agg_df_temp
        else:
            lp_vol_mv_agg_df['PRICE_TIER'] = '0'

        lp_vol_mv_agg_df['LM_PRICE_REIMB_CLAIM'] = lp_vol_mv_agg_df.LM_PRICE_REIMB/lp_vol_mv_agg_df.LM_CLAIMS
        lp_vol_mv_agg_df['LM_PHARM_PRICE_REIMB_CLAIM'] = lp_vol_mv_agg_df.LM_PHARM_PRICE_REIMB/lp_vol_mv_agg_df.LM_PHARM_CLAIMS
        
        # Create pricing UNC value so that the bounds can be different than effective price
        lp_vol_mv_agg_df['PRICING_UC_UNIT'] = lp_vol_mv_agg_df['UC_UNIT'].copy()
        
        if p.REMOVE_KRG_WMT_UC:
            lp_vol_mv_agg_df.loc[(lp_vol_mv_agg_df.CHAIN_GROUP.isin(['KRG', 'WMT', 'NONPREF_OTH'])), 'PRICING_UC_UNIT'] = \
                lp_vol_mv_agg_df.loc[(lp_vol_mv_agg_df.CHAIN_GROUP.isin(['KRG', 'WMT', 'NONPREF_OTH'])), 'PRICING_UC_UNIT'] * 500
           
        if p.REMOVE_SMALL_CAPPED_UC:
            lp_vol_mv_agg_df.loc[(lp_vol_mv_agg_df.CHAIN_GROUP.isin(list(set(p.SMALL_CAPPED_PHARMACY_LIST['GNRC']+p.SMALL_CAPPED_PHARMACY_LIST['BRND']+
                                                                             p.NON_CAPPED_PHARMACY_LIST['GNRC']+p.NON_CAPPED_PHARMACY_LIST['BRND']+
                                                                             p.COGS_PHARMACY_LIST['GNRC']+p.COGS_PHARMACY_LIST['BRND']+
                                                                             p.PSAO_LIST['GNRC']+p.PSAO_LIST['BRND'])))), 'PRICING_UC_UNIT'] = \
                lp_vol_mv_agg_df.loc[(lp_vol_mv_agg_df.CHAIN_GROUP.isin(list(set(p.SMALL_CAPPED_PHARMACY_LIST['GNRC']+p.SMALL_CAPPED_PHARMACY_LIST['BRND']+
                                                                             p.NON_CAPPED_PHARMACY_LIST['GNRC']+p.NON_CAPPED_PHARMACY_LIST['BRND']+
                                                                             p.COGS_PHARMACY_LIST['GNRC']+p.COGS_PHARMACY_LIST['BRND']+
                                                                             p.PSAO_LIST['GNRC']+p.PSAO_LIST['BRND'])))), 'PRICING_UC_UNIT'] * 500
        if 'BG_FLAG' in lp_vol_mv_agg_df.columns:
            old_len = len(lp_vol_mv_agg_df)
            old_claims = lp_vol_mv_agg_df.CLAIMS.sum(axis=0)
            temp_df = lp_vol_mv_agg_df.copy()
            
            uc_cols = ['CLIENT', 'BREAKOUT', 'MEASUREMENT', 'GPI_NDC', 'CHAIN_GROUP', 'CHAIN_SUBGROUP',
                                 'REGION', 'PHARMACY_TYPE', 'BG_FLAG', 'MAC1026_UNIT_PRICE', 'UC_UNIT', 'SOFT_CONST_BENCHMARK_PRICE']

            temp_df_filtered = temp_df[uc_cols].loc[temp_df.PRICE_MUTABLE == 1, :]

            gnrc_group = temp_df_filtered[temp_df_filtered['BG_FLAG'] == 'G']
            gnrc_group = gnrc_group.groupby(['GPI_NDC', 'CLIENT', 'REGION', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'MEASUREMENT'],
                                               as_index=False) \
                .agg(MAC1026_UNIT_PRICE_gnrc=('MAC1026_UNIT_PRICE', 'max'))

            brnd_group = temp_df_filtered[temp_df_filtered['BG_FLAG'] == 'B']
            brnd_group = brnd_group.groupby(['GPI_NDC', 'CLIENT', 'REGION', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'MEASUREMENT'],
                                               as_index=False) \
                .agg(UC_UNIT_brnd=('UC_UNIT', 'first'),
                    SOFT_CONST_BENCHMARK_PRICE_brnd=('SOFT_CONST_BENCHMARK_PRICE', 'min'))

            brnd_gnrc_merged = brnd_group.merge(gnrc_group, on=['GPI_NDC', 'CLIENT', 'REGION', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'MEASUREMENT'], how='inner', validate='1:1')
            
            brnd_gnrc_merged = brnd_gnrc_merged[(brnd_gnrc_merged['MAC1026_UNIT_PRICE_gnrc'] > brnd_gnrc_merged['UC_UNIT_brnd']) & (brnd_gnrc_merged['UC_UNIT_brnd'] > 0)]
            
            brnd_gnrc_merged['UC_UNIT_brnd'] = brnd_gnrc_merged['MAC1026_UNIT_PRICE_gnrc'] * 1.1
            brnd_gnrc_merged.drop(columns = ['MAC1026_UNIT_PRICE_gnrc','SOFT_CONST_BENCHMARK_PRICE_brnd'], inplace=True)
            brnd_gnrc_merged['BG_FLAG'] = 'B'
            
            temp_df = temp_df.merge(brnd_gnrc_merged, on=['GPI_NDC', 'CLIENT', 'REGION', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'MEASUREMENT', 'BG_FLAG'], how='left', validate='1:1')
            temp_df['PRICING_UC_UNIT'] = temp_df['UC_UNIT_brnd'].fillna(temp_df['PRICING_UC_UNIT'])
            temp_df.drop(columns = ['UC_UNIT_brnd'], inplace=True)
            
            assert len(temp_df) == old_len, "Unintentionally adding or dropping rows"
            assert np.abs(temp_df.CLAIMS.sum(axis=0) - old_claims) < 0.0001, "Making sure no claims are dropped"
            
            lp_vol_mv_agg_df = temp_df.copy(deep=True)
            del temp_df, temp_df_filtered, gnrc_group, brnd_group, brnd_gnrc_merged
               
        logger.info('Finished input reads')
        logger.info('--------------------')

    #    ##### Price Mutable Flag ######
        logger.info('Setting current prices immutable')

        lp_vol_mv_agg_df.loc[:,'PRICE_MUTABLE'] = 1
        lp_vol_mv_agg_df.loc[:,'IMMUTABLE_REASON'] = ''
        lp_vol_mv_agg_df['PRICE_MUTABLE'].where(lp_vol_mv_agg_df.CURRENT_MAC_PRICE > 0, 0, inplace = True)
        lp_vol_mv_agg_df['IMMUTABLE_REASON'].where(lp_vol_mv_agg_df.CURRENT_MAC_PRICE > 0, 'NON_MAC_DRUGS', inplace = True)
        
        if p.BRND_PRICE_FREEZE:
            lp_vol_mv_agg_df.loc[(lp_vol_mv_agg_df.BG_FLAG == 'B'),'PRICE_MUTABLE'] = 0
            lp_vol_mv_agg_df.loc[(lp_vol_mv_agg_df.BG_FLAG == 'B'),'IMMUTABLE_REASON'] = 'BRAND PRICE FREEZE'
            
        if p.GNRC_PRICE_FREEZE:
            lp_vol_mv_agg_df.loc[(lp_vol_mv_agg_df.BG_FLAG == 'G'),'PRICE_MUTABLE'] = 0
            lp_vol_mv_agg_df.loc[(lp_vol_mv_agg_df.BG_FLAG == 'G'),'IMMUTABLE_REASON'] = 'GENERIC PRICE FREEZE'
        
        if not p.PRICE_ZERO_PROJ_QTY:
            lp_vol_mv_agg_df.loc[:,'PRICE_MUTABLE'] = 0
            lp_vol_mv_agg_df['PRICE_MUTABLE'].where(lp_vol_mv_agg_df.FULLAWP_ADJ_PROJ_EOY == 0, lp_vol_mv_agg_df.PRICE_MUTABLE, inplace = True)
            lp_vol_mv_agg_df['IMMUTABLE_REASON'].where(lp_vol_mv_agg_df.FULLAWP_ADJ_PROJ_EOY != 0, "NO_PROJECTED_UTILIZATION", inplace = True)

        # Anything marked immutable -- undo U&C price changes.
        if p.UNC_OPT:
            undo_unc = (lp_vol_mv_agg_df['PRICE_MUTABLE']==0) & (lp_vol_mv_agg_df['PRICE_CHANGED_UC'])
            lp_vol_mv_agg_df.loc[undo_unc & lp_vol_mv_agg_df['RAISED_PRICE_UC'], 'CURRENT_MAC_PRICE'] = \
                lp_vol_mv_agg_df.loc[undo_unc & lp_vol_mv_agg_df['RAISED_PRICE_UC'], 'PRE_UC_MAC_PRICE']
            lp_vol_mv_agg_df.loc[undo_unc, 'PRICE_CHANGED_UC'] = False

        if p.FLOOR_PRICE:
            floor_gpi = standardize_df(pd.read_csv(p.FILE_INPUT_PATH + p.FLOOR_GPI_LIST, dtype = p.VARIABLE_TYPE_DIC))
            qa_dataframe(floor_gpi, dataset = 'FLOOR_GPI_LIST_AT_{}'.format(os.path.basename(__file__)))
            lp_vol_mv_agg_df['EFF_CAPPED_PRICE_ACTUAL'] = lp_vol_mv_agg_df['EFF_CAPPED_PRICE']
            lp_vol_mv_agg_df.loc[
                (lp_vol_mv_agg_df.CURRENT_MAC_PRICE > 0) & (lp_vol_mv_agg_df.GPI.isin(floor_gpi.GPI)), 'EFF_CAPPED_PRICE'] = \
                lp_vol_mv_agg_df.loc[
                    (lp_vol_mv_agg_df.CURRENT_MAC_PRICE > 0) & (
                        lp_vol_mv_agg_df.GPI.isin(floor_gpi.GPI)), 'MAC1026_UNIT_PRICE']

            lp_vol_mv_agg_df.loc[
                (lp_vol_mv_agg_df.CURRENT_MAC_PRICE > 0) & (lp_vol_mv_agg_df.GPI.isin(floor_gpi.GPI) & (lp_vol_mv_agg_df.MAC1026_GPI_FLAG == 1)), 'EFF_CAPPED_PRICE'] = \
                lp_vol_mv_agg_df.loc[
                    (lp_vol_mv_agg_df.CURRENT_MAC_PRICE > 0) & (
                        lp_vol_mv_agg_df.GPI.isin(floor_gpi.GPI)) & (
                            lp_vol_mv_agg_df.MAC1026_GPI_FLAG == 1), 'EFF_CAPPED_PRICE_ACTUAL']


            lp_vol_mv_agg_df['CURRENT_MAC_PRICE_ACTUAL'] = lp_vol_mv_agg_df['CURRENT_MAC_PRICE'].copy()
            lp_vol_mv_agg_df.loc[(lp_vol_mv_agg_df.CURRENT_MAC_PRICE > 0) & (
                lp_vol_mv_agg_df.GPI.isin(floor_gpi.GPI)), 'CURRENT_MAC_PRICE'] = lp_vol_mv_agg_df.loc[
                (lp_vol_mv_agg_df.CURRENT_MAC_PRICE > 0) & (lp_vol_mv_agg_df.GPI.isin(floor_gpi.GPI)), 'MAC1026_UNIT_PRICE']
            lp_vol_mv_agg_df['PRICE_MUTABLE'] = lp_vol_mv_agg_df.apply(
                lambda df: 0 if ((df.GPI in list(floor_gpi.GPI)) and (df.CURRENT_MAC_PRICE > 0)) else df.PRICE_MUTABLE,
                axis=1)
            lp_vol_mv_agg_df['IMMUTABLE_REASON'] = lp_vol_mv_agg_df.apply(
                lambda df: 'FLOOR_GPI' if ((df.GPI in list(floor_gpi.GPI)) and (df.CURRENT_MAC_PRICE > 0)) else '',
                axis=1)
            
            if p.UNC_OPT:
                undo_unc = (lp_vol_mv_agg_df['PRICE_MUTABLE']==0) & (lp_vol_mv_agg_df['PRICE_CHANGED_UC'])
                # We don't reset the price because it was already reset above
                lp_vol_mv_agg_df.loc[undo_unc, 'PRICE_CHANGED_UC'] = False

        if p.CLIENT_TIERS:
            lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df.PRICE_TIER =="4", 'PRICE_MUTABLE'] = 0
            lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df.PRICE_TIER =="4", 'IMMUTABLE_REASON'] = 'CLIENT_SUGGESTED_PRICE_TIERS'
        
        #The adjusted Price reimbursed is based on the current MAC prices and nor the historical prices (since that can change)
        lp_vol_mv_agg_df['PRICE_REIMB_ADJ'] = lp_vol_mv_agg_df['QTY'] * lp_vol_mv_agg_df['OLD_MAC_PRICE'] 
        lp_vol_mv_agg_df['PRICE_REIMB_CLAIM'] = lp_vol_mv_agg_df['PRICE_REIMB_ADJ'] / lp_vol_mv_agg_df['CLAIMS']
        lp_vol_mv_agg_df['PHARM_PRICE_REIMB_ADJ'] = lp_vol_mv_agg_df['PHARM_QTY'] * lp_vol_mv_agg_df['OLD_MAC_PRICE']
        lp_vol_mv_agg_df['PHARM_PRICE_REIMB_CLAIM'] = lp_vol_mv_agg_df['PHARM_PRICE_REIMB_ADJ'] / lp_vol_mv_agg_df['PHARM_CLAIMS']
        
        if p.TRUECOST_CLIENT:
            lp_vol_mv_agg_df['PRICE_REIMB_CLAIM'] = np.where((lp_vol_mv_agg_df['CLAIMS'] == 0),
                                                             lp_vol_mv_agg_df['PRICE_REIMB_ADJ'],
                                                            (lp_vol_mv_agg_df['PRICE_REIMB_ADJ'] + lp_vol_mv_agg_df['DISP_FEE']) / lp_vol_mv_agg_df['CLAIMS']
                                                            )
            lp_vol_mv_agg_df['PHARM_PRICE_REIMB_CLAIM'] = (lp_vol_mv_agg_df['PHARM_PRICE_REIMB_ADJ'] + lp_vol_mv_agg_df['PHARMACY_DISP_FEE']) / lp_vol_mv_agg_df['PHARM_CLAIMS']

        if p.READ_FROM_BQ:
            gpi_awp_df = uf.read_BQ_data(
                                    project_id=p.BQ_INPUT_PROJECT_ID,
                                    dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                                    table_id='awp_history_table',
                                    query=BQ.gpi_awp_table_custom.format(
                                        _project_id=p.BQ_INPUT_PROJECT_ID,
                                        _dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                                        _table_id='awp_history_table',
                                    ),
                                    custom=True)
            ndc_awp_df = uf.read_BQ_data(
                                    project_id=p.BQ_INPUT_PROJECT_ID,
                                    dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                                    table_id='awp_history_table',
                                    query=BQ.ndc_awp_table_custom.format(
                                        _project_id=p.BQ_INPUT_PROJECT_ID,
                                        _dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                                        _table_id='awp_history_table',
                                    ),
                                    custom=True)
        else:
            raise Exception("p.READ_FROM_BQ has to be True")

        lp_vol_mv_agg_df.loc[:,'AWP_LIST'] = np.nan

        old_len = len(lp_vol_mv_agg_df)
        
        lp_vol_mv_agg_df_gpi_temp = lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df['NDC'] == '***********'].merge(gpi_awp_df, on = ['GPI','BG_FLAG'], how='left')        
        lp_vol_mv_agg_df_ndc_temp = lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df['NDC'] != '***********'].merge(ndc_awp_df, on = ['GPI','NDC','BG_FLAG'], how='left')
        
        assert len(lp_vol_mv_agg_df_ndc_temp) + len(lp_vol_mv_agg_df_gpi_temp) == old_len, "len(lp_vol_mv_agg_df) == old_len"
        lp_vol_mv_agg_df = pd.concat([lp_vol_mv_agg_df_gpi_temp, lp_vol_mv_agg_df_ndc_temp]).reset_index(drop = True)
        
        lp_vol_mv_agg_df.loc[:,'AWP_LIST'] = lp_vol_mv_agg_df[['AWP_LIST_NDC', 'AWP_LIST_GPI']].bfill(axis=1).iloc[:, 0]
                
        # These new columns are the ones that should be used for pricing.  The change is to incorporate UNC excluded claims and correctly apply bounds to them.
        lp_vol_mv_agg_df['PRICING_QTY_PROJ_EOY'] = lp_vol_mv_agg_df[['QTY_PROJ_EOY', 'PHARM_QTY_PROJ_EOY']].max(axis=1, skipna=True)
        lp_vol_mv_agg_df['PRICING_CLAIMS_PROJ_EOY'] = lp_vol_mv_agg_df[['CLAIMS_PROJ_EOY', 'PHARM_CLAIMS_PROJ_EOY']].max(axis=1, skipna=True)
        lp_vol_mv_agg_df['PRICING_CLAIMS'] = lp_vol_mv_agg_df[['CLAIMS', 'PHARM_CLAIMS']].max(axis=1, skipna=True)
        lp_vol_mv_agg_df['PRICING_QTY'] = lp_vol_mv_agg_df[['QTY', 'PHARM_QTY']].max(axis=1, skipna=True)
        lp_vol_mv_agg_df['PRICING_PRICE_REIMB_CLAIM'] = lp_vol_mv_agg_df[['PRICE_REIMB_CLAIM', 'PHARM_PRICE_REIMB_CLAIM']].max(axis=1, skipna=True)   
        
        lp_vol_mv_agg_df['PHARM_AVG_AWP'] = lp_vol_mv_agg_df['PHARM_FULLAWP_ADJ'] / lp_vol_mv_agg_df['PHARM_QTY']
        lp_vol_mv_agg_df['TEMP_PRICING_AVG_AWP'] = lp_vol_mv_agg_df[['AVG_AWP', 'PHARM_AVG_AWP']].max(axis=1, skipna=True)
        
        lp_vol_mv_agg_df['TEMP_PRICING_AVG_AWP'].replace(0, np.nan, inplace=True)
        lp_vol_mv_agg_df['PRICING_AVG_AWP'] = lp_vol_mv_agg_df[['TEMP_PRICING_AVG_AWP', 'AWP_LIST']].bfill(axis=1).iloc[:, 0]
        # edge case mostly seen on retail: high unit cost & no AWP info leads to "AWP standard discount" ceilings << current price
        lp_vol_mv_agg_df.loc[(lp_vol_mv_agg_df.PRICING_AVG_AWP.isna()) & 
                             (lp_vol_mv_agg_df.BG_FLAG == 'B') & 
                             (lp_vol_mv_agg_df.CURRENT_MAC_PRICE > 1_000*(1 - p.BRAND_NON_MAC_RATE)),
                             'PRICING_AVG_AWP'] = lp_vol_mv_agg_df.loc[(lp_vol_mv_agg_df.PRICING_AVG_AWP.isna()) & 
                                                                       (lp_vol_mv_agg_df.BG_FLAG == 'B') & 
                                                                       (lp_vol_mv_agg_df.CURRENT_MAC_PRICE > 1_000*(1 - p.BRAND_NON_MAC_RATE)),
                                                                       'CURRENT_MAC_PRICE']/(1 - p.BRAND_NON_MAC_RATE)
        
        lp_vol_mv_agg_df.loc[(lp_vol_mv_agg_df.PRICING_AVG_AWP.isna()) & (lp_vol_mv_agg_df.CURRENT_MAC_PRICE > 1_000*(1 - p.RETAIL_NON_MAC_RATE)),
                             'PRICING_AVG_AWP'] = lp_vol_mv_agg_df.loc[(lp_vol_mv_agg_df.PRICING_AVG_AWP.isna()) & 
                                                                       (lp_vol_mv_agg_df.CURRENT_MAC_PRICE > 1_000*(1 - p.RETAIL_NON_MAC_RATE)),
                                                                       'CURRENT_MAC_PRICE']/(1 - p.MAIL_NON_MAC_RATE)

        lp_vol_mv_agg_df['PRICING_AVG_AWP'] = lp_vol_mv_agg_df['PRICING_AVG_AWP'].fillna(1_000).astype('float64')

        if p.TRUECOST_CLIENT:
            avg_price_ok = lp_vol_mv_agg_df[lp_vol_mv_agg_df['CHAIN_SUBGROUP'].str.contains('OK')].groupby(['GPI','BG_FLAG'])['PRICING_AVG_AWP'].max().reset_index()
            avg_price_ok.rename(columns={'PRICING_AVG_AWP':'OK_PRICING_AVG_AWP'}, inplace = True)
            old_len = len(lp_vol_mv_agg_df)
            lp_vol_mv_agg_df = lp_vol_mv_agg_df.merge(avg_price_ok, on = ['GPI','BG_FLAG'], how = 'left')
            assert len(lp_vol_mv_agg_df) == old_len, "len(lp_vol_mv_agg_df) == old_len"
            lp_vol_mv_agg_df['PRICING_AVG_AWP'] = np.where((lp_vol_mv_agg_df['PRICING_AVG_AWP'] < lp_vol_mv_agg_df['OK_PRICING_AVG_AWP']) 
                                                           & (lp_vol_mv_agg_df['MEASUREMENT'] == 'R30'),
                                                           lp_vol_mv_agg_df['OK_PRICING_AVG_AWP'], lp_vol_mv_agg_df['PRICING_AVG_AWP'])

            lp_vol_mv_agg_df.drop(columns = ['OK_PRICING_AVG_AWP'], inplace = True)

        lp_vol_mv_agg_df.drop(columns=['TEMP_PRICING_AVG_AWP'], inplace=True)
        
        old_len = len(lp_vol_mv_agg_df)
        old_claims = lp_vol_mv_agg_df.PRICING_CLAIMS.sum(axis=0)

        temp_df = (lp_vol_mv_agg_df.assign(AWP_x_CLAIMS=lp_vol_mv_agg_df.PRICING_AVG_AWP * lp_vol_mv_agg_df.PRICING_CLAIMS)
                                    .groupby(['CLIENT','REGION','MAC_LIST','GPI','NDC','BG_FLAG'])
                                    .agg(CLAIMS_SUM = ('CLAIMS', np.nansum),
                                         AWP_x_CLAIMS_SUM = ('AWP_x_CLAIMS', np.nansum),
                                         MAX_PRICING_AVG_AWP = ('PRICING_AVG_AWP', np.nanmax))
                                    .reset_index()
                                    .assign(TEMP_PRICING_AVG_AWP =lambda df:
                                            np.where(df.CLAIMS_SUM > 0, df.AWP_x_CLAIMS_SUM / df.CLAIMS_SUM, df.MAX_PRICING_AVG_AWP)
                                           )
                                    .drop(columns=['CLAIMS_SUM', 'AWP_x_CLAIMS_SUM', 'MAX_PRICING_AVG_AWP'])
                  )
                                
        temp_df = pd.merge(temp_df, lp_vol_mv_agg_df, how='left', on = ['CLIENT','REGION','MAC_LIST','GPI','NDC','BG_FLAG'])
        assert len(temp_df) == old_len, "Unintentionally adding or dropping rows"
        assert np.abs(temp_df.PRICING_CLAIMS.sum(axis=0) - old_claims) < 0.0001, "Making sure no AWP is dropped"
        
        temp_df.drop(columns=['PRICING_AVG_AWP'], inplace=True)
        temp_df['PRICING_AVG_AWP'] = temp_df['TEMP_PRICING_AVG_AWP']
        temp_df.drop(columns=['TEMP_PRICING_AVG_AWP'], inplace=True)
        
        if 'BG_FLAG' in temp_df.columns:
            avg_awp_cols = ['CLIENT', 'BREAKOUT', 'MEASUREMENT', 'GPI_NDC', 'CHAIN_GROUP', 'CHAIN_SUBGROUP',
                                 'REGION', 'PHARMACY_TYPE', 'BG_FLAG', 'MAC1026_UNIT_PRICE', 'PRICING_AVG_AWP', 'SOFT_CONST_BENCHMARK_PRICE']

            temp_df_filtered = temp_df[avg_awp_cols].loc[temp_df.PRICE_MUTABLE == 1, :]

            gnrc_group = temp_df_filtered[temp_df_filtered['BG_FLAG'] == 'G']
            gnrc_group = gnrc_group.groupby(['GPI_NDC', 'CLIENT', 'REGION', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'MEASUREMENT'],
                                               as_index=False) \
                .agg(MAC1026_UNIT_PRICE_gnrc=('MAC1026_UNIT_PRICE', 'max'))

            brnd_group = temp_df_filtered[temp_df_filtered['BG_FLAG'] == 'B']
            brnd_group = brnd_group.groupby(['GPI_NDC', 'CLIENT', 'REGION', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'MEASUREMENT'],
                                               as_index=False) \
                .agg(PRICING_AVG_AWP_brnd=('PRICING_AVG_AWP', 'min'),
                    SOFT_CONST_BENCHMARK_PRICE_brnd=('SOFT_CONST_BENCHMARK_PRICE', 'min'))

            brnd_gnrc_merged = brnd_group.merge(gnrc_group, on=['GPI_NDC', 'CLIENT', 'REGION', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'MEASUREMENT'], how='inner', validate='1:1')
            
            brnd_gnrc_merged = brnd_gnrc_merged[brnd_gnrc_merged['MAC1026_UNIT_PRICE_gnrc'] > (1 - p.BRAND_NON_MAC_RATE) * brnd_gnrc_merged['PRICING_AVG_AWP_brnd']]
            brnd_gnrc_merged['PRICING_AVG_AWP_brnd'] = brnd_gnrc_merged['SOFT_CONST_BENCHMARK_PRICE_brnd'].copy()
            brnd_gnrc_merged.drop(columns = ['MAC1026_UNIT_PRICE_gnrc', 'SOFT_CONST_BENCHMARK_PRICE_brnd'], inplace=True)
            brnd_gnrc_merged['BG_FLAG'] = 'B'
            
            temp_df = temp_df.merge(brnd_gnrc_merged, on=['GPI_NDC', 'CLIENT', 'REGION', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'MEASUREMENT', 'BG_FLAG'], how='left', validate='1:1')
            temp_df['PRICING_AVG_AWP'] = temp_df['PRICING_AVG_AWP_brnd'].fillna(temp_df['PRICING_AVG_AWP'])
            temp_df.drop(columns = ['PRICING_AVG_AWP_brnd'], inplace=True)
        
            assert len(temp_df) == old_len, "Unintentionally adding or dropping rows"
            assert np.abs(temp_df.PRICING_CLAIMS.sum(axis=0) - old_claims) < 0.0001, "Making sure no AWP is dropped"
            del temp_df_filtered, gnrc_group, brnd_group, brnd_gnrc_merged
        
        lp_vol_mv_agg_df = temp_df.copy(deep=True)
        del temp_df
        
        ### drop any prices where U&C conflicts with CostSaver
        ### We should have caught these in preprocessing, but just in case! 
        ### We remove them here, rather than in bounds setting, so we can properly estimate U&C value.
        if p.UNC_OPT and p.INTERCEPTOR_OPT:
            conflict_mask = (lp_vol_mv_agg_df['RAISED_PRICE_UC'] & 
                             ((lp_vol_mv_agg_df['CURRENT_MAC_PRICE'] > lp_vol_mv_agg_df['INTERCEPT_HIGH'])
                              | (lp_vol_mv_agg_df['CURRENT_MAC_PRICE'] < lp_vol_mv_agg_df['INTERCEPT_LOW']))) | (
                            (lp_vol_mv_agg_df['PRICE_CHANGED_UC'] & ~lp_vol_mv_agg_df['RAISED_PRICE_UC'] & 
                             ((lp_vol_mv_agg_df['MAC_PRICE_UPPER_LIMIT_UC'] > lp_vol_mv_agg_df['INTERCEPT_HIGH'])
                               | (lp_vol_mv_agg_df['MAC_PRICE_UPPER_LIMIT_UC'] < lp_vol_mv_agg_df['INTERCEPT_LOW']))))
            print("Erasing", conflict_mask.sum(), "out of", lp_vol_mv_agg_df['PRICE_CHANGED_UC'].sum())
            lp_vol_mv_agg_df.loc[conflict_mask & lp_vol_mv_agg_df['RAISED_PRICE_UC'], 'CURRENT_MAC_PRICE'] = \
                lp_vol_mv_agg_df.loc[conflict_mask & lp_vol_mv_agg_df['RAISED_PRICE_UC'], 'PRE_UC_MAC_PRICE']
            lp_vol_mv_agg_df.loc[conflict_mask, 'PRICE_CHANGED_UC'] = False
            lp_vol_mv_agg_df.loc[conflict_mask, 'RAISED_PRICE_UC'] = False
            lp_vol_mv_agg_df.loc[conflict_mask, 'IS_TWOSTEP_UNC'] = False
            lp_vol_mv_agg_df.loc[conflict_mask, 'IS_MAINTENANCE_UC'] = False
            lp_vol_mv_agg_df.loc[conflict_mask, 'MATCH_VCML'] = False

        ### prices for reg 34 should be non-mutable ######
        lp_vol_mv_agg_df['PRICE_MUTABLE'].where(lp_vol_mv_agg_df.REGION != 'REG_34', 0, inplace = True)
        lp_vol_mv_agg_df['IMMUTABLE_REASON'].where(lp_vol_mv_agg_df.REGION != 'REG_34', 'REG_34_REGION', inplace = True)
        
        #We do not set prices for rural nonpreferred pharmacies
        lp_vol_mv_agg_df['PRICE_MUTABLE'].where(lp_vol_mv_agg_df.CHAIN_GROUP != 'RURAL_NONPREF_OTH', 0, inplace = True)
        lp_vol_mv_agg_df['IMMUTABLE_REASON'].where(lp_vol_mv_agg_df.CHAIN_GROUP != 'RURAL_NONPREF_OTH', 'CHAIN_GROUP_RURAL_NONPREF_OTH', inplace = True)

        ### Prices for levothyroxine should never be changed
        #Note: this should be moved to specialty exclusions
        lp_vol_mv_agg_df['PRICE_MUTABLE'].where(lp_vol_mv_agg_df.GPI.str[0:3] != '281', 0, inplace = True)
        lp_vol_mv_agg_df['IMMUTABLE_REASON'].where(lp_vol_mv_agg_df.GPI.str[0:3] != '281', 'LEVOTHYROXINE_PRICE_FREEZE', inplace = True)

        if p.LIMITED_BO:
            lp_vol_mv_agg_df.loc[~lp_vol_mv_agg_df.REGION.isin(p.BO_LIST), 'PRICE_MUTABLE'] = 0
            lp_vol_mv_agg_df.loc[~lp_vol_mv_agg_df.REGION.isin(p.BO_LIST), 'IMMUTABLE_REASON'] = "BREAKOUT_FREEZE"
        

        # Undo any U&C price changes for prices marked immutable above
        if p.UNC_OPT:
            undo_unc = (lp_vol_mv_agg_df['PRICE_MUTABLE']==0) & (lp_vol_mv_agg_df['PRICE_CHANGED_UC'])
            lp_vol_mv_agg_df.loc[undo_unc & lp_vol_mv_agg_df['RAISED_PRICE_UC'], 'CURRENT_MAC_PRICE'] = \
                lp_vol_mv_agg_df.loc[undo_unc & lp_vol_mv_agg_df['RAISED_PRICE_UC'], 'PRE_UC_MAC_PRICE']
            lp_vol_mv_agg_df.loc[undo_unc, 'PRICE_CHANGED_UC'] = False

        specialty_exclusions = standardize_df(pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.SPECIALTY_EXCLUSION_FILE), dtype = p.VARIABLE_TYPE_DIC))
        qa_dataframe(specialty_exclusions, dataset = 'SPECIALTY_EXCLUSION_FILE_AT_{}'.format(os.path.basename(__file__)))

        specialty_exclusions_gpi_list = specialty_exclusions[['GPI', 'BG_FLAG']]
        lp_vol_mv_agg_df.loc[
            lp_vol_mv_agg_df.set_index(['GPI','BG_FLAG']).index.isin(specialty_exclusions_gpi_list.set_index(['GPI', 'BG_FLAG']).index),'IMMUTABLE_REASON'] = 'SPECIALTY_EXCLUSION'
        
        mac_price_override = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.MAC_PRICE_OVERRIDE_FILE, dtype = p.VARIABLE_TYPE_DIC))
        if p.CLIENT_LOB != 'AETNA' and not p.TRUECOST_CLIENT:
            qa_dataframe(mac_price_override, dataset = 'MAC_PRICE_OVERRIDE_FILE_AT_{}'.format(os.path.basename(__file__)))

        lp_vol_mv_agg_df.loc[:,'PRICE_OVRD_AMT'] = np.NaN

        if p.MAC_PRICE_OVERRIDE:
            ndc_prices = mac_price_override[mac_price_override['NDC'] != '***********']
            ndc_prices = ndc_prices.drop_duplicates(subset = ['CLIENT','REGION','GPI','BG_FLAG']).drop(columns=['VCML_ID','NDC'])
            ndc_prices_gpi_list = ndc_prices.loc[ndc_prices.CLIENT == p.CUSTOMER_ID[0], ['GPI', 'BG_FLAG']]
            lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df.set_index(['GPI', 'BG_FLAG']).index.isin(ndc_prices_gpi_list.set_index(['GPI', 'BG_FLAG']).index),
                                 'IMMUTABLE_REASON' ] = 'NDC_OVERRIDE'
            
            single_vcml = mac_price_override.groupby(['CLIENT','REGION','GPI','BG_FLAG']).agg({'VCML_ID':'count'}).reset_index()
            no_vcml = mac_price_override['VCML_ID'].unique().shape[0]
            single_vcml = single_vcml[single_vcml['VCML_ID'] != no_vcml].drop(columns=['VCML_ID'])
            single_vcml_gpi_list = single_vcml.loc[single_vcml.CLIENT == p.CUSTOMER_ID[0], ['GPI', 'BG_FLAG']]
            lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df.set_index(['GPI', 'BG_FLAG']).index.isin(single_vcml_gpi_list.set_index(['GPI', 'BG_FLAG']).index),
                                'IMMUTABLE_REASON'] = 'GPI_OVERRIDE'
            
            specialty_exclusions = pd.concat([specialty_exclusions, ndc_prices, single_vcml])
            
            if p.PRICE_OVERRIDE: 
                price_override = standardize_df(pd.read_csv(os.path.join(p.FILE_INPUT_PATH, p.PRICE_OVERRIDE_FILE), dtype = p.VARIABLE_TYPE_DIC))
                qa_dataframe(price_override, dataset = 'PRICE_OVERRIDE_FILE_AT_{}'.format(os.path.basename(__file__)))
                price_override_gpi_list = price_override.loc[price_override.CLIENT == p.CUSTOMER_ID[0], ['GPI', 'BG_FLAG']]
                lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df.set_index(['GPI', 'BG_FLAG']).index.isin(price_override_gpi_list.set_index(['GPI', 'BG_FLAG']).index),
                                'IMMUTABLE_REASON'] = 'CUSTOM_PRICE_OVERRIDE'

                mac_price_override = pd.concat([mac_price_override, price_override])
                mac_price_override = mac_price_override.drop_duplicates(subset=['CLIENT','REGION','VCML_ID','GPI','NDC','BG_FLAG','PRICE_OVRD_AMT'])
                overlapped_gpis = mac_price_override[mac_price_override.duplicated(subset=['CLIENT','REGION','VCML_ID','GPI','NDC','BG_FLAG'])]
                if len(overlapped_gpis)!=0:
                    overlapped_gpis.to_csv(os.path.join(p.FILE_OUTPUT_PATH, 'overlapped_gpis_in_overrides' + p.DATA_ID + '.csv'), index=False)
                assert len(mac_price_override[mac_price_override.duplicated(subset=['CLIENT','REGION','VCML_ID','GPI','NDC', 'BG_FLAG'])])==0, "PRICE_OVERRIDE and MAC_PRICE_OVERRIDE have overlapped GPI"
                
            lp_vol_mv_agg_df = price_overrider_function(mac_price_override, lp_vol_mv_agg_df)
            
        else:
            mac_price_override = mac_price_override.drop_duplicates(subset = ['CLIENT','REGION','GPI','BG_FLAG']).drop(columns=['VCML_ID','NDC'])
            mac_price_override_gpi_list = mac_price_override.loc[mac_price_override.CLIENT == p.CUSTOMER_ID[0], ['GPI', 'BG_FLAG']].values 
            lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df.set_index(['GPI', 'BG_FLAG']).index.isin(mac_price_override_gpi_list.set_index(['GPI', 'BG_FLAG']).index),
                                'IMMUTABLE_REASON'] = 'MAC_PRICE_OVERRIDE'

            specialty_exclusions = pd.concat([specialty_exclusions, mac_price_override]) 
            
            if p.PRICE_OVERRIDE:
                price_override = standardize_df(pd.read_csv(os.path.join(p.FILE_INPUT_PATH, p.PRICE_OVERRIDE_FILE), dtype = p.VARIABLE_TYPE_DIC))
                qa_dataframe(price_override, dataset = 'PRICE_OVERRIDE_FILE_AT_{}'.format(os.path.basename(__file__)))
                price_override_gpi_list = price_override.loc[price_override.CLIENT == p.CUSTOMER_ID[0], ['GPI', 'BG_FLAG']]
                lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df.set_index(['GPI', 'BG_FLAG']).index.isin(price_override_gpi_list.set_index(['GPI', 'BG_FLAG']).index),
                                    'IMMUTABLE_REASON'] = 'CUSTOM_PRICE_OVERRIDE'

                lp_vol_mv_agg_df = price_overrider_function(price_override, lp_vol_mv_agg_df)
        if p.UNC_OPT:
            undo_unc = (lp_vol_mv_agg_df['PRICE_MUTABLE']==0) & (lp_vol_mv_agg_df['PRICE_CHANGED_UC'])
            # No price changes here, since the override function took care of it
            lp_vol_mv_agg_df.loc[undo_unc, 'PRICE_CHANGED_UC'] = False
                
        wmt_unc_override = standardize_df(pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.WMT_UNC_PRICE_OVERRIDE_FILE), dtype = p.VARIABLE_TYPE_DIC))

        if len(wmt_unc_override.index)>0:
            qa_dataframe(wmt_unc_override, dataset = 'WMT_UNC_PRICE_OVERRIDE_FILE_AT_{}'.format(os.path.basename(__file__)))
            

        lp_vol_mv_agg_df = pd.merge(lp_vol_mv_agg_df, wmt_unc_override[['CLIENT','GPI_NDC','BG_FLAG','MAC','UNC_OVRD_AMT']], \
                                    how = 'left', on = ['CLIENT','GPI_NDC','BG_FLAG','MAC'])

        if p.HANDLE_INFEASIBLE:
            #tries to read the infeasibility exclusions file. This file contains GPIs confirmed to cause infeasibilities in the LP if included.
            infeasible_exclusions = standardize_df(pd.read_csv(os.path.join(p.FILE_INPUT_PATH, p.INFEASIBLE_EXCLUSION_FILE), dtype = p.VARIABLE_TYPE_DIC))
            if 'BG_FLAG' in infeasible_exclusions.columns: #ADHOC chnage for BG_FLAG
                infeasible_exclusions = infeasible_exclusions.drop(columns = ['BG_FLAG']).drop_duplicates()
            infeasible_exclusions_gnrc = infeasible_exclusions.copy()
            infeasible_exclusions_gnrc['BG_FLAG'] = 'G'
            infeasible_exclusions_brnd = infeasible_exclusions.copy()
            infeasible_exclusions_brnd['BG_FLAG'] = 'B'
            infeasible_exclusions = pd.concat([infeasible_exclusions_gnrc,infeasible_exclusions_brnd])
            infeasible_exclusions_gpi_list = infeasible_exclusions.loc[infeasible_exclusions.CLIENT == p.CUSTOMER_ID[0],[ 'GPI', 'BG_FLAG']]
            lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df.set_index(['GPI']).index.isin(infeasible_exclusions_gpi_list.set_index(['GPI', 'BG_FLAG']).index),
                                    'IMMUTABLE_REASON'] = 'INFEASIBLE_GPI'
            
            specialty_exclusions = pd.concat([specialty_exclusions, infeasible_exclusions])
            
        if p.HANDLE_CONFLICT_GPI and not p.CONFLICT_GPI_AS_TIERS: #ask
            gpi_conflict_exclusions = standardize_df(pd.read_csv(os.path.join(p.FILE_INPUT_PATH,p.CONFLICT_GPI_LIST_FILE), dtype = p.VARIABLE_TYPE_DIC))
            gpi_conflict_exclusions[['CLIENT', 'REGION', 'GPI']].drop_duplicates(inplace=True)
            
            gpi_conflict_exclusions['BG_FLAG'] = 'G'
            gpi_conflict_exclusions_brnd = gpi_conflict_exclusions.copy()
            gpi_conflict_exclusions_brnd['BG_FLAG'] = 'B'
            gpi_conflict_exclusions = pd.concat([gpi_conflict_exclusions,gpi_conflict_exclusions_brnd])
                
            conflict_gpis_to_exclude = gpi_conflict_exclusions.loc[gpi_conflict_exclusions.CLIENT == p.CUSTOMER_ID[0], ['GPI','BG_FLAG']]
            lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df.set_index(['GPI','BG_FLAG']).index.isin(conflict_gpis_to_exclude.set_index(['GPI','BG_FLAG']).index),
                                    'IMMUTABLE_REASON'] = 'CONFLICTING_CONTRAINTS_GPI'
            
            specialty_exclusions = pd.concat([specialty_exclusions, gpi_conflict_exclusions[['CLIENT','REGION','GPI','BG_FLAG']]])

        for client in specialty_exclusions.CLIENT.unique():
            if client == 'ALL':
                gpis_to_exclude = specialty_exclusions.loc[specialty_exclusions.CLIENT == client, ['GPI', 'BG_FLAG']]
                lp_vol_mv_agg_df.loc[
                    lp_vol_mv_agg_df.set_index(['GPI', 'BG_FLAG']).index.isin(gpis_to_exclude.set_index(['GPI', 'BG_FLAG']).index),
                    'PRICE_MUTABLE'
                ] = 0
            else:
                for region in specialty_exclusions.loc[specialty_exclusions.CLIENT == client, 'REGION'].unique():
                    if region == 'ALL':
                        gpis_to_exclude = specialty_exclusions.loc[
                            (specialty_exclusions.CLIENT == client) & (specialty_exclusions.REGION == 'ALL'), 
                            ['GPI', 'BG_FLAG']]
                        lp_vol_mv_agg_df.loc[ (lp_vol_mv_agg_df.CLIENT == client) &
                            (lp_vol_mv_agg_df.set_index(['GPI', 'BG_FLAG']).index.isin(gpis_to_exclude.set_index(['GPI', 'BG_FLAG']).index)),'PRICE_MUTABLE'] = 0
                    else:
                        gpis_to_exclude = specialty_exclusions.loc[
                            (specialty_exclusions.CLIENT == client) & (specialty_exclusions.REGION == region), ['GPI', 'BG_FLAG']]
                        lp_vol_mv_agg_df.loc[
                            (lp_vol_mv_agg_df.CLIENT == client) &
                            (lp_vol_mv_agg_df.REGION == region) &
                            (lp_vol_mv_agg_df.set_index(['GPI', 'BG_FLAG']).index.isin(gpis_to_exclude.set_index(['GPI', 'BG_FLAG']).index)),'PRICE_MUTABLE'] = 0
                        
        if (p.CLIENT_NAME_TABLEAU.startswith('WTW') and not p.REMOVE_WTW_RESTRICTION) or (p.CLIENT_NAME_TABLEAU.startswith('AON')):
            wtw_frezze = lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df['CURRENT_MAC_PRICE'] >= 1.25*lp_vol_mv_agg_df['BEG_PERIOD_PRICE'], ['CLIENT','REGION','GPI', 'BG_FLAG']]
            wtw_frezze['WTW_PRICE_MUTABLE'] = 0
            wtw_frezze.drop_duplicates(inplace=True)
            lp_vol_mv_agg_df_temp = pd.merge(lp_vol_mv_agg_df, wtw_frezze, how='left', on = ['CLIENT','REGION','GPI', 'BG_FLAG'])
            lp_vol_mv_agg_df_temp.loc[lp_vol_mv_agg_df_temp['WTW_PRICE_MUTABLE'] == 0, 'PRICE_MUTABLE'] = 0
            lp_vol_mv_agg_df_temp.loc[lp_vol_mv_agg_df_temp['WTW_PRICE_MUTABLE'] == 0, 'IMMUTABLE_REASON'] = "WTW_AON_PRICE_FREEZE"
            lp_vol_mv_agg_df_temp.drop(columns=['WTW_PRICE_MUTABLE'], inplace=True)
            assert len(lp_vol_mv_agg_df_temp) == len(lp_vol_mv_agg_df), "len(lp_vol_mv_agg_df_temp) == len(lp_vol_mv_agg_df)"
            lp_vol_mv_agg_df = lp_vol_mv_agg_df_temp
        
        lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df.GPI_ONLY == 0, 'PRICE_MUTABLE'] = 0
        lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df.GPI_ONLY == 0, 'IMMUTABLE_REASON'] = 'NDC_PRICING'
        
        if p.INTERCEPTOR_OPT or p.COSTSAVER_CLIENT:
            lp_vol_mv_agg_df.loc[(lp_vol_mv_agg_df.PRICE_MUTABLE == 0) & (lp_vol_mv_agg_df.CURRENT_MAC_PRICE >  lp_vol_mv_agg_df.VENDOR_PRICE) & (lp_vol_mv_agg_df.DESIRE_KEEP_SEND == 1.0), 'VENDOR_CONFLICT'] = True
            lp_vol_mv_agg_df.loc[(lp_vol_mv_agg_df.PRICE_MUTABLE == 0) & (lp_vol_mv_agg_df.CURRENT_MAC_PRICE <=  lp_vol_mv_agg_df.VENDOR_PRICE) & (lp_vol_mv_agg_df.DESIRE_KEEP_SEND == 0.0), 'VENDOR_CONFLICT'] = True
            lp_vol_mv_agg_df.loc[(lp_vol_mv_agg_df.PRICE_MUTABLE == 0) & (lp_vol_mv_agg_df.CURRENT_MAC_PRICE >  lp_vol_mv_agg_df.VENDOR_PRICE) & (lp_vol_mv_agg_df.DESIRE_KEEP_SEND == 1.0), 'EXPECTED_KEEP_SEND'] = 0.0
            lp_vol_mv_agg_df.loc[(lp_vol_mv_agg_df.PRICE_MUTABLE == 0) & (lp_vol_mv_agg_df.CURRENT_MAC_PRICE <= lp_vol_mv_agg_df.VENDOR_PRICE) & (lp_vol_mv_agg_df.DESIRE_KEEP_SEND == 0.0), 'EXPECTED_KEEP_SEND'] = 1.0
        
        
        lp_vol_mv_agg_df.loc[(lp_vol_mv_agg_df.PRICE_MUTABLE == 0), 'INTERCEPT_REASON'] = 'PRICE_FREEZE'
        
        immutable_report = lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df.PRICE_MUTABLE == 0, ['GPI','NDC','BG_FLAG','MAC_LIST','PRICE_MUTABLE','IMMUTABLE_REASON']].drop_duplicates()
        immutable_report.to_csv(p.FILE_OUTPUT_PATH + p.DATA_ID +'_immutable_gpi_report.csv', index=False)
                

        # Because U&C module does not include a full set of exclusions,
        # we undo any U&C-related changes here.
        if p.UNC_OPT:
            undo_unc = (lp_vol_mv_agg_df['PRICE_MUTABLE']==0) & (lp_vol_mv_agg_df['PRICE_CHANGED_UC'])
            lp_vol_mv_agg_df.loc[undo_unc & lp_vol_mv_agg_df['RAISED_PRICE_UC'], 'CURRENT_MAC_PRICE'] = \
                lp_vol_mv_agg_df.loc[undo_unc & lp_vol_mv_agg_df['RAISED_PRICE_UC'], 'PRE_UC_MAC_PRICE']
            lp_vol_mv_agg_df.loc[undo_unc, 'PRICE_CHANGED_UC'] = False

        # Flag cleanup
        if p.UNC_OPT:
            lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df['PRICE_MUTABLE']==0, 'RAISED_PRICE_UC'] = False
            lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df['PRICE_MUTABLE']==0, 'IS_TWOSTEP_UNC'] = False
            lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df['PRICE_MUTABLE']==0, 'IS_MAINTENANCE_UC'] = False
            lp_vol_mv_agg_df.loc[lp_vol_mv_agg_df['PRICE_MUTABLE']==0, 'MATCH_VCML'] = False
        
        LOWER_SCALE_FACTOR = p.GPI_LOW_FAC
        UPPER_SCALE_FACTOR = p.GPI_UP_FAC

        # generate pricing bounds based on the scale factor
        price_lambdas = []
        if month in p.LP_RUN:
            ###################################################################################
            ###### Define Linear Optimization problem #########################################
            ###################################################################################
            logger.info('--------------------')
            logger.info('Start Defining LP')

            lp_vol_mv_agg_df['GPI_CHANGE_EXCEPT'] = 0

            if p.STRENGTH_PRICE_CHANGE_EXCEPTION and (month <= 6):
                pricing_viol = pd.read_csv(os.path.join(p.FILE_INPUT_PATH, 'const_price_viol_unique.csv'), dtype = p.VARIABLE_TYPE_DIC)
                pricing_viol = standardize_df(pricing_viol)
                qa_dataframe(pricing_viol, dataset = 'const_price_viol_unique_AT_{}'.format(os.path.basename(__file__)))
                pricing_viol.GPI = '00' + pricing_viol.GPI.astype(str)
                pricing_viol.GPI = pricing_viol.GPI.str[-12:]
                pricing_gpis = pricing_viol.GPI.unique()
                lp_vol_mv_agg_df.loc[(lp_vol_mv_agg_df.GPI.str[0:12].isin(pricing_gpis)) &
                                     (lp_vol_mv_agg_df.CLIENT == 'SSI'), 'GPI_CHANGE_EXCEPT'] = 1
            if p.FORCE_FLOOR:
                lp_vol_mv_agg_df['CLIENT_REGION'] = lp_vol_mv_agg_df['CLIENT']+'_'+lp_vol_mv_agg_df['REGION']
                has_pref_oth = lp_vol_mv_agg_df[lp_vol_mv_agg_df['CHAIN_GROUP']=='PREF_OTHER']['CLIENT_REGION'].unique()
                lp_vol_mv_agg_df['HAS_PREF_OTH'] = lp_vol_mv_agg_df['CLIENT_REGION'].isin(has_pref_oth)
                has_cvssp = lp_vol_mv_agg_df[lp_vol_mv_agg_df['CHAIN_SUBGROUP']=='CVSSP']['CLIENT_REGION'].unique()
                lp_vol_mv_agg_df['HAS_CVSSP'] = lp_vol_mv_agg_df['CLIENT_REGION'].isin(has_cvssp)
                # logic here: it's in the given list; or the client has PREF_OTH and PREF_OTH is in the floor pharmacy list
                # and this is a CVS subgroup that must maintain state parity with PREF OTH; 
                # or the client doesn't have PREF_OTH but it does have NONPREF_OTH, and NONPREF_OTH is in the floor
                # pharmacy list, and this is a CVS subgroup that must maintain parity with NONPREF_OTH.
                match_independent = (
                    (lp_vol_mv_agg_df['HAS_PREF_OTH'] & ('PREF_OTH' in p.FORCE_FLOOR_PHARMACY_SUBGROUP_LIST))
                    | (~lp_vol_mv_agg_df['HAS_PREF_OTH'] & ('NONPREF_OTH' in p.FORCE_FLOOR_PHARMACY_SUBGROUP_LIST))
                )
                force_floor_vcmls = lp_vol_mv_agg_df[lp_vol_mv_agg_df['CHAIN_GROUP'].isin(p.FORCE_FLOOR_PHARMACY_SUBGROUP_LIST)
                                                     | ((lp_vol_mv_agg_df['CHAIN_SUBGROUP']=='CVSSP') & match_independent )
                                                     | ((lp_vol_mv_agg_df['CHAIN_SUBGROUP']=='CVS') & match_independent 
                                                        & ~lp_vol_mv_agg_df['HAS_CVSSP'])
                                                    ]['MAC_LIST'].unique()
                lp_vol_mv_agg_df['FLOOR_VCML'] = lp_vol_mv_agg_df['MAC_LIST'].isin(force_floor_vcmls)
                lp_vol_mv_agg_df['MATCH_CVSSP'] = lp_vol_mv_agg_df['HAS_CVSSP'] & (match_independent | ('CVSSP' in p.FORCE_FLOOR_PHARMACY_SUBGROUP_LIST))
                lp_vol_mv_agg_df.drop(columns=['CLIENT_REGION', 'HAS_PREF_OTH', 'HAS_CVSSP'], inplace=True)

            pricing_cols = ['CLIENT','REGION', 'BREAKOUT','MEASUREMENT','CHAIN_GROUP','CHAIN_SUBGROUP','GPI_NDC', 'BG_FLAG',
                            'MAC_PRICE_UNIT_ADJ', 'MAC1026_UNIT_PRICE', 'AVG_AWP', 'PRICE_MUTABLE',
                            'PRICING_UC_UNIT', 'PRICE_REIMB_CLAIM', 'CLAIMS', 'CLAIMS_PROJ_EOY', 'QTY', 'QTY_PROJ_EOY',
                            'GPI_CHANGE_EXCEPT', 'CLIENT_MAX_PRICE', 'CLIENT_MIN_PRICE', 'PRICE_TIER', 'BREAKOUT_AWP_MAX',
                            'ZBD_UPPER_LIMIT', 'GOODRX_UPPER_LIMIT', 'INTERCEPT_HIGH', 'INTERCEPT_LOW',
                            'PRICING_QTY_PROJ_EOY','PRICING_CLAIMS_PROJ_EOY', 'PRICING_CLAIMS', 'PRICING_QTY', 
                            'PRICING_PRICE_REIMB_CLAIM','PRICING_AVG_AWP','UNC_OVRD_AMT','PRICE_OVRD_AMT',
                            'NON_MAC_RATE', 'IS_MAC', 'IN_NONCAP_OK_VCML','HAS_PHARM_GUARANTEE'
                           ]

            if p.UNC_OPT:
                pricing_cols += ['PRICE_CHANGED_UC','MAC_PRICE_UPPER_LIMIT_UC','RAISED_PRICE_UC','IS_TWOSTEP_UNC','CURRENT_MAC_PRICE','IS_MAINTENANCE_UC']

            if p.PHARMACY_EXCLUSION:
                pricing_cols += ['MAC_LIST']

            if p.CLIENT_NAME_TABLEAU.startswith(('WTW','AON')):
                pricing_cols += ['BEG_PERIOD_PRICE']
            
            #COSTSAVER FLAG: Not removing this as we would surely need this
            if p.INTERCEPTOR_OPT or p.COSTSAVER_CLIENT:
                lp_vol_mv_agg_df = add_target_ingcost(lp_vol_mv_agg_df, client_guarantees, client_rate_col = 'RATE', target_cols=['TARG_INGCOST_ADJ_ZBD_FRAC', 'PHARM_TARG_INGCOST_ADJ_ZBD_FRAC'])
                
                pricing_cols += ['MAC_LIST','CURRENT_KEEP_SEND','EXPECTED_KEEP_SEND','VENDOR_PRICE',\
                                 'CLAIMS_ZBD_FRAC','QTY_ZBD_FRAC','FULLAWP_ADJ_ZBD_FRAC', 'TARG_INGCOST_ADJ_ZBD_FRAC',\
                                 'PHARM_CLAIMS_ZBD_FRAC','PHARM_QTY_ZBD_FRAC','PHARM_FULLAWP_ADJ_ZBD_FRAC','PHARM_TARG_INGCOST_ADJ_ZBD_FRAC']
                
            if not p.INTERCEPTOR_OPT:
                lp_vol_mv_agg_df.loc[:, 'INTERCEPT_HIGH'] = 10000*lp_vol_mv_agg_df['MAC_PRICE_UNIT_ADJ']
                lp_vol_mv_agg_df.loc[:, 'INTERCEPT_LOW'] = 0        

            if p.FORCE_FLOOR:
                pricing_cols += ['FLOOR_VCML', 'MATCH_CVSSP']
             
            # to prevent duplicates in pricing columns list
            pricing_cols = list(set(pricing_cols))
                

            logger.info('--------------------')
            logger.info('Start GPI Level Price Bounds')
            
            print(lp_vol_mv_agg_df.PRICE_TIER.unique())

            if p.HANDLE_CONFLICT_GPI and p.CONFLICT_GPI_AS_TIERS:
                temp_all=pd.DataFrame()
                for pt in lp_vol_mv_agg_df.PRICE_TIER.unique():
                   
                    temp = lp_vol_mv_agg_df[lp_vol_mv_agg_df.PRICE_TIER==pt].copy()
                    temp[['Price_Bounds', 'lb_name', 'ub_name','lower_bound_ordered','upper_bound_ordered']] = generate_price_bounds(LOWER_SCALE_FACTOR, UPPER_SCALE_FACTOR, temp[pricing_cols]).values
                    temp_all = pd.concat([temp_all,temp],ignore_index=True)
                lp_vol_mv_agg_df = temp_all.copy()
            
            else:
                lp_vol_mv_agg_df[['Price_Bounds', 'lb_name', 'ub_name','lower_bound_ordered','upper_bound_ordered']] = \
                    generate_price_bounds(LOWER_SCALE_FACTOR, UPPER_SCALE_FACTOR, lp_vol_mv_agg_df[pricing_cols])     
                         
                
            logger.info('End GPI Level Price Bounds')
            lp_vol_mv_agg_df['lb_ub'] = (lp_vol_mv_agg_df
                                          .Price_Bounds
                                          .map(lambda x: x[0] > x[1])
                                          .astype(int))

            if lp_vol_mv_agg_df['lb_ub'].sum() == 0:
                logger.info("No issues with pricing constraints - Lower bound less than Upper bound")
            else:
                logger.info("Check pricing bounds")
            logger.info('--------------------')

            lp_data_df = lp_vol_mv_agg_df#.copy()
            lp_data_df['lb'], lp_data_df['ub'] = zip(*lp_data_df['Price_Bounds'])

            # Soft price constraints are applied to all drugs
            # Rest index to ensure index values are contiguous range of numbers from 0 to len(lp_data_df) and we also add that index as a new column into lp_data_df
            lp_data_df.reset_index(inplace=True) 

            # Changing the name of this column. This new column is meant to uniquely identify each entry in lp_data_df
            lp_data_df.rename(columns={'index':'Dec_Var_ID'},inplace=True)
            columns = lp_data_df.columns.values.tolist()
            # Moving the 'Dec_Var_ID' column to the end of dataframe so it would maintain the previous order of the main columns in TOTAL_OUTPUT table (CLIENT, REGION, ...) and align this column position
            # with the  'Price_Decision_Var' and 'Dec_Var_Name'.
            columns = columns[1:] + columns[:1]
            lp_data_df = lp_data_df[columns]

            lp_data_df.to_csv(os.path.join(p.FILE_OUTPUT_PATH, 'TOTAL_OUTPUT_DEBUG_' + p.DATA_ID + '.csv'), index=False)
            lp_data_df.drop(columns=['lb', 'ub'],inplace=True)

            price_lambdas = lp_data_df.loc[(lp_data_df.PRICE_MUTABLE==1)][['Dec_Var_ID']].reset_index(drop=True)
            price_lambdas['LAMBDA_OVER'] = price_lambdas['Dec_Var_ID'].apply(lambda x: pulp.LpVariable('Price_' + str(x) + '_lambda_over', lowBound=0))
            price_lambdas['LAMBDA_UNDER'] = price_lambdas['Dec_Var_ID'].apply(lambda x: pulp.LpVariable('Price_' + str(x) + '_lambda_under', lowBound=0))


        # output files
        with open(eoy_days_out, 'wb') as f:
            pickle.dump(eoy_days, f)
        with open(proj_days_out, 'wb') as f:
            pickle.dump(proj_days, f)
        with open(lp_vol_mv_agg_df_out, 'wb') as f:
            pickle.dump(lp_vol_mv_agg_df, f)
        with open(mac_list_df_out, 'wb') as f:
            pickle.dump(mac_list_df, f)
        with open(chain_region_mac_mapping_out, 'wb') as f:
            pickle.dump(chain_region_mac_mapping, f)
        with open(lp_vol_mv_agg_df_actual_out, 'wb') as f:
            pickle.dump(lp_vol_mv_agg_df_actual, f)
        with open(client_list_out, 'wb') as f:
            pickle.dump(client_list, f)
        with open(breakout_df_out, 'wb') as f:
            pickle.dump(breakout_df, f)
        with open(client_guarantees_out, 'wb') as f:
            pickle.dump(client_guarantees, f)
        with open(pharmacy_guarantees_out, 'wb') as f:
            pickle.dump(pharmacy_guarantees, f)
        with open(pref_pharm_list_out, 'wb') as f:
            pickle.dump(pref_pharm_list, f)
        with open(oc_pharm_surplus_out, 'wb') as f:
            pickle.dump(oc_pharm_surplus, f)
        with open(other_client_pharm_lageoy_out, 'wb') as f:
            pickle.dump(other_client_pharm_lageoy, f)
        with open(oc_eoy_pharm_perf_out, 'wb') as f:
            pickle.dump(oc_eoy_pharm_perf, f)
        with open(generic_launch_df_out, 'wb') as f:
            pickle.dump(generic_launch_df, f)
        with open(oc_pharm_dummy_out, 'wb') as f:
            pickle.dump(oc_pharm_dummy, f)
        with open(dummy_perf_dict_out, 'wb') as f:
            pickle.dump(dummy_perf_dict, f)
        with open(perf_dict_col_out, 'wb') as f:
            pickle.dump(perf_dict_col, f)
        with open(gen_launch_eoy_dict_out, 'wb') as f:
            pickle.dump(gen_launch_eoy_dict, f)
        with open(gen_launch_lageoy_dict_out, 'wb') as f:
            pickle.dump(gen_launch_lageoy_dict, f)
        with open(ytd_perf_pharm_actuals_dict_out, 'wb') as f:
            pickle.dump(ytd_perf_pharm_actuals_dict, f)
        with open(performance_dict_out, 'wb') as f:
            pickle.dump(performance_dict, f)
        with open(act_performance_dict_out, 'wb') as f:
            pickle.dump(act_performance_dict, f)
        with open(lp_data_df_out, 'wb') as f:
            pickle.dump(lp_data_df, f)
        with open(price_lambdas_out, 'wb') as f:
            pickle.dump(price_lambdas, f)
        with open(brand_surplus_ytd_out, 'wb') as f:
            pickle.dump(brand_surplus_ytd_dict,f)
        with open(brand_surplus_lag_out, 'wb') as f:
            pickle.dump(brand_surplus_lag_dict,f)
        with open(brand_surplus_eoy_out, 'wb') as f:
            pickle.dump(brand_surplus_eoy_dict,f)
        with open(specialty_surplus_ytd_out, 'wb') as f:
            pickle.dump(specialty_surplus_ytd_dict,f)
        with open(specialty_surplus_lag_out, 'wb') as f:
            pickle.dump(specialty_surplus_lag_dict,f)
        with open(specialty_surplus_eoy_out, 'wb') as f:
            pickle.dump(specialty_surplus_eoy_dict,f)
        with open(disp_fee_surplus_ytd_out, 'wb') as f:
            pickle.dump(disp_fee_surplus_ytd_dict,f)
        with open(disp_fee_surplus_lag_out, 'wb') as f:
            pickle.dump(disp_fee_surplus_lag_dict,f)
        with open(disp_fee_surplus_eoy_out, 'wb') as f:
            pickle.dump(disp_fee_surplus_eoy_dict,f)
        return (lag_price_col, pharm_lag_price_col)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'Opt Preprocessing', repr(e), error_loc)
        raise e
######## END  Preprocessing ################################################################


########################### Consistent Strength Pricing constraints #########################
def consistent_strength_pricing_constraints(
    params_file_in: str,
    unc_flag: bool,
    lp_data_df_in: InputPath('pickle'),
    price_lambdas_in: InputPath('pickle'),
    lp_data_df_out: OutputPath('pickle'),
    t_cost_out: OutputPath('pickle'),
    cons_strength_cons_out: OutputPath('pickle'),
    loglevel: str = 'INFO'
    # kube_run: bool = True,
):
    import sys
    import os
    sys.path.append('/')
    import time
    import logging
    import pickle
    import pandas as pd
    import pulp
    import util_funcs as uf
    import BQ
    import duckdb
    from google.cloud import bigquery

    uf.write_params(params_file_in)
    import CPMO_parameters as p
    from CPMO_lp_functions import generatePricingDecisionVariables
    from CPMO_shared_functions import update_run_status
    update_run_status(i_error_type='Started building constraints') 
    try:
        # file inputs
        ############## NO LP MOd Changes segment- same as original code################
        with open(lp_data_df_in, 'rb') as f:
            lp_data_df = pickle.load(f)
        with open(price_lambdas_in, 'rb') as f:
            price_lambdas_df = pickle.load(f)

        out_path = os.path.join(p.FILE_LOG_PATH, 'ClientPharmacyMacOptimization.log')
        logger = uf.log_setup(log_file_path=out_path, loglevel=loglevel)
        logger.info('--------------------')
        logger.info("Starting building consistent strength pricing constraints")

        start = time.time()
        t_cost = []
        
        pharmacy_guarantees = pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH,p.PHARM_GUARANTEE_FILE), dtype = p.VARIABLE_TYPE_DIC)
        
        
        zero_num_pair_df=lp_data_df[(lp_data_df.PRICE_MUTABLE==1)&(lp_data_df.PRICING_QTY_PROJ_EOY == 0)].reset_index(drop = True)
        zero_num_pair_lambdas_df= pd.merge(zero_num_pair_df, price_lambdas_df,on = 'Dec_Var_ID', how = 'left')

        if p.ZERO_QTY_TIGHT_BOUNDS:
            zero_num_pair_lambdas_df['lambda_over_constraint'],zero_num_pair_lambdas_df['lambda_under_constraint']=\
            zero_num_pair_lambdas_df.LAMBDA_OVER*p.ZERO_QTY_WEIGHT,zero_num_pair_lambdas_df.LAMBDA_UNDER*p.ZERO_QTY_WEIGHT
        else:
            zero_num_pair_lambdas_df['lambda_over_constraint'],zero_num_pair_lambdas_df['lambda_under_constraint']=zero_num_pair_lambdas_df.LAMBDA_OVER*0.01,zero_num_pair_lambdas_df.LAMBDA_UNDER*0.01

        if p.HANDLE_CONFLICT_GPI and p.CONFLICT_GPI_AS_TIERS:
        ##df2 on which we do the constraint creation calcs
            num_pair_conflict_df=lp_data_df[(lp_data_df.PRICING_QTY_PROJ_EOY != 0) & (lp_data_df.PRICE_TIER=='CONFLICT') & (lp_data_df.PRICE_MUTABLE==1)].reset_index(drop = True)
            num_pair_conflict_lambdas_df= pd.merge(num_pair_conflict_df, price_lambdas_df,on = 'Dec_Var_ID', how = 'left')
            num_pair_conflict_lambdas_df['lambda_over_constraint'],num_pair_conflict_lambdas_df['lambda_under_constraint']=num_pair_conflict_lambdas_df.LAMBDA_OVER*10, num_pair_conflict_lambdas_df.LAMBDA_UNDER*10

        ##df3 on which we do the constraint creation calcs
            num_pair_no_conflict_df=lp_data_df[(lp_data_df.PRICING_QTY_PROJ_EOY!= 0) & (lp_data_df.PRICE_TIER!='CONFLICT')& (lp_data_df.PRICE_MUTABLE==1)].reset_index(drop = True)
            num_pair_no_conflict_lambdas_df= pd.merge(num_pair_no_conflict_df, price_lambdas_df,on = 'Dec_Var_ID', how = 'left')
            num_pair_no_conflict_lambdas_df['lambda_over_constraint'],num_pair_no_conflict_lambdas_df['lambda_under_constraint']=num_pair_no_conflict_lambdas_df.LAMBDA_OVER*0.01, num_pair_no_conflict_lambdas_df.LAMBDA_UNDER*0.01
        #first dataframe creation for inputting contraints into Gurobi
        #it's the concatenation of df1,df2,df3 and df4 created above--For Input to Gurobi!
            price_lambda_under_over_df=pd.concat([zero_num_pair_lambdas_df,num_pair_conflict_lambdas_df,num_pair_no_conflict_lambdas_df])

        else:
        ##df4 on which we do the constraint creation calcs
            num_pair_df = lp_data_df[(lp_data_df.PRICING_QTY_PROJ_EOY != 0)& (lp_data_df.PRICE_MUTABLE==1)].reset_index(drop = True)            
            num_pair_lambdas_df= pd.merge(num_pair_df, price_lambdas_df,on = 'Dec_Var_ID', how = 'left')
            num_pair_lambdas_df['lambda_over_constraint'],num_pair_lambdas_df['lambda_under_constraint']=num_pair_lambdas_df.LAMBDA_OVER*0.01, num_pair_lambdas_df.LAMBDA_UNDER*0.01
            #first dataframe creation for inputting contraints into Gurobi
            #it's the concatenation of df1,df2,df3 and df4 created above--For Input to Gurobi!
            price_lambda_under_over_df=pd.concat([zero_num_pair_lambdas_df,num_pair_lambdas_df])
        
        lambda_over_list=price_lambda_under_over_df['lambda_over_constraint'].tolist()
        lambda_under_list=price_lambda_under_over_df['lambda_under_constraint'].tolist()
        t_cost=[item for pair in zip(lambda_over_list, lambda_under_list + [0]) for item in pair] #preparation for pulp. Can be removed upon migration to Gurobi!
        
        #############****************PART ONE: Preparation of t_cost list ends##########
        
        #############****************PART TWO: Preparation of SV/Decision Variable Constraints- 

        logger.info('Generating Decision Variables')
        logger.info('--------------------')
        
         #The main complexity # 1 in this part is to vectorize for loops that compute constraint variables in the source code
        #The main complexity #2 in this part of the code is to compute special constraint variables for GPI pairs- where a contraint is created
        ## if an additional row exists for a GPI_12 with a different GPI. For example- 11000401003 is GPI_12 for which GPIs 1100040100310 and 1100040100315 both exist.
        ##if only one of the GPIs in the above example exist- like only 1100040100310 or 1100040100315- then the constraint is not computed! this can be achieved by using 
        #df.shift() functionality of pandas!
        ##The 3rd complexity here is that from a Gurobi perspective- this constraint is at a different grain from the Lambda_Under and Lambda_Over Price contraints. Nothing to be done on this front- just
        ## informational. 

        #addition of 4 more columns to lp_data_df- GPI12 ,GPI_Strength, LOW_B and UP_B- these will be used in computations below!
        lp_data_df = generatePricingDecisionVariables(lp_data_df)
        lp_data_df['GPI_12'],lp_data_df['GPI_Strength'] = lp_data_df.GPI.str[0:12],lp_data_df.GPI.str[12:]
        lp_data_df['LOW_B'],lp_data_df['UP_B'] = zip(*lp_data_df.Price_Bounds)

        print('Printing query results')
        ##duckdb usage started- to sort lp_data_df by FULLAWP_ADJ_PROJ_EOY and hace easy SQL like dialect to fetch values for previous row GPIs for constraint creation
        rep_gpi_df=duckdb.query('select * from\
                       (select CLIENT, BREAKOUT,MEASUREMENT,GPI, NDC, GPI_NDC, BG_FLAG, GPI_ONLY,\
                       CHAIN_GROUP, CHAIN_SUBGROUP, REGION, PHARMACY_TYPE,\
                       Price_Bounds,MAC_PRICE_UNIT_ADJ,GPI_12, GPI_Strength,\
                       MAC1026_UNIT_PRICE, CURRENT_MAC_PRICE,Price_Decision_Var,Dec_Var_Name,LOW_B,UP_B,\
                       rank() over (partition by CLIENT, BREAKOUT,REGION, MEASUREMENT, CHAIN_GROUP, CHAIN_SUBGROUP, GPI, BG_FLAG\
                       order by FULLAWP_ADJ_PROJ_EOY desc) as FULLAWP_ADJ_PROJ_EOY_rank from lp_data_df where PRICE_MUTABLE=1) where FULLAWP_ADJ_PROJ_EOY_rank=1').to_df()

        ##price_constraints_df- dataframe on which constraint creation will start. It is a subset(both in terms of rows and columns) of lp_data_df
        price_constraints_df=duckdb.query("""select * from rep_gpi_df where MAC1026_UNIT_PRICE > 0 and MEASUREMENT != 'M30'""")


        ##creation of gpi_sorted_df: to create count_window/count number of rows from price_constraints_df for a given partition of GPI_12,CLIENT, BREAKOUT,REGION, MEASUREMENT, CHAIN_SUBGROUP. 
        gpi_sorted_df=duckdb.query("select *, rank() OVER (PARTITION BY GPI_12,BG_FLAG,CLIENT, BREAKOUT,REGION, MEASUREMENT, CHAIN_SUBGROUP ORDER BY MAC1026_UNIT_PRICE asc,GPI_Strength\
                               asc) as rank_mac1026_GPI_Strength from (select GPI_12,BG_FLAG,CLIENT, BREAKOUT,REGION, MEASUREMENT, CHAIN_SUBGROUP, MAC1026_UNIT_PRICE, GPI_Strength,\
                               GPI, NDC, GPI_NDC, GPI_ONLY,CHAIN_GROUP, PHARMACY_TYPE,Price_Bounds,LOW_B,UP_B,MAC_PRICE_UNIT_ADJ,GPI_Strength, MAC1026_UNIT_PRICE, CURRENT_MAC_PRICE,Price_Decision_Var,Dec_Var_Name,\
                               count() OVER (PARTITION BY GPI_12,BG_FLAG, CLIENT, BREAKOUT,REGION, MEASUREMENT, CHAIN_SUBGROUP) as count_window\
                               from price_constraints_df)").to_df()


        ##mutation of gpi_sorted_df: keep only those rows from gpi_sorted_df that have more than 1 rows for a given partition of GPI_12,CLIENT, BREAKOUT,REGION, MEASUREMENT, CHAIN_SUBGROUP
        ##this filtering is done to fetch the previous row's GPI and Price_Decision_Var value to create constraints
        gpi_sorted_df=duckdb.query("select *,concat_ws('_', 'SV', REGION, MEASUREMENT, CHAIN_SUBGROUP, GPI, BG_FLAG, PREV_ROW_GPI) as SV_DEC_VAR from(select *,\
                               lag(GPI) OVER (PARTITION BY GPI_12,BG_FLAG, CLIENT, BREAKOUT,REGION, MEASUREMENT, CHAIN_SUBGROUP ORDER BY MAC1026_UNIT_PRICE asc, GPI_Strength asc) as PREV_ROW_GPI,\
                               lag(Price_Decision_Var) OVER (PARTITION BY GPI_12,BG_FLAG, CLIENT, BREAKOUT,REGION, MEASUREMENT, CHAIN_SUBGROUP ORDER BY MAC1026_UNIT_PRICE asc,GPI_Strength asc) as PREV_ROW_PRICE_DECISION_VAR,\
                               lag(LOW_B) OVER (PARTITION BY GPI_12,BG_FLAG, CLIENT, BREAKOUT,REGION, MEASUREMENT, CHAIN_SUBGROUP ORDER BY MAC1026_UNIT_PRICE asc, GPI_Strength asc) as PREV_ROW_LOW_B,\
                               lag(UP_B) OVER (PARTITION BY GPI_12,BG_FLAG, CLIENT, BREAKOUT,REGION, MEASUREMENT, CHAIN_SUBGROUP ORDER BY MAC1026_UNIT_PRICE asc,GPI_Strength asc) as PREV_ROW_UP_B\
                               from gpi_sorted_df where count_window>1) where PREV_ROW_GPI is not null").to_df()

        ##LP mod change: converted the SV_DEC_VAR created above into a pulp variable to further create the pulp affine expression
        #preparation for pulp. Can be removed upon migration to Gurobi!
        SV_DEC_VAR_PULP_LIST,PREV_ROW_PRICE_DECISION_VAR_PULP_LIST,Price_Decision_Var_PULP_LIST=pd.Series([pulp.LpVariable(pv,lowBound=0) for pv in gpi_sorted_df.SV_DEC_VAR]).tolist(),\
                                                                                                pd.Series([pulp.LpVariable(pv,lowBound=gpi_sorted_df.PREV_ROW_LOW_B[i], upBound=gpi_sorted_df.PREV_ROW_UP_B[i]) for i, pv in enumerate(gpi_sorted_df.PREV_ROW_PRICE_DECISION_VAR)]).tolist(),\
                                                                                                pd.Series([pulp.LpVariable(pv,lowBound=gpi_sorted_df.LOW_B[i], upBound=gpi_sorted_df.UP_B[i]) for i, pv in enumerate(gpi_sorted_df.Price_Decision_Var)]).tolist()
    
        ##LP mod change: added the dec_var_name_sv_pulp as a column to the gpi_sorted_df dataframe- this is because we can't use dec_var_name_sv as it is string. We 
        ##had to convert it into a pulp variable for constraint creations and then add a separate column dec_var_name_sv_pulp to gpi_sorted_df
        gpi_sorted_df['SV_DEC_VAR_PULP'],gpi_sorted_df['PREV_ROW_PRICE_DECISION_VAR_PULP'],gpi_sorted_df['Price_Decision_Var_PULP']=\
        SV_DEC_VAR_PULP_LIST,PREV_ROW_PRICE_DECISION_VAR_PULP_LIST,Price_Decision_Var_PULP_LIST

        ##LP mod change: added a contraint per the original code-- for pulp export!-- can be removed during gurobi migration!
        gpi_sorted_df['SV_DEC_VAR_CONSTRAINT']=gpi_sorted_df['SV_DEC_VAR_PULP']*10000
        gpi_sorted_df['PRICE_DECISION_CONSTRAINT']= gpi_sorted_df['PREV_ROW_PRICE_DECISION_VAR_PULP']-gpi_sorted_df['Price_Decision_Var_PULP']-gpi_sorted_df['SV_DEC_VAR_PULP']
        
        dec_var_name_sv_list=gpi_sorted_df['SV_DEC_VAR_CONSTRAINT'].tolist() #preparation for pulp. Can be removed upon migration to Gurobi!
        t_cost=t_cost+dec_var_name_sv_list #preparation for pulp. Can be removed upon migration to Gurobi!
        cons_strength_cons=gpi_sorted_df['PRICE_DECISION_CONSTRAINT'].tolist() #preparation for pulp. Can be removed upon migration to Gurobi!

        
        logger.info("Ending building consistent strength pricing constraints")
        end = time.time()
        logger.info("Run time: %f mins", (end - start)/60.)
        logger.info('--------------------')
        
        #############****************PART TWO: Preparation of SV/Decision Variable Constraints- ENDS***********
        # file outputs
        with open(lp_data_df_out, 'wb') as f:
            pickle.dump(lp_data_df, f)
        with open(t_cost_out, 'wb') as f:
            pickle.dump(t_cost, f) #preparation for pulp. Can be removed upon migration to Gurobi!. Need to be replaced with price_lambda_under_over_df for gurobi migration
        with open(cons_strength_cons_out, 'wb') as f:
            pickle.dump(cons_strength_cons, f) #preparation for pulp. Can be removed upon migration to Gurobi!. Need to be replaced with price_lambda_under_over_df for gurobi migration

        return t_cost, cons_strength_cons
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'Consistent Strength Pricing Constraints', repr(e), error_loc)
        raise e
########################### End Consistent Strength Pricing constraints #########################



###### 2. Client level Contraints ###########
# this constraint function itself does not need vectorization but invokes generateGuaranteeConstraintEbit in the CPMO_LP file. That function has been fully optimized
def client_level_constraints(
    params_file_in: str,
    unc_flag: bool,
    lp_data_df_in: InputPath('pickle'),
    client_guarantees_in: InputPath('pickle'),
    pharmacy_guarantees_in: InputPath('pickle'),
    performance_dict_in: InputPath('pickle'),
    breakout_df_in: InputPath('pickle'),
    client_list_in: InputPath('pickle'),
    oc_eoy_pharm_perf_in: InputPath('pickle'),
    gen_launch_eoy_dict_in: InputPath('pickle'),
    brand_surplus_eoy_in: InputPath('pickle'),
    specialty_surplus_eoy_in: InputPath('pickle'),
    disp_fee_surplus_eoy_in: InputPath('pickle'),
    price_lambdas_in: InputPath('pickle'),
    # total_pharm_list_in: InputPath('pickle'),
    # agreement_pharmacy_list_in: InputPath('pickle'),
    lambda_df_out: OutputPath('pickle'),
    client_constraint_list_out: OutputPath('pickle'),
    client_constraint_target_out: OutputPath('pickle'),
    loglevel: str = 'INFO'
    # kube_run: bool = True,
):

    import sys
    import os
    import logging
    sys.path.append('/')
    import pickle
    import pandas as pd
    import pulp
    import util_funcs as uf
    import BQ
    from google.cloud import bigquery
    

    uf.write_params(params_file_in)
    import CPMO_parameters as p
    from CPMO_lp_functions import generateGuaranteeConstraintEbit, generatePricingDecisionVariables, generateLambdaDecisionVariables_ebit
    from CPMO_shared_functions import update_run_status
    try:
        out_path = os.path.join(p.FILE_LOG_PATH, 'ClientPharmacyMacOptimization.log')
        logger = uf.log_setup(log_file_path=out_path, loglevel=loglevel)

        # file inputs
        with open(lp_data_df_in, 'rb') as f:
            lp_data_df = pickle.load(f)
        with open(client_guarantees_in, 'rb') as f:
            client_guarantees = pickle.load(f)
        with open(pharmacy_guarantees_in, 'rb') as f:
            pharmacy_guarantees = pickle.load(f)
        with open(performance_dict_in, 'rb') as f:
            performance_dict = pickle.load(f)
        with open(breakout_df_in, 'rb') as f:
            breakout_df = pickle.load(f)
        with open(client_list_in, 'rb') as f:
            client_list = pickle.load(f)
        with open(oc_eoy_pharm_perf_in, 'rb') as f:
            oc_eoy_pharm_perf = pickle.load(f)
        with open(gen_launch_eoy_dict_in, 'rb') as f:
            gen_launch_eoy_dict = pickle.load(f)
        with open(price_lambdas_in, 'rb') as f:
            price_lambdas_df = pickle.load(f)
        with open(brand_surplus_eoy_in,'rb') as f:
            brand_surplus_eoy_dict = pickle.load(f)
        with open(specialty_surplus_eoy_in,'rb') as f:
            specialty_surplus_eoy_dict = pickle.load(f)
        with open(disp_fee_surplus_eoy_in,'rb') as f:
            disp_fee_surplus_eoy_dict = pickle.load(f)
        # with open(total_pharm_list_in, 'rb') as f:  # now part of params file
        #     total_pharm_list = pickle.load(f)
        # with open(agreement_pharmacy_list_in, 'rb') as f:  # not needed
        #     agreement_pharmacy_list = pickle.load(f)

        lp_data_df = generatePricingDecisionVariables(lp_data_df)
        lp_data_df['GPI_12'] = lp_data_df.GPI.str[0:12]
        lp_data_df['GPI_Strength'] = lp_data_df.GPI.str[12:]

        constraint_cols = ['CLIENT', 'BREAKOUT', 'MEASUREMENT', 'REGION','BG_FLAG','Dec_Var_ID', 'Price_Decision_Var', 'QTY_PROJ_EOY','PRICING_QTY_PROJ_EOY',
                            'PHARMACY_TYPE', 'CHAIN_GROUP', 'CHAIN_SUBGROUP','PHRM_GRTE_TYPE','PHARMACY_RATE',
                           'FULLAWP_ADJ_PROJ_EOY', 'PRICE_MUTABLE', 
                           'PHARM_EFF_CAPPED_PRICE','PRICE_TIER',
                            'MAC_PRICE_UNIT_ADJ','SOFT_CONST_BENCHMARK_PRICE', 
                           'EFF_UNIT_PRICE', 'CURRENT_MAC_PRICE', 
                           'EFF_CAPPED_PRICE',
                           'PHARM_FULLAWP_ADJ_PROJ_EOY', 'PHARM_FULLAWP_ADJ_PROJ_LAG', 
                           'PHARM_QTY_PROJ_EOY','PHARM_QTY_PROJ_LAG',
                           'PHARM_FULLAWP_ADJ','PHARM_FULLNADAC_ADJ','PHARM_FULLACC_ADJ',
                           'PHARM_AVG_AWP','PHARM_AVG_NADAC','PHARM_AVG_ACC',
                            'PHARMACY_PERF_NAME', 'PRICE_REIMB_LAG'
                             ,'GPI','AVG_AWP','QTY','CLAIMS'
                           # additional target ingredient cost columns
                          ,'PHARM_TARG_INGCOST_ADJ','PHARM_TARG_INGCOST_ADJ_PROJ_EOY','PHARM_TARG_INGCOST_ADJ_PROJ_LAG',
                          'TARG_INGCOST_ADJ','TARG_INGCOST_ADJ_PROJ_EOY','TARG_INGCOST_ADJ_PROJ_LAG']
        
        if p.INTERCEPTOR_OPT or p.COSTSAVER_CLIENT:
            constraint_cols = constraint_cols + ['VENDOR_PRICE', 'CURRENT_KEEP_SEND', 'EXPECTED_KEEP_SEND','QTY_ZBD_FRAC',\
                                                 'FULLAWP_ADJ_ZBD_FRAC','PHARM_QTY_ZBD_FRAC','PHARM_FULLAWP_ADJ_ZBD_FRAC'\
                                                # additional target ingredient cost columns
                                                ,'PHARM_TARG_INGCOST_ADJ_ZBD_FRAC', 'TARG_INGCOST_ADJ_ZBD_FRAC'
                                                ]
        lambda_decision_var = generateLambdaDecisionVariables_ebit(breakout_df, list(set(p.PHARMACY_LIST['GNRC']+p.PHARMACY_LIST['BRND'])))
        lambda_df = lambda_decision_var[lambda_decision_var['Lambda_Level'] == 'CLIENT'] #superfluous from old code
        price_df = lp_data_df[constraint_cols]
        client_constraint_list, client_constraint_target  = generateGuaranteeConstraintEbit(price_df,
                                                                                            lambda_df,
                                                                                            client_guarantees,
                                                                                            pharmacy_guarantees,
                                                                                            performance_dict,
                                                                                            breakout_df,
                                                                                            client_list,
                                                                                            list(set(p.AGREEMENT_PHARMACY_LIST['GNRC']+
                                                                                                     p.AGREEMENT_PHARMACY_LIST['BRND']+
                                                                                                     p.COGS_PHARMACY_LIST['GNRC']+
                                                                                                     p.COGS_PHARMACY_LIST['BRND'])),
                                                                                            p.COGS_PHARMACY_LIST,
                                                                                            p.COGS_BUFFER,
                                                                                            oc_eoy_pharm_perf,
                                                                                            gen_launch_eoy_dict,
                                                                                            brand_surplus_eoy_dict,
                                                                                            specialty_surplus_eoy_dict,
                                                                                            disp_fee_surplus_eoy_dict,
                                                                                            price_lambdas_df)
        
        # file outputs
        with open(lambda_df_out, 'wb') as f:
            pickle.dump(lambda_df, f)
        with open(client_constraint_list_out, 'wb') as f:
            pickle.dump(client_constraint_list, f)
        with open(client_constraint_target_out, 'wb') as f:
            pickle.dump(client_constraint_target, f)

        return (lambda_df, client_constraint_list, client_constraint_target, lambda_decision_var)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'Client Level Constraints', repr(e), error_loc)
        raise e
###### End Client Level Constraints ##########


def preferred_pricing_less_than_non_preferred_pricing_constraints(
    params_file_in: str,
    unc_flag: bool,
    lp_data_df_in: InputPath('pickle'),
    pref_pharm_list_in: InputPath('pickle'),
    # total_pharm_list_in: InputPath('pickle'), 
    pref_lt_non_pref_cons_list_out: OutputPath('pickle'),
    anomaly_gpi_out: OutputPath('pickle'),
    loglevel: str = 'INFO'
    # kube_run: bool = True,
):
    import sys
    import os
    sys.path.append('/')
    import time
    import logging
    import pandas as pd
    import pickle
    import pulp
    import util_funcs as uf
    import BQ

    uf.write_params(params_file_in)
    import CPMO_parameters as p
    from CPMO_lp_functions import generatePricingDecisionVariables
    from CPMO_shared_functions import update_run_status    
    try:
        # file inputs
        with open(lp_data_df_in, 'rb') as f:
            lp_data_df = pickle.load(f)
        with open(pref_pharm_list_in, 'rb') as f:
            pref_pharm_list = pickle.load(f)
        # with open(total_pharm_list_in, 'rb') as f:
        #     total_pharm_list = pickle.load(f)

        # lp_data_df = lp_data_df_in

        out_path = os.path.join(p.FILE_LOG_PATH, 'ClientPharmacyMacOptimization.log')
        logger = uf.log_setup(log_file_path=out_path, loglevel=loglevel)

        logger.info('--------------------')
        logger.info("Preferred Pricing Less than Non Preferred Pricing")
        start = time.time()

        lp_data_df = generatePricingDecisionVariables(lp_data_df)
        lp_data_df['GPI_12'] = lp_data_df.GPI.str[0:12]
        lp_data_df['GPI_Strength'] = lp_data_df.GPI.str[12:]

        price_constraints_col = ['CLIENT', 'BREAKOUT', 'MEASUREMENT', 'GPI', 'NDC', 'BG_FLAG', 'GPI_NDC', 'GPI_ONLY',
                                    'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'REGION', 'PHARMACY_TYPE',
                                    'Price_Decision_Var', 'Price_Bounds', 'MAC_PRICE_UNIT_ADJ', 'Dec_Var_Name']

        price_constraints_df = lp_data_df.loc[(lp_data_df.PRICE_MUTABLE==1) &
                                                (lp_data_df.MEASUREMENT != 'M30') &
                                                (lp_data_df.BREAKOUT != 'SELECT'),price_constraints_col]

        
        ##################################################### vectorization starts here #####################################################
        
        # initiate
        pref_lt_non_pref_cons_list = [] 
        anomaly_gpi = pd.DataFrame(columns = ['CLIENT', 'REGION', 'GPI_NDC', 'BG_FLAG', 'EXCLUDE_REASON'])

        # some filtering
        price_constraints_df = price_constraints_df.drop(columns = ['PHARMACY_TYPE','MAC_PRICE_UNIT_ADJ']) # remove columns that are never needed
        price_constraints_df = price_constraints_df[price_constraints_df['MEASUREMENT'] != 'M30'] # remove mail
        
        price_constraints_df = price_constraints_df[~price_constraints_df['REGION'].isin(['REG_34', 'REG_5'])] # remove regions that are not used
        price_constraints_df['MEASUREMENT'] = price_constraints_df['MEASUREMENT'].str[:3] # get first three characters of "MEASUREMENT"
        price_constraints_df['Price_LB'] = price_constraints_df['Price_Bounds'].str[0] # split tuples in column "Price_Bounds" into two columns for lower and upper bounds
        price_constraints_df['Price_UB'] = price_constraints_df['Price_Bounds'].str[1]
        price_constraints_df.drop(columns=['Price_Bounds'],inplace=True) # cleanup
        df_1 = price_constraints_df.copy() # df_1 is a "cleaned-up" copy of the original dataframe, and we will work from it
        df_1.reset_index(drop=True, inplace=True) # reset index to make 0, 1, 2...

        #################################################################################################################################################################
        ################## The following section aims to label each row to indicate whether its "CHAIN_GROUP" is a preferred or non-preferred pharmacy ##################

        # Some cleaning work. (This converts "pref_pharm_list" into a clean format such that each row contains a list of pharmacy names for one unique CLIENT-REGION pair)
        pref_pharm_list = pref_pharm_list[['CLIENT','REGION','PREF_PHARMS']] # remove columns that we don't need
        exploded_pref_pharm_list = pref_pharm_list.explode('PREF_PHARMS').drop_duplicates() # explode each value in column "PREF_PHARMS" to a new row
        grouped_pref_pharm_list = exploded_pref_pharm_list.groupby(['CLIENT','REGION'])\
                                  .agg(list).reset_index() # after this, we have one list of strings in column "PREF_PHARMS" for each CLIENT-REGION pair
        grouped_pref_pharm_list['pharmacy_list'] = [list(set(p.PHARMACY_LIST['GNRC']+p.PHARMACY_LIST['BRND']))] * len(grouped_pref_pharm_list) # Create a new column containing ALL pharmacy names in preparation for next step

        # Next, we want to find the list of preferred and non-preferred pharmacies for each CLIENT-REGION pair
        dfA = grouped_pref_pharm_list.explode('PREF_PHARMS') # explode column "PREF_PHARMS"
        dfA.reset_index(drop=True, inplace=True) # reset index to make 0, 1, 2...
        dfA.reset_index(drop=False, inplace=True) # turn index into a column
        dfB = grouped_pref_pharm_list.explode('pharmacy_list') # explode column "pharmacy_list"
        dfB.reset_index(drop=True, inplace=True)
        dfB.reset_index(drop=False, inplace=True)
        matches = dfA.merge(dfB, how = 'right', left_on = ['PREF_PHARMS','CLIENT','REGION'],
                            right_on = ['pharmacy_list','CLIENT','REGION']) # Do a right merge. For each CLIENT-REGION pair, this creates a row for each "pharmacy_list" regardless of "PREF_PHARMS" values 
        rows_mistmatch = matches[matches['PREF_PHARMS_x'].isnull()] # next, find rows where "PREF_PHARMS" did not match "pharmacy_list" in the merge above. These are non-preferred pharmacies (for the current CLIENT-REGION pair)
        grouped_with_non_pref_list = rows_mistmatch.groupby(['REGION','CLIENT'])['pharmacy_list_y'].\
                                     agg(list).reset_index() # for each REGION-CLIENT pair, aggregate the non-preferred pharmacies that we have identified back into one row
        grouped_with_non_pref_list.rename(columns={'pharmacy_list_y':'NONPREF_PHARMS'}, inplace=True) # cleanup
        grouped_pref_pharm_list.drop(columns=['pharmacy_list'],inplace=True) # cleanup
        pref_nonpref_lists = grouped_pref_pharm_list.merge(grouped_with_non_pref_list, how='inner',
                                                  on=['REGION','CLIENT']) # merge results above so that each CLIENT-REGION pair has columns "PREF_PHARMS" and "NONPREF_PHARMS"
        df_2 = pref_nonpref_lists.merge(df_1, how='inner', on=['REGION','CLIENT']) # df_2 is the cleaned original dataframe (df_1) with columns "PREF_PHARMS" and "NONPREF_PHARMS" appended

        # Next, removes rows in df_2 where CHAIN_GROUP does not belong in either "PREF_PHARMS" or "NONPREF_PHARMS" (which means CHAIN_GROUP is not in P.PHARMACY_LIST)
        # (Note: if data does not contain such cases, feel free to remove this section and set df_3 = df_2. For now, keeping this snippet because Melanie says data is not yet always perfect)
        PREF_PHARMS_exploded = df_2.explode('PREF_PHARMS')
        mask1 = PREF_PHARMS_exploded['CHAIN_GROUP'].isin(PREF_PHARMS_exploded['PREF_PHARMS'])
        NONPREF_PHARMS_exploded = df_2.explode('NONPREF_PHARMS')
        mask2 = NONPREF_PHARMS_exploded['CHAIN_GROUP'].isin(NONPREF_PHARMS_exploded['NONPREF_PHARMS'])
        df_2['keep'] = (mask1.groupby(mask1.index).any()) | (mask2.groupby(mask2.index).any())
        df_3 = df_2[df_2['keep'] == True].copy()
        df_3.drop(columns=['keep'],inplace=True)
        df_3.reset_index(drop=True,inplace=True)

        # Add a new column "chain_is_pref" to df_3 to indicate whether the CHAIN_GROUP belongs in PREF_PHARMS list (if true, it's preferred pharmacy; if False, it's non-preferred)
        PREF_PHARMS_exploded = df_3.explode('PREF_PHARMS')
        mask = PREF_PHARMS_exploded['CHAIN_GROUP'].isin(PREF_PHARMS_exploded['PREF_PHARMS'])
        df_3['chain_is_pref'] = (mask.groupby(mask.index).any())
        df_3.drop(columns=['PREF_PHARMS', 'NONPREF_PHARMS', 'BREAKOUT'], inplace=True)

        ######################################################################### End of section ########################################################################
        #################################################################################################################################################################

        # For each ['CLIENT','GPI','MEASUREMENT','REGION'] group, self-merge df_3 such that "chain_is_pref" is "true" for the left dataframe, and "false" for the right dataframe.
        # (Important to understand) This essentially creates a row for each pair of preferred pharmacy - nonpreferred pharmacy within each ['CLIENT','GPI','MEASUREMENT','REGION'] group.
        # After self-merging, the preferred subchain and nonpreferred subchain for each pair of "preferred pharmacy - nonpreferred pharmacy" will be in columns "CHAIN_SUBGROUP_pref" and "CHAIN_SUBGROUP_nonpref"
        pref_df = df_3[df_3['chain_is_pref'] == True]
        nonpref_df = df_3[df_3['chain_is_pref'] == False]
        self_merged = pref_df.merge(nonpref_df, on = ['CLIENT','GPI','BG_FLAG','MEASUREMENT','REGION'], suffixes=('_pref', '_nonpref'), how = 'inner')
        self_merged.drop(columns=['chain_is_pref_pref', 'chain_is_pref_nonpref'],inplace=True)


        #################################################################################################################################################################
        ############################################## Next we jump into the 4 If/Else/Elif clauses in the original code ################################################

        ############### If/Else Clause No.1 : Original Code: (len(pref_gpi.loc[pref_gpi.GPI_ONLY == 0, 'GPI_NDC']) > 0) & (len(npref_gpi.loc[npref_gpi.GPI_ONLY == 0, 'GPI_NDC']) > 0)):  
        ###############

        # Performs filtering: the current preferred subchain and nonpreferred subchain must both contain rows with GPI_ONLY == 0.
        # In terms of our self-merged dataframe, this requires that the current group has rows with GPI_ONLY == 0 in both left (pref) and right (nonpref) halves.
        self_merged['GPI_pref_is_0'] = (self_merged['GPI_ONLY_pref'] == 0) # create a column to label True/False for wheather GPI_ONLY == 0, to make it easier to do agg functions later
        self_merged['GPI_nonpref_is_0'] = (self_merged['GPI_ONLY_nonpref'] == 0)
        grouped = self_merged.groupby(['CLIENT','GPI','BG_FLAG','MEASUREMENT','REGION','CHAIN_GROUP_pref','CHAIN_GROUP_nonpref','CHAIN_SUBGROUP_pref','CHAIN_SUBGROUP_nonpref'])\
                             .agg(condition1 = ('GPI_pref_is_0', 'sum'), condition2 = ('GPI_nonpref_is_0', 'sum')).reset_index()
        grouped = grouped[(grouped['condition1'] > 0) & (grouped['condition2'] > 0)] # both conditions need to be satisfied
        grouped = grouped.drop(columns = ['condition1','condition2']) # cleanups
        df_4 = self_merged.merge(grouped, on = ['CLIENT','GPI','BG_FLAG','MEASUREMENT','REGION','CHAIN_GROUP_pref','CHAIN_GROUP_nonpref','CHAIN_SUBGROUP_pref','CHAIN_SUBGROUP_nonpref'],
                                 how = 'inner') # df_4 is the self-merged dataframe filtered to satisfy the conditions above

        # Next, filter to only keep rows where the left (pref) and right (nonpreft) halves have equal NDC values, i.e. the preferred and the nonpreferred subchains have the same NDC
        # and get the variable and price bounds from each group
        df_5 = df_4[df_4['NDC_pref'] == df_4['NDC_nonpref']].copy()
        df_5.drop(columns=['NDC_nonpref'],inplace=True) # cleanups
        df_5.rename(columns={'NDC_pref':'NDC'}, inplace=True) # cleanups
        final = df_5.groupby(['CLIENT','GPI','BG_FLAG','MEASUREMENT','REGION','CHAIN_GROUP_pref','CHAIN_GROUP_nonpref',
                                         'CHAIN_SUBGROUP_pref','CHAIN_SUBGROUP_nonpref','NDC'])\
                                .agg(var1 = ('Price_Decision_Var_pref', 'first'), pref_lower_bound = ('Price_LB_pref', 'first'),
                                     var2 = ('Price_Decision_Var_nonpref', 'first'), non_pref_upper_bound = ('Price_UB_nonpref', 'first')).reset_index()

        # Next, create price constraints, or record anomalies
        bad_rows = final[final['pref_lower_bound'] > final['non_pref_upper_bound']].copy() # price bounds have an issue
        if not bad_rows.empty: # iterate problematic rows to record anomalies
            for i in range(bad_rows.shape[0]):
                logger.info(bad_rows['GPI'].iloc[i] + '-' + bad_rows['NDC'].iloc[i] + '-' + bad_rows['BG_FLAG'].iloc[i] + '-' + bad_rows['REGION'].iloc[i] + ': ' + \
                                    bad_rows['CHAIN_SUBGROUP_pref'].iloc[i] + '-' + bad_rows['CHAIN_SUBGROUP_nonpref'].iloc[i])
            new_anomaly_df = pd.DataFrame({
                'CLIENT': bad_rows['CLIENT'].astype(str), 
                'REGION': str('ALL'),
                'GPI_NDC': bad_rows['GPI'].astype(str) + '_' + bad_rows['NDC'].astype(str),
                'BG_FLAG': bad_rows['BG_FLAG'].astype(str),
                'EXCLUDE_REASON': bad_rows['CHAIN_SUBGROUP_pref'].astype(str) + '_' + bad_rows['CHAIN_GROUP_nonpref'].astype(str)
            })
            anomaly_gpi = pd.concat([new_anomaly_df, anomaly_gpi], ignore_index = True) # add anomaly info to anomaly_gpi dataframe
        good_rows = final[final['pref_lower_bound'] <= final['non_pref_upper_bound']].copy()
        if not good_rows.empty:
            for i in range(good_rows.shape[0]):
                pref_lt_non_pref_cons_list.append(good_rows.var1.iloc[i] - good_rows.var2.iloc[i] <= 0)
                logger.info(good_rows.var1.iloc[i] - good_rows.var2.iloc[i])


        ############### If/Else Clause No.2: Original Code (len(npref_gpi.loc[npref_gpi.GPI_ONLY == 0, 'GPI_NDC']) > 0)###############

        # Performs filtering: the current nonpreferred subchain contain rows with GPI_ONLY == 0, but the preferred subchain does not
        grouped = self_merged.groupby(['CLIENT','GPI','BG_FLAG','MEASUREMENT','REGION','CHAIN_GROUP_pref','CHAIN_GROUP_nonpref','CHAIN_SUBGROUP_pref','CHAIN_SUBGROUP_nonpref'])\
                             .agg(condition1 = ('GPI_pref_is_0', 'sum'), condition2 = ('GPI_nonpref_is_0', 'sum')).reset_index()
        grouped = grouped[(grouped['condition1'] == 0) & (grouped['condition2'] > 0)]
        grouped = grouped.drop(columns = ['condition1','condition2'])
        df_4 = self_merged.merge(grouped, on = ['CLIENT','GPI','BG_FLAG','MEASUREMENT','REGION','CHAIN_GROUP_pref','CHAIN_GROUP_nonpref','CHAIN_SUBGROUP_pref','CHAIN_SUBGROUP_nonpref'],
                                 how = 'inner')

        # checks that the current nonpreferred subchain has no more than 1 row in the original (not self-merged) dataframe
        test = df_4.groupby(['CLIENT','GPI','BG_FLAG','MEASUREMENT','REGION','CHAIN_GROUP_pref','CHAIN_GROUP_nonpref','CHAIN_SUBGROUP_pref','CHAIN_SUBGROUP_nonpref',
                             'NDC_nonpref', 'GPI_NDC_nonpref', 'GPI_ONLY_nonpref','Dec_Var_Name_nonpref','Price_LB_nonpref', 'Price_UB_nonpref'], as_index = False).size()
                             # This is a groupby on the 8 keys and all nonpref columns.
                             # After this groupby, we will have one row per group if the current nonpreferred subchain has only 1 row in the original dataframe
        bad_groups = test[test['size'] != 1] # if any group has more than 1 row, it's problematic
        merged_bad_groups = bad_groups.merge(df_4, on = ['CLIENT','GPI','BG_FLAG','MEASUREMENT','REGION','CHAIN_GROUP_pref','CHAIN_GROUP_nonpref','CHAIN_SUBGROUP_pref','CHAIN_SUBGROUP_nonpref',
                      'NDC_nonpref', 'GPI_NDC_nonpref', 'GPI_ONLY_nonpref','Dec_Var_Name_nonpref','Price_LB_nonpref', 'Price_UB_nonpref'], how = 'inner') # merge back just so that we can access the "Dec_Var_Name_pref" column needed for logger
        if merged_bad_groups.shape[0] > 0: # if there are problematic groups, iterate to log the error information
            for i in range(merged_bad_groups.shape[0]):
                logger.info('ERROR with ' + merged_bad_groups['Dec_Var_Name_pref'].iloc[i])
                assert False, "len(pref_gpi.Price_Decision_Var)==1" # maybe change the error message if pref_gpi no longer makes sense

        # Get the variable and price bounds from each pref-nonpref subchain pair
        final = df_4.groupby(['CLIENT','GPI','BG_FLAG','MEASUREMENT','REGION','CHAIN_GROUP_pref','CHAIN_GROUP_nonpref',
                                         'CHAIN_SUBGROUP_pref','CHAIN_SUBGROUP_nonpref','NDC_nonpref'])\
                               .agg(var1 = ('Price_Decision_Var_pref', 'first'), pref_lower_bound = ('Price_LB_pref', 'first'),
                                    var2 = ('Price_Decision_Var_nonpref', 'first'), non_pref_upper_bound = ('Price_UB_nonpref', 'first')).reset_index()
        final.rename(columns={'NDC_nonpref':'NDC'}, inplace=True)

        # Next, create price constraints, or record anomalies
        bad_rows = final[final['pref_lower_bound'] > final['non_pref_upper_bound']].copy()
        if not bad_rows.empty:
            for i in range(bad_rows.shape[0]):      
                logger.info(bad_rows['GPI'].iloc[i] + '-' + bad_rows['NDC'].iloc[i] + '-' + bad_rows['BG_FLAG'].iloc[i] + '-' + bad_rows['REGION'].iloc[i] + ': ' + \
                            bad_rows['CHAIN_SUBGROUP_pref'].iloc[i] + '-' + bad_rows['CHAIN_SUBGROUP_nonpref'].iloc[i])
            new_anomaly_df = pd.DataFrame({
                'CLIENT': bad_rows['CLIENT'].astype(str), 
                'REGION': str('ALL'),
                'GPI_NDC': bad_rows['GPI'].astype(str) + '_' + bad_rows['NDC'].astype(str),
                'BG_FLAG': bad_rows['BG_FLAG'].astype(str),
                'EXCLUDE_REASON': bad_rows['CHAIN_SUBGROUP_pref'].astype(str) + '_' + bad_rows['CHAIN_GROUP_nonpref'].astype(str)
            })
            anomaly_gpi = pd.concat([new_anomaly_df, anomaly_gpi], ignore_index = True)
        good_rows = final[final['pref_lower_bound'] <= final['non_pref_upper_bound']].copy()
        if not good_rows.empty:
            for i in range(good_rows.shape[0]):
                pref_lt_non_pref_cons_list.append(good_rows.var1.iloc[i] - good_rows.var2.iloc[i] <= 0)
                logger.info(good_rows.var1.iloc[i] - good_rows.var2.iloc[i])


        ############### If/Else Clause No.3 Original Code: (len(pref_gpi.loc[pref_gpi.GPI_ONLY == 0, 'GPI_NDC']) > 0)###############

        # Performs filtering: the current preferred subchain contain rows with GPI_ONLY == 0, but the nonpreferred subchain does not
        grouped = self_merged.groupby(['CLIENT','GPI','BG_FLAG','MEASUREMENT','REGION','CHAIN_GROUP_pref','CHAIN_GROUP_nonpref','CHAIN_SUBGROUP_pref','CHAIN_SUBGROUP_nonpref'])\
                             .agg(condition1 = ('GPI_pref_is_0', 'sum'), condition2 = ('GPI_nonpref_is_0', 'sum')).reset_index()
        grouped = grouped[(grouped['condition1'] > 0) & (grouped['condition2'] == 0)]
        grouped = grouped.drop(columns = ['condition1','condition2'])
        df_4 = self_merged.merge(grouped, on = ['CLIENT','GPI','BG_FLAG','MEASUREMENT','REGION','CHAIN_GROUP_pref','CHAIN_GROUP_nonpref','CHAIN_SUBGROUP_pref','CHAIN_SUBGROUP_nonpref'],
                                 how = 'inner')

        # checks that the current preferred subchain has no more than 1 row in the original (not self-merged) dataframe
        test = df_4.groupby(['CLIENT','GPI','BG_FLAG','MEASUREMENT','REGION','CHAIN_GROUP_pref','CHAIN_GROUP_nonpref','CHAIN_SUBGROUP_pref','CHAIN_SUBGROUP_nonpref',
                      'NDC_pref', 'GPI_NDC_pref', 'GPI_ONLY_pref','Dec_Var_Name_pref','Price_LB_pref', 'Price_UB_pref'],as_index = False).size()
        bad_groups = test[test['size'] != 1]
        merged_bad_groups = bad_groups.merge(df_4, on = ['CLIENT','GPI','BG_FLAG','MEASUREMENT','REGION','CHAIN_GROUP_pref','CHAIN_GROUP_nonpref','CHAIN_SUBGROUP_pref','CHAIN_SUBGROUP_nonpref',
                      'NDC_pref', 'GPI_NDC_pref', 'GPI_ONLY_pref','Dec_Var_Name_pref','Price_LB_pref', 'Price_UB_pref'], how = 'inner')
        if merged_bad_groups.shape[0] > 0:
            for i in range(merged_bad_groups.shape[0]):
                logger.info('ERROR with ' + merged_bad_groups['Dec_Var_Name_nonpref'].iloc[i])
                assert False, "len(npref_gpi.Price_Decision_Var)==1"

        # Get the variable and price bounds from each pref-nonpref subchain pair
        final = df_4.groupby(['CLIENT','GPI','BG_FLAG','MEASUREMENT','REGION','CHAIN_GROUP_pref','CHAIN_GROUP_nonpref',
                                         'CHAIN_SUBGROUP_pref','CHAIN_SUBGROUP_nonpref','NDC_pref'])\
                               .agg(var1 = ('Price_Decision_Var_pref', 'first'), pref_lower_bound = ('Price_LB_pref', 'first'),
                                    var2 = ('Price_Decision_Var_nonpref', 'first'), non_pref_upper_bound = ('Price_UB_nonpref', 'first')).reset_index()
        final.rename(columns={'NDC_pref':'NDC'}, inplace=True)

        # Next, create price constraints, or record anomalies
        bad_rows = final[final['pref_lower_bound'] > final['non_pref_upper_bound']].copy()
        if not bad_rows.empty:
            for i in range(bad_rows.shape[0]):
                logger.info(bad_rows['GPI'].iloc[i] + '-' + bad_rows['NDC'].iloc[i] + '-' + bad_rows['BG_FLAG'].iloc[i] + '-' + bad_rows['REGION'].iloc[i] + ': ' + \
                            bad_rows['CHAIN_SUBGROUP_pref'].iloc[i] + '-' + bad_rows['CHAIN_SUBGROUP_nonpref'].iloc[i])
            new_anomaly_df = pd.DataFrame({
                'CLIENT': bad_rows['CLIENT'].astype(str), 
                'REGION': str('ALL'),
                'GPI_NDC': bad_rows['GPI'].astype(str) + '_' + bad_rows['NDC'].astype(str),
                'BG_FLAG': bad_rows['BG_FLAG'].astype(str),
                'EXCLUDE_REASON': bad_rows['CHAIN_SUBGROUP_pref'].astype(str) + '_' + bad_rows['CHAIN_GROUP_nonpref'].astype(str)
            })
            anomaly_gpi = pd.concat([new_anomaly_df, anomaly_gpi], ignore_index = True)
        good_rows = final[final['pref_lower_bound'] <= final['non_pref_upper_bound']].copy()
        if not good_rows.empty:
            for i in range(good_rows.shape[0]):
                pref_lt_non_pref_cons_list.append(good_rows.var1.iloc[i] - good_rows.var2.iloc[i] <= 0)
                logger.info(good_rows.var1.iloc[i] - good_rows.var2.iloc[i])


        ############### If/Else Clause No.4: current preferred subchain and nonpreferred subchain both do not contain rows with GPI_ONLY == 0 ###############

        # Performs filtering: the current preferred subchain and nonpreferred subchain both do not contain rows with GPI_ONLY == 0.
        grouped = self_merged.groupby(['CLIENT','GPI','BG_FLAG','MEASUREMENT','REGION','CHAIN_GROUP_pref','CHAIN_GROUP_nonpref','CHAIN_SUBGROUP_pref','CHAIN_SUBGROUP_nonpref'])\
                             .agg(condition1 = ('GPI_pref_is_0', 'sum'), condition2 = ('GPI_nonpref_is_0', 'sum')).reset_index()
        grouped = grouped[(grouped['condition1'] == 0) & (grouped['condition2'] == 0)]
        grouped = grouped.drop(columns = ['condition1','condition2'])
        df_4 = self_merged.merge(grouped, on = ['CLIENT','GPI','BG_FLAG','MEASUREMENT','REGION','CHAIN_GROUP_pref','CHAIN_GROUP_nonpref','CHAIN_SUBGROUP_pref','CHAIN_SUBGROUP_nonpref'],
                                 how = 'inner')

        # Get the variable and price bounds from each pref-nonpref subchain pair
        merged = df_4.groupby(['CLIENT','GPI','BG_FLAG','MEASUREMENT','REGION','CHAIN_GROUP_pref','CHAIN_GROUP_nonpref','CHAIN_SUBGROUP_pref','CHAIN_SUBGROUP_nonpref'])\
                                .agg(var1 = ('Price_Decision_Var_pref', 'first'), pref_lower_bound = ('Price_LB_pref', 'first'),
                                     var2 = ('Price_Decision_Var_nonpref', 'first'), non_pref_upper_bound = ('Price_UB_nonpref', 'first')).reset_index() # calling it "merged" just to be consistent with previous if/else clauses

        # Next, create price constraints, or record anomalies
        bad_rows = merged[merged['pref_lower_bound'] > merged['non_pref_upper_bound']].copy()
        if not bad_rows.empty:
            for i in range(bad_rows.shape[0]):
                logger.info(bad_rows['GPI'].iloc[i] + '_***********_' + '-' + bad_rows['BG_FLAG'].iloc[i] + bad_rows['REGION'].iloc[i] + ': ' + \
                            bad_rows['CHAIN_SUBGROUP_pref'].iloc[i] + '-' + bad_rows['CHAIN_SUBGROUP_nonpref'].iloc[i])
            new_anomaly_df = pd.DataFrame({
                'CLIENT': bad_rows['CLIENT'].astype(str), 
                'REGION': str('ALL'),
                'GPI_NDC': bad_rows['GPI'].astype(str) + '_***********',
                'BG_FLAG': bad_rows['BG_FLAG'].astype(str),
                'EXCLUDE_REASON': bad_rows['CHAIN_SUBGROUP_pref'].astype(str) + '_' + bad_rows['CHAIN_GROUP_nonpref'].astype(str)
            })
            anomaly_gpi = pd.concat([new_anomaly_df, anomaly_gpi], ignore_index = True)
        good_rows = merged[merged['pref_lower_bound'] <= merged['non_pref_upper_bound']].copy()
        if not good_rows.empty:
            for i in range(good_rows.shape[0]):
                pref_lt_non_pref_cons_list.append(good_rows.var1.iloc[i] - good_rows.var2.iloc[i] <= 0)
                
        logger.info("End Preferred Pricing Less than Non Preferred Pricing")
        end = time.time()
        logger.info("Run time: {} mins".format((end - start)/60.))
        logger.info('--------------------')

        # output files
        with open(pref_lt_non_pref_cons_list_out, 'wb') as f:
            pickle.dump(pref_lt_non_pref_cons_list, f)
        with open(anomaly_gpi_out, 'wb') as f:
            pickle.dump(anomaly_gpi, f)

        return pref_lt_non_pref_cons_list, anomaly_gpi
    
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'Preferred LT Non-Preferred Pricing Constraints', repr(e), error_loc)
        raise e
###### End Preferred Pricing less than Non Preferred Pricing ########



###### Measure Specific Pricing (ie M < R90 < R30) ########
def specific_pricing_constraints(
        params_file_in: str,
        unc_flag: bool,
        lp_data_df_in: InputPath('pickle'),
        # total_pharm_list_in: InputPath('pickle'),
        meas_specific_price_cons_list_out: OutputPath('pickle'),
        anomaly_mes_gpi_out: OutputPath('pickle'),
        loglevel: str = 'INFO'
        # kubmeas_specific_price_cons_listrue,
):
    import sys
    import os
    sys.path.append('/')
    import time
    import logging
    import pandas as pd
    import pickle
    import pulp
    import util_funcs as uf
    import BQ

    uf.write_params(params_file_in)
    import CPMO_parameters as p
    from CPMO_lp_functions import generatePricingDecisionVariables
    from CPMO_shared_functions import update_run_status

    try:
        out_path = os.path.join(p.FILE_LOG_PATH, 'ClientPharmacyMacOptimization.log')
        logger = uf.log_setup(log_file_path=out_path, loglevel=loglevel)

        # file inputs ##########################################
        with open(lp_data_df_in, 'rb') as f:
            lp_data_df = pickle.load(f)
        ########################################################
        
    
        logger.info('--------------------')
        logger.info("M <= Retail Pricing")
        start = time.time()

        # If the client contract allows mail prices to go above retail, apply p.MAIL_UNRESTRICTED_CAP to the
        # M <= Retail pricing constraints. The R90 <= R30 constraint will prevent Mail prices from exceeding
        # the MAIL_UNRESTRICTED_CAP limit when compared to R90. If there is no R90 price, the cap is limited
        # against the R30 price.
        if p.MAIL_MAC_UNRESTRICTED:
            m30_cons_cap = p.MAIL_UNRESTRICTED_CAP
        else:
            m30_cons_cap = 1.0
        assert m30_cons_cap >= 1.0, "p.MAIL_UNRESTRICTED_CAP < 1 is not recommended without business justification."
        
        ######### Modification of lp_data_df and creation of price_contraints_df ##########################
        lp_data_df = generatePricingDecisionVariables(lp_data_df)
        lp_data_df['GPI_12'] = lp_data_df.GPI.str[0:12]
        lp_data_df['GPI_Strength'] = lp_data_df.GPI.str[12:]
        price_constraints_col = ['CLIENT', 'BREAKOUT', 'MEASUREMENT', 'GPI_NDC', 'BG_FLAG', 'CHAIN_GROUP', 'CHAIN_SUBGROUP',
                                 'REGION', 'PHARMACY_TYPE',
                                 'Price_Decision_Var', 'Price_Bounds', 'MAC_PRICE_UNIT_ADJ', 'Dec_Var_Name']
        price_constraints_df = lp_data_df[price_constraints_col].loc[lp_data_df.PRICE_MUTABLE == 1, :]
        price_constraints_df.MEASUREMENT = price_constraints_df.MEASUREMENT.replace(
            {'R30P': 'R30', 'R30N': 'R30', 'R90P': 'R90', 'R90N': 'R90'})
        ####################################################################################################
        
        meas_specific_price_cons_list = []
        meas_specific_price_cons_list_rev = []
        anomaly_mes_gpi = pd.DataFrame(columns=['CLIENT', 'REGION', 'GPI_NDC', 'BG_FLAG', 'EXCLUDE_REASON'])

        price_constraints_df['Price_LB'] = price_constraints_df['Price_Bounds'].str[0]
        price_constraints_df['Price_UB'] = price_constraints_df['Price_Bounds'].str[1]

        M30_filtered = price_constraints_df[price_constraints_df['MEASUREMENT'] == 'M30']
        M30_group = M30_filtered.groupby(['GPI_NDC', 'BG_FLAG', 'CLIENT', 'REGION'], as_index=False) \
            .agg(cons_M30=('Price_Decision_Var', 'first'),
                 LB_M30=('Price_LB', 'max'),
                 UB_M30=('Price_UB', 'min'))
        R90_filtered = price_constraints_df[price_constraints_df['MEASUREMENT'] == 'R90']
        R90_group = R90_filtered.groupby(['GPI_NDC', 'BG_FLAG', 'CLIENT', 'REGION', 'CHAIN_GROUP', 'CHAIN_SUBGROUP'],
                                         as_index=False) \
            .agg(cons_R90=('Price_Decision_Var', 'first'),
                 LB_R90=('Price_LB', 'first'),
                 UB_R90=('Price_UB', 'first'))
        R90_group = R90_group[R90_group['CHAIN_GROUP'].isin(list(set(p.PHARMACY_LIST['GNRC']+p.PHARMACY_LIST['BRND'])))]
        R90_M30_merged = R90_group.merge(M30_group, on=['GPI_NDC', 'BG_FLAG', 'CLIENT', 'REGION'], how='inner')
        R90_M30_merged['price_cons'] = "" + R90_M30_merged['cons_M30'] - m30_cons_cap * R90_M30_merged['cons_R90']
        R90_M30_merged['price_cons_rev'] = "" - R90_M30_merged['cons_M30'] + p.MAIL_RETAIL_BOUND * R90_M30_merged[
            'cons_R90']
        
        bad_rows = pd.DataFrame([])

        if not p.MAIL_MAC_UNRESTRICTED:
            #### segment 1: creating the list of Price Decision vars constraints where M30 lower bound <= R90 upper bound
            bad_rows = R90_M30_merged[R90_M30_merged['LB_M30'] > R90_M30_merged['UB_R90']]
        
        if not bad_rows.empty:
            for index, row in bad_rows.iterrows():
                
                logger.info(str(R90_M30_merged.GPI_NDC[index]) + '-' + str(R90_M30_merged.BG_FLAG[index]) + '-' + str(R90_M30_merged.REGION[index]) + '-' + str(
                    R90_M30_merged.CHAIN_SUBGROUP[index]) + ': ' + 'M - R90')
            new_anomaly_df = pd.DataFrame({
                'CLIENT': bad_rows['CLIENT'].astype(str),
                'REGION': bad_rows['REGION'].astype(str),
                'GPI_NDC': bad_rows['GPI_NDC'].astype(str),
                'BG_FLAG': bad_rows['BG_FLAG'].astype(str),
                'EXCLUDE_REASON': bad_rows['CHAIN_SUBGROUP'].astype(str) + ': M - R90'
            })
            anomaly_mes_gpi = pd.concat([new_anomaly_df, anomaly_mes_gpi], ignore_index=True)

        if not p.MAIL_MAC_UNRESTRICTED: 
        	good_rows = R90_M30_merged[R90_M30_merged['LB_M30'] <= R90_M30_merged['UB_R90']]
        else: 
        	good_rows = R90_M30_merged.copy()

        if not good_rows.empty:
            target = good_rows.price_cons
            for i in range(target.shape[0]):
                meas_specific_price_cons_list.append(target.iloc[i] <= 0)
        ########### segment 1 ends #####################################################################################    
        
        if not p.MAIL_MAC_UNRESTRICTED:
            #### segment 2: creating the list of Price Decision vars constraints where R90 lower bound <= M30 upper bound
            bad_rows = R90_M30_merged[p.MAIL_RETAIL_BOUND * R90_M30_merged['LB_R90'] > R90_M30_merged['UB_M30']]

        if not bad_rows.empty:
            for index, row in bad_rows.iterrows():
                
                logger.info(str(R90_M30_merged.GPI_NDC[index]) + '-' + str(R90_M30_merged.BG_FLAG[index]) + '-' + str(R90_M30_merged.REGION[index]) + '-' + str(
                    R90_M30_merged.CHAIN_SUBGROUP[index]) + ': ' + 'R90 - M')
            new_anomaly_df = pd.DataFrame({
                'CLIENT': bad_rows['CLIENT'].astype(str),
                'REGION': bad_rows['REGION'].astype(str),
                'GPI_NDC': bad_rows['GPI_NDC'].astype(str),
                'BG_FLAG': bad_rows['BG_FLAG'].astype(str),
                'EXCLUDE_REASON': bad_rows['CHAIN_SUBGROUP'].astype(str) + ': R90 - M'
            })
            anomaly_mes_gpi = pd.concat([new_anomaly_df, anomaly_mes_gpi], ignore_index=True)

        if not p.MAIL_MAC_UNRESTRICTED: 
        	good_rows = R90_M30_merged[p.MAIL_RETAIL_BOUND * R90_M30_merged['LB_R90'] <= R90_M30_merged['UB_M30']]
        else: 
         	good_rows = R90_M30_merged.copy()

        if not good_rows.empty:
            target = good_rows.price_cons_rev
            for i in range(target.shape[0]):
                meas_specific_price_cons_list_rev.append(target.iloc[i] <= 0)
        ########### segment 2 ends #####################################################################################

        #### segment 3: creating the list of Price Decision vars constraints where M30 lower bound <= R30 upper bound
        R30_filtered = price_constraints_df[price_constraints_df['MEASUREMENT'] == 'R30']
        R30_group = R30_filtered.groupby(['GPI_NDC', 'BG_FLAG', 'CLIENT', 'REGION', 'CHAIN_GROUP', 'CHAIN_SUBGROUP'],
                                         as_index=False) \
            .agg(cons_R30=('Price_Decision_Var', 'first'),
                 LB_R30=('Price_LB', 'first'),
                 UB_R30=('Price_UB', 'first'))
        R30_group = R30_group[R30_group['CHAIN_GROUP'].isin(list(set(p.PHARMACY_LIST['GNRC']+p.PHARMACY_LIST['BRND'])))]
        R30_M30_merged = R30_group.merge(M30_group, on=['GPI_NDC', 'BG_FLAG', 'CLIENT', 'REGION'], how='inner')
        R30_M30_merged['price_cons'] = "" + R30_M30_merged['cons_M30'] - m30_cons_cap * R30_M30_merged['cons_R30']
        R30_M30_merged['price_cons_rev'] = "" - R30_M30_merged['cons_M30'] + p.MAIL_RETAIL_BOUND * R30_M30_merged[
            'cons_R30']

        if not p.MAIL_MAC_UNRESTRICTED:
            bad_rows = R30_M30_merged[R30_M30_merged['LB_M30'] > R30_M30_merged['UB_R30']]
        
        if not bad_rows.empty:
            for index, row in bad_rows.iterrows():

                logger.info(str(R30_M30_merged.GPI_NDC[index]) + '-' + str(R30_M30_merged.BG_FLAG[index]) + '-' + str(R30_M30_merged.REGION[index]) + '-' + str(
                    R30_M30_merged.CHAIN_SUBGROUP[index]) + ': ' + 'M - R30')
            new_anomaly_df = pd.DataFrame({
                'CLIENT': bad_rows['CLIENT'].astype(str),
                'REGION': bad_rows['REGION'].astype(str),
                'GPI_NDC': bad_rows['GPI_NDC'].astype(str),
                'BG_FLAG': bad_rows['BG_FLAG'].astype(str),
                'EXCLUDE_REASON': bad_rows['CHAIN_SUBGROUP'].astype(str) + ': M - R30'
            })
            anomaly_mes_gpi = pd.concat([new_anomaly_df, anomaly_mes_gpi], ignore_index=True)

        if not p.MAIL_MAC_UNRESTRICTED: 
        	good_rows = R30_M30_merged[R30_M30_merged['LB_M30'] <= R30_M30_merged['UB_R30']]
        else: 
        	good_rows = R30_M30_merged.copy()
        if not good_rows.empty:
            target = good_rows.price_cons
            for i in range(target.shape[0]):
                meas_specific_price_cons_list.append(target.iloc[i] <= 0)
        ########### segment 3 ends #####################################################################################

        #### segment 4: creating the list of Price Decision vars constraints where R30 lower bound <= M30 upper bound
        if not p.MAIL_MAC_UNRESTRICTED:
            bad_rows = R30_M30_merged[p.MAIL_RETAIL_BOUND * R30_M30_merged['LB_R30'] > R30_M30_merged['UB_M30']]
        
        if not bad_rows.empty:
            for index, row in bad_rows.iterrows():

                logger.info(str(R30_M30_merged.GPI_NDC[index]) + '-' + str(R30_M30_merged.BG_FLAG[index]) + '-' + str(R30_M30_merged.REGION[index]) + '-' + str(
                    R30_M30_merged.CHAIN_SUBGROUP[index]) + ': ' + 'R30 - M')
            new_anomaly_df = pd.DataFrame({
                'CLIENT': bad_rows['CLIENT'].astype(str),
                'REGION': bad_rows['REGION'].astype(str),
                'GPI_NDC': bad_rows['GPI_NDC'].astype(str),
                'BG_FLAG': bad_rows['BG_FLAG'].astype(str),
                'EXCLUDE_REASON': bad_rows['CHAIN_SUBGROUP'].astype(str) + ': R30 - M'
            })
            anomaly_mes_gpi = pd.concat([new_anomaly_df, anomaly_mes_gpi], ignore_index=True)

        if not p.MAIL_MAC_UNRESTRICTED: 
            good_rows = R30_M30_merged[p.MAIL_RETAIL_BOUND * R30_M30_merged['LB_R30'] <= R30_M30_merged['UB_M30']]
        else: 
            good_rows = R30_M30_merged.copy()

        if not good_rows.empty:
            target = good_rows.price_cons_rev
            for i in range(target.shape[0]):
                meas_specific_price_cons_list_rev.append(target.iloc[i] <= 0)

        
        R30_R90_merged = R30_group.merge(R90_group, on=['GPI_NDC', 'BG_FLAG', 'CLIENT', 'REGION', 'CHAIN_GROUP', 'CHAIN_SUBGROUP'],
                                         how='inner')
        R30_R90_merged['price_cons'] = "" + R30_R90_merged['cons_R90'] - R30_R90_merged['cons_R30']

        
        bad_rows = R30_R90_merged[R30_R90_merged['LB_R90'] > R30_R90_merged['UB_R30']]

        if not bad_rows.empty:
            new_anomaly_df = pd.DataFrame({
                'CLIENT': bad_rows['CLIENT'].astype(str),
                'REGION': bad_rows['REGION'].astype(str),
                'GPI_NDC': bad_rows['GPI_NDC'].astype(str),
                'BG_FLAG': bad_rows['BG_FLAG'].astype(str),
                'EXCLUDE_REASON': bad_rows['CHAIN_GROUP'].astype(str) + ': R30 - R90'
            })
            anomaly_mes_gpi = pd.concat([new_anomaly_df, anomaly_mes_gpi], ignore_index=True)

        good_rows = R30_R90_merged[R30_R90_merged['LB_R90'] <= R30_R90_merged['UB_R30']]

        if not good_rows.empty:
            target = good_rows.price_cons
            for i in range(target.shape[0]):
                meas_specific_price_cons_list.append(target.iloc[i] <= 0)

        ########### segment 4 ends #####################################################################################
                
        #### snippet 5: creating the list of Price Decision vars constraints where R90OK lower bound <= R30 upper bound
        R90_OK_group = R90_group[R90_group['CHAIN_SUBGROUP'].str.contains('_R90OK')]
        R90_OK_group['MATCH_CHAIN_SUBGROUP'] = R90_OK_group['CHAIN_SUBGROUP'].str[:-6]
        
        R30_R90OK_merged = R30_group[~(R30_group['CHAIN_SUBGROUP'].str.contains('_EXTRL'))].merge(R90_OK_group, on=['GPI_NDC', 'BG_FLAG', 'CLIENT', 'REGION'],
                                         how='inner')
        R30_R90OK_merged['price_cons'] = "" + R30_R90OK_merged['cons_R90'] - R30_R90OK_merged['cons_R30']

        bad_rows = R30_R90OK_merged[R30_R90OK_merged['LB_R90'] > R30_R90OK_merged['UB_R30']]

        if not bad_rows.empty:
            new_anomaly_df = pd.DataFrame({
                'CLIENT': bad_rows['CLIENT'].astype(str),
                'REGION': bad_rows['REGION'].astype(str),
                'GPI_NDC': bad_rows['GPI_NDC'].astype(str),
                'BG_FLAG': bad_rows['BG_FLAG'].astype(str),
                'EXCLUDE_REASON': bad_rows['MATCH_CHAIN_SUBGROUP'].astype(str) + ': R30 - R90OK'
            })
            anomaly_mes_gpi = pd.concat([new_anomaly_df, anomaly_mes_gpi], ignore_index=True)

        good_rows = R30_R90OK_merged[R30_R90OK_merged['LB_R90'] <= R30_R90OK_merged['UB_R30']]

        if not good_rows.empty:
            target = good_rows.price_cons
            for i in range(target.shape[0]):
                meas_specific_price_cons_list.append(target.iloc[i] <= 0)

        ########### segment 5 ends #####################################################################################

        logger.info("End M <= R90 <= R30 Pricing")
        end = time.time()
        logger.info("Run time: {} mins".format((end - start) / 60.))
        logger.info('--------------------')

        # file outputs ########################################################
        with open(meas_specific_price_cons_list_out, 'wb') as f:
            pickle.dump(meas_specific_price_cons_list, f)
        with open(anomaly_mes_gpi_out, 'wb') as f:
            pickle.dump(anomaly_mes_gpi, f)
        #####################################################################
            
        return meas_specific_price_cons_list, anomaly_mes_gpi

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'Measure Specific Pricing Constraints', repr(e), error_loc)
        raise e

###### End Measure Specific Pricing (ie M < R90 < R30) ########



###### Brand Generic Pricing (ie Generic < Brand) ########
def brand_generic_pricing_constraints(
        params_file_in: str,
        unc_flag: bool,
        lp_data_df_in: InputPath('pickle'),
        brnd_gnrc_price_cons_list_out: OutputPath('pickle'),
        anomaly_bg_gpi_out: OutputPath('pickle'),
        loglevel: str = 'INFO'
):
    import sys
    import os
    sys.path.append('/')
    import time
    import logging
    import pandas as pd
    import pickle
    import pulp
    import util_funcs as uf
    import BQ

    uf.write_params(params_file_in)
    import CPMO_parameters as p
    from CPMO_lp_functions import generatePricingDecisionVariables
    from CPMO_shared_functions import update_run_status

    try:
        out_path = os.path.join(p.FILE_LOG_PATH, 'ClientPharmacyMacOptimization.log')
        logger = uf.log_setup(log_file_path=out_path, loglevel=loglevel)

        # file inputs ##########################################
        with open(lp_data_df_in, 'rb') as f:
            lp_data_df = pickle.load(f)
        ########################################################
        
    
        logger.info('--------------------')
        logger.info("Generic <= Brand Pricing")
        start = time.time()

        # If the client contract/business allows generic prices to go above brand, apply p.GENERIC_UNRESTRICTED_CAP to the
        # Generic < Brand pricing constraints.
        if p.BRAND_GENERIC_UNRESTRICTED:
            gnrc_cons_cap = p.GENERIC_UNRESTRICTED_CAP
        else:
            gnrc_cons_cap = 1.0
        assert gnrc_cons_cap >= 1.0, "p.GENERIC_UNRESTRICTED_CAP < 1 is not recommended without business justification."
        
        ######### Modification of lp_data_df and creation of price_contraints_df ##########################
        lp_data_df = generatePricingDecisionVariables(lp_data_df)
        lp_data_df['GPI_12'] = lp_data_df.GPI.str[0:12]
        lp_data_df['GPI_Strength'] = lp_data_df.GPI.str[12:]
        
        price_constraints_col = ['CLIENT', 'BREAKOUT', 'MEASUREMENT', 'GPI_NDC', 'CHAIN_GROUP', 'CHAIN_SUBGROUP',
                                 'REGION', 'PHARMACY_TYPE', 'BG_FLAG', 'MAC1026_UNIT_PRICE', 'IMMUTABLE_REASON',
                                 'Price_Decision_Var', 'Price_Bounds', 'MAC_PRICE_UNIT_ADJ', 'Dec_Var_Name']
        
        if p.TRUECOST_CLIENT:
            price_constraints_col += ['PCD_IDX', 'NET_COST_GUARANTEE_UNIT']
            
        if p.GNRC_PRICE_FREEZE:
            price_constraints_df = lp_data_df[price_constraints_col].loc[(lp_data_df.PRICE_MUTABLE == 1) | (lp_data_df['BG_FLAG'] == 'G'), :]
        elif p.BRND_PRICE_FREEZE:
            price_constraints_df = lp_data_df[price_constraints_col].loc[(lp_data_df.PRICE_MUTABLE == 1) | (lp_data_df['BG_FLAG'] == 'B'), :]
        else:
            price_constraints_df = lp_data_df[price_constraints_col].loc[(lp_data_df.PRICE_MUTABLE == 1), :]
            
        price_constraints_df.MEASUREMENT = price_constraints_df.MEASUREMENT.replace(
            {'R30P': 'R30', 'R30N': 'R30', 'R90P': 'R90', 'R90N': 'R90'})
        
        if not p.TRUECOST_CLIENT:
            price_constraints_df['NET_COST_GUARANTEE_UNIT'] = 1.0
            price_constraints_df['PCD_IDX'] = '1'
        ####################################################################################################
        
        brnd_gnrc_price_cons_list = []
        brnd_gnrc_price_cons_list_rev = []
        anomaly_bg_gpi = pd.DataFrame(columns=['CLIENT', 'REGION', 'GPI_NDC', 'EXCLUDE_REASON'])

        price_constraints_df['Price_LB'] = price_constraints_df['Price_Bounds'].str[0]
        price_constraints_df['Price_UB'] = price_constraints_df['Price_Bounds'].str[1]

        gnrc_filtered = price_constraints_df[price_constraints_df['BG_FLAG'] == 'G']
        gnrc_group = gnrc_filtered.groupby(['GPI_NDC', 'CLIENT', 'REGION', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'MEASUREMENT'],
                                           as_index=False) \
            .agg(cons_gnrc=('Price_Decision_Var', 'first'),
                 LB_gnrc=('Price_LB', 'first'),
                 UB_gnrc=('Price_UB', 'first'),
                 MAC1026_UNIT_PRICE_gnrc=('MAC1026_UNIT_PRICE', 'first'),
                 NET_COST_GUARANTEE_gnrc=('NET_COST_GUARANTEE_UNIT', 'first'))
        
        brnd_filtered = price_constraints_df[price_constraints_df['BG_FLAG'] == 'B']
        brnd_group = brnd_filtered.groupby(['GPI_NDC', 'CLIENT', 'REGION', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'MEASUREMENT'],
                                           as_index=False) \
            .agg(cons_brnd=('Price_Decision_Var', 'first'),
                 LB_brnd=('Price_LB', 'first'),
                 UB_brnd=('Price_UB', 'first'),
                 PCD_IDX_brnd=('PCD_IDX', 'first'),
                 NET_COST_GUARANTEE_brnd=('NET_COST_GUARANTEE_UNIT', 'first'))
        
        brnd_gnrc_merged = brnd_group.merge(gnrc_group, on=['GPI_NDC', 'CLIENT', 'REGION', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'MEASUREMENT'], how='inner', validate='1:1')
            
        if p.TRUECOST_CLIENT:
            # filter out when gen floor > brand guarantee
            brnd_gnrc_merged = brnd_gnrc_merged[~((brnd_gnrc_merged['LB_gnrc'] > gnrc_cons_cap * brnd_gnrc_merged['UB_brnd']) & 
                                                  (brnd_gnrc_merged['MAC1026_UNIT_PRICE_gnrc'] > brnd_gnrc_merged['NET_COST_GUARANTEE_brnd']))]
            
            # filter out when gen PCD > brand NTM guarantee
            brnd_gnrc_merged = brnd_gnrc_merged[~((brnd_gnrc_merged['LB_gnrc'] > gnrc_cons_cap * brnd_gnrc_merged['UB_brnd']) & 
                  (brnd_gnrc_merged['PCD_IDX_brnd'].astype(str) != '1') & 
                  (brnd_gnrc_merged['NET_COST_GUARANTEE_gnrc'] > brnd_gnrc_merged['NET_COST_GUARANTEE_brnd']))]
        
        brnd_gnrc_merged['price_cons'] = "" + brnd_gnrc_merged['cons_gnrc'] - gnrc_cons_cap * brnd_gnrc_merged['cons_brnd']
        
        #### segment 1: creating the list of Price Decision vars constraints where Generic lower bound <= Brand upper bound
        bad_rows = brnd_gnrc_merged[brnd_gnrc_merged['LB_gnrc'] > gnrc_cons_cap * brnd_gnrc_merged['UB_brnd']]
        if not bad_rows.empty:
            for index, row in bad_rows.iterrows():
                
                logger.info(str(brnd_gnrc_merged.GPI_NDC[index]) + '-' + str(brnd_gnrc_merged.REGION[index]) + '-' + str(
                    brnd_gnrc_merged.CHAIN_SUBGROUP[index]) + '-' + str(brnd_gnrc_merged.MEASUREMENT[index]) + ': ' + 'GNRC - BRND')
            new_anomaly_df = pd.concat([pd.DataFrame({
                'CLIENT': bad_rows['CLIENT'].astype(str),
                'REGION': bad_rows['REGION'].astype(str),
                'GPI_NDC': bad_rows['GPI_NDC'].astype(str),
                'BG_FLAG': 'G',
                'EXCLUDE_REASON': bad_rows['CHAIN_SUBGROUP'].astype(str) + '_' + bad_rows['MEASUREMENT'].astype(str) + ': GNRC - BRND'
            }), pd.DataFrame({
                'CLIENT': bad_rows['CLIENT'].astype(str),
                'REGION': bad_rows['REGION'].astype(str),
                'GPI_NDC': bad_rows['GPI_NDC'].astype(str),
                'BG_FLAG': 'B',
                'EXCLUDE_REASON': bad_rows['CHAIN_SUBGROUP'].astype(str) + '_' + bad_rows['MEASUREMENT'].astype(str) + ': GNRC - BRND'
            })], ignore_index=True)
            anomaly_bg_gpi = pd.concat([new_anomaly_df, anomaly_bg_gpi], ignore_index=True)

        good_rows = brnd_gnrc_merged[brnd_gnrc_merged['LB_gnrc'] <= gnrc_cons_cap * brnd_gnrc_merged['UB_brnd']]
        if not good_rows.empty:
            target = good_rows.price_cons
            for i in range(target.shape[0]):
                brnd_gnrc_price_cons_list.append(target.iloc[i] <= 0)
       ########### segment 1 ends #####################################################################################    
        
        logger.info("End Gnrc < Brnd Pricing")
        end = time.time()
        logger.info("Run time: {} mins".format((end - start) / 60.))
        logger.info('--------------------')

        # file outputs ########################################################
        with open(brnd_gnrc_price_cons_list_out, 'wb') as f:
            pickle.dump(brnd_gnrc_price_cons_list, f)
        with open(anomaly_bg_gpi_out, 'wb') as f:
            pickle.dump(anomaly_bg_gpi, f)
        #####################################################################
            
        return brnd_gnrc_price_cons_list, anomaly_bg_gpi

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'Brand Generic Pricing Constraints', repr(e), error_loc)
        raise e

###### End Brand Generic Pricing (ie Gnrc < Brnd) ########


###### Adjudication Cap over Guarantee Pricing ########
def adj_cap_constraints(
        params_file_in: str,
        unc_flag: bool,
        lp_data_df_in: InputPath('pickle'),
        adj_cap_price_cons_list_out: OutputPath('pickle'),
        anomaly_adj_cap_gpi_out: OutputPath('pickle'),
        loglevel: str = 'INFO'
):
    import sys
    import os
    sys.path.append('/')
    import time
    import logging
    import pandas as pd
    import pickle
    import pulp
    import util_funcs as uf
    import BQ
    import numpy as np

    uf.write_params(params_file_in)
    import CPMO_parameters as p
    from CPMO_lp_functions import generatePricingDecisionVariables
    from CPMO_shared_functions import update_run_status

    try:
        out_path = os.path.join(p.FILE_LOG_PATH, 'ClientPharmacyMacOptimization.log')
        logger = uf.log_setup(log_file_path=out_path, loglevel=loglevel)

        # file inputs ##########################################
        with open(lp_data_df_in, 'rb') as f:
            lp_data_df = pickle.load(f)
        ########################################################
        
        logger.info('--------------------')
        logger.info("Pricing below cap * guaranteed price for the year")
        start = time.time()
        
        ######### Modification of lp_data_df and creation of price_contraints_df ##########################
        lp_data_df = generatePricingDecisionVariables(lp_data_df)
        lp_data_df['GPI_12'] = lp_data_df.GPI.str[0:12]
        lp_data_df['GPI_Strength'] = lp_data_df.GPI.str[12:]
        
        lp_data_df['DISP_FEE_PROJ_LAG'] = lp_data_df['CLAIMS_PROJ_LAG'] * lp_data_df['AVG_DISP_FEE']
        lp_data_df['DISP_FEE_PROJ_EOY'] = lp_data_df['CLAIMS_PROJ_EOY'] * lp_data_df['AVG_DISP_FEE']
        lp_data_df['TARGET_DISP_FEE_PROJ_LAG'] = lp_data_df['CLAIMS_PROJ_LAG'] * lp_data_df['AVG_TARGET_DISP_FEE']
        lp_data_df['TARGET_DISP_FEE_PROJ_EOY'] = lp_data_df['CLAIMS_PROJ_EOY'] * lp_data_df['AVG_TARGET_DISP_FEE']
        
        price_constraints_col = ['CLIENT', 'BREAKOUT', 'MEASUREMENT', 'GPI', 'GPI_NDC', 'CHAIN_GROUP', 'CHAIN_SUBGROUP',
                                 'REGION', 'PHARMACY_TYPE', 'BG_FLAG', 'TARG_INGCOST_ADJ', 'TARG_INGCOST_ADJ_PROJ_LAG', 
                                 'TARG_INGCOST_ADJ_PROJ_EOY', 'TARGET_DISP_FEE', 'TARGET_DISP_FEE_PROJ_LAG', 
                                 'TARGET_DISP_FEE_PROJ_EOY','PRICE_REIMB', 'LAG_REIMB','QTY_PROJ_LAG', 
                                 'QTY_PROJ_EOY', 'DISP_FEE', 'DISP_FEE_PROJ_LAG', 'DISP_FEE_PROJ_EOY', 'NADAC_WAC_YTD', 
                                 'NADAC', 'WAC', 'PCD_IDX', 'Price_Decision_Var', 'Price_Bounds', 'Dec_Var_Name']
        
        adj_cap_price_cons_list = []
        anomaly_adj_cap_gpi = pd.DataFrame(columns=['CLIENT', 'REGION', 'GPI_NDC', 'EXCLUDE_REASON'])

        
        if p.TRUECOST_CLIENT and p.BRAND_OPT:
            price_constraints_df = lp_data_df[price_constraints_col].loc[(lp_data_df.CLAIMS > 0) & 
                                                                         (lp_data_df.PCD_IDX == 1) & 
                                                                         (lp_data_df.GPI_ONLY != 0), :]
            price_constraints_df = price_constraints_df[price_constraints_df['BG_FLAG'] == 'B'] # remove once we know generic adj cap

            price_constraints_df.MEASUREMENT = price_constraints_df.MEASUREMENT.replace(
                {'R30P': 'R30', 'R30N': 'R30', 'R90P': 'R90', 'R90N': 'R90'})

            ####################################################################################################
        
            price_constraints_df['Price_LB'] = price_constraints_df['Price_Bounds'].str[0]
            price_constraints_df['Price_UB'] = price_constraints_df['Price_Bounds'].str[1]
            
            # As per contract, GPI's adjudication are compared against guarantee or, if greater, NADAC (or WAC, if NADAC is unavailable).
            # NADAC and WAC will be calculated as of the adjudication date
            price_constraints_df.loc[price_constraints_df['NADAC'] == 0, 'NADAC'] = None
            price_constraints_df.loc[price_constraints_df['WAC'] == 0, 'WAC'] = None
            
            price_constraints_df['TARG_INGCOST_NW_YTD'] = price_constraints_df[['TARG_INGCOST_ADJ','NADAC_WAC_YTD']].max(axis=1)
            price_constraints_df['TARG_INGCOST_NW_LAG'] = np.nanmax([price_constraints_df['TARG_INGCOST_ADJ_PROJ_LAG'],
                                                                     price_constraints_df.NADAC.combine_first(price_constraints_df.WAC) * \
                                                                     price_constraints_df.QTY_PROJ_LAG],  axis=0)
            price_constraints_df['TARG_INGCOST_NW_EOY'] = np.nanmax([price_constraints_df['TARG_INGCOST_ADJ_PROJ_EOY'],
                                                                     price_constraints_df.NADAC.combine_first(price_constraints_df.WAC) * \
                                                                     price_constraints_df.QTY_PROJ_EOY], axis=0)

            grouped_df = price_constraints_df.groupby(['GPI', 'GPI_NDC', 'CLIENT', 'REGION', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'MEASUREMENT', 'BG_FLAG'],
                                               as_index=False) \
                .agg(cons=('Price_Decision_Var', 'first'),
                        LB=('Price_LB', 'first'),
                        UB=('Price_UB', 'first'),
                        TARGET_DISP_FEE=('TARGET_DISP_FEE', 'sum'),
                        TARGET_DISP_FEE_PROJ_LAG=('TARGET_DISP_FEE_PROJ_LAG', 'sum'),
                        TARGET_DISP_FEE_PROJ_EOY=('TARGET_DISP_FEE_PROJ_EOY', 'sum'),
                        TARG_INGCOST_NW_YTD=('TARG_INGCOST_NW_YTD', 'sum'),
                        TARG_INGCOST_NW_LAG=('TARG_INGCOST_NW_LAG', 'sum'),
                        TARG_INGCOST_NW_EOY=('TARG_INGCOST_NW_EOY', 'sum'),
                        PRICE_REIMB=('PRICE_REIMB', 'sum'),
                        LAG_REIMB=('LAG_REIMB', 'sum'),
                        QTY_PROJ_EOY=('QTY_PROJ_EOY', 'sum'),
                        DISP_FEE=('DISP_FEE', 'sum'),
                        DISP_FEE_PROJ_LAG=('DISP_FEE_PROJ_LAG', 'sum'),
                        DISP_FEE_PROJ_EOY=('DISP_FEE_PROJ_EOY', 'sum'))
            
            if p.FULL_YEAR:
                grouped_df['GUARANTEED_SCRIPT_PRICE'] = grouped_df['TARG_INGCOST_NW_EOY'] + grouped_df['TARGET_DISP_FEE_PROJ_EOY']
                grouped_df['ACTUAL_SCRIPT_PRICE_VAR'] = "" + grouped_df['cons'] * grouped_df['QTY_PROJ_EOY']
                grouped_df['ACTUAL_REIMB'] = grouped_df['DISP_FEE_PROJ_EOY']
                grouped_df['ACTUAL_SCRIPT_PRICE_LB'] = (grouped_df['LB'] * grouped_df['QTY_PROJ_EOY']) + grouped_df['DISP_FEE_PROJ_EOY']
            else:
                grouped_df['GUARANTEED_SCRIPT_PRICE'] = grouped_df['TARG_INGCOST_NW_YTD'] + grouped_df['TARG_INGCOST_NW_LAG'] + grouped_df['TARG_INGCOST_NW_EOY'] + grouped_df['TARGET_DISP_FEE'] + grouped_df['TARGET_DISP_FEE_PROJ_LAG'] + grouped_df['TARGET_DISP_FEE_PROJ_EOY']
                grouped_df['ACTUAL_SCRIPT_PRICE_VAR'] = "" + grouped_df['cons'] * grouped_df['QTY_PROJ_EOY']
                grouped_df['ACTUAL_REIMB'] = grouped_df['PRICE_REIMB'] + grouped_df['LAG_REIMB'] + grouped_df['DISP_FEE'] + grouped_df['DISP_FEE_PROJ_LAG'] + grouped_df['DISP_FEE_PROJ_EOY']
                grouped_df['ACTUAL_SCRIPT_PRICE_LB'] = grouped_df['PRICE_REIMB'] + grouped_df['LAG_REIMB'] + (grouped_df['LB'] * grouped_df['QTY_PROJ_EOY']) + grouped_df['DISP_FEE'] + grouped_df['DISP_FEE_PROJ_LAG'] + grouped_df['DISP_FEE_PROJ_EOY']

            contraints = grouped_df.groupby(['CLIENT', 'REGION', 'GPI', 'BG_FLAG'], as_index=False) \
                .agg(GPI_NDC=('GPI_NDC', 'min'),
                        GUARANTEED_SCRIPT_PRICE=('GUARANTEED_SCRIPT_PRICE', 'sum'),
                        ACTUAL_SCRIPT_PRICE_VAR=('ACTUAL_SCRIPT_PRICE_VAR', list),
                        ACTUAL_REIMB=('ACTUAL_REIMB', 'sum'),
                        ACTUAL_SCRIPT_PRICE_LB=('ACTUAL_SCRIPT_PRICE_LB', 'sum'))
            def create_price_cons(row):
                const = "" + row['ACTUAL_SCRIPT_PRICE_VAR'][0] + row['ACTUAL_REIMB']
                if len(row['ACTUAL_SCRIPT_PRICE_VAR']) > 1:
                    for i in range(1, len(row['ACTUAL_SCRIPT_PRICE_VAR'])):
                        const += row['ACTUAL_SCRIPT_PRICE_VAR'][i]
                if row['BG_FLAG'] == 'G':
                    return (const) /row['GUARANTEED_SCRIPT_PRICE'] - 1.499
                else:
                    return (const) /row['GUARANTEED_SCRIPT_PRICE'] - 1.149
            contraints['price_cons'] = contraints.apply(create_price_cons, axis=1)

            #### segment 1: creating the list of Price Decision vars constraints where GPI overall adjudication 
            #### is less than 1.15 * guaranteed price for Brand, or 1.5 * guaranteed price for Generic
            contraints['EXCLUDE_REASON'] = np.where(contraints['BG_FLAG'] == 'G', 'GNRC ADJ > 1.50 * GUARANTEE', 'BRND ADJ > 1.15 * GUARANTEE')
            bad_rows = contraints[((contraints['BG_FLAG'] == 'B') & (contraints['ACTUAL_SCRIPT_PRICE_LB'] / contraints['GUARANTEED_SCRIPT_PRICE'] > 1.149)) | 
                                  ((contraints['BG_FLAG'] == 'G') & (contraints['ACTUAL_SCRIPT_PRICE_LB'] / contraints['GUARANTEED_SCRIPT_PRICE'] > 1.499))]
            if not bad_rows.empty:
                for index, row in bad_rows.iterrows():
                    logger.info(str(contraints.GPI_NDC[index]) + ': ' + str(contraints.EXCLUDE_REASON[index]))
                new_anomaly_df = pd.DataFrame({
                    'CLIENT': bad_rows['CLIENT'].astype(str),
                    'REGION': bad_rows['REGION'].astype(str),
                    'GPI_NDC': bad_rows['GPI_NDC'].astype(str),
                    'BG_FLAG': bad_rows['BG_FLAG'].astype(str),
                    'EXCLUDE_REASON': bad_rows['EXCLUDE_REASON'].astype(str)
                })
                anomaly_adj_cap_gpi = pd.concat([new_anomaly_df, anomaly_adj_cap_gpi], ignore_index=True)

            good_rows = contraints[((contraints['BG_FLAG'] == 'B') & (contraints['ACTUAL_SCRIPT_PRICE_LB'] / contraints['GUARANTEED_SCRIPT_PRICE'] <= 1.149)) | 
                                  ((contraints['BG_FLAG'] == 'G') & (contraints['ACTUAL_SCRIPT_PRICE_LB'] / contraints['GUARANTEED_SCRIPT_PRICE'] <= 1.499))]
            
            if not good_rows.empty:
                target = good_rows.price_cons
                for i in range(target.shape[0]):
                    adj_cap_price_cons_list.append(target.iloc[i] <= 0)
           ########### segment 1 ends #####################################################################################    
        
        logger.info("End Adjudication < Guarantee Pricing * Cap")
        end = time.time()
        logger.info("Run time: {} mins".format((end - start) / 60.))
        logger.info('--------------------')

        # file outputs ########################################################
        with open(adj_cap_price_cons_list_out, 'wb') as f:
            pickle.dump(adj_cap_price_cons_list, f)
        with open(anomaly_adj_cap_gpi_out, 'wb') as f:
            pickle.dump(anomaly_adj_cap_gpi, f)
        #####################################################################
            
        return adj_cap_price_cons_list, anomaly_adj_cap_gpi

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'Adj Cap Pricing Constraints', repr(e), error_loc)
        raise e

###### End Adjudication < Guarantee Pricing * Cap ########


###### All Other Pharmacy Pricing greater than CVS Pricing ########
def cvs_parity_price_constraint(
        params_file_in: str,
        unc_flag: bool,
        lp_data_df_in: InputPath('pickle'),
        pref_pharm_list_in: InputPath('pickle'),
        pref_other_price_cons_list_out: OutputPath('pickle'),
        anomaly_pref_gpi_out: OutputPath('pickle'),
        loglevel: str = 'INFO'
        # # kube_run: bool = True,
):
    import sys
    import os
    import time
    import logging
    import pandas as pd
    import pickle
    import pulp
    import util_funcs as uf
    import BQ

    uf.write_params(params_file_in)
    sys.path.append('/')
    import CPMO_parameters as p
    from CPMO_lp_functions import generatePricingDecisionVariables
    from CPMO_shared_functions import update_run_status
    try:
        out_path = os.path.join(p.FILE_LOG_PATH, 'ClientPharmacyMacOptimization.log')
        logger = uf.log_setup(log_file_path=out_path, loglevel=loglevel)

        # input files
        with open(lp_data_df_in, 'rb') as f:
            lp_data_df = pickle.load(f)
        with open(pref_pharm_list_in, 'rb') as f:
            pref_pharm_list = pickle.load(f)

            logger.info('--------------------')
            logger.info("All Other Pharmacies >= CVS Pricing")
            start = time.time()

        lp_data_df = generatePricingDecisionVariables(lp_data_df)
        lp_data_df['GPI_12'] = lp_data_df.GPI.str[0:12]
        lp_data_df['GPI_Strength'] = lp_data_df.GPI.str[12:]

        price_constraints_col = ['CLIENT', 'BREAKOUT', 'MEASUREMENT', 'GPI', 'NDC', 'GPI_NDC', 'BG_FLAG', 'GPI_ONLY',
                                 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'REGION', 'PHARMACY_TYPE',
                                 'Price_Decision_Var', 'Price_Bounds', 'MAC_PRICE_UNIT_ADJ', 'Dec_Var_Name',
                                 'MAC_LIST'
                                ]

        price_constraints_df = lp_data_df[price_constraints_col].loc[
                               (lp_data_df.PRICE_MUTABLE == 1) & (lp_data_df.MEASUREMENT != 'M30'), :]

        if 'CVSSP' in price_constraints_df.CHAIN_SUBGROUP.values:
            pref_subchain = 'CVSSP'
        else:
            pref_subchain = 'CVS'
        # pref_other = 'PREF_OTH'

        pref_other_price_cons_list = []
        gpi_arr = price_constraints_df.GPI.unique()
        anomaly_pref_gpi = pd.DataFrame(columns=['CLIENT', 'REGION', 'GPI_NDC', 'BG_FLAG', 'EXCLUDE_REASON'])

        ##Filter 1: Since all the calculations below would need to be only on the retail breakouts- filtering the price_constraints_df to remove all MAIL breakouts
        
        ##Filter 2: cvs pricing is independent of X vcml prices, and therefore we are excluding them from constraint list just like we ignore MAIL. Failure to do this will lead to many unwanted conflicting gpis with CVS - EXTRL conflict
        
        price_constraints_df_no_mail = price_constraints_df[
            (price_constraints_df.MEASUREMENT != 'M30') & ~(price_constraints_df.CHAIN_SUBGROUP.str.contains('_EXTRL'))]

        ########################################################################################################

        ###preferred and non preferred pharmacy list preperation################################################
        preferred_chains = [x for i in pref_pharm_list['PREF_PHARM'].values for x in i.split(',') if
                            i not in ['none', 'None', 'NONE']]
        if p.CLIENT_TYPE == 'MEDD':
            gnrc_non_preferred_chains = [i for i in (p.NON_CAPPED_PHARMACY_LIST['GNRC'] + p.COGS_PHARMACY_LIST['GNRC']) if i not in preferred_chains]
            brnd_non_preferred_chains = [i for i in (p.NON_CAPPED_PHARMACY_LIST['BRND'] + p.COGS_PHARMACY_LIST['BRND']) if i not in preferred_chains]
        elif (p.CLIENT_TYPE == 'COMMERCIAL' or p.CLIENT_TYPE == 'MEDICAID'):
            gnrc_non_preferred_chains = [i for i in list(p.PHARMACY_LIST['GNRC']) if i not in preferred_chains]
            brnd_non_preferred_chains = [i for i in list(p.PHARMACY_LIST['BRND']) if i not in preferred_chains]
        cvs_rule_neccessary = False
        if ('CVS' in preferred_chains):
            if p.CLIENT_TYPE == 'MEDD':
                gnrc_pref_other_chains = [x for x in (p.NON_CAPPED_PHARMACY_LIST['GNRC'] + p.COGS_PHARMACY_LIST['GNRC']) if x != 'CVS']
                brnd_pref_other_chains = [x for x in (p.NON_CAPPED_PHARMACY_LIST['BRND'] + p.COGS_PHARMACY_LIST['BRND']) if x != 'CVS']
            elif (p.CLIENT_TYPE == 'COMMERCIAL' or p.CLIENT_TYPE == 'MEDICAID'):
                gnrc_pref_other_chains = [x for x in list(p.PHARMACY_LIST['GNRC']) if x != 'CVS'] # x for x in preferred_chains if x != 'CVS']
                brnd_pref_other_chains = [x for x in list(p.PHARMACY_LIST['BRND']) if x != 'CVS'] # x for x in preferred_chains if x != 'CVS']
            cvs_rule_neccessary = True

        elif len(preferred_chains) == 0:
            gnrc_pref_other_chains = [x for x in gnrc_non_preferred_chains if x != 'CVS']
            brnd_pref_other_chains = [x for x in brnd_non_preferred_chains if x != 'CVS']
            cvs_rule_neccessary = True

        ###preferred and non preferred dataframe preparation###################################################
        cvs_gpi = price_constraints_df_no_mail[
            price_constraints_df_no_mail.CHAIN_SUBGROUP == pref_subchain]  ###dataframe  with CVS or CVSSP as the chain subgroup
        gnrc_other_subchains = price_constraints_df_no_mail[
            price_constraints_df_no_mail.CHAIN_GROUP.isin(gnrc_pref_other_chains)].CHAIN_SUBGROUP.unique()
        brnd_other_subchains = price_constraints_df_no_mail[
            price_constraints_df_no_mail.CHAIN_GROUP.isin(brnd_pref_other_chains)].CHAIN_SUBGROUP.unique()
        other_gpi = price_constraints_df_no_mail.loc[((price_constraints_df_no_mail.CHAIN_SUBGROUP.isin(
            gnrc_other_subchains)) & (price_constraints_df_no_mail.BG_FLAG == 'G')) | ((price_constraints_df_no_mail.CHAIN_SUBGROUP.isin(
            brnd_other_subchains)) & (price_constraints_df_no_mail.BG_FLAG == 'B'))]  ##dataframe with other than CVS or CVSSP chain subgroups
        ########################################################################################################

        ##########master join between the prefererred and other chain_subgroup dataframes!
        cvs_other_master = cvs_gpi.merge(other_gpi, how='inner',
                                         on=['CLIENT', 'REGION', 'MEASUREMENT', 'BREAKOUT', 'GPI', 'NDC', 'BG_FLAG'],
                                         suffixes=('_pref', '_other'))
        cvs_other_master['pref_lower_bound'], cvs_other_master['pref_upper_bound'] = zip(
            *cvs_other_master.Price_Bounds_pref)
        cvs_other_master['pref_other_lower_bound'], cvs_other_master['pref_other_upper_bound'] = zip(
            *cvs_other_master.Price_Bounds_other)
        ##################################################################################

        ###################################################################################################################################################
        #####Condition 1: Both CVS and Other Pref sub chains have GPI_ONLY=0 for a given combination of 'CLIENT','REGION','MEASUREMENT','BREAKOUT','GPI','NDC'
        cvs_other_bothGPI0 = cvs_other_master.loc[
            (cvs_other_master.GPI_ONLY_pref == 0) & (cvs_other_master.GPI_ONLY_other == 0)]

        if cvs_other_bothGPI0.shape[0] > 0:
            ##appending the anomaly dataframe if the anomaly condition 1 is met
            anomaly_pref_gpi_1 = cvs_other_bothGPI0[
                cvs_other_bothGPI0.pref_lower_bound * p.PREF_OTHER_FACTOR > cvs_other_bothGPI0.pref_other_upper_bound]
            if anomaly_pref_gpi_1.shape[0] > 0:
                anomaly_pref_gpi_1['GPI_NDC'], anomaly_pref_gpi_1['BG_FLAG'], anomaly_pref_gpi_1['EXCLUDE_REASON'] = anomaly_pref_gpi_1[
                                                                                          'GPI'] + '_***********', \
                                                                                      anomaly_pref_gpi_1['BG_FLAG'], \
                                                                                      anomaly_pref_gpi_1[
                                                                                          'CHAIN_SUBGROUP_pref'] + '-' + \
                                                                                      anomaly_pref_gpi_1[
                                                                                          'CHAIN_SUBGROUP_other']
                anomaly_pref_gpi = anomaly_pref_gpi.append(anomaly_pref_gpi_1)
                # logger.info(str(anomaly_pref_gpi_1.GPI) + '-***********-' + str(anomaly_pref_gpi_1.REGION) + ': ' + str(anomaly_pref_gpi_1.CHAIN_SUBGROUP_pref)+'-'+str(anomaly_pref_gpi_1.CHAIN_SUBGROUP_other))
            ##normal-non anomaly condition 1
            non_anomaly_pref_gpi_1 = cvs_other_bothGPI0[
                cvs_other_bothGPI0.pref_lower_bound * p.PREF_OTHER_FACTOR <= cvs_other_bothGPI0.pref_other_upper_bound]
            non_anomaly_pref_gpi_1['price_cons'] = non_anomaly_pref_gpi_1[
                                                       'Price_Decision_Var_pref'] * p.PREF_OTHER_FACTOR - \
                                                   non_anomaly_pref_gpi_1['Price_Decision_Var_other']

            for i in range(non_anomaly_pref_gpi_1.shape[0]):
                pref_other_price_cons_list.append(non_anomaly_pref_gpi_1.price_cons.iloc[i] <= 0)

            ##appending the anomaly dataframe if the anomaly condition 2 is met
            anomaly_pref_gpi_2 = cvs_other_bothGPI0[
                cvs_other_bothGPI0.pref_upper_bound < p.RETAIL_RETAIL_BOUND * cvs_other_bothGPI0.pref_other_lower_bound]
            if anomaly_pref_gpi_2.shape[0] > 0:
                anomaly_pref_gpi_2['GPI_NDC'], anomaly_pref_gpi_2['BG_FLAG'], anomaly_pref_gpi_2['EXCLUDE_REASON'] = anomaly_pref_gpi_2[
                                                                                          'GPI'] + '_***********', \
                                                                                      anomaly_pref_gpi_2['BG_FLAG'], \
                                                                                      anomaly_pref_gpi_2[
                                                                                          'CHAIN_SUBGROUP_pref'] + '-' + \
                                                                                      anomaly_pref_gpi_2[
                                                                                          'CHAIN_SUBGROUP_other']
                anomaly_pref_gpi = anomaly_pref_gpi.append(anomaly_pref_gpi_2)
                ##logger.info(str(anomaly_pref_gpi_2.GPI) + '-***********-' + str(anomaly_pref_gpi_2.REGION) + ': ' + str(anomaly_pref_gpi_2.CHAIN_SUBGROUP_pref)+'-'+str(anomaly_pref_gpi_2.CHAIN_SUBGROUP_other))
            
            ##normal-non anomaly condition 2
            non_anomaly_pref_gpi_2 = cvs_other_bothGPI0[
                cvs_other_bothGPI0.pref_upper_bound >= p.RETAIL_RETAIL_BOUND * cvs_other_bothGPI0.pref_other_lower_bound]
            non_anomaly_pref_gpi_2['price_cons'] = -non_anomaly_pref_gpi_2[
                'Price_Decision_Var_pref'] + p.RETAIL_RETAIL_BOUND * non_anomaly_pref_gpi_2['Price_Decision_Var_other']
            for i in range(non_anomaly_pref_gpi_2.shape[0]):
                pref_other_price_cons_list.append(non_anomaly_pref_gpi_2.price_cons.iloc[i] <= 0)
        #####################Condition 1 ends##############################################################################################################
        ###################################################################################################################################################

        ###################################################################################################################################################
        #####Condition 2: Only Other Pref sub chains have GPI_ONLY=0 for a given combination of 'CLIENT','REGION','MEASUREMENT','BREAKOUT','GPI','NDC'
        cvs_other_only_other_GPI0 = cvs_other_master.loc[
            (cvs_other_master.GPI_ONLY_pref != 0) & (cvs_other_master.GPI_ONLY_other == 0)]

        if cvs_other_only_other_GPI0.shape[0] > 0:
            ##if there are more than one rows for the cvs subgroup that dont have GPI_ONLY=0 for a given combination of 'CLIENT','REGION','MEASUREMENT','BREAKOUT','GPI','NDC'
            bad_rows_master_1 = cvs_other_only_other_GPI0.groupby(
                ['CLIENT', 'REGION', 'MEASUREMENT', 'BREAKOUT', 'GPI', 'NDC', 'BG_FLAG', 'CHAIN_SUBGROUP_pref']) \
                .agg(count_rows=('CLIENT', 'count')).reset_index()
            bad_rows_actual_1 = bad_rows_master_1[bad_rows_master_1['count_rows'] > 1]
            ##to know what price decision variables correspond to the bad rows
            bad_rows_detailed = bad_rows_actual_1.merge(cvs_other_only_other_GPI0, how='inner',
                                                        on=['CLIENT', 'REGION', 'MEASUREMENT', 'BREAKOUT', 'GPI', 'NDC', 'BG_FLAG',
                                                            'CHAIN_SUBGROUP_pref'])
            if bad_rows_actual_1.shape[0] > 0:
                ##logger.info('ERROR with ' + bad_rows_detailed.Price_Decision_Var_pref.values)
                assert len(
                    bad_rows_detailed.Price_Decision_Var_pref) == 1, "len(bad_rows_detailed.Price_Decision_Var_pref)==1"

            ##if there is one or less rows for cvs subgroup that dont have GPI_ONLY=0 for a given combination of 'CLIENT','REGION','MEASUREMENT','BREAKOUT','GPI','NDC'
            anomaly_pref_gpi_3 = cvs_other_only_other_GPI0[
                cvs_other_only_other_GPI0.pref_lower_bound * p.PREF_OTHER_FACTOR > cvs_other_only_other_GPI0.pref_other_upper_bound]
            if anomaly_pref_gpi_3.shape[0] > 0:
                anomaly_pref_gpi_3['GPI_NDC'], anomaly_pref_gpi_3['BG_FLAG'], anomaly_pref_gpi_3['EXCLUDE_REASON'] = anomaly_pref_gpi_3[
                                                                                          'GPI'] + '_***********', \
                                                                                      anomaly_pref_gpi_3['BG_FLAG'], \
                                                                                      anomaly_pref_gpi_3[
                                                                                          'CHAIN_SUBGROUP_pref'] + '-' + \
                                                                                      anomaly_pref_gpi_3[
                                                                                          'CHAIN_SUBGROUP_other']
                anomaly_pref_gpi = anomaly_pref_gpi.append(anomaly_pref_gpi_3)
                # logger.info(str(anomaly_pref_gpi_3.GPI) + '-***********-' + str(anomaly_pref_gpi_3.REGION) + ': ' + str(anomaly_pref_gpi_3.CHAIN_SUBGROUP_pref)+'-'+str(anomaly_pref_gpi_3.CHAIN_SUBGROUP_other))
            ##normal-non anomaly condition 1
            non_anomaly_pref_gpi_3 = cvs_other_only_other_GPI0[
                cvs_other_only_other_GPI0.pref_lower_bound * p.PREF_OTHER_FACTOR <= cvs_other_only_other_GPI0.pref_other_upper_bound]
            non_anomaly_pref_gpi_3['price_cons'] = non_anomaly_pref_gpi_3[
                                                       'Price_Decision_Var_pref'] * p.PREF_OTHER_FACTOR - \
                                                   non_anomaly_pref_gpi_3['Price_Decision_Var_other']

            for i in range(non_anomaly_pref_gpi_3.shape[0]):
                pref_other_price_cons_list.append(non_anomaly_pref_gpi_3.price_cons.iloc[i] <= 0)

            ##appending the anomaly dataframe if the anomaly condition 2 is met
            anomaly_pref_gpi_4 = cvs_other_only_other_GPI0[
                cvs_other_only_other_GPI0.pref_upper_bound < p.RETAIL_RETAIL_BOUND * cvs_other_only_other_GPI0.pref_other_lower_bound]
            if anomaly_pref_gpi_4.shape[0] > 0:
                anomaly_pref_gpi_4['GPI_NDC'], anomaly_pref_gpi_4['BG_FLAG'], anomaly_pref_gpi_4['EXCLUDE_REASON'] = anomaly_pref_gpi_4[
                                                                                          'GPI'] + '_***********', \
                                                                                      anomaly_pref_gpi_4['BG_FLAG'], \
                                                                                      anomaly_pref_gpi_4[
                                                                                          'CHAIN_SUBGROUP_pref'] + '-' + \
                                                                                      anomaly_pref_gpi_4[
                                                                                          'CHAIN_SUBGROUP_other']
                anomaly_pref_gpi = anomaly_pref_gpi.append(anomaly_pref_gpi_4)
                ##logger.info(str(anomaly_pref_gpi_4.GPI) + '-***********-' + str(anomaly_pref_gpi_4.REGION) + ': ' + str(anomaly_pref_gpi_4.CHAIN_SUBGROUP_pref)+'-'+str(anomaly_pref_gpi_4.CHAIN_SUBGROUP_other))
            ##normal-non anomaly condition 2
            non_anomaly_pref_gpi_4 = cvs_other_only_other_GPI0[
                cvs_other_only_other_GPI0.pref_upper_bound >= p.RETAIL_RETAIL_BOUND * cvs_other_only_other_GPI0.pref_other_lower_bound]
            non_anomaly_pref_gpi_4['price_cons'] = -non_anomaly_pref_gpi_4[
                'Price_Decision_Var_pref'] + p.RETAIL_RETAIL_BOUND * non_anomaly_pref_gpi_4['Price_Decision_Var_other']
            for i in range(non_anomaly_pref_gpi_4.shape[0]):
                pref_other_price_cons_list.append(non_anomaly_pref_gpi_4.price_cons.iloc[i] <= 0)
        #####################Condition 2 ends##############################################################################################################
        ###################################################################################################################################################

        ###################################################################################################################################################
        #####Condition 3: Only CVS sub chains have GPI_ONLY=0 for a given combination of 'CLIENT','REGION','MEASUREMENT','BREAKOUT','GPI','NDC'
        cvs_other_only_cvs_GPI0 = cvs_other_master.loc[
            (cvs_other_master.GPI_ONLY_pref == 0) & (cvs_other_master.GPI_ONLY_other != 0)]

        if cvs_other_only_cvs_GPI0.shape[0] > 0:
            ##if there are more than one rows for the cvs subgroup that dont have GPI_ONLY=0 for a given combination of 'CLIENT','REGION','MEASUREMENT','BREAKOUT','GPI','NDC'
            bad_rows_master_2 = cvs_other_only_cvs_GPI0.groupby(
                ['CLIENT', 'REGION', 'MEASUREMENT', 'BREAKOUT', 'GPI', 'NDC', 'BG_FLAG', 'CHAIN_SUBGROUP_pref']) \
                .agg(count_rows=('CLIENT', 'count')).reset_index()
            bad_rows_actual_2 = bad_rows_master_2[bad_rows_master_2.count_rows > 1]
            ##to know what price decision variables correspond to the bad rows
            bad_rows_detailed = bad_rows_actual_2.merge(cvs_other_only_cvs_GPI0, how='inner',
                                                        on=['CLIENT', 'REGION', 'MEASUREMENT', 'BREAKOUT', 'GPI', 'NDC', 'BG_FLAG',
                                                            'CHAIN_SUBGROUP_pref'])
            if bad_rows_actual_2.shape[0] > 0:
                ##logger.info('ERROR with ' + bad_rows_detailed.Price_Decision_Var_pref.values)
                assert len(
                    bad_rows_detailed.Price_Decision_Var_other) == 1, "len(bad_rows_detailed.Price_Decision_Var_other)==1"

            ##if there is one or less rows for cvs subgroup that dont have GPI_ONLY=0 for a given combination of 'CLIENT','REGION','MEASUREMENT','BREAKOUT','GPI','NDC'
            anomaly_pref_gpi_5 = cvs_other_only_cvs_GPI0[
                cvs_other_only_cvs_GPI0.pref_lower_bound * p.PREF_OTHER_FACTOR > cvs_other_only_cvs_GPI0.pref_other_upper_bound]
            if anomaly_pref_gpi_5.shape[0] > 0:
                anomaly_pref_gpi_5['GPI_NDC'], anomaly_pref_gpi_5['BG_FLAG'], anomaly_pref_gpi_5['EXCLUDE_REASON'] = anomaly_pref_gpi_5[
                                                                                          'GPI'] + '_***********', \
                                                                                      anomaly_pref_gpi_5['BG_FLAG'], \
                                                                                      anomaly_pref_gpi_5[
                                                                                          'CHAIN_SUBGROUP_pref'] + '-' + \
                                                                                      anomaly_pref_gpi_5[
                                                                                          'CHAIN_SUBGROUP_other']
                anomaly_pref_gpi = anomaly_pref_gpi.append(anomaly_pref_gpi_5)
                ##logger.info(str(anomaly_pref_gpi_5.GPI) + '-***********-' + str(anomaly_pref_gpi_5.REGION) + ': ' + str(anomaly_pref_gpi_5.CHAIN_SUBGROUP_pref)+'-'+str(anomaly_pref_gpi_5.CHAIN_SUBGROUP_other))
            ##normal-non anomaly condition 1
            non_anomaly_pref_gpi_5 = cvs_other_only_cvs_GPI0[
                cvs_other_only_cvs_GPI0.pref_lower_bound * p.PREF_OTHER_FACTOR <= cvs_other_only_cvs_GPI0.pref_other_upper_bound]
            non_anomaly_pref_gpi_5['price_cons'] = non_anomaly_pref_gpi_5[
                                                       'Price_Decision_Var_pref'] * p.PREF_OTHER_FACTOR - \
                                                   non_anomaly_pref_gpi_5['Price_Decision_Var_other']

            for i in range(non_anomaly_pref_gpi_5.shape[0]):
                pref_other_price_cons_list.append(non_anomaly_pref_gpi_5.price_cons.iloc[i] <= 0)

            ##appending the anomaly dataframe if the anomaly condition 2 is met
            anomaly_pref_gpi_6 = cvs_other_only_cvs_GPI0[
                cvs_other_only_cvs_GPI0.pref_upper_bound < p.RETAIL_RETAIL_BOUND * cvs_other_only_cvs_GPI0.pref_other_lower_bound]
            if anomaly_pref_gpi_6.shape[0] > 0:
                anomaly_pref_gpi_6['GPI_NDC'], anomaly_pref_gpi_6['BG_FLAG'], anomaly_pref_gpi_6['EXCLUDE_REASON'] = anomaly_pref_gpi_6[
                                                                                          'GPI'] + '_***********', \
                                                                                      anomaly_pref_gpi_6['BG_FLAG'], \
                                                                                      anomaly_pref_gpi_6[
                                                                                          'CHAIN_SUBGROUP_pref'] + '-' + \
                                                                                      anomaly_pref_gpi_6[
                                                                                          'CHAIN_SUBGROUP_other']
                anomaly_pref_gpi = anomaly_pref_gpi.append(anomaly_pref_gpi_6)
                ##logger.info(str(anomaly_pref_gpi_6.GPI) + '-***********-' + str(anomaly_pref_gpi_6.REGION) + ': ' + str(anomaly_pref_gpi_6.CHAIN_SUBGROUP_pref)+'-'+str(anomaly_pref_gpi_6.CHAIN_SUBGROUP_other))
            ##normal-non anomaly condition 2
            non_anomaly_pref_gpi_6 = cvs_other_only_cvs_GPI0[
                cvs_other_only_cvs_GPI0.pref_upper_bound >= p.RETAIL_RETAIL_BOUND * cvs_other_only_cvs_GPI0.pref_other_lower_bound]
            non_anomaly_pref_gpi_6['price_cons'] = -non_anomaly_pref_gpi_6[
                'Price_Decision_Var_pref'] + p.RETAIL_RETAIL_BOUND * non_anomaly_pref_gpi_6['Price_Decision_Var_other']
            for i in range(non_anomaly_pref_gpi_6.shape[0]):
                pref_other_price_cons_list.append(non_anomaly_pref_gpi_6.price_cons.iloc[i] <= 0)
        #####################Condition 3 ends##############################################################################################################
        ###################################################################################################################################################

        ###################################################################################################################################################
        #####Condition 4: Both CVS and Other Pref sub chains have GPI_ONLY!=0 for a given combination of 'CLIENT','REGION','MEASUREMENT','BREAKOUT','GPI','NDC'
        cvs_other_bothGPINot0 = cvs_other_master.loc[
            (cvs_other_master.GPI_ONLY_pref != 0) & (cvs_other_master.GPI_ONLY_other != 0)]

        if cvs_other_bothGPINot0.shape[0] > 0:
            ##appending the anomaly dataframe if the anomaly condition 1 is met
            anomaly_pref_gpi_7 = cvs_other_bothGPINot0[
                cvs_other_bothGPINot0.pref_lower_bound * p.PREF_OTHER_FACTOR > cvs_other_bothGPINot0.pref_other_upper_bound]
            if anomaly_pref_gpi_7.shape[0] > 0:
                anomaly_pref_gpi_7['GPI_NDC'], anomaly_pref_gpi_7['BG_FLAG'], anomaly_pref_gpi_7['EXCLUDE_REASON'] = anomaly_pref_gpi_7[
                                                                                          'GPI'] + '_***********', \
                                                                                      anomaly_pref_gpi_7['BG_FLAG'], \
                                                                                      anomaly_pref_gpi_7[
                                                                                          'CHAIN_SUBGROUP_pref'] + '-' + \
                                                                                      anomaly_pref_gpi_7[
                                                                                          'CHAIN_SUBGROUP_other']
                anomaly_pref_gpi = anomaly_pref_gpi.append(anomaly_pref_gpi_7)
                # logger.info(str(anomaly_pref_gpi_7.GPI) + '-***********-' + str(anomaly_pref_gpi_7.REGION) + ': ' + str(anomaly_pref_gpi_7.CHAIN_SUBGROUP_pref)+'-'+str(anomaly_pref_gpi_7.CHAIN_SUBGROUP_other))
            ##normal-non anomaly condition 1
            non_anomaly_pref_gpi_7 = cvs_other_bothGPINot0[
                cvs_other_bothGPINot0.pref_lower_bound * p.PREF_OTHER_FACTOR <= cvs_other_bothGPINot0.pref_other_upper_bound]
            non_anomaly_pref_gpi_7['price_cons'] = non_anomaly_pref_gpi_7[
                                                       'Price_Decision_Var_pref'] * p.PREF_OTHER_FACTOR - \
                                                   non_anomaly_pref_gpi_7['Price_Decision_Var_other']

            for i in range(non_anomaly_pref_gpi_7.shape[0]):
                pref_other_price_cons_list.append(non_anomaly_pref_gpi_7.price_cons.iloc[i] <= 0)

            ##appending the anomaly dataframe if the anomaly condition 2 is met
            anomaly_pref_gpi_8 = cvs_other_bothGPINot0[
                cvs_other_bothGPINot0.pref_upper_bound < p.RETAIL_RETAIL_BOUND * cvs_other_bothGPINot0.pref_other_lower_bound]
            if anomaly_pref_gpi_8.shape[0] > 0:
                anomaly_pref_gpi_8['GPI_NDC'], anomaly_pref_gpi_8['BG_FLAG'], anomaly_pref_gpi_8['EXCLUDE_REASON'] = anomaly_pref_gpi_8[
                                                                                          'GPI'] + '_***********', \
                                                                                      anomaly_pref_gpi_8['BG_FLAG'], \
                                                                                      anomaly_pref_gpi_8[
                                                                                          'CHAIN_SUBGROUP_pref'] + '-' + \
                                                                                      anomaly_pref_gpi_8[
                                                                                          'CHAIN_SUBGROUP_other']
                anomaly_pref_gpi = anomaly_pref_gpi.append(anomaly_pref_gpi_8)
                ##logger.info(str(anomaly_pref_gpi_8.GPI) + '-***********-' + str(anomaly_pref_gpi_8.REGION) + ': ' + str(anomaly_pref_gpi_8.CHAIN_SUBGROUP_pref)+'-'+str(anomaly_pref_gpi_8.CHAIN_SUBGROUP_other))
            ##normal-non anomaly condition 2
            non_anomaly_pref_gpi_8 = cvs_other_bothGPINot0[
                cvs_other_bothGPINot0.pref_upper_bound >= p.RETAIL_RETAIL_BOUND * cvs_other_bothGPINot0.pref_other_lower_bound]
            non_anomaly_pref_gpi_8['price_cons'] = -non_anomaly_pref_gpi_8[
                'Price_Decision_Var_pref'] + p.RETAIL_RETAIL_BOUND * non_anomaly_pref_gpi_8['Price_Decision_Var_other']
            for i in range(non_anomaly_pref_gpi_8.shape[0]):
                pref_other_price_cons_list.append(non_anomaly_pref_gpi_8.price_cons.iloc[i] <= 0)

 ###################################################################################################################################################
        #####Extra condition: checking R30-R90 within CVSSP
        #####"R30match" necessary because some clients have shared VCMLs between R30 and R90 for e.g. NONPREF_OTH
        #####so R90 pricing can constrain R30 CVSSP
        #####For these extra conditions, we are checking for infeasibilities *only*
        #####and so for code complexity reasons we don't check for cases where the two dataframes have different NDC grain.
        if 'CVSSP' in price_constraints_df_no_mail.CHAIN_SUBGROUP.unique():
            other_gpi_r30match = price_constraints_df_no_mail[price_constraints_df_no_mail.MAC_LIST.isin(
                other_gpi[other_gpi.MEASUREMENT=='R30'].MAC_LIST) & (price_constraints_df_no_mail.MEASUREMENT!='R30')]
            cvs_other_master_r30match = cvs_gpi.merge(other_gpi_r30match, how='inner',
                                            on=['CLIENT', 'REGION', 'GPI', 'NDC', 'BG_FLAG'],
                                            suffixes=('_pref', '_other'))
            cvs_other_master_r30match = pd.concat([cvs_other_master[cvs_other_master.MEASUREMENT=='R30'], cvs_other_master_r30match])
            cvs_other_master_r30match['pref_lower_bound'], cvs_other_master_r30match['pref_upper_bound'] = zip(
                *cvs_other_master_r30match.Price_Bounds_pref)
            cvs_other_master_r30match['pref_other_lower_bound'], cvs_other_master_r30match['pref_other_upper_bound'] = zip(
                *cvs_other_master_r30match.Price_Bounds_other)

            cvs_other_master_r30match['R30nonprefub'] = cvs_other_master_r30match.groupby(['CLIENT', 'REGION', 'BREAKOUT', 'GPI', 'NDC', 'BG_FLAG'])['pref_other_upper_bound'].transform(min)
            anomaly_pref_gpi_9 = cvs_other_master_r30match[
                    (cvs_other_master_r30match.pref_lower_bound*p.PREF_OTHER_FACTOR > cvs_other_master_r30match.R30nonprefub)]
            if anomaly_pref_gpi_9.shape[0] > 0:
                anomaly_pref_gpi_9 = anomaly_pref_gpi_9.drop_duplicates(['GPI', 'NDC'])
                anomaly_pref_gpi_9['GPI_NDC'], anomaly_pref_gpi_9['BG_FLAG'], anomaly_pref_gpi_9['EXCLUDE_REASON'] = anomaly_pref_gpi_9[
                                                                                        'GPI'] + '_' + anomaly_pref_gpi_9['NDC'], \
                                                                                    anomaly_pref_gpi_9['BG_FLAG'], \
                                                                                    'CVSSP-CVSSP90Parity'
                anomaly_pref_gpi = anomaly_pref_gpi.append(anomaly_pref_gpi_9)

        ##### Extra condition: checking OK VCMLs against actual constraints for CVS/CVSSP R30
        ##### Situation that inspired this: NONPREF_OTH R90 had an upper limit around 0.7. CVSSP_R90OK had a lower limit around 0.8.
        ##### NONPREF_OTH R30 VCML was also used for R90, so CVSSP was limited to <0.7 via NONPREF_OTH R30=NONPREF_OTH R90 > CVSSP, 
        ##### and CVSSP_R90OK has to be <=CVSSP. So there was an infeasibility.
        ##### To find this, we need to A) check R90 VCMLs that are also used for R30 (in previous extra condition) as well as 
        ##### B) checking R90OK VCMLs against R30 VCMLs *even though we already do that in the R30-R90 check*,
        ##### because we need to be able to check it using the additional constraints that R90OK < CVSSP < all R30 will impose.
        if 'CVSSP' in price_constraints_df_no_mail.CHAIN_SUBGROUP.unique():
            cvs_other_master_okvcml = cvs_other_master[cvs_other_master.CHAIN_SUBGROUP_other.str.contains('_R90OK')]
            if not cvs_other_master_okvcml.empty:
                other_gpi_okvcmlmatch = price_constraints_df_no_mail[price_constraints_df_no_mail.MAC_LIST.isin(
                    cvs_other_master_okvcml.MAC_LIST_other)]
                other_gpi_r30 = other_gpi[other_gpi.MEASUREMENT=='R30']  ##dataframe with other than CVS or CVSSP chain subgroups
                other_other_okvcmlmatch = other_gpi_okvcmlmatch.merge(pd.concat([other_gpi_r30match, other_gpi_r30]), how='inner',
                                                 on=['CLIENT', 'REGION', 'GPI', 'NDC', 'BG_FLAG'],
                                                 suffixes=('_R90OK', '_other')
                                                                     ).merge(cvs_gpi[cvs_gpi['MEASUREMENT']=='R30'], 
                                                                             how='inner', on=['CLIENT', 'REGION', 'GPI', 'NDC', 'BG_FLAG'])
                other_other_okvcmlmatch['R90OK_lower_bound'], other_other_okvcmlmatch['R90OK_upper_bound'] = zip(
                    *other_other_okvcmlmatch.Price_Bounds_R90OK)
                other_other_okvcmlmatch['pref_other_lower_bound'], other_other_okvcmlmatch['pref_other_upper_bound'] = zip(
                    *other_other_okvcmlmatch.Price_Bounds_other)
                # CVS has no suffix because it doesn't have any collisions with the other column names that already DO have suffixes
                other_other_okvcmlmatch['CVS_lower_bound'], other_other_okvcmlmatch['CVS_upper_bound'] = zip(
                    *other_other_okvcmlmatch.Price_Bounds)

                anomaly_pref_gpi_10 = other_other_okvcmlmatch[
                    (other_other_okvcmlmatch.R90OK_lower_bound*p.PREF_OTHER_FACTOR > other_other_okvcmlmatch.pref_other_upper_bound)
                ]
                if anomaly_pref_gpi_10.shape[0] > 0:
                    anomaly_pref_gpi_10['GPI_NDC'], anomaly_pref_gpi_10['BG_FLAG'], anomaly_pref_gpi_10['EXCLUDE_REASON'] = anomaly_pref_gpi_10[
                                                                                              'GPI'] + '_' + anomaly_pref_gpi_10['NDC'], \
                                                                                            anomaly_pref_gpi_10['BG_FLAG'], \
                                                                                            anomaly_pref_gpi_10[
                                                                                                  'CHAIN_SUBGROUP_R90OK'] + '-' + \
                                                                                            anomaly_pref_gpi_10[
                                                                                                  'CHAIN_SUBGROUP_other']
                    anomaly_pref_gpi = anomaly_pref_gpi.append(anomaly_pref_gpi_10.drop_duplicates(subset=['CLIENT', 'REGION', 'GPI_NDC', 'EXCLUDE_REASON']))
                anomaly_pref_gpi_11 = other_other_okvcmlmatch[
                    (other_other_okvcmlmatch.R90OK_lower_bound*p.PREF_OTHER_FACTOR > other_other_okvcmlmatch.CVS_upper_bound)
                ]
                if anomaly_pref_gpi_11.shape[0] > 0:
                    anomaly_pref_gpi_11['GPI_NDC'], anomaly_pref_gpi_11['BG_FLAG'], anomaly_pref_gpi_11['EXCLUDE_REASON'] = anomaly_pref_gpi_11[
                                                                                              'GPI'] + '_' + anomaly_pref_gpi_11['NDC'], \
                                                                                            anomaly_pref_gpi_11['BG_FLAG'], \
                                                                                            anomaly_pref_gpi_11[
                                                                                                  'CHAIN_SUBGROUP_R90OK'] + '-' + \
                                                                                            anomaly_pref_gpi_11[
                                                                                                  'CHAIN_SUBGROUP']
                    anomaly_pref_gpi = anomaly_pref_gpi.append(anomaly_pref_gpi_11.drop_duplicates(subset=['CLIENT', 'REGION','GPI_NDC', 'EXCLUDE_REASON']))

        #####Extra condition: checking R30-R90 for CVSSP & CVS collar
        #####"R30match" necessary because some clients have shared VCMLs between R30 and R90 for e.g. NONPREF_OTH
        #####so R90 pricing can constrain R30 CVSSP
        if 'CVSSP' in price_constraints_df_no_mail.CHAIN_SUBGROUP.unique() and 'CVS' in price_constraints_df_no_mail.CHAIN_SUBGROUP.unique():
            other_gpi_r30match = price_constraints_df_no_mail[price_constraints_df_no_mail.MAC_LIST.isin(
                other_gpi[other_gpi.MEASUREMENT=='R30'].MAC_LIST) & (price_constraints_df_no_mail.MEASUREMENT!='R30')]
            # Here, we need to check both R30 and R90, so we're going to "fake" that these R30-match rows from R90 are R30
            # so we can join on measurement
            other_gpi_r30match['MEASUREMENT'] = 'R30'
            other_gpi_r30 = other_gpi[other_gpi['MEASUREMENT']=='R30']
            other_gpi_r30all = pd.concat([other_gpi_r30match, other_gpi_r30])
            cvs_collar_gpi = price_constraints_df_no_mail[price_constraints_df_no_mail.CHAIN_SUBGROUP=='CVS']
            cvs_other_collar_r30match = cvs_collar_gpi.merge(other_gpi_r30all, how='inner',
                                             on=['CLIENT', 'REGION', 'MEASUREMENT', 'GPI', 'NDC', 'BG_FLAG'],
                                             suffixes=('_collar', '_other'))
            cvs_other_collar_r30match['collar_lower_bound'], cvs_other_collar_r30match['collar_upper_bound'] = zip(
                *cvs_other_collar_r30match.Price_Bounds_collar)
            cvs_other_collar_r30match['pref_other_lower_bound'], cvs_other_collar_r30match['pref_other_upper_bound'] = zip(
                *cvs_other_collar_r30match.Price_Bounds_other)

            cvs_other_collar_r30match['nonprefub'] = cvs_other_collar_r30match.groupby(['CLIENT', 'REGION', 'MEASUREMENT', 'GPI', 'NDC', 'BG_FLAG'])['pref_other_upper_bound'].transform(min)
            anomaly_pref_gpi_12 = cvs_other_collar_r30match[
                    (cvs_other_collar_r30match.collar_lower_bound*p.PREF_OTHER_FACTOR/p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH 
                     > cvs_other_collar_r30match.nonprefub)]
            if anomaly_pref_gpi_12.shape[0] > 0:
                anomaly_pref_gpi_12 = anomaly_pref_gpi_12.drop_duplicates(['GPI', 'NDC'])
                anomaly_pref_gpi_12['GPI_NDC'], anomaly_pref_gpi_12['BG_FLAG'], anomaly_pref_gpi_12['EXCLUDE_REASON'] = anomaly_pref_gpi_12[
                                                                                          'GPI'] + '_' + anomaly_pref_gpi_12['NDC'], \
                                                                                       anomaly_pref_gpi_12['BG_FLAG'], \
                                                                                      'CVSSPParity-CVSCollar'
                anomaly_pref_gpi = anomaly_pref_gpi.append(anomaly_pref_gpi_12)

        
        logger.info("Ending All Other Pharmacies >= CVS Pricing")
        end = time.time()
        logger.info("Run time: {} mins".format((end - start) / 60.))
        logger.info('--------------------')

        # file outputs
        with open(pref_other_price_cons_list_out, 'wb') as f:
            pickle.dump(pref_other_price_cons_list, f)
        with open(anomaly_pref_gpi_out, 'wb') as f:
            pickle.dump(anomaly_pref_gpi, f)

        return pref_other_price_cons_list, anomaly_pref_gpi
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'CVS Parity Price Constraint', repr(e), error_loc)
        raise e
###### End All Other Pharmacies >= CVS pricing constraint ####################


###### State Parity and Non-State Parity Pricing Constraints ########
def state_parity_constraints(
    params_file_in: str,
    unc_flag: bool,
    lp_data_df_in: InputPath('pickle'), 
    parity_price_cons_list_out: OutputPath('pickle'),
    anomaly_state_parity_gpi_out: OutputPath('pickle'),
    loglevel: str = 'INFO'
    # # kube_run: bool = True,
):
    import sys
    import os
    import time
    import logging
    import pandas as pd
    import pickle
    import pulp
    import util_funcs as uf
    import BQ

    uf.write_params(params_file_in)
    sys.path.append('/')
    import CPMO_parameters as p
    from CPMO_lp_functions import generatePricingDecisionVariables
    from CPMO_shared_functions import update_run_status
    try:
        out_path = os.path.join(p.FILE_LOG_PATH, 'ClientPharmacyMacOptimization.log')
        logger = uf.log_setup(log_file_path=out_path, loglevel=loglevel)

        # input files
        with open(lp_data_df_in, 'rb') as f:
            lp_data_df = pickle.load(f)

            logger.info('--------------------')
            logger.info("State Parity vs Non-State Parity pricing constraints")
            start = time.time()

        lp_data_df = generatePricingDecisionVariables(lp_data_df)

        price_constraints_col = ['CLIENT', 'BREAKOUT', 'MEASUREMENT', 'GPI', 'NDC', 'GPI_NDC', 'BG_FLAG', 'GPI_ONLY',
                                    'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'REGION', 'PHARMACY_TYPE',
                                    'Price_Decision_Var', 'Price_Bounds', 'MAC_PRICE_UNIT_ADJ', 'Dec_Var_Name', 'AVG_AWP']

        price_constraints_df = lp_data_df[price_constraints_col].loc[(lp_data_df.PRICE_MUTABLE==1) & (lp_data_df.MEASUREMENT != 'M30'),:]
        
        
        # NOTE: order is important! First entry does not require parity, second entry must maintain parity.
        parity_subchain_pairs = [('CVS', 'CVSSP')]
        
        ###################################################### vectorization starts below ######################################################
        parity_price_cons_list = []
        anomaly_parity_gpi = pd.DataFrame(columns=('CLIENT', 'REGION', 'GPI_NDC', 'BG_FLAG'))

        # define CVS and CVSSP
        no_parity_subchain = parity_subchain_pairs[0][0]
        parity_subchain = parity_subchain_pairs[0][1]

        # vectorize a few things from the for loops
        df_1 = price_constraints_df[price_constraints_df.MEASUREMENT != 'M30'] # remove mail
        
        df_1['par'] = df_1['CHAIN_SUBGROUP'] == parity_subchain # make new column "par", this will help us filter for "parity_gpi" throughout the code
        df_1['no_par'] = df_1['CHAIN_SUBGROUP'] == no_parity_subchain # same as above

        # vectorize the first if clause (before the four parallel if clauses):
        # filter original dataframe to keep rows which belong to groups which contains at least one row ith 'par' = True AND at least one row with 'no_par' = True  
        grouped_df = df_1.groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'])\
                         .agg(any_par_in_group = ('par', 'any'),
                              any_no_par_in_group = ('no_par', 'any')).reset_index() # for each group, find whether there is any row where 'par' (or 'no_par') is True
        grouped_df['both_par_and_no_par_in_group'] = grouped_df['any_par_in_group'] & grouped_df['any_no_par_in_group'] # groups that have at least one row ith 'par' = True AND at least one row with 'no_par' = True
        merged = grouped_df.merge(df_1, on = ['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'], how = 'right') # merge so that the original dataframe has the column "both_par_and_no_par_in_group"
        df_2 = merged[merged['both_par_and_no_par_in_group'] == True] # filter original dataframe based on this criteria
        df_2 = df_2.drop(['any_par_in_group', 'any_no_par_in_group', 'both_par_and_no_par_in_group'], axis = 1) # drop useless columns
        

        ############ If Clause no.1 original code: (len(parity_gpi.loc[parity_gpi.GPI_ONLY == 0, 'GPI_NDC']) > 0) & (len(no_parity_gpi.loc[no_parity_gpi.GPI_ONLY == 0, 'GPI_NDC']) > 0)####
        # keep only only rows with GPI_ONLY = 0
        df_3 = df_2[df_2['GPI_ONLY'] == 0]
        # filter original dataframe to keep rows which belong to groups which contains at least one row ith 'par' = True AND at least one row with 'no_par' = True
        grouped_df = df_3.groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'])\
                         .agg(any_par_in_group = ('par', 'any'),
                              any_no_par_in_group = ('no_par', 'any')).reset_index()
        merged = grouped_df.merge(df_3, on = ['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'], how = 'right')
        df_4 = merged[(merged['any_par_in_group'] == True) & (merged['any_no_par_in_group'] == True)]
        df_4 = df_4.drop(['any_par_in_group', 'any_no_par_in_group'], axis = 1)
        # group by 5 columns including NDC and keep only groups that have at least one row ith 'par' = True AND at least one row with 'no_par' = True
        grouped_df = df_4.groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT','NDC'])\
                         .agg(any_par_in_group = ('par', 'any'),
                              any_no_par_in_group = ('no_par', 'any')).reset_index()
        grouped_df = grouped_df[(grouped_df['any_par_in_group'] == True) & (grouped_df['any_no_par_in_group'] == True)]
        grouped_df = grouped_df.drop(['any_par_in_group', 'any_no_par_in_group'], axis = 1)
        # get price bounds from parity_gpi and no_parity_gpi (which are from the level of df_2)
        grouped_df_parity = df_2[df_2['par'] == True].groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT','NDC'])\
                                                     .agg(parity_price_bounds = ('Price_Bounds', 'first')).reset_index()
        grouped_df_no_parity = df_2[df_2['no_par'] == True].groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT','NDC'])\
                                                           .agg(no_parity_price_bounds = ('Price_Bounds', 'first')).reset_index()
        # get price decision variables (needed later for constructing constraints)
        grouped_df_parity_var = df_2[df_2['par'] == True].groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT','NDC'])\
                                                         .agg(var1 = ('Price_Decision_Var', 'first')).reset_index()
        grouped_df_no_parity_var = df_2[df_2['no_par'] == True].groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT','NDC'])\
                                                               .agg(var2 = ('Price_Decision_Var', 'first')).reset_index()
        # merging dataframes above so that we have one row per group with all the information we need for each execution in each row.
        grouped_df = grouped_df.merge(grouped_df_parity, on = ['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT','NDC'], how = 'inner') # this basically adds the 'parity_price_bounds' column to the grouped_by df
        grouped_df = grouped_df.merge(grouped_df_no_parity, on = ['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT','NDC'], how = 'inner') # add the 'no_parity_price_bounds' column     
        grouped_df = grouped_df.merge(grouped_df_parity_var, on = ['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT','NDC'], how = 'inner') # add the 'var1' column
        grouped_df = grouped_df.merge(grouped_df_no_parity_var, on = ['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT','NDC'], how = 'inner') # add the 'var1' column
        grouped_df['parity_lower_bound'] = grouped_df['parity_price_bounds'].str[0] # get price lower bound
        grouped_df['parity_upper_bound'] = grouped_df['parity_price_bounds'].str[1] # get price upper bound
        grouped_df['no_parity_lower_bound'] = grouped_df['no_parity_price_bounds'].str[0]
        grouped_df['no_parity_upper_bound'] = grouped_df['no_parity_price_bounds'].str[1]
        grouped_df.drop(['no_parity_price_bounds','parity_price_bounds'], axis = 1, inplace = True) # drop useless columns
        # if clause 1-1 : original code: no_parity_lower_bound > p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH*parity_upper_bound:
        anomaly_rows = grouped_df[grouped_df['no_parity_lower_bound'] > p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH * grouped_df['parity_upper_bound']].copy()
        if not anomaly_rows.empty:
            for i in range(anomaly_rows.shape[0]):
                logger.info(anomaly_rows['GPI'].iloc[i] + '-' + anomaly_rows['NDC'].iloc[i] + '-' + anomaly_rows['BG_FLAG'].iloc[i] + '-' + anomaly_rows['REGION'].iloc[i]\
                                    + ': ' + str(parity_subchain) +'-'+str(no_parity_subchain))
        anomaly_rows['GPI_NDC'] = anomaly_rows['GPI'] + '_***********'
        anomaly_parity_gpi = pd.concat([anomaly_parity_gpi, anomaly_rows[['CLIENT','REGION','GPI_NDC','BG_FLAG']]]) # append to anomaly list
        good_rows = grouped_df[grouped_df['no_parity_lower_bound'] <= p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH * grouped_df['parity_upper_bound']].copy()                
        if not good_rows.empty:
            for i in range(good_rows.shape[0]):
                parity_price_cons_list.append(good_rows.var2.iloc[i] <= p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH * good_rows.var1.iloc[i])
        
        # if clause 1-2 : original code: no_parity_upper_bound < p.PARITY_PRICE_DIFFERENCE_COLLAR_LOW*parity_lower_bound
        anomaly_rows = grouped_df[grouped_df['no_parity_upper_bound'] < p.PARITY_PRICE_DIFFERENCE_COLLAR_LOW * grouped_df['parity_lower_bound']].copy()
        if not anomaly_rows.empty:
            for i in range(anomaly_rows.shape[0]):
                logger.info(anomaly_rows['GPI'].iloc[i] + '-' + anomaly_rows['NDC'].iloc[i] + '-' + anomaly_rows['BG_FLAG'].iloc[i] + '-' + anomaly_rows['REGION'].iloc[i]\
                                    + ': ' + str(parity_subchain) +'-'+str(no_parity_subchain))
        anomaly_rows['GPI_NDC'] = anomaly_rows['GPI'] + '_***********'
        anomaly_parity_gpi = pd.concat([anomaly_parity_gpi, anomaly_rows[['CLIENT','REGION','GPI_NDC','BG_FLAG']]])
        good_rows = grouped_df[grouped_df['no_parity_upper_bound'] >= p.PARITY_PRICE_DIFFERENCE_COLLAR_LOW * grouped_df['parity_lower_bound']].copy()
        if not good_rows.empty:
            for i in range(good_rows.shape[0]):
                parity_price_cons_list.append(good_rows.var2.iloc[i] >= p.PARITY_PRICE_DIFFERENCE_COLLAR_LOW * good_rows.var1.iloc[i])
        ######################################################################################################################################################

        ############ If Clause no.2 : Original Code(len(no_parity_gpi.loc[no_parity_gpi.GPI_ONLY == 0, 'GPI_NDC']) > 0) : parity is GPI############
        # vectorize the elif clause
        df_3 = df_2[df_2['GPI_ONLY'] == 0]
        grouped_df = df_3.groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'])\
                         .agg(any_no_par_in_group = ('no_par', 'any'),
                              total_par_in_group = ('par', 'sum')).reset_index()
        merged = grouped_df.merge(df_3, on = ['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'], how = 'right')
        df_4 = merged[(merged['any_no_par_in_group'] == True) & (merged['total_par_in_group'] == 0)]
        df_4 = df_4.drop(['any_no_par_in_group','total_par_in_group'], axis = 1)
        ### for the logger in the beginning (this part deserves its own iPad notes)
        all_groups = df_2.groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT']).agg(num_rows_par_True = ('par', 'sum')).reset_index() # get all original groups (so we can get parity_gpi)
        merged = all_groups.merge(df_4, on = ['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'], how = 'inner') # merge with df_4 so we know the par sum of each row in df_4
        df_4_filtered = merged[merged['num_rows_par_True']>1] # keep only df_4 rows with par sum > 1, these rows belong to problematic groups within df_4
        par_rows = df_2[df_2['par']==True] # get all parity_gpi rows
        keys = df_4_filtered.groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'], as_index = False).size() # get all groupby keys for df_4_filtered (i.e. all problematic group combinations in df_4)
        keys.drop(columns = 'size', inplace = True) # now "keys" contain all problematic group keys
        problem_rows = par_rows.merge(keys, how = 'inner') # find all parity_gpi rows that correspond to problematic groups in df_4
        problem_groups = problem_rows.groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'])['Dec_Var_Name'] # get all problematic groups and their Dec_Var_Name values
        for group_key, group_values in problem_groups:
            logger.info(('ERROR with '+ group_values).tolist())
            assert False, "len(parity_gpi.Price_Decision_Var)==1" # but do we want to change this error message to something else....???
        # vectorize the ndc for loop clause
        grouped_df = df_4.groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT','NDC'])\
                         .agg(any_no_par_in_group = ('no_par', 'any')).reset_index()
        grouped_df = grouped_df[grouped_df['any_no_par_in_group'] == True]
        grouped_df = grouped_df.drop(['any_no_par_in_group'], axis = 1)
        # get price bounds
        grouped_df_parity = df_2[df_2['par'] == True].groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'])\
                                                     .agg(parity_price_bounds = ('Price_Bounds', 'first')).reset_index() # 4 groupby keys, excluding ndc
        grouped_df_no_parity = df_2[df_2['no_par'] == True].groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT','NDC'])\
                                                           .agg(no_parity_price_bounds = ('Price_Bounds', 'first')).reset_index() # 5 groupby keys
        # get price decision variables (needed later for constructing constraints)
        grouped_df_parity_var = df_2[df_2['par'] == True].groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'])\
                                                         .agg(var1 = ('Price_Decision_Var', 'first')).reset_index() # 4 groupby keys, excluding ndc
        grouped_df_no_parity_var = df_2[df_2['no_par'] == True].groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT','NDC'])\
                                                               .agg(var2 = ('Price_Decision_Var', 'first')).reset_index() # 5 groupby keys
        # merging dataframes above so that we have one row per group with all the information we need for each execution in each row.
        grouped_df = grouped_df.merge(grouped_df_parity, on = ['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'], how = 'inner')
        grouped_df = grouped_df.merge(grouped_df_parity_var, on = ['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'], how = 'inner')
        grouped_df = grouped_df.merge(grouped_df_no_parity, on = ['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT','NDC'], how = 'inner')
        grouped_df = grouped_df.merge(grouped_df_no_parity_var, on = ['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT','NDC'], how = 'inner')
        grouped_df['parity_lower_bound'] = grouped_df['parity_price_bounds'].str[0]
        grouped_df['parity_upper_bound'] = grouped_df['parity_price_bounds'].str[1]
        grouped_df['no_parity_lower_bound'] = grouped_df['no_parity_price_bounds'].str[0]
        grouped_df['no_parity_upper_bound'] = grouped_df['no_parity_price_bounds'].str[1]
        grouped_df.drop(['no_parity_price_bounds','parity_price_bounds'], axis = 1, inplace = True)
        # if clause : original code: no_parity_lower_bound > p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH*parity_upper_bound:
        anomaly_rows = grouped_df[grouped_df['no_parity_lower_bound'] > p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH * grouped_df['parity_upper_bound']].copy()
        if not anomaly_rows.empty:
            for i in range(anomaly_rows.shape[0]):
                logger.info(anomaly_rows['GPI'].iloc[i] + '-' + anomaly_rows['NDC'].iloc[i] + '-' + anomaly_rows['BG_FLAG'].iloc[i] + '-' + anomaly_rows['REGION'].iloc[i]\
                                    + ': ' + str(parity_subchain) +'-'+str(no_parity_subchain))
        anomaly_rows['GPI_NDC'] = anomaly_rows['GPI'] + '_***********'
        anomaly_parity_gpi = pd.concat([anomaly_parity_gpi, anomaly_rows[['CLIENT','REGION','GPI_NDC','BG_FLAG']]])
        good_rows = grouped_df[grouped_df['no_parity_lower_bound'] <= p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH * grouped_df['parity_upper_bound']].copy()
        if not good_rows.empty:
            for i in range(good_rows.shape[0]):
                parity_price_cons_list.append(good_rows.var2.iloc[i] <= p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH * good_rows.var1.iloc[i])
        # if clause: original code: no_parity_upper_bound < p.PARITY_PRICE_DIFFERENCE_COLLAR_LOW*parity_lower_bound:
        anomaly_rows = grouped_df[grouped_df['no_parity_upper_bound'] < p.PARITY_PRICE_DIFFERENCE_COLLAR_LOW * grouped_df['parity_lower_bound']].copy()
        if not anomaly_rows.empty:
            for i in range(anomaly_rows.shape[0]):
                logger.info(anomaly_rows['GPI'].iloc[i] + '-' + anomaly_rows['NDC'].iloc[i] + '-' + anomaly_rows['BG_FLAG'].iloc[i] + '-' + anomaly_rows['REGION'].iloc[i]\
                                    + ': ' + str(parity_subchain) +'-'+str(no_parity_subchain))
        anomaly_rows['GPI_NDC'] = anomaly_rows['GPI'] + '_***********'
        anomaly_parity_gpi = pd.concat([anomaly_parity_gpi, anomaly_rows[['CLIENT','REGION','GPI_NDC','BG_FLAG']]])
        good_rows = grouped_df[grouped_df['no_parity_upper_bound'] >= p.PARITY_PRICE_DIFFERENCE_COLLAR_LOW * grouped_df['parity_lower_bound']].copy()
        if not good_rows.empty:
            for i in range(good_rows.shape[0]):
                parity_price_cons_list.append(good_rows.var2.iloc[i] >= p.PARITY_PRICE_DIFFERENCE_COLLAR_LOW * good_rows.var1.iloc[i])
        #######################################################################################################################################################

        ############ If Clause no.3 : Original Code: (len(parity_gpi.loc[parity_gpi.GPI_ONLY == 0, 'GPI_NDC']) > 0): no_parity is GPI######################

        df_3 = df_2[df_2['GPI_ONLY'] == 0]
        grouped_df = df_3.groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'])\
                         .agg(any_par_in_group = ('par', 'any'),
                              total_no_par_in_group = ('no_par', 'sum')).reset_index()
        merged = grouped_df.merge(df_3, on = ['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'], how = 'right')
        df_4 = merged[(merged['any_par_in_group'] == True) & (merged['total_no_par_in_group'] == 0)]
        df_4 = df_4.drop(['any_par_in_group','total_no_par_in_group'], axis = 1)

        all_groups = df_2.groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT']).agg(num_rows_no_par_True = ('no_par', 'sum')).reset_index()
        merged = all_groups.merge(df_4, on = ['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'], how = 'inner')
        df_4_filtered = merged[merged['num_rows_no_par_True']>1]
        par_rows = df_2[df_2['no_par']==True]
        keys = df_4_filtered.groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'], as_index = False).size()
        keys.drop(columns = 'size', inplace = True)
        problem_rows = par_rows.merge(keys, how = 'inner')
        problem_groups = problem_rows.groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'])['Dec_Var_Name']
        for group_key, group_values in problem_groups:
            logger.info(('ERROR with '+ group_values).tolist())
            assert False, "len(no_parity_gpi.Price_Decision_Var)==1" # maybe change the error message
        # vectorize the ndc for loop clause
        grouped_df = df_4.groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT','NDC'])\
                         .agg(any_par_in_group = ('par', 'any')).reset_index()
        grouped_df = grouped_df[grouped_df['any_par_in_group'] == True]
        grouped_df = grouped_df.drop(['any_par_in_group'], axis = 1)
        # get price bounds
        grouped_df_no_parity = df_2[df_2['no_par'] == True].groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'])\
                                                     .agg(no_parity_price_bounds = ('Price_Bounds', 'first')).reset_index() # 4 groupby keys, excluding ndc
        grouped_df_parity = df_2[df_2['par'] == True].groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT','NDC'])\
                                                           .agg(parity_price_bounds = ('Price_Bounds', 'first')).reset_index() # 5 groupby keys
        # get price decision variables (needed later for constructing constraints)
        grouped_df_no_parity_var = df_2[df_2['no_par'] == True].groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'])\
                                                         .agg(var2 = ('Price_Decision_Var', 'first')).reset_index() # 4 groupby keys, excluding ndc
        grouped_df_parity_var = df_2[df_2['par'] == True].groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT','NDC'])\
                                                               .agg(var1 = ('Price_Decision_Var', 'first')).reset_index() # 5 groupby keys
        # merging dataframes above so that we have one row per group with all the information we need for each execution in each row.
        grouped_df = grouped_df.merge(grouped_df_no_parity, on = ['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'], how = 'inner')
        grouped_df = grouped_df.merge(grouped_df_no_parity_var, on = ['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'], how = 'inner')
        grouped_df = grouped_df.merge(grouped_df_parity, on = ['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT','NDC'], how = 'inner')
        grouped_df = grouped_df.merge(grouped_df_parity_var, on = ['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT','NDC'], how = 'inner')
        grouped_df['parity_lower_bound'] = grouped_df['parity_price_bounds'].str[0]
        grouped_df['parity_upper_bound'] = grouped_df['parity_price_bounds'].str[1]
        grouped_df['no_parity_lower_bound'] = grouped_df['no_parity_price_bounds'].str[0]
        grouped_df['no_parity_upper_bound'] = grouped_df['no_parity_price_bounds'].str[1]
        grouped_df.drop(['no_parity_price_bounds','parity_price_bounds'], axis = 1, inplace = True)
        # if clause 3-1: Original Code: no_parity_lower_bound > p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH*parity_upper_bound
        anomaly_rows = grouped_df[grouped_df['no_parity_lower_bound'] > p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH * grouped_df['parity_upper_bound']].copy()
        if not anomaly_rows.empty:
            for i in range(anomaly_rows.shape[0]):
                logger.info(anomaly_rows['GPI'].iloc[i] + '-' + anomaly_rows['NDC'].iloc[i] + '-' + anomaly_rows['BG_FLAG'].iloc[i] + '-' + anomaly_rows['REGION'].iloc[i]\
                                    + '_' + str(parity_subchain))
        anomaly_rows['GPI_NDC'] = anomaly_rows['GPI'] + '_***********'
        anomaly_parity_gpi = pd.concat([anomaly_parity_gpi, anomaly_rows[['CLIENT','REGION','GPI_NDC','BG_FLAG']]])
        good_rows = grouped_df[grouped_df['no_parity_lower_bound'] <= p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH * grouped_df['parity_upper_bound']].copy()
        if not good_rows.empty:
            for i in range(good_rows.shape[0]):
                parity_price_cons_list.append(good_rows.var2.iloc[i] <= p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH * good_rows.var1.iloc[i])            
        # if clause 3-2: original Code: no_parity_upper_bound < p.PARITY_PRICE_DIFFERENCE_COLLAR_LOW*parity_lower_bound
        anomaly_rows = grouped_df[grouped_df['no_parity_upper_bound'] < p.PARITY_PRICE_DIFFERENCE_COLLAR_LOW * grouped_df['parity_lower_bound']].copy()
        if not anomaly_rows.empty:
            for i in range(anomaly_rows.shape[0]):
                logger.info(anomaly_rows['GPI'].iloc[i] + '-' + anomaly_rows['NDC'].iloc[i] + '-' + anomaly_rows['BG_FLAG'].iloc[i] + '-' + anomaly_rows['REGION'].iloc[i]\
                                    + '_' + str(parity_subchain) +'-'+str(no_parity_subchain))
        anomaly_rows['GPI_NDC'] = anomaly_rows['GPI'] + '_***********'
        anomaly_parity_gpi = pd.concat([anomaly_parity_gpi, anomaly_rows[['CLIENT','REGION','GPI_NDC','BG_FLAG']]])
        good_rows = grouped_df[grouped_df['no_parity_upper_bound'] >= p.PARITY_PRICE_DIFFERENCE_COLLAR_LOW * grouped_df['parity_lower_bound']].copy()
        if not good_rows.empty:
            for i in range(good_rows.shape[0]):
                parity_price_cons_list.append(good_rows.var2.iloc[i] >= p.PARITY_PRICE_DIFFERENCE_COLLAR_LOW * good_rows.var1.iloc[i])
        #######################################################################################################################################################

        ############ If Clause no.4 ###########
        # elif clause: we want groups where all rows with "par" True has GPI != 0, AND all rows with "no_par" = True has GPI != 0 as well
        df_2['condition1'] = (df_2['par'] | df_2['no_par']) & (df_2['GPI_ONLY'] != 0)
        df_2['condition2'] = ~(df_2['par'] | df_2['no_par'])
        df_2['test'] = df_2['condition1'] | df_2['condition2'] # "test" is true if row is neither "par" or "no_par", OR if it's "par" or "no_par" but with GPI != 0
        groups = df_2.groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'])\
                     .agg(keep = ('test', 'all')).reset_index() # "keep" indicates whether all rows in group satisfies "test" condition and hence we should keep this group
        groups = groups[groups['keep'] == True]
        df_4 = groups.merge(df_2, on = ['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'], how = 'inner') # (I call the filtered dataframe df_4 just to be consistent with previous clauses)
        # get price bounds
        grouped_df_parity = df_2[df_2['par'] == True].groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'])\
                                                     .agg(parity_price_bounds = ('Price_Bounds', 'first')).reset_index()
        grouped_df_no_parity = df_2[df_2['no_par'] == True].groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'])\
                                                           .agg(no_parity_price_bounds = ('Price_Bounds', 'first')).reset_index()
        # get price decision variables (needed later for constructing constraints)
        grouped_df_parity_var = df_2[df_2['par'] == True].groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'])\
                                                         .agg(var1 = ('Price_Decision_Var', 'first')).reset_index()
        grouped_df_no_parity_var = df_2[df_2['no_par'] == True].groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'])\
                                                               .agg(var2 = ('Price_Decision_Var', 'first')).reset_index()
        # merging dataframes above so that we have one row per group with all the information we need for each execution in each row.
        grouped_df = df_4.groupby(['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT']).agg(placeholder = ('par', 'sum')).reset_index()
        grouped_df = grouped_df.drop(['placeholder'], axis = 1)  
        grouped_df = grouped_df.merge(grouped_df_no_parity, on = ['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'], how = 'inner')
        grouped_df = grouped_df.merge(grouped_df_no_parity_var, on = ['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'], how = 'inner')
        grouped_df = grouped_df.merge(grouped_df_parity, on = ['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'], how = 'inner')
        grouped_df = grouped_df.merge(grouped_df_parity_var, on = ['GPI','BG_FLAG','CLIENT','REGION','MEASUREMENT'], how = 'inner')
        grouped_df['parity_lower_bound'] = grouped_df['parity_price_bounds'].str[0]
        grouped_df['parity_upper_bound'] = grouped_df['parity_price_bounds'].str[1]
        grouped_df['no_parity_lower_bound'] = grouped_df['no_parity_price_bounds'].str[0]
        grouped_df['no_parity_upper_bound'] = grouped_df['no_parity_price_bounds'].str[1]
        grouped_df.drop(['no_parity_price_bounds','parity_price_bounds'], axis = 1, inplace = True)                    
        # if clause 4-1
        anomaly_rows = grouped_df[grouped_df['no_parity_lower_bound'] > p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH * grouped_df['parity_upper_bound']].copy()
        if not anomaly_rows.empty:
            for i in range(anomaly_rows.shape[0]):
                logger.info(anomaly_rows['GPI'].iloc[i] + '-***********-' + anomaly_rows['BG_FLAG'].iloc[i] + '-' + anomaly_rows['REGION'].iloc[i]\
                                    + ': ' + str(parity_subchain))
        anomaly_rows['GPI_NDC'] = anomaly_rows['GPI'] + '_***********'
        anomaly_parity_gpi = pd.concat([anomaly_parity_gpi, anomaly_rows[['CLIENT','REGION','GPI_NDC','BG_FLAG']]])
        good_rows = grouped_df[grouped_df['no_parity_lower_bound'] <= p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH * grouped_df['parity_upper_bound']].copy()
        if not good_rows.empty:
            for i in range(good_rows.shape[0]):
                parity_price_cons_list.append(good_rows.var2.iloc[i] <= p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH * good_rows.var1.iloc[i])  
        # if clause 4-2
        anomaly_rows = grouped_df[grouped_df['no_parity_upper_bound'] < p.PARITY_PRICE_DIFFERENCE_COLLAR_LOW * grouped_df['parity_lower_bound']].copy()
        if not anomaly_rows.empty:
            for i in range(anomaly_rows.shape[0]):
                logger.info(anomaly_rows['GPI'].iloc[i] + '-***********-' + anomaly_rows['BG_FLAG'].iloc[i] + '-' + anomaly_rows['REGION'].iloc[i]\
                                    + ': ' + str(parity_subchain))
        anomaly_rows['GPI_NDC'] = anomaly_rows['GPI'] + '_***********'
        anomaly_parity_gpi = pd.concat([anomaly_parity_gpi, anomaly_rows[['CLIENT','REGION','GPI_NDC','BG_FLAG']]])
        good_rows = grouped_df[grouped_df['no_parity_upper_bound'] >= p.PARITY_PRICE_DIFFERENCE_COLLAR_LOW * grouped_df['parity_lower_bound']].copy()
        if not good_rows.empty:
            for i in range(good_rows.shape[0]):
                parity_price_cons_list.append(good_rows.var2.iloc[i] >= p.PARITY_PRICE_DIFFERENCE_COLLAR_LOW * good_rows.var1.iloc[i])
        
#######################################################################################################################################################

        anomaly_parity_gpi['EXCLUDE_REASON'] = 'State Parity Constraints'

        logger.info("Ending State Parity vs Non-State Parity pricing constraints")
        end = time.time()
        logger.info("Run time: {} mins".format((end - start)/60.))
        logger.info('--------------------')
        # anomaly_parity_gpi = pd.DataFrame(columns=('CLIENT', 'REGION', 'GPI_NDC', 'BG_FLAG'))

        # file outputs
        with open(parity_price_cons_list_out, 'wb') as f:
            pickle.dump(parity_price_cons_list, f)
        with open(anomaly_state_parity_gpi_out, 'wb') as f:
            pickle.dump(anomaly_parity_gpi, f)

        return parity_price_cons_list, anomaly_parity_gpi
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'State Parity Pricing Constraint', repr(e), error_loc)
        raise e
###### End State Parity and Non-State Parity Pricing Constraints ####################


########################### Consistent MAC constraints #########################
def consistent_mac_constraints(
        params_file_in: str,
        unc_flag: bool,
        lp_data_df_in: InputPath('pickle'),
        mac_cons_list_out: OutputPath('pickle'),
        anomaly_const_mac_out: OutputPath('pickle'),
        loglevel: str = 'INFO'
        # kube_run: bool = True,
):
    import sys
    import os
    sys.path.append('/')
    import time
    import logging
    import pandas as pd
    import numpy as np
    import pickle
    import pulp
    import util_funcs as uf
    import BQ

    uf.write_params(params_file_in)
    import CPMO_parameters as p
    from qa_checks import qa_dataframe
    from CPMO_shared_functions import standardize_df
    from CPMO_lp_functions import generatePricingDecisionVariables
    from CPMO_shared_functions import update_run_status
    try:
        out_path = os.path.join(p.FILE_LOG_PATH, 'ClientPharmacyMacOptimization.log')
        logger = uf.log_setup(log_file_path=out_path, loglevel=loglevel)

        # file inputs
        with open(lp_data_df_in, 'rb') as f:
            lp_data_df = pickle.load(f)

            logger.info('--------------------')
            logger.info("Starting building consistent MAC constraints")
            start = time.time()

        mac_constraints = standardize_df(pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.MAC_CONSTRAINT_FILE), dtype=p.VARIABLE_TYPE_DIC))
        mac_constraints = standardize_df(mac_constraints)
        qa_dataframe(mac_constraints, dataset='MAC_CONSTRAINT_FILE_AT_{}'.format(os.path.basename(__file__)))
        # Old mac_constraints files will have CHAIN_GROUP rather than CHAIN_SUBGROUP.
        # However, in this situation, the PHARMACY_PERF_NAME is the CHAIN_GROUP, so this shouldn't break anything.
        mac_constraints_melt = pd.melt(mac_constraints, id_vars=['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT'],
                                       var_name='CHAIN_SUBGROUP', value_name='Constraint')

        max_constraint = mac_constraints_melt.Constraint.max()

        lp_data_df = generatePricingDecisionVariables(lp_data_df)
        lp_data_df['GPI_12'] = lp_data_df.GPI.str[0:12]
        lp_data_df['GPI_Strength'] = lp_data_df.GPI.str[12:]

        mac_constraints_col = ['CLIENT', 'BREAKOUT', 'MEASUREMENT', 'GPI_NDC', 'BG_FLAG',
                               'CHAIN_SUBGROUP', 'REGION', 'PHARMACY_TYPE', 'MAC_PRICE_UNIT_ADJ',
                               'Price_Bounds', 'Price_Decision_Var', 'Dec_Var_Name', 'FULLAWP_ADJ']
        price_constraints_df = lp_data_df[mac_constraints_col].loc[lp_data_df.PRICE_MUTABLE == 1, :]

        mac_cons_list = []
        anomaly_const_mac = pd.DataFrame(columns=['CLIENT', 'REGION', 'GPI_NDC', 'BG_FLAG', 'EXCLUDE_REASON'])

        ############################################# vectorization begins #############################################

        #############Filling the  mac_cons_list - good data ########################################################################################
        ##only the non 0 constraint values are needed, 0 values can be discarded
        mac_constraints_melt_df = mac_constraints_melt[mac_constraints_melt['Constraint'] != 0].reset_index()
        if mac_constraints_melt_df['BREAKOUT'].str.contains('G.*B|B.*G').any():
            if p.GENERIC_OPT and not p.BRAND_OPT:
                mac_constraints_melt_df['BG_FLAG'] = 'G'
            elif not p.GENERIC_OPT and p.BRAND_OPT:
                mac_constraints_melt_df['BG_FLAG'] = 'B'
            elif p.GENERIC_OPT and p.BRAND_OPT:
                mac_constraints_melt_df_g = mac_constraints_melt_df.copy()
                mac_constraints_melt_df_g['BG_FLAG'] = 'G'
                mac_constraints_melt_df_b = mac_constraints_melt_df.copy()
                mac_constraints_melt_df_b['BG_FLAG'] = 'B'
                mac_constraints_melt_df = pd.concat([mac_constraints_melt_df_g, mac_constraints_melt_df_b])
        else:
            mac_constraints_melt_df['BG_FLAG'] = np.where(mac_constraints_melt_df['BREAKOUT'].str.contains('G'), 'G', 'B')
        
        # creating a hash table sort of data structure to map a chain_subgroup to a number- this will help in creating combinations of chain subgroups required below
        ##the hash table will help removing duplicate combinations- like if i already got CVS as CHAIN_SUBGROUP_mac1 and KRG as CHAIN_SUBGROUP_mac2 for a given client, GPI_NDC and constraint, I dont need
        ##another row with everything the same but CVS as CHAIN_SUBGROUP_mac2 and KRG as CHAIN_SUBGROUP_mac1
        mac_constraints_melt_df['CHAIN_SUBGROUP_INDEX'] = mac_constraints_melt_df.index

        ##join to fetch the constraint value from mac_constraint_melt and add it to price_constraints_df to further to the comparisons
        mac_price_constraints_df = price_constraints_df.merge(mac_constraints_melt_df, how='inner',
                                                              on=['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT',
                                                                  'CHAIN_SUBGROUP','BG_FLAG'])

        mac_price_constraints_self_join_df = mac_price_constraints_df.merge(mac_price_constraints_df, how='inner',
                                                                            on=['CLIENT', 'Constraint', 'GPI_NDC','BG_FLAG'],
                                                                            suffixes=('_mac1', '_mac2'))

        ##removing duplicate combinations- like if i already got CVS as CHAIN_SUBGROUP_mac1 and KRG as CHAIN_SUBGROUP_mac2 for a given client, GPI_NDC and constraint, I dont need
        ##another row with everything the same but CVS as CHAIN_SUBGROUP_mac2 and KRG as CHAIN_SUBGROUP_mac1
        mac_price_constraints_self_join_df_filter = mac_price_constraints_self_join_df[
            mac_price_constraints_self_join_df['CHAIN_SUBGROUP_INDEX_mac1'] < mac_price_constraints_self_join_df[
                'CHAIN_SUBGROUP_INDEX_mac2']]
        mac_price_constraints_self_join_df_filter['price_cons'] = mac_price_constraints_self_join_df_filter[
                                                                      'Price_Decision_Var_mac1'] - \
                                                                  mac_price_constraints_self_join_df_filter[
                                                                      'Price_Decision_Var_mac2']
        
        if mac_price_constraints_self_join_df_filter.shape[0]>0:
            mac_price_constraints_self_join_df_filter['lower_bound1'], mac_price_constraints_self_join_df_filter['upper_bound1']  = zip(*mac_price_constraints_self_join_df_filter.Price_Bounds_mac1)
            mac_price_constraints_self_join_df_filter['lower_bound2'], mac_price_constraints_self_join_df_filter['upper_bound2']  = zip(*mac_price_constraints_self_join_df_filter.Price_Bounds_mac2)

            mac_price_constraints_self_join_df_filter['max_lower_bound']=mac_price_constraints_self_join_df_filter[['lower_bound1', 'lower_bound2']].max(axis=1)
            mac_price_constraints_self_join_df_filter['min_upper_bound']=mac_price_constraints_self_join_df_filter[['upper_bound1', 'upper_bound2']].min(axis=1)

            non_anomaly_df=mac_price_constraints_self_join_df_filter[mac_price_constraints_self_join_df_filter['max_lower_bound']<=mac_price_constraints_self_join_df_filter['min_upper_bound']]
            logger.info(f'Subgroup mappings for the same values in mac constraints file {non_anomaly_df}')

            mac_cons_list_no_condition=non_anomaly_df['price_cons'].to_list()
            
            if len(mac_cons_list_no_condition)>0:
                for i in mac_cons_list_no_condition:
                    mac_cons_list.append(i == 0)
                    
        #######################################################################################################################################
        #############Finding anomalies#########################################################################################################

            anomaly_df = mac_price_constraints_self_join_df_filter[mac_price_constraints_self_join_df_filter['max_lower_bound'] > mac_price_constraints_self_join_df_filter[
                'min_upper_bound']]
            if anomaly_df.shape[0] > 0:
                anomaly_const_mac['CLIENT'] = anomaly_df['CLIENT']
                anomaly_const_mac['REGION'] = anomaly_df['REGION_mac1']
                anomaly_const_mac['GPI_NDC'] = anomaly_df['GPI_NDC']
                anomaly_const_mac['BG_FLAG'] = anomaly_df['BG_FLAG']
                anomaly_const_mac['EXCLUDE_REASON'] = 'consistent mac price'


        logger.info("Ending building consistent MAC constraints")
        end = time.time()
        logger.info("Run time: {} mins".format((end - start) / 60.))
        logger.info('---------------------------------------------------------------')

        # file outputs
        with open(mac_cons_list_out, 'wb') as f:
            pickle.dump(mac_cons_list, f)
        with open(anomaly_const_mac_out, 'wb') as f:
            pickle.dump(anomaly_const_mac, f)

        return mac_cons_list, anomaly_const_mac
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'Consistent MAC Constraints', repr(e), error_loc)
        raise e
########################### End Consistent MAC constraints #########################


########################### Start Aggregate MAC price change constraints #########################
def agg_mac_constraints(
    params_file_in: str,
    month: int,
    unc_flag: bool,
    lp_data_df_in: InputPath('pickle'),
    agg_mac_cons_list_out: OutputPath('pickle'),
    loglevel: str = 'INFO'
    # kube_run: bool = True,
):
    import sys
    import os
    import time
    import logging
    import pandas as pd
    import pickle
    import pulp
    from pulp import LpAffineExpression
    import util_funcs as uf
    import BQ

    sys.path.append('/')
    uf.write_params(params_file_in)
    import CPMO_parameters as p
    from CPMO_lp_functions import generatePricingDecisionVariables
    from CPMO_shared_functions import update_run_status
    try:
        out_path = os.path.join(p.FILE_LOG_PATH, 'ClientPharmacyMacOptimization.log')
        logger = uf.log_setup(log_file_path=out_path, loglevel=loglevel)

        # file inputs
        with open(lp_data_df_in, 'rb') as f:
            lp_data_df = pickle.load(f)

        agg_mac_cons_list = []
        if p.AGG_UP_FAC >= 0:
            logger.info('--------------------')
            logger.info("Starting building aggregate MAC price change constraints")
            start = time.time()

            lp_data_df = generatePricingDecisionVariables(lp_data_df)
            lp_data_df['GPI_12'] = lp_data_df.GPI.str[0:12]
            lp_data_df['GPI_Strength'] = lp_data_df.GPI.str[12:]

            price_constraints_col = ['CLIENT', 'BREAKOUT', 'MEASUREMENT', 'QTY_PROJ_EOY', 'GPI_NDC', 'BG_FLAG',
                                        'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'REGION', 'PHARMACY_TYPE',
                                        'Price_Decision_Var', 'MAC_PRICE_UNIT_ADJ', 'Dec_Var_Name']

            price_constraints_df = lp_data_df[price_constraints_col].loc[lp_data_df.PRICE_MUTABLE==1,:]
            if month == 6:
                exempt_reg_list = ['REG_PLUS', 'REG_ALLURE']
            else:
                exempt_reg_list = []

            ########################## VECTORIZATION STARTS BELOW ##########################
            # remove rows where 'REGION' is in exempt list
            price_constraints_df = price_constraints_df[~price_constraints_df['REGION'].isin(exempt_reg_list)]

            # make new column which is the product of 'MAC_PRICE_UNIT_ADJ' and 'QTY_PROJ_EOY'
            price_constraints_df['ing_cost'] = price_constraints_df['MAC_PRICE_UNIT_ADJ'] * price_constraints_df['QTY_PROJ_EOY']

            # group by the 6 columns and then do agg sum
            groups = price_constraints_df.groupby(['CLIENT','BREAKOUT','REGION','MEASUREMENT','CHAIN_GROUP','CHAIN_SUBGROUP','BG_FLAG'], as_index=False)\
                                            .agg(lower_bound = ('ing_cost', 'sum'), upper_bound = ('ing_cost', 'sum'))
            groups['group_index'] = groups.index # add new column to indicate the group index, so that later we can iterate through each group
            groups['lower_bound'] = groups['lower_bound'] * (1 - p.AGG_LOW_FAC)
            groups['upper_bound'] = groups['upper_bound'] * (1 + p.AGG_UP_FAC)

            # RIGHT merge groupby result with original dataframe, so that every row in the original dataframe has the corresponding lower bound and upper bound.
            merged = groups.merge(price_constraints_df, on=['CLIENT','BREAKOUT','REGION','MEASUREMENT','CHAIN_GROUP','CHAIN_SUBGROUP','BG_FLAG'], how='right')

            # Next, iterate through each group to create "mac_cons":
            # Specifically, multiply merged.Price_Decision_Var with merged.QTY_PROJ_EOY values for each row and add them up to create an expression
            for i in range(groups.shape[0]):
                merged_current_i = merged[merged['group_index'] == i].copy() # create a copy so that in the next line I don't modify the original dataframe

                # create a new column with tuples that contain two variables to be multiplied in PuLP
                merged_current_i['tuple']=list(zip(merged_current_i['Price_Decision_Var'], merged_current_i['QTY_PROJ_EOY']))
                # convert column of tuples to list of tuples
                tuple_list = merged_current_i['tuple'].tolist()
                # directly create Lp variable from list of tuples (creating linear combinations such as 1x+2y+3z)
                mac_cons = LpAffineExpression(tuple_list)      

                # Next, create two constraints for the current group, for lower and upper bounds, respectively. 
                # Since all rows for this "group" have the same "lower_bound" and "upper_bound", we'll just take the first value:
                agg_mac_cons_list.append(mac_cons >= merged_current_i.lower_bound.values[0]) 
                agg_mac_cons_list.append(mac_cons <= merged_current_i.upper_bound.values[0])
                
            ########################## VECTORIZATION ENDS HERE ##########################
            logger.info("Ending building aggregate MAC price change constraints")
            end = time.time()
            logger.info("Run time: {} mins".format((end - start)/60.))
            logger.info('--------------------')

        # file outputs
        with open(agg_mac_cons_list_out, 'wb') as f:
            pickle.dump(agg_mac_cons_list, f)

        return agg_mac_cons_list
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'Aggregate MAC Price Change Constraints', repr(e), error_loc)
        raise e
########################### End Aggregate MAC price change constraints #########################



######################### Equal Package Size Contraints #########################
def equal_package_size_constraints(
    params_file_in: str,
    unc_flag: bool,
    lp_data_df_in: InputPath('pickle'),
    eq_pkg_sz_cons_list_out: OutputPath('pickle'),
    anomaly_const_pkg_sz_out: OutputPath('pickle'),
    loglevel: str = 'INFO'
    # kube_run: bool = True,
):
    import sys
    import os
    sys.path.append('/')
    import time
    import logging
    import pandas as pd
    import pickle
    import pulp
    import util_funcs as uf
    import BQ

    uf.write_params(params_file_in)
    import CPMO_parameters as p
    from CPMO_lp_functions import generatePricingDecisionVariables
    from CPMO_shared_functions import update_run_status
    try:
        out_path = os.path.join(p.FILE_LOG_PATH, 'ClientPharmacyMacOptimization.log')
        logger = uf.log_setup(log_file_path=out_path, loglevel=loglevel)

        # input files
        with open(lp_data_df_in, 'rb') as f:
            lp_data_df = pickle.load(f)

        logger.info('--------------------')
        logger.info("Equal Package Size Equal Price Constraints")
        start = time.time()

        lp_data_df = generatePricingDecisionVariables(lp_data_df)
        lp_data_df['GPI_12'] = lp_data_df.GPI.str[0:12]
        lp_data_df['GPI_Strength'] = lp_data_df.GPI.str[12:]

        price_constraints_col = ['CLIENT', 'BREAKOUT', 'MEASUREMENT', 'QTY_PROJ_EOY', 'GPI', 'GPI_NDC', 'NDC', 'BG_FLAG',
                                    'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'REGION', 'PHARMACY_TYPE', 'Price_Bounds',
                                    'Price_Decision_Var', 'MAC_PRICE_UNIT_ADJ', 'Dec_Var_Name', 'PKG_SZ']

        price_constraints_df = lp_data_df[price_constraints_col].loc[(lp_data_df.PRICE_MUTABLE==1)
                                            & (lp_data_df.GPI_ONLY==0)
                                            & (lp_data_df.PKG_SZ>0),:]

        #pref_chain = 'CVS'
        #pref_other = 'PREF_OTH'

        ############################################ VECTORIZATION STARTS HERE ############################################
        # initialization
        eq_pkg_sz_cons_list = []
        no_const_pkg_sz = []

        price_constraints_df['not_stars'] = (price_constraints_df['NDC'] != '***********').astype(int) # make new column to indicate whether column 'NDC' is '***********'
        grouped_df = price_constraints_df.groupby('GPI', as_index = False).agg(keep_GPI = ('not_stars', 'sum')) # group original df by GPI and sum "not_stars" for each group
        price_constraints_df = grouped_df.merge(price_constraints_df, on=['GPI'], how='right') # merge so that original df has a column indicating sum of "not_stars" for each group 
        price_constraints_df = price_constraints_df[price_constraints_df['keep_GPI'] != 0] # rows with 'keep_GPI' == 0 means this GPI group has all 'NDC' being '***********', and should be discarded

        # next, use group-by instead of for loops
        price_constraints_df.reset_index(drop=True, inplace=True) # right now the indices are not continuous, let's reset index so it goes 0, 1, 2...
        grouped_df = price_constraints_df.groupby(['CLIENT','BREAKOUT','MEASUREMENT','GPI','CHAIN_GROUP',
                                                   'CHAIN_SUBGROUP','REGION','PKG_SZ','BG_FLAG']).ngroup().reset_index(name='group_index') # group by and assign group_index to each group
        price_constraints_df.reset_index(drop=False, inplace=True) # create a new column called "index", because we need it for merging

        # merge every row of the orignal dataframe with the grouped by dataframe, ON index matches. This allows us to get the group (belonging) index of each row in original dataframe.
        merged = grouped_df.merge(price_constraints_df, on=['index'], how='right') 
        # we no longer need the separate "index" column, drop it
        merged.drop('index',axis=1,inplace=True)
        # Next, remove rows which correspond to group size = 1. 
        merged = merged[merged.duplicated(subset='group_index', keep=False)] # "Keep=False" so that all duplicated rows are identified
        # since we removed some rows, redo the indexing so it goes 0, 1, 2...
        merged.reset_index(drop=False, inplace=True)
        # so far we have essentially (1) found group index for each row, and (2) removed groups that have only one row.

        # get price lower bound and upper bound
        merged['Price_LB'] = merged['Price_Bounds'].str[0]
        merged['Price_UB'] = merged['Price_Bounds'].str[1]
        # remove columns we will no longer need, to make things cleaner...
        merged.drop(columns=['BREAKOUT','MEASUREMENT','CHAIN_GROUP','CHAIN_SUBGROUP',
                             'PHARMACY_TYPE','GPI_NDC','QTY_PROJ_EOY','NDC','Price_Bounds','index'],inplace=True)
        # add a column "index", we need this for self merging later. This will become "index_left" and "index_right" after self merging
        merged.reset_index(drop=False, inplace=True)

        # instead of iterating i and then iterating j like in the original code, we do self merging, while only keeping rows where left index is smaller than right index
        self_merged = merged.merge(merged, on ='group_index', suffixes=('_left', '_right'))
        self_merged = self_merged[self_merged.index_left < self_merged.index_right].reset_index(drop=True)

        # construct a few columns which are needed later
        self_merged['price_cons'] = self_merged['Price_Decision_Var_left'] - self_merged['Price_Decision_Var_right']
        self_merged['MAC_PRICE_UNIT_ADJ_equal'] = (self_merged['MAC_PRICE_UNIT_ADJ_left'] == self_merged['MAC_PRICE_UNIT_ADJ_right'])
        self_merged['bound_anormal'] = (self_merged[['Price_LB_left','Price_LB_right']].max(axis=1) > self_merged[['Price_UB_left','Price_UB_right']].min(axis=1))

        # make the "no_const_pkg_sz" list
        self_merged['tuples'] = list(zip(self_merged['Dec_Var_Name_left'], self_merged['Dec_Var_Name_right']))
        no_const_pkg_sz = self_merged.loc[self_merged['MAC_PRICE_UNIT_ADJ_equal']==False, 'tuples'].tolist()

        # make the "anomaly_const_pkg_sz" dataframe
        self_merged_anormal = self_merged[(self_merged['MAC_PRICE_UNIT_ADJ_equal'] == True) & (self_merged['bound_anormal'] == True)]
        anomaly_const_pkg_sz = pd.DataFrame({
            'CLIENT': self_merged_anormal['CLIENT_left'].astype(str), 
            'REGION': self_merged_anormal['REGION_left'].astype(str),
            'GPI_NDC': self_merged_anormal['GPI_left'].astype(str) + '_' + self_merged_anormal['PKG_SZ_left'].astype(str),
            'BG_FLAG': self_merged_anormal['BG_FLAG_left'].astype(str),
            'EXCLUDE_REASON': 'consistent pkg size'
        })
        # for logger
        for index, row in anomaly_const_pkg_sz.iterrows(): 
            var1 = self_merged.Dec_Var_Name_left[index]
            var2 = self_merged.Dec_Var_Name_right[index]
            logger.info('Error with ' + str(var1) + ' and ' + str(var2) + ' consistent pkg size')

        # make Lp Constraints
        eq_pkg_sz_cons_list = self_merged.loc[(self_merged['MAC_PRICE_UNIT_ADJ_equal'] == True) & (self_merged['bound_anormal'] == False) ,'price_cons'].tolist() # first collect rows to be appended
        for i in range(len(eq_pkg_sz_cons_list)):
            eq_pkg_sz_cons_list[i] = (eq_pkg_sz_cons_list[i] == 0)

        ############################################ VECTORIZATION ENDS HERE ############################################    
        logger.info("Ending Equal Package Size Equal Price Constraints")
        end = time.time()
        logger.info("Run time: {} mins".format((end - start)/60.))
        logger.info('--------------------')

        # file outputs
        with open(eq_pkg_sz_cons_list_out, 'wb') as f:
            pickle.dump(eq_pkg_sz_cons_list, f)
        with open(anomaly_const_pkg_sz_out, 'wb') as f:
            pickle.dump(anomaly_const_pkg_sz, f)

        return eq_pkg_sz_cons_list, anomaly_const_pkg_sz
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'Equal Package Size Constraints', repr(e), error_loc)
        raise e
###### End Equal Package Size Contraints ####################


###### Same Difference Package Size Constraints ########
def same_difference_package_size_constraints(
    params_file_in: str,
    unc_flag: bool,
    lp_data_df_in: InputPath('pickle'),
    sm_diff_pkg_sz_cons_list_out: OutputPath('pickle'),
    loglevel: str = 'INFO'
    # kube_run: bool = True,
):
    import sys
    import os
    sys.path.append('/')
    import time
    import logging
    import pandas as pd
    import pickle
    import pulp
    import util_funcs as uf
    import BQ

    uf.write_params(params_file_in)
    import CPMO_parameters as p
    from CPMO_lp_functions import generatePricingDecisionVariables
    from CPMO_shared_functions import update_run_status
    try:
        out_path = os.path.join(p.FILE_LOG_PATH, 'ClientPharmacyMacOptimization.log')
        logger = uf.log_setup(log_file_path=out_path, loglevel=loglevel)

        # file inputs
        with open(lp_data_df_in, 'rb') as f:
            lp_data_df = pickle.load(f)

        logger.info('--------------------')
        logger.info("Same Difference Package Size Constraints")
        start = time.time()

        lp_data_df = generatePricingDecisionVariables(lp_data_df)
        lp_data_df['GPI_12'] = lp_data_df.GPI.str[0:12]
        lp_data_df['GPI_Strength'] = lp_data_df.GPI.str[12:]

        price_constraints_col = ['CLIENT', 'BREAKOUT', 'MEASUREMENT', 'QTY_PROJ_EOY', 'GPI', 'GPI_NDC', 'NDC', 'BG_FLAG',
                                    'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'REGION', 'PHARMACY_TYPE', 'Price_Bounds',
                                    'Price_Decision_Var', 'MAC_PRICE_UNIT_ADJ', 'Dec_Var_Name', 'PKG_SZ']
        price_constraints_df = lp_data_df[price_constraints_col].loc[(lp_data_df.PRICE_MUTABLE==1) & (lp_data_df.GPI_ONLY==0),:]

        #pref_chain = 'CVS'
        #pref_other = 'PREF_OTH'

        sm_diff_pkg_sz_cons_list = []
        gpi_arr = price_constraints_df.loc[price_constraints_df.NDC != '***********'].GPI.unique()
        anomaly_same_difference = []
        for gpi in gpi_arr:
            gpi_df = price_constraints_df[price_constraints_df.GPI == gpi]
            for client in gpi_df.CLIENT.unique():
                for breakout in gpi_df.loc[gpi_df.CLIENT == client, 'BREAKOUT'].unique():
                    region_arr = gpi_df.loc[(gpi_df.CLIENT==client) & (gpi_df.BREAKOUT==breakout)].REGION.unique()
                    for reg in ['REGIONS']: # region_arr:
                        pharm_arr = gpi_df.loc[(gpi_df.CLIENT==client)
                                                & (gpi_df.BREAKOUT==breakout)
                                                & (gpi_df.REGION==reg)].CHAIN_GROUP.unique()
                        for pharm in pharm_arr:
                            pharm_sub_arr = gpi_df.loc[(gpi_df.CLIENT==client)
                                                & (gpi_df.BREAKOUT==breakout)
                                                & (gpi_df.REGION==reg)
                                                & (gpi_df.CHAIN_GROUP==pharm)].CHAIN_SUBGROUP.unique()
                            for pharm_sub in pharm_sub_arr:
                                bg_flag_arr = gpi_df.loc[(gpi_df.CLIENT==client)
                                                        & (gpi_df.BREAKOUT==breakout)
                                                        & (gpi_df.REGION==reg)
                                                        & (gpi_df.CHAIN_GROUP==pharm)
                                                        & (gpi_df.CHAIN_SUBGROUP==pharm_sub)].BG_FLAG.unique()
                                for bg_flag in bg_flag_arr:
                                    measure_arr = gpi_df.loc[(gpi_df.CLIENT==client)
                                                            & (gpi_df.BREAKOUT==breakout)
                                                            & (gpi_df.REGION==reg)
                                                            & (gpi_df.CHAIN_GROUP==pharm)
                                                            & (gpi_df.CHAIN_SUBGROUP==pharm_sub)
                                                            & (gpi_df.BG_FLAG==bg_flag)].MEASUREMENT.unique()
                                    for measure in ['R30']: #measure_arr:
                                        gpi_meas_df = gpi_df.loc[(gpi_df.CLIENT==client)
                                                            & (gpi_df.BREAKOUT==breakout)
                                                            & (gpi_df.REGION==reg)
                                                            & (gpi_df.CHAIN_GROUP==pharm)
                                                            & (gpi_df.CHAIN_SUBGROUP==pharm_sub)
                                                            & (gpi_df.BG_FLAG==bg_flag)
                                                            & (gpi_df.MEASUREMENT==measure)]

                                        ndcs = gpi_meas_df.NDC.unique()
                                        if len(ndcs) > 1:
                                            for i in range(len(ndcs)-1):
                                                for j in range(i+1, len(ndcs)):
                                                    price_cons = ""
                                                    if (gpi_meas_df.loc[gpi_meas_df.NDC == ndcs[i], 'MAC_PRICE_UNIT_ADJ'].values[0] - gpi_meas_df.loc[gpi_meas_df.NDC == ndcs[j], 'MAC_PRICE_UNIT_ADJ'].values[0]) >= 0:
                                                        price_cons += gpi_meas_df.loc[gpi_meas_df.NDC == ndcs[i], 'Price_Decision_Var'].values[0] - gpi_meas_df.loc[gpi_meas_df.NDC == ndcs[j], 'Price_Decision_Var'].values[0]
                                                    else:
                                                        price_cons += gpi_meas_df.loc[gpi_meas_df.NDC == ndcs[j], 'Price_Decision_Var'].values[0] - gpi_meas_df.loc[gpi_meas_df.NDC == ndcs[i], 'Price_Decision_Var'].values[0]

                                                    logger.info(price_cons)
                                                    curr_diff = gpi_meas_df.loc[gpi_meas_df.NDC == ndcs[i], 'MAC_PRICE_UNIT_ADJ'].values[0] - gpi_meas_df.loc[gpi_meas_df.NDC == ndcs[j], 'MAC_PRICE_UNIT_ADJ'].values[0]

                                                    sm_diff_pkg_sz_cons_list.append(price_cons >= (curr_diff/1000))
                                                    sm_diff_pkg_sz_cons_list.append(price_cons <= (curr_diff/.0001))

        logger.info("Ending Same Difference Package Size Constraints")
        end = time.time()
        logger.info("Run time: {} mins".format((end - start)/60.))
        logger.info('--------------------')

        # file outputs
        with open(sm_diff_pkg_sz_cons_list_out, 'wb') as f:
            pickle.dump(sm_diff_pkg_sz_cons_list, f)

        return sm_diff_pkg_sz_cons_list
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'Same Difference Package Size Constraints', repr(e), error_loc)
        raise e
###### End Same Difference Package Size Constraints ####################

def leakage_opt(
    params_file_in: str,
    lp_data_df_in: InputPath('pickle'),
    leakage_cost_list_out: OutputPath('pickle'),
    leakage_const_list_out: OutputPath('pickle'),
    loglevel: str = 'INFO'
):
    '''
    Adds a penalty to the objective function proportional to the amount of leakage generated by a given price,
    in order to encourage lowering the price of GPIs that generate leakage. Leakage is the share of a claim's 
    total cost paid out by a member that's above the pharmacy cost/MAC1026 at non-capped pharmacies in locked
    clients. 
    
    To calculate the penalty, the "Big M" method is employed. Leakage is normally calculated non-linearly when
    a plan design uses a copay structure. For example, no leakage occurs in all cases when member share is less
    than the MAC1026 cost. Leakage increases linearly between the MAC1026 cost and the maximumm copay cost as
    the drug price increases, after which leakage plateaus and remains at the max copay cost even as the drug price
    goes up. For coinsurance designs, leakage increases linearly with the drug cost when the member share is greater
    than the MAC1026 cost.
    
    For both copay and complex designs, we break up the leakage function into three parts and use the Big M technique to control 
    which part of the leakage function is used to calculate leakage. Which part of the function is activated depends 
    on where the pricing decision variable falls within the three situations described in the previous paragraph. The 
    Big M technique is not used for coinsurance designs, and is controlled using simple linear constraints.
    
    Complex designs have both a copay and a coinsurance for a single claim. For example, for a $20 claim with a $10 copay 
    and 20% coinsurance, a member would pay the first full $10, and then would pay an additional $2 on the remaining $10 of the claim. 
    The client would pay the remaining $8. If other complex structures exist that differ from this, this function would need
    to be updated to handle any specific differences.
    
    Relevant Parameters:
        p.LEAKAGE_PENALTY: Weight used to control how much leakage penalizes the overall objective function. A value of 0.5 would halve the
                           the amount of leakage used to penalize the objective function, 1.0 would be dollar per dollar weight, and 2.0 would double
                           the weight. Current default is 5.0 based on FULL_YEAR test runs.
        p.NON_CAPPED_PHARMACY_LIST: List used to determine which pharmacies have GPIs that generate leakage. Be sure this aligns with
                                    the logic used in the gpi_leakage_rankings table and value reporting.
        p.COGS_PHARMACY_LIST: In rare cases an independent or PSAO pharmacy may be placed here, so it's included to make sure all are captured.
        p.LEAKAGE_LIST: Sets the list of what GPIs should have the leakage penalty applied. Default is 'All', meaning all GPIs will have the penalty 
                        calculated. Other options are 'Client' and 'Legacy'.
        p.LEAKAGE_RANK: Limit which GPIs have a leakage penalty applied. Reduce this if solve time becomes too long. Not applicable when LEAKAGE_LIST is 'All'
        p.LEAKAGE_COPAY_SIMPLE_BREAKPOINT: Used to set which column is used for the maximum copay breakpoint of the leakage function for plan
                                           designs that have copays only. Average, median, minimum, or maximum copay. 
        p.LEAKAGE_COPAY_COMPLEX_BREAKPOINT: Used to set which column is used for the maximum copay breakpoint of the leakage function for plan
                                            designs that are complex. Average, median, minimum, or maximum copay.
        p.LEAKAGE_COINSURANCE_SIMPLE: Used to set which column is used as the coinsurance factor for the leakage function for plan designs that 
                                      have coinsurances only. Average, median, minimum, or maximum coinsurance. 
        p.LEAKAGE_COINSURANCE_COMPLEX: Used to set which column is used as the coinsurance factor for the leakage function for complex plan 
                                       designs. Average, median, minimum, or maximum coinsurance.

    Outputs:
        Pickle files containing objects to be added to the overall object submitted to the solver.
        leakage_cost_list: Objects to be added to the overall objective function
        leakage_const_list: Objects to be added to the overall set of constraints
    '''
    import sys
    import os
    sys.path.append('/')
    import time
    import pandas as pd
    import pulp
    import pickle
    import util_funcs as uf
    uf.write_params(params_file_in)
    import CPMO_parameters as p
    from CPMO_shared_functions import update_run_status
    from CPMO_lp_functions import generatePricingDecisionVariables, generateLeakageDecisionVariables

    update_run_status(i_error_type='Started leakage_opt')

    try:
        out_path = os.path.join(p.FILE_LOG_PATH, 'ClientPharmacyMacOptimization.log')
        logger = uf.log_setup(log_file_path=out_path, loglevel=loglevel)

        # Retrieve LP data and list of GPIs to calculate leakage for
        with open(lp_data_df_in, 'rb') as f:
            lp_data_df = pickle.load(f) 
        
        if p.LEAKAGE_LIST != 'All':
            gpi_list = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + 'LEAKAGE_GPI_DF_{}.csv'.format(p.DATA_ID), dtype=p.VARIABLE_TYPE_DIC)['GPI'].tolist()
        
        # Set parameters
        penalty = p.LEAKAGE_PENALTY
        copay_smpl_col = p.LEAKAGE_COPAY_SIMPLE_BREAKPOINT
        copay_cmpx_col = p.LEAKAGE_COPAY_COMPLEX_BREAKPOINT
        coins_smpl_col = p.LEAKAGE_COINSURANCE_SIMPLE
        coins_cmpx_col = p.LEAKAGE_COINSURANCE_COMPLEX

        # Separate upper and lower bounds for use in creating Big M values
        lp_data_df['LOW_B'],lp_data_df['UP_B'] = zip(*lp_data_df.Price_Bounds)
        
        # Create decision variables
        lp_data_df = generatePricingDecisionVariables(lp_data_df) 
        lp_data_df['delta1'] = generateLeakageDecisionVariables(lp_data_df, 'bin', 1)
        lp_data_df['delta2'] = generateLeakageDecisionVariables(lp_data_df, 'bin', 2)
        lp_data_df['delta3'] = generateLeakageDecisionVariables(lp_data_df, 'bin', 3)
        lp_data_df['delta4'] = generateLeakageDecisionVariables(lp_data_df, 'bin', 4)
        lp_data_df['delta5'] = generateLeakageDecisionVariables(lp_data_df, 'bin', 5)
        lp_data_df['delta6'] = generateLeakageDecisionVariables(lp_data_df, 'bin', 6)
        lp_data_df['leakage_copay'] = generateLeakageDecisionVariables(lp_data_df, 'con', nm = 'copay')
        lp_data_df['leakage_coins'] = generateLeakageDecisionVariables(lp_data_df, 'con', nm = 'coins')
        lp_data_df['leakage_complex'] = generateLeakageDecisionVariables(lp_data_df, 'con', nm ='complex')

        # Filter for rows we want to have the leakage penalty applied to
        if p.LEAKAGE_LIST == 'All':
            lp_data_df = lp_data_df.loc[((lp_data_df.PRICE_MUTABLE==1) & (lp_data_df.QTY_PROJ_EOY > 0) &
                                        (lp_data_df.CHAIN_GROUP.isin(p.NON_CAPPED_PHARMACY_LIST['GNRC'] + p.COGS_PHARMACY_LIST['GNRC'])) &
                                        (lp_data_df.BG_FLAG == 'G')),:]        
        else:
            lp_data_df = lp_data_df.loc[((lp_data_df.PRICE_MUTABLE==1) & (lp_data_df.QTY_PROJ_EOY > 0) &
                                        (lp_data_df.CHAIN_GROUP.isin(p.NON_CAPPED_PHARMACY_LIST['GNRC'] + p.COGS_PHARMACY_LIST['GNRC'])) & 
                                        (lp_data_df.BG_FLAG == 'G') &
                                        (lp_data_df.GPI.isin(gpi_list)))
                                        ,:]

        logger.info('--------------------')
        logger.info("Preparing Leakage Optimization")
        start = time.time()
            
        # Create lists to export to the the overall LP problem
        leakage_cost_list = []
        leakage_const_list = []
        
        # The code used to create the constraints are written to maintain the form of linear equations, and aren't simplified in order
        # to maintain comparability with broader conventions in using the Big M technique. The particular equation used for each contstraint
        # is in a comment before each relevant code chunk. First, we create the objects that can be calculated before solve-time (referred to
        # below as constants), then create the constraints, which are dependent on the status of the decision variables.
        
        # Create constants used across all plan design types
        lp_data_df['M1'] = lp_data_df['UP_B'] * lp_data_df['QTY_PROJ_EOY']
        lp_data_df['d0'] = 0
        lp_data_df['d3'] = lp_data_df['M1']
        lp_data_df['a1'] = 0
        lp_data_df['b1'] = 0
        lp_data_df['b2'] = 1
        lp_data_df['b3'] = 0
        lp_data_df['d1'] = lp_data_df['MAC1026_UNIT_PRICE'] * lp_data_df['QTY_PROJ_EOY']
        lp_data_df['a2'] = -lp_data_df['MAC1026_UNIT_PRICE'] * lp_data_df['QTY_PROJ_EOY']
        
        # Create constants used for copay only designs
        lp_data_df['d2'] = lp_data_df[copay_smpl_col] * lp_data_df['QTY_PROJ_EOY']
        lp_data_df['a3'] = lp_data_df[copay_smpl_col] * lp_data_df['QTY_PROJ_EOY'] - lp_data_df['MAC1026_UNIT_PRICE'] * lp_data_df['QTY_PROJ_EOY']
        lp_data_df['M2'] = lp_data_df[['d2', 'd1']].max(axis=1)
        lp_data_df['M3'] = lp_data_df['UP_B'] * lp_data_df['QTY_PROJ_EOY'] + lp_data_df[['d2', 'd1']].max(axis=1)
    
        # Create constants used for complex designs
        lp_data_df['d4'] = lp_data_df[copay_cmpx_col] * lp_data_df['QTY_PROJ_EOY']
        lp_data_df['b4'] = lp_data_df[coins_cmpx_col]
        lp_data_df['a4'] = (lp_data_df[copay_cmpx_col] * lp_data_df['QTY_PROJ_EOY'] - lp_data_df['MAC1026_UNIT_PRICE'] * lp_data_df['QTY_PROJ_EOY'] - 
                            lp_data_df[coins_cmpx_col] * (lp_data_df[copay_cmpx_col] * lp_data_df['QTY_PROJ_EOY']))
        lp_data_df['M4'] = lp_data_df[['d4', 'd1']].max(axis=1)
        lp_data_df['M5'] = lp_data_df['UP_B'] * lp_data_df['QTY_PROJ_EOY'] + lp_data_df[['d4', 'd1']].max(axis=1)
        
        # Create constant used for coinsurance only designs
        lp_data_df['b5'] = lp_data_df[coins_smpl_col]

        # Create constraints for each row if they have non-zero weight. The quantity weight columns tell us what percent of quantity within a
        # given row has copay only, coinsurance only, or complex plan design utilization so that we can adjust the aggregated leakage calculation
        # proportionally.
        for i in range(len(lp_data_df)): 
            if lp_data_df['COPAY_ONLY_QTY_WT'].iloc[i] > 0:
                # Find the ceiling of Big M: IC*qty - di - M*(1-deltai) <= 0   
                leakage_const_list += pd.Series([pulp.LpConstraint(lp_data_df['Price_Decision_Var'].iloc[i] * lp_data_df['QTY_PROJ_EOY'].iloc[i] - 
                                                                   lp_data_df['d1'].iloc[i] - lp_data_df['M1'].iloc[i] * (1 - lp_data_df['delta1'].iloc[i]),
                                                                   sense = pulp.LpConstraintLE, 
                                                                   rhs = 0)]).to_list() # Function 1: IC<=mac1026
                leakage_const_list += pd.Series([pulp.LpConstraint(lp_data_df['Price_Decision_Var'].iloc[i] * lp_data_df['QTY_PROJ_EOY'].iloc[i] - 
                                                                   lp_data_df['d2'].iloc[i] - lp_data_df['M1'].iloc[i] * (1 - lp_data_df['delta2'].iloc[i]),
                                                                   sense = pulp.LpConstraintLE, 
                                                                   rhs = 0)]).to_list() # Function 2: IC>=mac1026
                leakage_const_list += pd.Series([pulp.LpConstraint(lp_data_df['Price_Decision_Var'].iloc[i] * lp_data_df['QTY_PROJ_EOY'].iloc[i] - 
                                                                   lp_data_df['d3'].iloc[i] - lp_data_df['M1'].iloc[i] * (1 - lp_data_df['delta3'].iloc[i]),
                                                                   sense = pulp.LpConstraintLE, 
                                                                   rhs = 0)]).to_list() # Function 3: IC>=copay

                # Find the floor of Big M: IC*qty - d(i-1) + M*(1-deltai) >= 0
                leakage_const_list += pd.Series([pulp.LpConstraint(lp_data_df['Price_Decision_Var'].iloc[i] * lp_data_df['QTY_PROJ_EOY'].iloc[i] - 
                                                                   lp_data_df['d0'].iloc[i] + lp_data_df['M2'].iloc[i] * (1- lp_data_df['delta1'].iloc[i]),  
                                                                   sense = pulp.LpConstraintGE,
                                                                   rhs =  0)]).to_list() # Function 1: IC>=0
                leakage_const_list += pd.Series([pulp.LpConstraint(lp_data_df['Price_Decision_Var'].iloc[i] * lp_data_df['QTY_PROJ_EOY'].iloc[i] - 
                                                                   lp_data_df['d1'].iloc[i] + lp_data_df['M2'].iloc[i] * (1- lp_data_df['delta2'].iloc[i]),  
                                                                   sense = pulp.LpConstraintGE,
                                                                   rhs =  0)]).to_list() # Function 2: IC>=mac1026
                leakage_const_list += pd.Series([pulp.LpConstraint(lp_data_df['Price_Decision_Var'].iloc[i] * lp_data_df['QTY_PROJ_EOY'].iloc[i] - 
                                                                   lp_data_df['d2'].iloc[i] + lp_data_df['M2'].iloc[i] * (1- lp_data_df['delta3'].iloc[i]),  
                                                                   sense = pulp.LpConstraintGE,
                                                                   rhs =  0)]).to_list() # Function 3: IC>=copay

                # Functions that calculate leakage: leakage_var - ai - bi*IC*qty + M(1-deltai) >= 0 
                leakage_const_list += pd.Series([pulp.LpConstraint(lp_data_df['leakage_copay'].iloc[i] - lp_data_df['a1'].iloc[i] - lp_data_df['b1'].iloc[i] * 
                                                                   lp_data_df['Price_Decision_Var'].iloc[i] * lp_data_df['QTY_PROJ_EOY'].iloc[i] + 
                                                                   lp_data_df['M3'].iloc[i]*(1-lp_data_df['delta1'].iloc[i]),
                                                                   sense = pulp.LpConstraintGE, 
                                                                   rhs = 0)]).to_list() # Function 1    
                leakage_const_list += pd.Series([pulp.LpConstraint(lp_data_df['leakage_copay'].iloc[i] - lp_data_df['a2'].iloc[i] - lp_data_df['b2'].iloc[i] * 
                                                                   lp_data_df['Price_Decision_Var'].iloc[i] * lp_data_df['QTY_PROJ_EOY'].iloc[i] + 
                                                                   lp_data_df['M3'].iloc[i]*(1-lp_data_df['delta2'].iloc[i]),
                                                                   sense = pulp.LpConstraintGE, 
                                                                   rhs = 0)]).to_list() # Function 2    
                leakage_const_list += pd.Series([pulp.LpConstraint(lp_data_df['leakage_copay'].iloc[i] - lp_data_df['a3'].iloc[i] - lp_data_df['b3'].iloc[i] *
                                                                   lp_data_df['Price_Decision_Var'].iloc[i] * lp_data_df['QTY_PROJ_EOY'].iloc[i] + 
                                                                   lp_data_df['M3'].iloc[i]*(1-lp_data_df['delta3'].iloc[i]),
                                                                   sense = pulp.LpConstraintGE, 
                                                                   rhs = 0)]).to_list() # Function 3  
                
                # Assure only one delta variable is allowed to be set to 1. Note these are binary variables.
                leakage_const_list += pd.Series([pulp.LpConstraint(lp_data_df['delta1'].iloc[i] + lp_data_df['delta2'].iloc[i] + 
                                                                   lp_data_df['delta3'].iloc[i] == 1)]).to_list()

                # Leakage cannot be less than 0
                leakage_const_list += pd.Series([pulp.LpConstraint(lp_data_df['leakage_copay'].iloc[i], sense = pulp.LpConstraintGE, rhs = 0)]).to_list()

            if lp_data_df['COPAY_COINS_QTY_WT'].iloc[i] > 0:
                # Find the ceiling of Big M: IC*qty - di - M*(1-deltai) <= 0     
                leakage_const_list += pd.Series([pulp.LpConstraint(lp_data_df['Price_Decision_Var'].iloc[i] * lp_data_df['QTY_PROJ_EOY'].iloc[i] - 
                                                                   lp_data_df['d1'].iloc[i] - lp_data_df['M1'].iloc[i] * (1 - lp_data_df['delta4'].iloc[i]),
                                                                   sense = pulp.LpConstraintLE, 
                                                                   rhs = 0)]).to_list() # Function 1: IC<=mac1026
                leakage_const_list += pd.Series([pulp.LpConstraint(lp_data_df['Price_Decision_Var'].iloc[i] * lp_data_df['QTY_PROJ_EOY'].iloc[i] - 
                                                                   lp_data_df['d4'].iloc[i] - lp_data_df['M1'].iloc[i] * (1 - lp_data_df['delta5'].iloc[i]),
                                                                   sense = pulp.LpConstraintLE, 
                                                                   rhs = 0)]).to_list() # Function 2: IC>=mac1026
                leakage_const_list += pd.Series([pulp.LpConstraint(lp_data_df['Price_Decision_Var'].iloc[i] * lp_data_df['QTY_PROJ_EOY'].iloc[i] - 
                                                                   lp_data_df['d3'].iloc[i] - lp_data_df['M1'].iloc[i] * (1 - lp_data_df['delta6'].iloc[i]),
                                                                   sense = pulp.LpConstraintLE, 
                                                                   rhs = 0)]).to_list() # Function 3: IC>=copay

                # Find the floor of Big M: IC*qty - d(i-1) + M*(1-deltai) >= 0
                leakage_const_list += pd.Series([pulp.LpConstraint(lp_data_df['Price_Decision_Var'].iloc[i] * lp_data_df['QTY_PROJ_EOY'].iloc[i] - 
                                                                   lp_data_df['d0'].iloc[i] + lp_data_df['M4'].iloc[i] * (1- lp_data_df['delta4'].iloc[i]),  
                                                                   sense = pulp.LpConstraintGE,
                                                                   rhs =  0)]).to_list() # Function 1: IC>=0
                leakage_const_list += pd.Series([pulp.LpConstraint(lp_data_df['Price_Decision_Var'].iloc[i] * lp_data_df['QTY_PROJ_EOY'].iloc[i] - 
                                                                   lp_data_df['d1'].iloc[i] + lp_data_df['M4'].iloc[i] * (1- lp_data_df['delta5'].iloc[i]),  
                                                                   sense = pulp.LpConstraintGE,
                                                                   rhs =  0)]).to_list() # Function 2: IC>=mac1026
                leakage_const_list += pd.Series([pulp.LpConstraint(lp_data_df['Price_Decision_Var'].iloc[i] * lp_data_df['QTY_PROJ_EOY'].iloc[i] - 
                                                                   lp_data_df['d4'].iloc[i] + lp_data_df['M4'].iloc[i] * (1- lp_data_df['delta6'].iloc[i]),  
                                                                   sense = pulp.LpConstraintGE,
                                                                   rhs =  0)]).to_list() # Function 3: IC>=copay

                # Functions that calculate leakage: leakage_var - ai - bi*IC*qty + M(1-deltai) >= 0 
                leakage_const_list += pd.Series([pulp.LpConstraint(lp_data_df['leakage_complex'].iloc[i] - lp_data_df['a1'].iloc[i] - lp_data_df['b1'].iloc[i] * 
                                                                   lp_data_df['Price_Decision_Var'].iloc[i] * lp_data_df['QTY_PROJ_EOY'].iloc[i] + 
                                                                   lp_data_df['M5'].iloc[i]*(1-lp_data_df['delta4'].iloc[i]),
                                                                   sense = pulp.LpConstraintGE, 
                                                                   rhs = 0)]).to_list() # Function 1    
                leakage_const_list += pd.Series([pulp.LpConstraint(lp_data_df['leakage_complex'].iloc[i] - lp_data_df['a2'].iloc[i] - lp_data_df['b2'].iloc[i] * 
                                                                   lp_data_df['Price_Decision_Var'].iloc[i] * lp_data_df['QTY_PROJ_EOY'].iloc[i] + 
                                                                   lp_data_df['M5'].iloc[i]*(1-lp_data_df['delta5'].iloc[i]),
                                                                   sense = pulp.LpConstraintGE, 
                                                                   rhs = 0)]).to_list() # Function 2    
                leakage_const_list += pd.Series([pulp.LpConstraint(lp_data_df['leakage_complex'].iloc[i] - lp_data_df['a4'].iloc[i] - lp_data_df['b4'].iloc[i] *
                                                                   lp_data_df['Price_Decision_Var'].iloc[i] * lp_data_df['QTY_PROJ_EOY'].iloc[i] + 
                                                                   lp_data_df['M5'].iloc[i]*(1-lp_data_df['delta6'].iloc[i]),
                                                                   sense = pulp.LpConstraintGE, 
                                                                   rhs = 0)]).to_list() # Function 3   
                
                # Assure only one delta variable is allowed to be set to 1. Note these are binary variables.               
                leakage_const_list += pd.Series([pulp.LpConstraint(lp_data_df['delta4'].iloc[i] + lp_data_df['delta5'].iloc[i] + 
                                                                   lp_data_df['delta6'].iloc[i] == 1)]).to_list()
                
                # Leakage cannot be less than 0                
                leakage_const_list += pd.Series([pulp.LpConstraint(lp_data_df['leakage_complex'].iloc[i], sense = pulp.LpConstraintGE, rhs = 0)]).to_list()
                
            if lp_data_df['COINS_ONLY_QTY_WT'].iloc[i] > 0:
                # Calculate leakage due to coinsurance: Leakage_var - IC * coinsurance - mac1026 >= 0
                leakage_const_list += pd.Series([pulp.LpConstraint(lp_data_df['leakage_coins'].iloc[i] - 
                                                                   (lp_data_df['Price_Decision_Var'].iloc[i] * lp_data_df['QTY_PROJ_EOY'].iloc[i] *
                                                                    lp_data_df['b5'].iloc[i] - 
                                                                    lp_data_df['MAC1026_UNIT_PRICE'].iloc[i] * lp_data_df['QTY_PROJ_EOY'].iloc[i]),
                                                                   sense = pulp.LpConstraintGE,
                                                                   rhs = 0)]).to_list()     
                
                # Leakage cannot be less than 0  
                leakage_const_list += pd.Series([pulp.LpConstraint(lp_data_df['leakage_coins'].iloc[i], sense = pulp.LpConstraintGE, rhs = 0)]).to_list()
             
            # Adjust the weight of leakage coming from each plan design type by utilization, then weight the final value by the leakage penalty parameter.
            leakage_cost_list += pd.Series([pulp.LpAffineExpression(penalty * (lp_data_df['COPAY_ONLY_QTY_WT'].iloc[i] * lp_data_df['leakage_copay'].iloc[i] + 
                                                                               lp_data_df['COINS_ONLY_QTY_WT'].iloc[i] * lp_data_df['leakage_coins'].iloc[i] +
                                                                               lp_data_df['COPAY_COINS_QTY_WT'].iloc[i] * lp_data_df['leakage_complex'].iloc[i]))
                                           ]).to_list()
        
        logger.info("Ending prep for Leakage Optimization")
        end = time.time()
        logger.info("Run time: {} mins".format((end - start)/60.))
        logger.info('--------------------') 
            
        # Pickle outputs
        with open(leakage_cost_list_out, 'wb') as f:
            pickle.dump(leakage_cost_list, f)
        with open(leakage_const_list_out, 'wb') as f:
            pickle.dump(leakage_const_list, f)
            
        return leakage_cost_list, leakage_const_list

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'Prepare Leakage Optimization', repr(e), error_loc)
        raise e  
        
###### Same Drug Therapeutic Class Constraints ########
def same_therapeutic_constraints(
    params_file_in: str,
    unc_flag: bool,
    lp_data_df_in: InputPath('pickle'),
    sm_thera_class_cons_list_out: OutputPath('pickle'),
    anomaly_sm_thera_gpi_out: OutputPath('pickle'),
    loglevel: str = 'INFO'
    # kube_run: bool = True,
):
    import sys
    import numpy as np
    import os
    sys.path.append('/')
    import time
    import logging
    import pandas as pd
    import warnings
    from pandas.core.common import SettingWithCopyWarning
    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    import pickle
    import pulp
    import util_funcs as uf
    import BQ

    uf.write_params(params_file_in)
    import CPMO_parameters as p
    from CPMO_lp_functions import generatePricingDecisionVariables
    from CPMO_shared_functions import update_run_status
    try:
        out_path = os.path.join(p.FILE_LOG_PATH, 'ClientPharmacyMacOptimization.log')
        logger = uf.log_setup(log_file_path=out_path, loglevel=loglevel)

        # file inputs
        with open(lp_data_df_in, 'rb') as f:
            lp_data_df = pickle.load(f)

        logger.info('--------------------')
        logger.info("Same Drug Therapeutic Class Constraints")
        start = time.time()

        lp_data_df = generatePricingDecisionVariables(lp_data_df)
        # Drug Therapeutic subclass
        lp_data_df['GPI_8'] = lp_data_df.GPI.str[0:8]
        lp_data_df['lb'],lp_data_df['ub'] = zip(*lp_data_df.Price_Bounds)

        price_constraints_df = pd.DataFrame()
        if not p.TRUECOST_CLIENT:
            price_constraints_col = ['CLIENT', 'BREAKOUT', 'MEASUREMENT', 'GPI_NDC', 'GPI' ,'NDC', 'BG_FLAG', 'GPI_8',
                                        'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'REGION',  'Price_Bounds', 'lb_name', 'ub_name', 'lb', 'ub', 'UC_UNIT',
                                        'Price_Decision_Var', 'MAC_PRICE_UNIT_ADJ', 'Dec_Var_Name']
            price_constraints_df = lp_data_df[price_constraints_col].loc[(lp_data_df.PRICE_MUTABLE==1) 
                                                                        & (lp_data_df.GPI_ONLY==1) 
                                                                        & (lp_data_df.BG_FLAG =='B')
                                                                        ]
       
        sm_thera_class_cons_list = []
        anomaly_sm_thera_gpi = pd.DataFrame(columns=('CLIENT', 'REGION', 'GPI_NDC', 'BG_FLAG','EXCLUDE_REASON'))
        
        if len(price_constraints_df) != 0 :
            ############################################# vectorization begins #############################################
            price_constraints_df_sorted = price_constraints_df.sort_values(by='GPI_NDC').reset_index()
            price_constraints_df_sorted['INDEX'] = price_constraints_df_sorted.index

            #Self join to create the constraint pair
            price_constraints_self_join_df = price_constraints_df_sorted.merge(price_constraints_df_sorted, how = 'inner',
                                                                       on = ['CLIENT','BREAKOUT','MEASUREMENT', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'REGION', 'BG_FLAG', 'GPI_8'],
                                                                       suffixes = ('_1', '_2'))

            #Remove duplicate constraint pair
            price_constraints_self_join_df_filter = price_constraints_self_join_df.loc[
                price_constraints_self_join_df['INDEX_1'] < price_constraints_self_join_df['INDEX_2']]

            #Identify conflict: basic idea is to see if the LB2_UP2 * curr_ratio overlap with LB1_UP1
            price_constraints_self_join_df_filter['curr_ratio'] = price_constraints_self_join_df_filter['MAC_PRICE_UNIT_ADJ_1'] / price_constraints_self_join_df_filter['MAC_PRICE_UNIT_ADJ_2']


            if p.SM_THERA_COLLAR:
                # assert 0 <= p.SM_THERA_COLLAR_LOW <= 1, f"SM_THERA_COLLAR_LOW = {p.SM_THERA_COLLAR_LOW} is not within the range [{0}, {1}], please check and reset parameter in the range"
                # assert 0 <= p.SM_THERA_COLLAR_HIGH <= 1, f"SM_THERA_COLLAR_High = {p.SM_THERA_COLLAR_HIGH} is not within the range [{0}, {1}], please check and reset parameter in the range"
                price_constraints_self_join_df_filter['curr_ratio_low'] = np.where(
                    price_constraints_self_join_df_filter['curr_ratio'] < 1,
                    price_constraints_self_join_df_filter['curr_ratio'] * (1-p.SM_THERA_COLLAR_LOW),
                    np.maximum(price_constraints_self_join_df_filter['curr_ratio'] * (1-p.SM_THERA_COLLAR_LOW),1)
                )
                price_constraints_self_join_df_filter['curr_ratio_high'] = np.where(
                    price_constraints_self_join_df_filter['curr_ratio'] < 1,
                    np.minimum(price_constraints_self_join_df_filter['curr_ratio'] * (1+p.SM_THERA_COLLAR_HIGH),1),
                    price_constraints_self_join_df_filter['curr_ratio'] * (1+p.SM_THERA_COLLAR_HIGH)
                )
            else:
                price_constraints_self_join_df_filter['curr_ratio_low'] = price_constraints_self_join_df_filter['curr_ratio']
                price_constraints_self_join_df_filter['curr_ratio_high'] = price_constraints_self_join_df_filter['curr_ratio']


            ######Segment 1 bad rows: PRICE1_UB < curr_ratio_low * PRICE2_LB######
            bad_row1_mask = price_constraints_self_join_df_filter['ub_1'] < price_constraints_self_join_df_filter['curr_ratio_low'] * price_constraints_self_join_df_filter['lb_2']
            bad_row1 = price_constraints_self_join_df_filter.loc[bad_row1_mask]
            bad_row1['EXCLUDE_REASON'] = 'SM_THERA: PRICE1_UB < curr_ratio_low * PRICE2_LB'
            bad_row1['GPI_NDC'] = bad_row1['GPI_NDC_1']
            if not bad_row1.empty:
                logger.info(f'Warning: There are {len(bad_row1)} pairs that has PRICE1_UB < curr_ratio_low * PRICE2_LB')
            anomaly_sm_thera_gpi = pd.concat([anomaly_sm_thera_gpi, bad_row1[['CLIENT','REGION','GPI_NDC','BG_FLAG','EXCLUDE_REASON']]])


            ######Segment 2 bad rows: PRICE1_LB > curr_ratio_high * PRICE2_UB######
            bad_row2_mask = price_constraints_self_join_df_filter['lb_1'] > price_constraints_self_join_df_filter['curr_ratio_high'] * price_constraints_self_join_df_filter['ub_2']
            bad_row2 = price_constraints_self_join_df_filter.loc[bad_row2_mask]
            bad_row2['EXCLUDE_REASON'] = 'SM_THERA: PRICE1_LB > curr_ratio_high * PRICE2_UB'
            bad_row2['GPI_NDC'] = bad_row2['GPI_NDC_1']
            if not bad_row1.empty:
                logger.info(f'Warning: There are {len(bad_row2)} pairs that has PRICE1_LB > curr_ratio_high * PRICE2_UB')
            anomaly_sm_thera_gpi = pd.concat([anomaly_sm_thera_gpi, bad_row2[['CLIENT','REGION','GPI_NDC','BG_FLAG','EXCLUDE_REASON']]]).drop_duplicates() 


            ######Segment 3 good rows######
            good_rows = price_constraints_self_join_df_filter.loc[~(bad_row1_mask) & ~(bad_row2_mask)]
            assert len(price_constraints_self_join_df_filter)==len(good_rows)+len(bad_row1)+len(bad_row2), f"Rows are not being bucketed correctly for constraint setup, please check"
            #create constraints
            good_rows['price_cons1'] = good_rows['Price_Decision_Var_1']-good_rows['curr_ratio_high']*good_rows['Price_Decision_Var_2']
            good_rows['price_cons2'] = good_rows['Price_Decision_Var_1']-good_rows['curr_ratio_low']*good_rows['Price_Decision_Var_2']

            if not good_rows.empty:
                price_cons1 = good_rows.price_cons1
                price_cons2 = good_rows.price_cons2
                for i in range(good_rows.shape[0]):
                    # When we do not add collar to ratio, ratio_high and ratio_low are the same
                    # hence we could use either cons1 or cons2 and set it to zero
                    # This is a more efficient way of setting only 1 constraint per good row, instead of 2 constraints per good row
                    if good_rows.curr_ratio_high.iloc[i] == good_rows.curr_ratio_low.iloc[i]:
                        sm_thera_class_cons_list.append(price_cons1.iloc[i]==0)
                    elif good_rows.curr_ratio.iloc[i]<=1 and (not good_rows.curr_ratio_high.iloc[i] == good_rows.curr_ratio_low.iloc[i]):
                        # If current ratio if smaller or equal to 1, only need to enforce the high_collar constraint to maintain order
                        # If wish to limit how far away the price could be interms of ratio, un-comment the lower_collar constraint to allow it to be enforced

                        sm_thera_class_cons_list.append(price_cons1.iloc[i]<=0)
                        # sm_thera_class_cons_list.append(price_cons2.iloc[i]>=0)
                    elif good_rows.curr_ratio.iloc[i] > 1 and (not good_rows.curr_ratio_high.iloc[i] == good_rows.curr_ratio_low.iloc[i]):
                        # If current ratio if larger than 1, only need to enforce the lower_collar constraint to maintain order
                        # If wish to limit how far away the price could be interms of ratio, uncomment the higher_collar constraint to allow it to be enforced

                        # sm_thera_class_cons_list.append(price_cons1.iloc[i]<=0)
                        sm_thera_class_cons_list.append(price_cons2.iloc[i]>=0)
                    
        
        logger.info("Ending Same Drug Therapeutic Class Constraints")
        end = time.time()
        logger.info("Run time: {} mins".format((end - start)/60.))
        logger.info('--------------------')

        # file outputs
        with open(sm_thera_class_cons_list_out, 'wb') as f:
            pickle.dump(sm_thera_class_cons_list, f)
        with open(anomaly_sm_thera_gpi_out, 'wb') as f:
            pickle.dump(anomaly_sm_thera_gpi, f)

        return sm_thera_class_cons_list, anomaly_sm_thera_gpi
    
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'Same Drug Therapeutic Class Constraints', repr(e), error_loc)
        raise e
###### End Same Drug Therapeutic Constraints ####################
        
### Generate Conflicting GPI ######################
def generate_conflict_gpi(
    params_file_in: str,
    month: int,
    unc_flag: bool,
    #handle_conflict_gpi: bool,
     # pricing_constraints outputs
    t_cost_in: InputPath('pickle'),
    cons_strength_cons_in: InputPath('pickle'),
    # Client Constraints outputs
    client_constraint_list_in: InputPath('pickle'),
    client_constraint_target_in: InputPath('pickle'),
    # Preferred Pricing less than Non Preferred Pricing outputs
    pref_lt_non_pref_cons_list_in: InputPath('pickle'),
    # Measure Specific Pricing Constraints outputs
    meas_specific_price_cons_list_in: InputPath('pickle'),
    # Brand Generic Pricing Constraints outputs
    brnd_gnrc_price_cons_list_in: InputPath('pickle'),
    # Adjudication Cap Pricing Constraints outputs
    adj_cap_price_cons_list_in: InputPath('pickle'),
    # All other pharmacies greater than CVS Pricing outputs
    pref_other_price_cons_list_in: InputPath('pickle'),
    # State parity and non-parity outputs
    parity_price_cons_list_in: InputPath('pickle'),
    # MAC Constraints outputs
    mac_cons_list_in: InputPath('pickle'),
    # Aggregate MAC price change constraints outputs
    agg_mac_cons_list_in: InputPath('pickle'),
    # Equal Package Size Contraints outputs
    eq_pkg_sz_cons_list_in: InputPath('pickle'),
    # Same Difference Package Size Constraints outputs
    sm_diff_pkg_sz_cons_list_in: InputPath('pickle'),  
    # Leakage Optimization Constraints outputs
    leakage_cost_list_in: InputPath('pickle'),
    leakage_const_list_in: InputPath('pickle'),
    # Same Therapeutic Constraints outputs
    sm_thera_class_cons_list_in: InputPath('pickle'),
    # Other vars
    lambda_df_in: InputPath('pickle'),
    breakout_df_in: InputPath('pickle'),
    # total_pharm_list_in: InputPath('pickle'),
    lp_data_df_in: InputPath('pickle'),
    lp_vol_mv_agg_df_in: InputPath('pickle'),   
    anomaly_gpi_in: InputPath('pickle'),
    anomaly_mes_gpi_in: InputPath('pickle'),
    anomaly_bg_gpi_in: InputPath('pickle'),
    anomaly_adj_cap_gpi_in: InputPath('pickle'),
    anomaly_pref_gpi_in: InputPath('pickle'),
    anomaly_const_pkg_sz_in: InputPath('pickle'),
    anomaly_state_parity_gpi_in: InputPath('pickle'),
    anomaly_const_mac_in: InputPath('pickle'),
    anomaly_sm_thera_gpi_in: InputPath('pickle'),
    # lp_data_output_df_out: OutputPath('pickle'),
    # # pilot_output_columns_out: OutputPath('pickle'),
    # total_output_columns_out: OutputPath('pickle'),
    # lambda_output_df_out: OutputPath('pickle'),
    conflict_gpi_out: OutputPath('pickle'),
    loglevel: str = 'INFO'
    # kube_run: bool = True
):
    import sys
    import os
    import shutil
    sys.path.append('/')
    import time
    import logging
    import numpy as np
    import pandas as pd
    import pickle
    import util_funcs as uf
    import BQ
    import json
    import datetime as dt

    uf.write_params(params_file_in)
    import CPMO_parameters as p
    from CPMO_shared_functions import standardize_df, update_run_status
    from util_funcs import file_exists_gcs

    update_run_status(i_error_type='Started generate_conflict_gpi') 
    try:
        out_path = os.path.join(p.FILE_LOG_PATH, 'ClientPharmacyMacOptimization.log')
        logger = uf.log_setup(log_file_path=out_path, loglevel=loglevel)

        with open(anomaly_gpi_in, 'rb') as f:
            anomaly_gpi = pickle.load(f)
        with open(anomaly_mes_gpi_in, 'rb') as f:
            anomaly_mes_gpi = pickle.load(f)
        with open(anomaly_bg_gpi_in, 'rb') as f:
            anomaly_bg_gpi = pickle.load(f)
        with open(anomaly_adj_cap_gpi_in, 'rb') as f:
            anomaly_adj_cap_gpi = pickle.load(f)
        with open(anomaly_pref_gpi_in, 'rb') as f:
            anomaly_pref_gpi = pickle.load(f)
        with open(anomaly_const_pkg_sz_in, 'rb') as f:
            anomaly_const_pkg_sz = pickle.load(f)
        with open(anomaly_state_parity_gpi_in, 'rb') as f:
            anomaly_state_parity_gpi = pickle.load(f)       
        with open(anomaly_const_mac_in, 'rb') as f:
            anomaly_const_mac = pickle.load(f)
        if p.LEAKAGE_OPT and p.LOCKED_CLIENT:
            with open(leakage_cost_list_in, 'rb') as f:
                leakage_cost_list = pickle.load(f)
            with open(leakage_const_list_in, 'rb') as f:
                leakage_const_list = pickle.load(f)
        with open(anomaly_sm_thera_gpi_in, 'rb') as f:
            anomaly_sm_thera_gpi = pickle.load(f)
        # with open(total_pharm_lianomaly_gst_in, 'rb') as f:  # not needed since PHARMACY_LIST now in params
        #     total_pharm_list = pickle.load(f)
        
        full_anomaly_gpi = pd.concat([anomaly_gpi, anomaly_mes_gpi, anomaly_bg_gpi, anomaly_adj_cap_gpi, anomaly_pref_gpi, anomaly_const_pkg_sz, anomaly_state_parity_gpi, anomaly_const_mac, anomaly_sm_thera_gpi], 
                                     ignore_index = True)
        
        full_anomaly_gpi.to_csv(os.path.join(p.FILE_OUTPUT_PATH, p.CONFLICT_GPI_LIST_FILE_THIS_RUN), index=False)
        
        if len(full_anomaly_gpi) != 0:
            full_anomaly_gpi[['GPI', 'NDC']] = full_anomaly_gpi['GPI_NDC'].str.split('_', expand=True, n=1)
            full_anomaly_gpi = full_anomaly_gpi.groupby(['CLIENT', 'REGION', 'GPI', 'BG_FLAG'])[['EXCLUDE_REASON']].agg('; '.join).reset_index()
            print(full_anomaly_gpi)

            if p.HANDLE_CONFLICT_GPI:
                orig_full_anomaly_gpi = standardize_df(pd.read_csv(os.path.join(p.FILE_INPUT_PATH,p.CONFLICT_GPI_LIST_FILE), dtype = p.VARIABLE_TYPE_DIC))
                full_anomaly_gpi_out = pd.concat([full_anomaly_gpi, orig_full_anomaly_gpi])
            else:
                full_anomaly_gpi_out = full_anomaly_gpi
            full_anomaly_gpi_out = full_anomaly_gpi_out.groupby(['CLIENT', 'REGION', 'GPI', 'BG_FLAG'])[['EXCLUDE_REASON']].agg('; '.join).reset_index()
            full_anomaly_gpi_out.to_csv(os.path.join(p.FILE_OUTPUT_PATH, p.CONFLICT_GPI_LIST_FILE), index=False)
            full_anomaly_gpi_out.to_csv(os.path.join(p.FILE_INPUT_PATH, p.CONFLICT_GPI_LIST_FILE), index=False)

            if p.WRITE_TO_BQ:
                uf.write_to_bq(
                    full_anomaly_gpi_out,
                    project_output = p.BQ_OUTPUT_PROJECT_ID,
                    dataset_output = p.BQ_OUTPUT_DATASET,
                    table_id = "anomaly_gpi_out",
                    client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                    timestamp_param = p.TIMESTAMP,
                    run_id = p.AT_RUN_ID,
                    schema = None) # TODO: write proper schema
                
            if 'gs://' == params_file_in[:5]:  # gcp storage path
                print("GPIs with violating constraints have been identified and reported. The aglorithem will copy the file conflict_exclusion_gpis_{0}_{1}_{2}.csv from the output folder to the Input path to exclude these GPIs from the next LP run and set HANDLE_CONFLICT_GPI = True. Run the model again.".format(p.CUSTOMER_ID[0], dt.date.today().strftime("%B"), p.RUN_TYPE_TABLEAU))     
            else:
                assert len(full_anomaly_gpi) == 0, "GPIs with violating constraints have been identified and reported. From the output folder copy the file conflict_exclusion_gpis_{0}_{1}_{2}.csv to the Input path to exclude these GPIs from the next LP run and set HANDLE_CONFLICT_GPI = True (or modify the UP/DOWN GPI_FAC). Run the model again.".format(p.CUSTOMER_ID[0], dt.date.today().strftime("%B"), p.RUN_TYPE_TABLEAU)

        if file_exists_gcs(file_location=os.path.join(p.FILE_INPUT_PATH, p.CONFLICT_GPI_LIST_FILE)):
            final_conf_gpi = pd.read_csv(os.path.join(p.FILE_INPUT_PATH, p.CONFLICT_GPI_LIST_FILE))
            num_conf_gpi = final_conf_gpi.shape[0]
        elif not p.HANDLE_CONFLICT_GPI:
            num_conf_gpi = 0
        update_run_status(num_conflict_gpi=num_conf_gpi)
        
        full_anomaly_gpi_out = full_anomaly_gpi
        with open(conflict_gpi_out, 'wb') as f:
            pickle.dump(full_anomaly_gpi_out, f)
        return full_anomaly_gpi_out

    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'Generate Conflict GPI', repr(e), error_loc)
        raise e

### Linear Programming Problem ######################
def run_solver(
    params_file_in: str,
    month: int,
    unc_flag: bool,
    # pricing_constraints outputs
    t_cost_in: InputPath('pickle'),
    cons_strength_cons_in: InputPath('pickle'),
    # Client Constraints outputs
    client_constraint_list_in: InputPath('pickle'),
    client_constraint_target_in: InputPath('pickle'),
    # Preferred Pricing less than Non Preferred Pricing outputs
    pref_lt_non_pref_cons_list_in: InputPath('pickle'),
    # Measure Specific Pricing Constraints outputs
    meas_specific_price_cons_list_in: InputPath('pickle'),
    # Brand Generic Pricing Constraints outputs
    brnd_gnrc_price_cons_list_in: InputPath('pickle'),
    # Adjudication Cap Pricing Constraints outputs
    adj_cap_price_cons_list_in: InputPath('pickle'),
    # All other pharmacies greater than CVS Pricing outputs
    pref_other_price_cons_list_in: InputPath('pickle'),
    # State parity and non-parity outputs
    parity_price_cons_list_in: InputPath('pickle'),
    # MAC Constraints outputs
    mac_cons_list_in: InputPath('pickle'),
    # Aggregate MAC price change constraints outputs
    agg_mac_cons_list_in: InputPath('pickle'),
    # Equal Package Size Contraints outputs
    eq_pkg_sz_cons_list_in: InputPath('pickle'),
    # Same Difference Package Size Constraints outputs
    sm_diff_pkg_sz_cons_list_in: InputPath('pickle'),  
    # Leakage Optimization Constraints outputs
    leakage_cost_list_in: InputPath('pickle'),
    leakage_const_list_in: InputPath('pickle'),
    # Same Therapeutic Constraints outputs
    sm_thera_class_cons_list_in: InputPath('pickle'),
    # Other vars
    lambda_df_in: InputPath('pickle'),
    breakout_df_in: InputPath('pickle'),
    # total_pharm_list_in: InputPath('pickle'),
    lp_data_df_in: InputPath('pickle'),
    lp_vol_mv_agg_df_in: InputPath('pickle'),
    anomaly_gpi_in: InputPath('pickle'),
    anomaly_mes_gpi_in: InputPath('pickle'),
    anomaly_bg_gpi_in: InputPath('pickle'),
    anomaly_adj_cap_gpi_in: InputPath('pickle'),
    anomaly_pref_gpi_in: InputPath('pickle'),
    anomaly_const_pkg_sz_in: InputPath('pickle'),
    anomaly_state_parity_gpi_in: InputPath('pickle'),
    anomaly_const_mac_in: InputPath('pickle'),
    lp_data_output_df_out: OutputPath('pickle'),
    anomaly_sm_thera_gpi_in: InputPath('pickle'),
    # pilot_output_columns_out: OutputPath('pickle'),
    total_output_columns_out: OutputPath('pickle'),
    lambda_output_df_out: OutputPath('pickle'),
    loglevel: str = 'INFO'
    # kube_run: bool = True
):
    import sys
    import os
    import shutil
    sys.path.append('/')
    import time
    import logging
    import pulp
    import numpy as np
    import pandas as pd
    import pickle
    import util_funcs as uf
    from datetime import datetime
    import BQ
    import json
    import datetime as dt
    import gurobipy

    uf.write_params(params_file_in)
    import CPMO_parameters as p
    from qa_checks import qa_dataframe
    from CPMO_shared_functions import (
        check_price_increase_decrease_initial, check_agg_price_cons, standardize_df, determine_effective_price
    )
    from CPMO_lp_functions import (
        generateCost_new, generateLambdaDecisionVariables_ebit, generatePricingDecisionVariables
    )
    from CPMO_plan_liability import createPlanCostObj
    from CPMO_shared_functions import update_run_status, round_to
    update_run_status(i_error_type='Started run_solver') 

    try:
        # TODO: fix logging HERE
        out_path = os.path.join(p.FILE_LOG_PATH, 'ClientPharmacyMacOptimization.log')
        logger = uf.log_setup(log_file_path=out_path, loglevel=loglevel)

        with open(t_cost_in, 'rb') as f:
            t_cost = pickle.load(f)
        with open(cons_strength_cons_in, 'rb') as f:
            cons_strength_cons = pickle.load(f)
        with open(client_constraint_list_in, 'rb') as f:
            client_constraint_list = pickle.load(f)
        with open(client_constraint_target_in, 'rb') as f:
            client_constraint_target = pickle.load(f)
        with open(pref_lt_non_pref_cons_list_in, 'rb') as f:
            pref_lt_non_pref_cons_list = pickle.load(f)
        with open(meas_specific_price_cons_list_in, 'rb') as f:
            meas_specific_price_cons_list = pickle.load(f)
        with open(brnd_gnrc_price_cons_list_in, 'rb') as f:
            brnd_gnrc_price_cons_list = pickle.load(f)
        with open(adj_cap_price_cons_list_in, 'rb') as f:
            adj_cap_price_cons_list = pickle.load(f)
        with open(pref_other_price_cons_list_in, 'rb') as f:
            pref_other_price_cons_list = pickle.load(f)
        with open(parity_price_cons_list_in, 'rb') as f:
            parity_price_cons_list = pickle.load(f)
        with open(mac_cons_list_in, 'rb') as f:
            mac_cons_list = pickle.load(f)
        with open(agg_mac_cons_list_in, 'rb') as f:
            agg_mac_cons_list = pickle.load(f)
        with open(eq_pkg_sz_cons_list_in, 'rb') as f:
            eq_pkg_sz_cons_list = pickle.load(f)
        with open(sm_diff_pkg_sz_cons_list_in, 'rb') as f:
            sm_diff_pkg_sz_cons_list = pickle.load(f)
        with open(lambda_df_in, 'rb') as f:
            lambda_df = pickle.load(f)
        with open(lp_data_df_in, 'rb') as f:
            lp_data_df = pickle.load(f)
        with open(breakout_df_in, 'rb') as f:
            breakout_df = pickle.load(f)
        with open(lp_vol_mv_agg_df_in, 'rb') as f:
            lp_vol_mv_agg_df = pickle.load(f)
        with open(anomaly_gpi_in, 'rb') as f:
            anomaly_gpi = pickle.load(f)
        with open(anomaly_mes_gpi_in, 'rb') as f:
            anomaly_mes_gpi = pickle.load(f)
        with open(anomaly_bg_gpi_in, 'rb') as f:
            anomaly_bg_gpi = pickle.load(f)
        with open(anomaly_adj_cap_gpi_in, 'rb') as f:
            anomaly_adj_cap_gpi = pickle.load(f)
        with open(anomaly_pref_gpi_in, 'rb') as f:
            anomaly_pref_gpi = pickle.load(f)
        with open(anomaly_const_pkg_sz_in, 'rb') as f:
            anomaly_const_pkg_sz = pickle.load(f)
        with open(anomaly_state_parity_gpi_in, 'rb') as f:
            anomaly_state_parity_gpi = pickle.load(f)       
        with open(anomaly_const_mac_in, 'rb') as f:
            anomaly_const_mac = pickle.load(f)
        if p.LEAKAGE_OPT and p.LOCKED_CLIENT:
            with open(leakage_cost_list_in, 'rb') as f:
                leakage_cost_list = pickle.load(f)
            with open(leakage_const_list_in, 'rb') as f:
                leakage_const_list = pickle.load(f)
        # with open(total_pharm_lianomaly_gst_in, 'rb') as f:  # not needed since PHARMACY_LIST now in params
        #     total_pharm_list = pickle.load(f)
        with open(sm_thera_class_cons_list_in, 'rb') as f:
            sm_thera_class_cons_list = pickle.load(f)
        with open(anomaly_sm_thera_gpi_in, 'rb') as f:
            anomaly_sm_thera_gpi = pickle.load(f)

        lambda_decision_var = generateLambdaDecisionVariables_ebit(breakout_df, list(set(p.PHARMACY_LIST['GNRC']+p.PHARMACY_LIST['BRND'])))

        # reinitialize variables (to take care of variable hash ids being different on unpickling)
        var_pool = {}  # keep track of current LpVariables
        for var in lambda_decision_var.Lambda_Over:
            assert var.name not in var_pool.keys(), 'Variable already in var_pool'
            var_pool[var.name] = var
        for var in lambda_decision_var.Lambda_Under:
            assert var.name not in var_pool.keys(), 'Variable already in var_pool'
            var_pool[var.name] = var

        def re_init(con_list, var_pool={}):
            '''re-initialize variables not in var_pool for each constraint in list'''
            new_con_list = []
            for cnst in con_list:
                re_items = []
                for var, val in list(cnst.items()):
                    if var.name not in var_pool.keys():  # re-initialize
                        re_var = pulp.LpVariable.from_dict(**var.to_dict())
                        var_pool[var.name] = re_var
                        re_items.append((re_var, val))
                    else:  # set to correct var
                        re_items.append((var_pool[var.name], val))  # use same var_pool var if it's there
                if type(cnst) == pulp.LpConstraint:
                    new_cnst = pulp.LpConstraint(e=re_items, sense=cnst.sense, rhs=-cnst.constant, name=cnst.name)
                elif type(cnst) == pulp.LpAffineExpression:
                    new_cnst = pulp.LpAffineExpression(e=re_items, constant=cnst.constant, name=cnst.name)
                else:
                    raise Exception(f'Item must be of type LpConstraint or LpAffineExpression, instead got: {cnst}')
                new_con_list.append(new_cnst)
            return new_con_list, var_pool

        t_cost, var_pool = re_init(t_cost, var_pool)
        cons_strength_cons, var_pool = re_init(cons_strength_cons, var_pool)
        client_cons_list = [(client_constraint_list[i] == client_constraint_target[i]) for i in range(len(client_constraint_list))]
        client_cons_list, var_pool = re_init(client_cons_list, var_pool)
        pref_lt_non_pref_cons_list, var_pool = re_init(pref_lt_non_pref_cons_list, var_pool)
        meas_specific_price_cons_list, var_pool = re_init(meas_specific_price_cons_list, var_pool)
        brnd_gnrc_price_cons_list, var_pool = re_init(brnd_gnrc_price_cons_list, var_pool)
        adj_cap_price_cons_list, var_pool = re_init(adj_cap_price_cons_list, var_pool)
        mac_cons_list, var_pool = re_init(mac_cons_list, var_pool)
        pref_other_price_cons_list, var_pool = re_init(pref_other_price_cons_list, var_pool)
        agg_mac_cons_list, var_pool = re_init(agg_mac_cons_list, var_pool)
        eq_pkg_sz_cons_list, var_pool = re_init(eq_pkg_sz_cons_list, var_pool)
        sm_diff_pkg_sz_cons_list, var_pool = re_init(sm_diff_pkg_sz_cons_list, var_pool)
        parity_price_cons_list, var_pool = re_init(parity_price_cons_list, var_pool)
        sm_thera_class_cons_list, var_pool = re_init(sm_thera_class_cons_list, var_pool)
        if p.LEAKAGE_OPT and p.LOCKED_CLIENT:
            leakage_cost_list, var_pool = re_init(leakage_cost_list, var_pool)
            leakage_const_list, var_pool = re_init(leakage_const_list, var_pool)
        logging.debug('Creating Objective Function')
        logging.debug('--------------------')

        ##### Objective Function ##################

        prob = pulp.LpProblem(str(p.CUSTOMER_ID) + 'MAC_Optimization', pulp.LpMinimize)
        obj_cost = generateCost_new(lambda_decision_var[lambda_decision_var['Lambda_Level'] == 'CLIENT'], p.COST_GAMMA, p.OVER_REIMB_GAMMA, 0)

        total_cost = ""
        total_cost += obj_cost
        
        if p.INCLUDE_PLAN_LIABILITY:
            plan_liab_df, plan_lambdas, icl_cons, plan_cost_cons  = createPlanCostObj(lp_data_df.loc[(lp_data_df.QTY_PROJ_EOY > 0)&(lp_data_df.PRICE_MUTABLE == 1)].copy())

            if p.READ_IN_PLAN_LIAB_WIEGHTS:
                plan_weights = standardize_df(pd.read_csv(p.FILE_INPUT_PATH + p.PLAN_LIAB_WEIGHT_FILE), dtype = p.VARIABLE_TYPE_DIC)
                qa_dataframe(plan_weights, dataset = 'PLAN_LIAB_WEIGHT_FILE_AT_{}'.format(os.path.basename(__file__)))
                plan_lambdas_weights = pd.merge(plan_lambdas, plan_weights, how='left', on=['CLIENT', 'BREAKOUT', 'REGION'])
                assert len(plan_lambdas) == len(plan_lambdas_weights), "len(plan_lambdas) == len(plan_lambdas_weights)"

                plan_lambdas_weights['WEIGHT'] = plan_lambdas_weights['WEIGHT'].fillna(1.0)
            else:
                plan_lambdas_weights = plan_lambdas.copy()
                plan_lambdas_weights['WEIGHT'] = 1.0

            for i in range(0,len(plan_lambdas_weights)):
                plan_lambda = plan_lambdas_weights.iloc[i]['Plan_Liab_Lambda']
                plan_weighting = plan_lambdas_weights.iloc[i]['WEIGHT'] * p.PLAN_LIAB_WEIGHT
                #print(plan_lambda)
                #print(plan_weighting)
                total_cost += plan_weighting * plan_lambda

        logger.info('------------------------')
        logger.info("Main Objective Function:")
        logger.info(str(total_cost))
        logger.info('------------------------')      

        if p.LEAKAGE_OPT and p.LOCKED_CLIENT:
            total_cost += leakage_cost_list         
        
        #Class to encode the modle to a json object.
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return super(NpEncoder, self).default(obj)

        def export_model_to_json(prob,file_path,encoder=NpEncoder):
            prob_dict = prob.to_dict()

            with open(file_path, 'w') as f:
                json.dump(prob_dict, f, cls=encoder)

        if 'gs://' in p.FILE_OUTPUT_PATH:  # (google storage case)
            TEMP_WORK_DIR = 'temp_work_dir_' + str(datetime.now())
            os.makedirs(TEMP_WORK_DIR, exist_ok=True)
        #Function writes the LP to different file types based on the extension in the name passed to it.
        def _writeLP(fname):
            if 'gs://' in p.FILE_OUTPUT_PATH:  # (google storage case)
                if '.MPS' in fname:
                    prob.writeMPS(os.path.join(TEMP_WORK_DIR, fname))
                elif '.json' in fname:
                    export_model_to_json(prob,os.path.join(TEMP_WORK_DIR, fname),NpEncoder)
                else:
                    prob.writeLP(os.path.join(TEMP_WORK_DIR, fname), max_length = 250)
                bucket = p.FILE_OUTPUT_PATH[5:].split('/', 1)[0]
                uf.upload_blob(bucket, os.path.join(TEMP_WORK_DIR, fname), os.path.join(p.FILE_LP_PATH, fname))
            else:  # (local directory case)
                if '.MPS' in fname:
                    prob.writeMPS(os.path.join(p.FILE_LP_PATH, fname))
                elif '.json' in fname:
                    export_model_to_json(prob,os.path.join(p.FILE_LP_PATH, fname),NpEncoder)
                else:
                    prob.writeLP(os.path.join(p.FILE_LP_PATH, fname), max_length = 250)

        for tc in t_cost:
            total_cost += tc


        prob += total_cost
        if p.WRITE_LP_FILES:
            _writeLP(str(p.CUSTOMER_ID) + "_objective" + str(p.TIMESTAMP) + ".lp" )
        # Plan liability constraints
        if p.INCLUDE_PLAN_LIABILITY:
            for cons in icl_cons:
                prob += (cons <=0)
            for cons in plan_cost_cons:
                prob += (cons <=0)
        # Consistent Strength Pricing constraints
        for cons in cons_strength_cons:
            prob += (cons <= 0)
        if p.WRITE_LP_FILES:
            _writeLP(str(p.CUSTOMER_ID) + "_Const_Strength" + str(p.TIMESTAMP) + ".lp" )
        # Client level Contraints
        for cnst in client_cons_list:
            prob += cnst
        if p.WRITE_LP_FILES:
            _writeLP(str(p.CUSTOMER_ID) + "_EBIT_Cons" + str(p.TIMESTAMP) + ".lp" )
        # Preferred Pricing less than Non Preferred Pricing
        for constraint in pref_lt_non_pref_cons_list:
            prob += constraint
        if p.WRITE_LP_FILES:
            _writeLP(str(p.CUSTOMER_ID) + "_PrefPricing" + str(p.TIMESTAMP) + ".lp" )
        # Measure Specific Pricing (ie M < R90 < R30)
        for cnst in meas_specific_price_cons_list:
            prob += cnst
        if p.WRITE_LP_FILES:
            _writeLP(str(p.CUSTOMER_ID) + "_ChannelOrdering" + str(p.TIMESTAMP) + ".lp" )
        # Brand Generic Pricing (ie Generic < Brand)
        for cnst in brnd_gnrc_price_cons_list:
            prob += cnst
        if p.WRITE_LP_FILES:
            _writeLP(str(p.CUSTOMER_ID) + "_BrandGeneric" + str(p.TIMESTAMP) + ".lp" )
        # Adjudication > Guarantee Pricing * Cap
        for cnst in adj_cap_price_cons_list:
            prob += cnst
        if p.WRITE_LP_FILES:
            _writeLP(str(p.CUSTOMER_ID) + "_AdjCap" + str(p.TIMESTAMP) + ".lp" )
        # All other pharmacies >= CVS pricing constraint
        for cnst in pref_other_price_cons_list:
            prob += cnst
        if p.WRITE_LP_FILES:
            _writeLP(str(p.CUSTOMER_ID) + "_CVSParity" + str(p.TIMESTAMP) + ".lp" )
        # Parity price difference hard constraints
        for cnst in parity_price_cons_list:
            prob += cnst
        if p.WRITE_LP_FILES:
            _writeLP(str(p.CUSTOMER_ID) + "_ParityPrice" + str(p.TIMESTAMP) + ".lp" )
        # Consistent MAC constraints
        for cnst in mac_cons_list:
            prob += cnst
        if p.WRITE_LP_FILES:
            _writeLP(str(p.CUSTOMER_ID) + "_Const_MAC2" + str(p.TIMESTAMP) + ".lp" )
        # Aggregate MAC price change constraints
        if p.AGG_UP_FAC >= 0:
            for cnst in agg_mac_cons_list:
                prob += cnst
            if p.WRITE_LP_FILES:
                _writeLP(str(p.CUSTOMER_ID) + "_Agg" + str(p.TIMESTAMP) + ".lp" )
        # Equal Package Size Contraints
        for cnst in eq_pkg_sz_cons_list:
            prob += cnst
        if p.WRITE_LP_FILES:
            _writeLP(str(p.CUSTOMER_ID) + "_Pkg_Sz" + str(p.TIMESTAMP) + ".lp" )
        # Same Difference Package Size Constraints
        for cnst in sm_diff_pkg_sz_cons_list:
            prob += cnst
        if p.WRITE_LP_FILES:
            _writeLP(str(p.CUSTOMER_ID) + "_SameDiff" + str(p.TIMESTAMP) + ".lp" )
        # Same Therapeutic Constraints
        for cnst in sm_thera_class_cons_list:
            prob += cnst
        if p.WRITE_LP_FILES:
            _writeLP(str(p.CUSTOMER_ID) + "_SameTherapeutic" + str(p.TIMESTAMP) + ".lp" )
        # Leakage optimization constraints
        if p.LEAKAGE_OPT and p.LOCKED_CLIENT:
            for cnst in leakage_const_list:
                prob += cnst
            if p.WRITE_LP_FILES:
                _writeLP(str(p.CUSTOMER_ID) + "_leakage_opt" + str(p.TIMESTAMP) + ".lp")
        if p.WRITE_LP_FILES:
            fname_json = str(p.CUSTOMER_ID[0]) + "_total_prob.json"
            _writeLP(fname_json)
            fname_lp = str(p.CUSTOMER_ID[0]) + "_total_prob.lp"
            _writeLP(fname_lp)

        ### Run the solver #####
        start = time.time()
        logger.info('--------------------')
        logger.info('Starting Solver')

        pulp.LpSolverDefault.msg = 1
        solver = pulp.GUROBI()
        gurobi_run = False
        try:
            logger.info('Attempting Gurobi')
            optimization_result = prob.solve(solver)
            gurobi_run = True
        except Exception as e:
            print(e)
            if p.LEAKAGE_OPT == False:
                logger.info('Gurobi attempt failed, falling back to PuLP')
                try:
                    optimization_result = prob.solve()
                except Exception as e:
                    print(e)
                    assert False, 'Error running PuLP too'
            else:
                assert False, 'Error within Gurobi, PuLP not attempted as LEAKAGE_OPT == True for the client'
        end = time.time()
        logger.info("Solver Done")
        if optimization_result == pulp.LpStatusOptimal:
            logger.info("Run time: {} mins".format((end - start)/60.))
            logger.info("Status:")
            logger.info(pulp.LpStatus[prob.status])
            #assert optimization_result == pulp.LpStatusOptimal
            logger.info("Optimal Solution to the problem: %f", pulp.value(prob.objective))
            logger.info('--------------------')

        elif not gurobi_run:       
            logger.info('Handling Infeasibility')
            #Handle infeasibility
            handled_infeasibility = True

            def conflicting_gpi(prob):
                '''A function identifying conflicting constraints for the interim infeasibile solution, determining if the conflicting constraint is a hard constraint, iterating over the variables in the hard constraints and extract GPI from the variable'''
                cols = ['gpi', 'ndc', 'bg_flag', 'client', 'client2', 'breakout', 'measurement', 'region', 'chain', 'subchain']
                suspect_gpi_df = pd.DataFrame(columns = cols)
                for c in prob.constraints.values():
                    if not c.valid(0):
                        ##get variables objects from suspect constraints
                        constr_vars = c.toDict()['coefficients']
                        ##get variable names
                        constr_vars_names = [var['name'] for var in constr_vars]

                        ##check if the constraint is a soft constraint;
                        ##currently, soft constraints have lambda_ or sv_ variables
                        ##any constraint that only includes P_ variables is a hard constraint
                        suspect_constr = [name.startswith("P_") for name in constr_vars_names]
                        ##if all the variables in the constraint were of types P_;
                        if all(suspect_constr):
                            ##extract GPI, NDC, CLIENT, BREAKOUT, MEASUREMENT, REGION, CHAIN from constraint variables
                            price_vars = [price.replace('NONPREF_OTH', 'NONPREF-OTH').replace('PREF_OTH','PREF-OTH').replace('_R90OK', '-R90OK').replace('_EXTRL', '-EXTRL').split('_')[1:] for price in constr_vars_names]
                            suspect_gpi_df = suspect_gpi_df.append(pd.DataFrame(price_vars, columns = cols))
                return suspect_gpi_df

            def violating_gpi(prob):
                '''A function identifying decision variables that are set out of bounds in the interim feasibile solution, determining if that variable is a price variable and not a "soft cap variable"'''
                cols = ['gpi', 'ndc', 'bg_flag', 'client', 'client2', 'breakout', 'measurement', 'region', 'chain', 'subchain']
                suspect_gpi_df = pd.DataFrame(columns = cols)
                for v in prob.variables():
                    if not v.valid(0):
                        ##get violating variable names
                        ##check if they are price variables and not SV_ or lambda_ types
                        if v.name.startswith("P_"):
                            var_row = v.name.replace('NONPREF_OTH', 'NONPREF-OTH').replace('PREF_OTH','PREF-OTH').replace('_R90OK', '-R90OK').replace('_EXTRL', '-EXTRL').split('_')[1:]
                            suspect_gpi_df = suspect_gpi_df.append(pd.DataFrame([var_row], columns = cols))
                return suspect_gpi_df

            def rm_gpi_from_constr(prob_json, suspect_gpi_df):
                '''Function to remove hard constraints from the problem json model given a list of suspected GPIs. Any hard constraints that has a variable corresponding to one of the suspected GPIs is removed from the problem. At the same time, a dictionary of those constraints and the suspected GPIs is saved for later use'''
                costr_lst = prob_json['constraints'].copy()
                gpi_bg_flag_lst = suspect_gpi_df[['gpi', 'bg_flag']].drop_duplicates().values.tolist()
                constr_gpi_dict = {tuple(gpi_bg): [] for gpi_bg in gpi_bg_lst}
                for constr in prob_json['constraints']:
                    constr_var_lst = [var['name'] for var in constr['coefficients']]
                    hard_constr = [name.startswith("P_") for name in constr_var_lst]
                    if all(hard_constr):
                        for gpi, bg_flag in gpi_bg_lst:
                            if any(gpi in var and bg_flag in var for var in constr_var_lst):
                                costr_lst.remove(constr)
                                constr_gpi_dict[(gpi, bg_flag)].append(constr)
                                break
                return costr_lst, constr_gpi_dict

            cols = ['gpi', 'ndc', 'bg_flag', 'client', 'client2', 'breakout', 'measurement', 'region', 'chain', 'subchain']
            suspect_gpi_df = pd.DataFrame(columns = cols)

            #run the lp for first time
            orig_prob_json = prob.to_dict()#read_model_json(orig_path_to_model)
            prob_json = orig_prob_json.copy()
            #var, prob = pulp.LpProblem.from_dict(orig_prob_json)

            prob.solve()
            print(f"status: {prob.status}, {pulp.LpStatus[prob.status]}")
            print(f"objective: {prob.objective.value()}")
            status = pulp.LpStatus[prob.status]

            conflicting_gpi(prob).to_csv(os.path.join(p.FILE_OUTPUT_PATH,'possible_conflicting_gpi.csv'), index=False)
            violating_gpi(prob).to_csv(os.path.join(p.FILE_OUTPUT_PATH,'possible_violating_gpi.csv'), index=False)
            
            while status == "Infeasible":
                #create initial df of suspect GPIs
                new_found_gpi_df = pd.concat([conflicting_gpi(prob), violating_gpi(prob)])
                new_found_gpi_df.drop_duplicates(inplace = True, ignore_index = True)
                if new_found_gpi_df.empty:
                    #infeasibility lies in the interaction of different a mix of soft and hard constraints
                    #this needs further research because something like this should not happen!
                    #or there are other hard constraints which are not identified in this notebook
                    assert optimization_result == pulp.LpStatusOptimal, "Error!: No more GPIs are suspected but the infeasibility issue persists. Infeasibility is caused by another issue which will need to be looked at separately. GPI Exclusion method will not work in this case."

                suspect_gpi_df = suspect_gpi_df.append(new_found_gpi_df)

                logger.info(f"unique number of new suspect GPIs found: {len(new_found_gpi_df[['gpi', 'bg_flag']].drop_duplicates())}")
                suspect_gpi_master_lst = list(new_found_gpi_df[['gpi', 'bg_flag']].drop_duplicates())

                logger.info(f"Identifying hard constraints to be removed from the original problem...")
                constr_lst, constr_gpi_dict = rm_gpi_from_constr(prob_json,
                                                     suspect_gpi_df.loc[suspect_gpi_df[['gpi', 'bg_flag']].apply(tuple, axis=1).isin(suspect_gpi_master_lst), ])

                logger.info(f"Solving the reduced problem...")
                prob_json['constraints'] = constr_lst
                var, prob = pulp.LpProblem.from_dict(prob_json)
                prob.solve()
                logger.info(f"status: {prob.status}, {pulp.LpStatus[prob.status]}")
                logger.info(f"objective: {prob.objective.value()}")
                status = pulp.LpStatus[prob.status]

                if pulp.LpStatus[prob.status] == "Optimal":
                    safe_gpi_bg_flag = []
                    unsafe_gpi_bg_flag = []
                    logger.info("Add back hard constraint corresponding to suspect GPIs one by one...")
                for gpi, bg_flag in suspect_gpi_master_lst:
                    constr_lst.extend(constr_gpi_dict[(gpi, bg_flag)])
                    prob_json['constraints'] = constr_lst
                    new_var, new_prob = pulp.LpProblem.from_dict(prob_json)
                    new_prob.solve()
                    if pulp.LpStatus[new_prob.status] == "Optimal":
                        safe_gpi.append((gpi, bg_flag))
                        logger.info(f"{gpi} with BG_FLAG {bg_flag} from the suspected list is safe")
                    else:
                        unsafe_gpi.append((gpi, bg_flag))
                        logger.info(f"{gpi} with BG_FLAG {bg_flag} from the suspected list is unsafe")
                        for constr in constr_gpi_dict[(gpi, bg_flag)]:
                            constr_lst.remove(constr)
                logger.info(f"List of unsafe GPIs: {unsafe_gpi}")

            region = 'ALL'
            gpi_exclusion = pd.DataFrame(columns = ['CLIENT', 'REGION', 'GPI', 'BG_FLAG'])
            gpi_exclusion[['GPI', 'BG_FLAG']] = pd.DataFrame(unsafe_gpi_bg_flag, columns=['GPI', 'BG_FLAG'])
            gpi_exclusion['CLIENT'] = p.CUSTOMER_ID[0]
            gpi_exclusion['REGION'] = region
            gpi_exclusion.to_csv(os.path.join(p.FILE_OUTPUT_PATH,p.INFEASIBLE_EXCLUSION_FILE), index=False)
            gpi_exclusion.to_csv(os.path.join(p.FILE_INPUT_PATH,p.INFEASIBLE_EXCLUSION_FILE), index=False)
            logger.info(f'File with {len(gpi_exclusion)} gpi(s) sent to the shared input folder.')
            
            if p.WRITE_TO_BQ:
                uf.write_to_bq(
                    gpi_exclusion,
                    project_output = p.BQ_OUTPUT_PROJECT_ID,
                    dataset_output = p.BQ_OUTPUT_DATASET,
                    table_id = "infeasible_gpi_out",
                    client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                    timestamp_param = p.TIMESTAMP,
                    run_id = p.AT_RUN_ID,
                    schema = None) # TODO: write proper shechema

            assert optimization_result == pulp.LpStatusOptimal, "This run is infeasible. Good news though, there is a solution. Go to the Output Folder of this run and copy infeasible_exlusion_gpis_{0}.csv into the Input path and set HANDLE_INFEASIBLE = True. Run the model again.".format(p.CUSTOMER_ID[0])
            
        elif gurobi_run:
            logger.info('Handling Infeasibility')
            
            if p.FILE_OUTPUT_PATH[:5] == 'gs://':
                # If outputting to GCS bucket do below
                from google.cloud import storage
                storage_client = storage.Client()
                bucket_name =  p.FILE_OUTPUT_PATH[5:].split('/', 1)[0]
                source_blob_name = p.FILE_LP_PATH + str(p.CUSTOMER_ID[0]) + "_total_prob.lp"

                source_blob_name = source_blob_name[len('gs://' + bucket_name + "/"):]
                bucket = storage_client.bucket(bucket_name)

                blob = bucket.get_blob(source_blob_name) 
                file_name = str(p.CUSTOMER_ID[0]) + "_total_prob.lp"
                blob.download_to_filename(file_name)
            else:
                # If running on VM or a local directory, do below
                file_name = p.FILE_LP_PATH + str(p.CUSTOMER_ID[0]) + "_total_prob.lp"
            
            # Convert to gurobi model
            prob = gurobipy.read(file_name)

            # Create list of error outputs to check for during while loop
            err_list = [gurobipy.GRB.INFEASIBLE, gurobipy.GRB.INF_OR_UNBD, gurobipy.GRB.UNBOUNDED]

            # Get initial state of the optimization problem (should be infeasible)
            prob.optimize()

            # Create objects for saving out GPIs causing infeasibility
            inf_gpi_bg_flag = []
            gpi_exclusion = pd.DataFrame(columns = ['CLIENT','REGION','GPI', 'BG_FLAG'])
            
            # Find infeasible GPIs
            while prob.status in err_list:
                # Find the constraints causing infeasibility
                prob.computeIIS()
                
                logger.info("Constraints causing infeasibility:")
                for c in prob.getConstrs():
                    if c.IISConstr: logger.info(f'{c.constrname}: {prob.getRow(c)} {c.Sense} {c.RHS}')
                                        
                for v in prob.getVars():
                    if v.IISUB or v.IISLB:
                        inf_gpi_bg_flag.append((v.varname[2:16], v.varname[16:18])) 
                logger.info(f"GPIs causing infeasibility: {set(inf_gpi_bg_flag)}")
                
                # Remove problematic constraints for rerun
                for c in prob.getConstrs():
                    if c.IISConstr:
                        prob.remove(c)

                # Run model again to determine if it's still infeasible. If so, go through loop again.
                prob.optimize()
            
            # Save infeasible GPIs to df
            gpi_exclusion[['GPI', 'BG_FLAG']] = pd.DataFrame(inf_gpi_bg_flag, columns=['GPI', 'BG_FLAG'])
            gpi_exclusion.loc[:,'CLIENT'] = p.CUSTOMER_ID[0]
            gpi_exclusion.loc[:,'REGION'] = 'ALL'
            gpi_exclusion = gpi_exclusion.drop_duplicates()
                
            if len(gpi_exclusion)>0:
                gpi_exclusion.to_csv(os.path.join(p.FILE_OUTPUT_PATH, p.INFEASIBLE_EXCLUSION_FILE), index = False)
                gpi_exclusion.to_csv(os.path.join(p.FILE_INPUT_PATH, p.INFEASIBLE_EXCLUSION_FILE), index = False)
                logger.info(f'File with {len(gpi_exclusion)} gpi(s) sent to the shared input folder.')

            assert False, "This run is infeasible. Good news though, there is a solution. Run the model again with HANDLE_INFEASIBLE set to True."
        
        else:
            assert False, "Unhandled error running solvers."
            
        ## Retrieve lambda and price variables ##
        # Get list of lambda variables we care about
        lambda_ = []
        for row in range(lambda_df.shape[0]):
            lambda_.append(lambda_df.iloc[row].Lambda_Over.name)
            lambda_.append(lambda_df.iloc[row].Lambda_Under.name)

        if p.INCLUDE_PLAN_LIABILITY:
            for plan_lambda_name in plan_lambdas.Plan_Liab_Lambda_String.values:
                lambda_.append(plan_lambda_name)

        lambda_name = []
        lambda_val = []
        price_var_name = []
        price_var_val = []

        # Store variables based on whether they are a lambda of interest or not
        for v in prob.variables():
            if v.name in lambda_:
                # logger.info(v.name, "=", v.varValue)
                lambda_name.append(v.name)
                lambda_val.append(v.varValue)  
                
            else:
                # logger.info(v.name, "=", v.varValue)
                price_var_name.append(v.name)
                price_var_val.append(v.varValue)

        price_array = np.concatenate((np.array(price_var_name).reshape(-1,1), np.array(price_var_val).reshape(-1,1)), axis=1) 
        price_output_df = pd.DataFrame(price_array, columns=['Dec_Var_Name', 'New_Price'])  
        price_output_df['New_Price'] = pd.to_numeric(price_output_df['New_Price'], errors='raise')
        
        # Create dataframe of lambda variables
        lambda_array = np.concatenate((np.array(lambda_name).reshape(-1,1), np.array(lambda_val).reshape(-1,1)), axis=1)
        lambda_output_df = pd.DataFrame(lambda_array, columns=['Lambda_Dec_Var', 'Value'])

        # lp_data_output_cols = [
        #     'CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'GPI',
        #     'CHAIN_GROUP', 'GO_LIVE', 'MAC_LIST', 'CURRENT_MAC_PRICE', 'GPI_ONLY',
        #     'CLAIMS', 'QTY', 'FULLAWP_ADJ', 'PRICE_REIMB', 'LM_CLAIMS', 'LM_QTY',
        #     'LM_FULLAWP_ADJ', 'LM_PRICE_REIMB', 'CLAIMS_PROJ_LAG', 'QTY_PROJ_LAG',
        #     'FULLAWP_ADJ_PROJ_LAG', 'CLAIMS_PROJ_EOY', 'QTY_PROJ_EOY',
        #     'FULLAWP_ADJ_PROJ_EOY', 'UC_UNIT', 'UC_UNIT25', 'CURR_AWP',
        #     'CURR_AWP_MIN', 'CURR_AWP_MAX', 'NDC', 'GPI_NDC', 'PKG_SZ', 'AVG_AWP',
        #     'BREAKOUT_AWP_MAX', '1026_NDC_PRICE', '1026_GPI_PRICE',
        #     'MAC1026_UNIT_PRICE', 'MAC1026_GPI_FLAG', 'PHARMACY_TYPE',
        #     'PRICE_MUTABLE', 'PRICE_REIMB_UNIT', 'EFF_UNIT_PRICE',
        #     'MAC_PRICE_UNIT_ADJ', 'EFF_CAPPED_PRICE', 'PRICE_REIMB_ADJ',
        #     'OLD_MAC_PRICE', 'LAG_REIMB', 'PRICE_REIMB_LAG', 'FULLAWP_ADJ_YTDLAG',
        #     'CLIENT_MIN_PRICE', 'CLIENT_MAX_PRICE', 'PRICE_TIER',
        #     'LM_PRICE_REIMB_CLAIM', 'PRICE_REIMB_CLAIM', 'GPI_CHANGE_EXCEPT',
        #     'Price_Bounds', 'lb_ub', 'Price_Decision_Var', 'Dec_Var_Name', 'GPI_12',
        #     'GPI_Strength', 'New_Price'
        # ]
        # Merge the new prices onto the old dataframe
        lp_data_output_df = pd.merge(lp_data_df,price_output_df, how='left', on=['Dec_Var_Name'])
        if p.TRUECOST_CLIENT:
            lp_data_output_df.loc[(lp_data_output_df['PRICE_MUTABLE'] == 0), 'New_Price'] = None
        # TO BE REMOVED JANUARY 2024 FOR BETTER SOLUTION
        # Commented below piece of code to resolve New_Price>0 while prices are immutable, 
        # revisit if corner case occurs else this WS hotfix has been phased out
        # if lp_data_output_df['New_Price'].isna().any():
        #      #picking a random column, any without nans will work
        #     lp_data_output_df['num_gpi_rows'] = lp_data_output_df.groupby(['GPI', 'BG_FLAG'])['CLIENT'].transform('count')
        #     lp_data_output_df.loc[(lp_data_output_df['num_gpi_rows']==1) & (lp_data_output_df['New_Price'].isna()),
        #                           'New_Price'] = lp_data_output_df.loc[(lp_data_output_df['num_gpi_rows']==1) 
        #                                                                & (lp_data_output_df['New_Price'].isna()),
        #                                                                'CURRENT_MAC_PRICE']

        assert len(lp_data_output_df.loc[(lp_data_output_df.New_Price > 0) & (lp_data_output_df.PRICE_MUTABLE == 0)]) == 0, "len(lp_data_output_df.loc[(lp_data_output_df.New_Price > 0) & (lp_data_output_df.PRICE_MUTABLE == 0)]) == 0"
        assert len(lp_data_df) == len(lp_data_output_df), "len(lp_data_df) == len(lp_data_output_df)"

        lp_data_output_df['EFF_UNIT_PRICE_new'] = determine_effective_price(lp_data_output_df, old_price='New_Price')
        lp_data_output_df.loc[lp_data_output_df.PRICE_MUTABLE == 0, 'EFF_UNIT_PRICE_new'] = lp_data_output_df.loc[lp_data_output_df.PRICE_MUTABLE == 0, 'EFF_UNIT_PRICE']

        #    if p.NDC_UPDATE:
        #        lp_data_output_df['EFF_CAPPED_PRICE'] = determine_effective_price(lp_data_output_df,
        #                                                                          old_price='OLD_MAC_PRICE',
        #                                                                          capped_only=True)

        #    else:
        #        lp_data_output_df['EFF_CAPPED_PRICE'] = determine_effective_price(lp_data_output_df,
        #                                                                          old_price='CURRENT_MAC_PRICE',
        #                                                                          capped_only=True)

        #    lp_data_output_df['EFF_CAPPED_PRICE'] = lp_data_output_df.apply(lambda df: df['EFF_CAPPED_PRICE'] if df['EFF_CAPPED_PRICE'] > 0 else df['PRICE_REIMB_UNIT'], axis=1)

        lp_data_output_df['EFF_CAPPED_PRICE_new'] = determine_effective_price(lp_data_output_df,
                                                                              old_price='New_Price',
                                                                              uc_unit='UC_UNIT25',
                                                                              capped_only=True)
        lp_data_output_df.loc[lp_data_output_df.PRICE_MUTABLE == 0, 'EFF_CAPPED_PRICE_new'] = lp_data_output_df.loc[lp_data_output_df.PRICE_MUTABLE == 0, 'EFF_CAPPED_PRICE']
        
        lp_data_output_df['CS_EFF_CAPPED_PRICE_new'] = determine_effective_price(lp_data_output_df,
                                                                              old_price='New_Price',
                                                                              uc_unit='UC_UNIT',
                                                                              capped_only=True)
        lp_data_output_df.loc[lp_data_output_df.PRICE_MUTABLE == 0, 'CS_EFF_CAPPED_PRICE_new'] = lp_data_output_df.loc[lp_data_output_df.PRICE_MUTABLE == 0, 'EFF_CAPPED_PRICE']


        lp_data_output_df.loc[lp_data_output_df.PRICE_MUTABLE == 0, 'New_Price'] = lp_data_output_df.loc[lp_data_output_df.PRICE_MUTABLE == 0, 'EFF_CAPPED_PRICE']

        if p.FLOOR_PRICE:
            floor_gpi = standardize_df(pd.read_csv(p.FILE_INPUT_PATH + p.FLOOR_GPI_LIST, dtype = p.VARIABLE_TYPE_DIC))
            lp_data_output_df.loc[(lp_data_output_df.GPI.isin(
                floor_gpi.GPI) & lp_data_output_df.CURRENT_MAC_PRICE > 0), 'CURRENT_MAC_PRICE'] = lp_data_output_df.loc[
                (lp_data_output_df.GPI.isin(floor_gpi.GPI) & lp_data_output_df.CURRENT_MAC_PRICE > 0), 'CURRENT_MAC_PRICE_ACTUAL']
            lp_data_output_df.loc[(lp_data_output_df.GPI.isin(
                floor_gpi.GPI) & lp_data_output_df.CURRENT_MAC_PRICE > 0), 'EFF_CAPPED_PRICE'] = lp_data_output_df.loc[
                (lp_data_output_df.GPI.isin(
                    floor_gpi.GPI) & lp_data_output_df.CURRENT_MAC_PRICE > 0), 'EFF_CAPPED_PRICE_ACTUAL']
            lp_data_output_df.loc[
                    (lp_data_output_df.GPI.isin(floor_gpi.GPI) & (lp_data_output_df.CURRENT_MAC_PRICE > 0)), 'PRICE_MUTABLE'] = 1



        #####Uncomment if you want to adjust MAC prices after the fact
        #    manual_override = standardize_df(pd.read_csv(p.FILE_INPUT_PATH + '20190523_NDC_prices_to_squash.csv', dtype = p.VARIABLE_TYPE_DIC))

        #    for i in range(len(manual_override)):
        #        lp_data_output_df.loc[(lp_data_output_df.CLIENT == manual_override.iloc[i].CLIENT) &
        #                              (lp_data_output_df.BREAKOUT == manual_override.iloc[i].BREAKOUT) &
        #                              (lp_data_output_df.REGION == manual_override.iloc[i].REGION) &
        #                              (lp_data_output_df.MEASUREMENT == manual_override.iloc[i].MEASUREMENT)&
        #                              (lp_data_output_df.CHAIN_GROUP == manual_override.iloc[i].CHAIN_GROUP) &
        #                              (lp_data_output_df.GPI_NDC == manual_override.iloc[i].GPI_NDC), 'New_Price'] = manual_override.iloc[i].New_Price

        #        lp_data_output_df.loc[(lp_data_output_df.CLIENT == manual_override.iloc[i].CLIENT) &
        #                              (lp_data_output_df.BREAKOUT == manual_override.iloc[i].BREAKOUT) &
        #                              (lp_data_output_df.REGION == manual_override.iloc[i].REGION) &
        #                              (lp_data_output_df.MEASUREMENT == manual_override.iloc[i].MEASUREMENT)&
        #                              (lp_data_output_df.CHAIN_GROUP == manual_override.iloc[i].CHAIN_GROUP) &
        #                              (lp_data_output_df.GPI_NDC == manual_override.iloc[i].GPI_NDC), 'EFF_UNIT_PRICE_new'] = manual_override.iloc[i].New_Price


        lp_data_output_df['Rounded_Price'] = lp_data_output_df['New_Price'].apply(round_to)

        # Find and save out rows where rounding up occurred; index hard coded based on default rounding to 4 decimal places
        if p.ROUND_DOWN:
            rounded_up_mask = lp_data_output_df.apply(lambda row: f"{row['New_Price']:.20f}"[6] == '9', axis=1)
            difference_mask = lp_data_output_df.apply(lambda row: abs(float(f"{row['New_Price']:.5f}") - float(f"{row['Rounded_Price']:.5f}")) > 0.00000001, axis=1)
            combined_mask = rounded_up_mask & difference_mask
            if combined_mask.any():
                rounded_up_rows = lp_data_output_df[combined_mask].copy()
                rounded_up_rows.to_csv(os.path.join(p.FILE_OUTPUT_PATH,'rounding_up_cases.csv'), index=False)
        
        #ACTUAL_KEEP_SEND: Based on Final MAC Price - compared to the GoodRx price
        if p.INTERCEPTOR_OPT or p.COSTSAVER_CLIENT:
            lp_data_output_df.loc[:,'ACTUAL_KEEP_SEND'] = 1.0
            lp_data_output_df.loc[
                (lp_data_output_df.CS_EFF_CAPPED_PRICE_new > lp_data_output_df.VENDOR_PRICE)
                # # VENDOR_PRICE has na's filled with 0, but nan VENDOR_PRICE means we don't want to send
                # # Shouldn't affect actual outcome since zbd fracs should already be set to 0, 
                # # but for recording purposes we add the below mask
                & (lp_data_output_df.VENDOR_PRICE > 0),
                'ACTUAL_KEEP_SEND'
            ] = 0.0
            
            mask = (lp_data_output_df['ACTUAL_KEEP_SEND'] != lp_data_output_df['EXPECTED_KEEP_SEND']) & (lp_data_output_df.UNC_OVRD_AMT.notna())
            lp_data_output_df.loc[mask, 'INTERCEPT_REASON'] = "UNC OVERRIDES"
            lp_data_output_df.loc[lp_data_output_df.UNC_OVRD_AMT <= lp_data_df.INTERCEPT_LOW, 'ACTUAL_KEEP_SEND'] = 1.0
            lp_data_output_df.loc[lp_data_output_df.UNC_OVRD_AMT > lp_data_output_df.INTERCEPT_HIGH, 'ACTUAL_KEEP_SEND'] = 0.0
            
            
            mask = (lp_data_output_df['ACTUAL_KEEP_SEND'] != lp_data_output_df['EXPECTED_KEEP_SEND']) & (lp_data_output_df.CS_EFF_CAPPED_PRICE_new < lp_data_output_df.New_Price)
            lp_data_output_df.loc[mask, 'INTERCEPT_REASON'] = "UNC/NMR CAPPED PRICES"
        
        # lp_data_output_df = pd.read_csv(p.FILE_OUTPUT_PATH + 'Model_11_Output_prices_0125.csv', dtype = p.VARIABLE_TYPE_DIC)
        lp_data_output_df['lb'] = lp_data_output_df.Price_Bounds.map(lambda x: x[0])
        lp_data_output_df['ub'] = lp_data_output_df.Price_Bounds.map(lambda x: x[1])


        # Save data in a temp file in case of crash

        # For Non UNC Clients: (WRITE_TO_BQ = True and unc_flag = False and UNC_ADJUST=False, hence below condition will evaluate to True
        # For UNC Clients:
        #   1st Iteration: (WRITE_TO_BQ = True) and unc_flag = False and UNC_ADJUST=True, hence below condition will evaluate to False
        #   2nd Iteration: (WRITE_TO_BQ = True) and unc_flag = True and UNC_ADJUST=True, hence below condition will evaluate to True
        if False: #if p.WRITE_TO_BQ and (unc_flag == True or p.UNC_ADJUST == False):
            uf.write_to_bq(
                lp_data_output_df,
                project_output = p.BQ_OUTPUT_PROJECT_ID,
                dataset_output = p.BQ_OUTPUT_DATASET,
                table_id = "model_last_run_subchain",
                client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                timestamp_param = p.TIMESTAMP,
                run_id = p.AT_RUN_ID,
                schema = None  # TODO: write proper schema
            )
        else:
            lp_data_output_df.to_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, 'last_run_' + p.DATA_ID + '.csv'), index=False)

        total_output_columns = []
        if p.WRITE_OUTPUT:
            total_output_columns = list(lp_data_output_df.columns)

        ########################### Price checks  #########################
        logger.info('--------------------')
        logger.info('Starting pricing checks')
        lp_output_mut = lp_data_output_df.loc[lp_data_output_df['PRICE_MUTABLE']==1]

        logger.info('Price increase percentage upheld: ')
        if lp_output_mut.apply(check_price_increase_decrease_initial, args=tuple([month]), axis=1).any().any():
            logger.info('False')
        else:
            logger.info('True')
        if p.AGG_UP_FAC >= 0:
            logger.info('Agg MAC price change upheld: ')
            if check_agg_price_cons(lp_output_mut, month):
                logger.info('False')
            else:
                logger.info('True')

        # file outputs
        with open(lp_data_output_df_out, 'wb') as f:
            pickle.dump(lp_data_output_df, f)
        # with open(pilot_output_columns_out, 'wb') as f:
        #     pickle.dump(pilot_output_columns, f)
        with open(total_output_columns_out, 'wb') as f:
            pickle.dump(total_output_columns, f)
        with open(lambda_output_df_out, 'wb') as f:
            pickle.dump(lambda_output_df, f)

        if 'gs://' in p.FILE_OUTPUT_PATH:  # (cleanup local temp dir)
            shutil.rmtree(TEMP_WORK_DIR)

        return (lp_data_output_df, lambda_output_df)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'Run LP Solver', repr(e), error_loc)
        raise e
######################## END run_solver ######################

######################## no_lp_run ######################
def no_lp_run(
    # params_file_in: str,
    unc_flag: bool,
    lp_vol_mv_agg_df_in: InputPath('pickle'),  
    lp_data_output_df_out: OutputPath('pickle'),
    loglevel: str = 'INFO'
    # kube_run: bool = True,
):
    from CPMO_shared_functions import update_run_status
    try:
        # input files
        with open(lp_vol_mv_agg_df_in, 'rb') as f:
            lp_vol_mv_agg_df = pickle.load(f)

            lp_data_output_df = lp_vol_mv_agg_df
            lp_data_output_df['New_Price'] = lp_data_output_df['MAC_PRICE_UNIT_ADJ']
            lp_data_output_df['EFF_UNIT_PRICE_new'] = lp_data_output_df['EFF_UNIT_PRICE']
            lp_data_output_df['EFF_CAPPED_PRICE_new'] = lp_data_output_df['EFF_CAPPED_PRICE']

        # output files
        with open(lp_data_output_df_out, 'wb') as f:
            pickle.dump(lp_data_output_df, f)

        return lp_data_output_df
    except Exception as e:
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'No LP Run', repr(e), error_loc)
        raise e
######################## End no_lp_run ######################

######################## lp_output ######################
def lp_output(
    params_file_in: str,
    # m: int, 
    unc_flag: bool,
    month: int,
    lag_price_col: str,
    pharm_lag_price_col: str,
    lp_data_output_df_in: InputPath('pickle'),
    performance_dict_in: InputPath('pickle'),
    act_performance_dict_in: InputPath('pickle'),
    ytd_perf_pharm_actuals_dict_in: InputPath('pickle'),
    client_list_in: InputPath('pickle'),
    client_guarantees_in: InputPath('pickle'),
    pharmacy_guarantees_in: InputPath('pickle'),
    oc_eoy_pharm_perf_in: InputPath('pickle'),
    gen_launch_eoy_dict_in: InputPath('pickle'),
    eoy_days_in: InputPath('pickle'),
    perf_dict_col_in: InputPath('pickle'),
    mac_list_df_in: InputPath('pickle'),
    lp_vol_mv_agg_df_actual_in: InputPath('pickle'),
    oc_pharm_dummy_in: InputPath('pickle'),
    dummy_perf_dict_in: InputPath('pickle'),
#     pilot_output_columns_in: InputPath('pickle'),
    generic_launch_df_in: InputPath('pickle'),
    pref_pharm_list_in: InputPath('pickle'),
    breakout_df_in: InputPath('pickle'),
    oc_pharm_surplus_in: InputPath('pickle'),
    proj_days_in: InputPath('pickle'),
    lambda_output_df_in: InputPath('pickle'),
    chain_region_mac_mapping_in: InputPath('pickle'),
    total_output_columns_in: InputPath('pickle'),
    brand_surplus_ytd_in: InputPath('pickle'),
    brand_surplus_lag_in: InputPath('pickle'),
    brand_surplus_eoy_in: InputPath('pickle'),
    specialty_surplus_ytd_in: InputPath('pickle'),
    specialty_surplus_lag_in: InputPath('pickle'),
    specialty_surplus_eoy_in: InputPath('pickle'),
    disp_fee_surplus_ytd_in: InputPath('pickle'),
    disp_fee_surplus_lag_in: InputPath('pickle'),
    disp_fee_surplus_eoy_in: InputPath('pickle'),
    # non_capped_pharmacy_list_in: InputPath('pickle'),
    # agreement_pharmacy_list_in: InputPath('pickle'),
    loglevel: str = 'INFO'
    # kube_run: bool = True,
):
    import os
    import shutil
    import sys
    sys.path.append('/')
    import logging
    import pickle
    import calendar
    import numpy as np
    import pandas as pd
    import datetime as dt
    import pulp
    import util_funcs as uf
    import BQ

    uf.write_params(params_file_in)
    import CPMO_parameters as p
    from qa_checks import qa_dataframe
    from CPMO_shared_functions import (
        calculatePerformance, dict_to_df, df_to_dict, standardize_df, check_agg_price_cons, write_spend_total_df
    )
    from CPMO_lp_functions import pharmacy_type_new
    from CPMO_plan_liability import generatePlanLiabilityOutput, calcPlanCost
    from CPMO_shared_functions import update_run_status, round_to, add_target_ingcost
    update_run_status(i_error_type='Started lp_output')
    try:
        out_path = os.path.join(p.FILE_LOG_PATH, 'ClientPharmacyMacOptimization.log')
        logger = uf.log_setup(log_file_path=out_path, loglevel=loglevel)

        # file inputs
        with open(lp_data_output_df_in, 'rb') as f:
            lp_data_output_df = pickle.load(f)
        with open(performance_dict_in, 'rb') as f:
            performance_dict = pickle.load(f)
        with open(act_performance_dict_in, 'rb') as f:
            act_performance_dict = pickle.load(f)
        with open(ytd_perf_pharm_actuals_dict_in, 'rb') as f:
            ytd_perf_pharm_actuals_dict = pickle.load(f)
        with open(client_list_in, 'rb') as f:
            client_list = pickle.load(f)
        with open(client_guarantees_in, 'rb') as f:
            client_guarantees = pickle.load(f)
        with open(pharmacy_guarantees_in, 'rb') as f:
            pharmacy_guarantees = pickle.load(f)
        with open(oc_eoy_pharm_perf_in, 'rb') as f:
            oc_eoy_pharm_perf = pickle.load(f)
        with open(gen_launch_eoy_dict_in, 'rb') as f:
            gen_launch_eoy_dict = pickle.load(f)
        with open(eoy_days_in, 'rb') as f:
            eoy_days = pickle.load(f)
        with open(perf_dict_col_in, 'rb') as f:
            perf_dict_col = pickle.load(f)
        with open(mac_list_df_in, 'rb') as f:
            mac_list_df = pickle.load(f)
        with open(lp_vol_mv_agg_df_actual_in, 'rb') as f:
            lp_vol_mv_agg_df_actual = pickle.load(f)
        with open(oc_pharm_dummy_in, 'rb') as f:
            oc_pharm_dummy = pickle.load(f)
        with open(dummy_perf_dict_in, 'rb') as f:
            dummy_perf_dict = pickle.load(f)
        # with open(pilot_output_columns_in, 'rb') as f:
        #     pilot_output_columns = pickle.load(f)
        with open(generic_launch_df_in, 'rb') as f:
            generic_launch_df = pickle.load(f)
        with open(pref_pharm_list_in, 'rb') as f:
            pref_pharm_list = pickle.load(f)
        with open(breakout_df_in, 'rb') as f:
            breakout_df = pickle.load(f)
        with open(oc_pharm_surplus_in, 'rb') as f:
            oc_pharm_surplus = pickle.load(f)
        with open(proj_days_in, 'rb') as f:
            proj_days = pickle.load(f)
        with open(lambda_output_df_in, 'rb') as f:
            lambda_output_df = pickle.load(f)
        with open(chain_region_mac_mapping_in, 'rb') as f:
            chain_region_mac_mapping = pickle.load(f)
        with open(total_output_columns_in, 'rb') as f:
            total_output_columns = pickle.load(f)
        with open(brand_surplus_ytd_in,'rb') as f:
            brand_surplus_ytd_dict = pickle.load(f)
        with open(brand_surplus_lag_in,'rb') as f:
            brand_surplus_lag_dict = pickle.load(f)
        with open(brand_surplus_eoy_in,'rb') as f:
            brand_surplus_eoy_dict = pickle.load(f)
        with open(specialty_surplus_ytd_in,'rb') as f:
            specialty_surplus_ytd_dict = pickle.load(f)
        with open(specialty_surplus_lag_in,'rb') as f:
            specialty_surplus_lag_dict = pickle.load(f)
        with open(specialty_surplus_eoy_in,'rb') as f:
            specialty_surplus_eoy_dict = pickle.load(f)           
        with open(disp_fee_surplus_ytd_in,'rb') as f:
            disp_fee_surplus_ytd_dict = pickle.load(f)
        with open(disp_fee_surplus_lag_in,'rb') as f:
            disp_fee_surplus_lag_dict = pickle.load(f)
        with open(disp_fee_surplus_eoy_in,'rb') as f:
            disp_fee_surplus_eoy_dict = pickle.load(f)
        # with open(non_capped_pharmacy_list_in, 'rb') as f:
        #     non_capped_pharmacy_list = pickle.load(f)
        # with open(agreement_pharmacy_list_in, 'rb') as f:
        #     agreement_pharmacy_list = pickle.load(f)

        ######################## Readjudication on New Prices ######################
        logger.info('--------------------')
        logger.info('Readjudicating on new prices')


        #    lp_data_output_df['Price_Reimb_Old'] = lp_data_output_df.QTY * lp_data_output_df.EFF_UNIT_PRICE
        #    future_performance_dict = calculatePerformance(lp_data_output_df, client_guarantees, pharmacy_guarantees,
        #                                               client_list, p.BIG_CAPPED_PHARMACY_LIST, oc_eoy_pharm_perf, gen_launch_eoy_dict,
        #                                               'Price_Reimb_Old')
        gnrc_noncap_cogs = p.NON_CAPPED_PHARMACY_LIST['GNRC'] + p.COGS_PHARMACY_LIST['GNRC']
        brnd_noncap_cogs = p.NON_CAPPED_PHARMACY_LIST['BRND'] + p.COGS_PHARMACY_LIST['BRND']
        
        all_noncap_cogs = set(gnrc_noncap_cogs + brnd_noncap_cogs)

        ytd_perf_df = dict_to_df(ytd_perf_pharm_actuals_dict, perf_dict_col)

        # For Non UNC Clients: (WRITE_TO_BQ = True and unc_flag = False and UNC_ADJUST=False, hence below condition will evaluate to True
        # For UNC Clients:
        #   1st Iteration: (WRITE_TO_BQ = True) and unc_flag = False and UNC_ADJUST=True, hence below condition will evaluate to False
        #   2nd Iteration: (WRITE_TO_BQ = True) and unc_flag = True and UNC_ADJUST=True, hence below condition will evaluate to True
        if p.WRITE_TO_BQ and (unc_flag == True or p.UNC_ADJUST == False):
            uf.write_to_bq(
                ytd_perf_df,
                project_output = p.BQ_OUTPUT_PROJECT_ID,
                dataset_output = p.BQ_OUTPUT_DATASET,
                table_id = "YTD_Performance",
                client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                timestamp_param = p.TIMESTAMP,
                run_id = p.AT_RUN_ID,
                schema = None # TODO: create schema
            )
        else:
            ytd_perf_df.to_csv(os.path.join(p.FILE_OUTPUT_PATH, 'YTD_Performance_' + str(month) + '_' + str(p.TIMESTAMP) + '.csv'), index=False)


        # Capped and floored for former prices but new prices just as they are
#         lp_data_output_df['Price_Reimb_Proj'] = lp_data_output_df.QTY_PROJ_EOY * lp_data_output_df.New_Price #This should match with LP
        
        #Simplified for coding: Actual spend_proj = new_mac*(Qty_proj*NonZbd + Qty_proj*Zbd*Keepsend_act) + Grx*(Qty_proj*Zbd*(1- Keepsend_act)) 
        if p.INTERCEPTOR_OPT or p.COSTSAVER_CLIENT:
            # the min() will always be VENDOR_PRICE if ACTUAL_KEEP_SEND=0,
            # but using the min will filter nans in the VENDOR_PRICE 
            lp_data_output_df['Price_Reimb_Proj'] = lp_data_output_df.QTY_PROJ_EOY * (lp_data_output_df.New_Price -lp_data_output_df[['New_Price', 'VENDOR_PRICE']].min(axis=1) * \
                                                                                      lp_data_output_df.QTY_ZBD_FRAC * (1 - lp_data_output_df.ACTUAL_KEEP_SEND)) 
        else:
            lp_data_output_df['Price_Reimb_Proj'] = lp_data_output_df.QTY_PROJ_EOY * lp_data_output_df.New_Price #This should match with LP
        
        #Pharmacy Spend calculations remain the same for Costsaver as the projected Qty is adjusted to exclude any send claims
        lp_data_output_df['Pharm_Price_Reimb_Proj'] = lp_data_output_df.PHARM_QTY_PROJ_EOY * lp_data_output_df.New_Price #This should match with LP
        
        #For Costsaver Pharm_AWP_EOY is already recalculated in costsaver functions
        new_proj_performance_eoy_dict = calculatePerformance(lp_data_output_df, client_guarantees, pharmacy_guarantees,
                                                   client_list, p.AGREEMENT_PHARMACY_LIST, oc_eoy_pharm_perf,
                                                   gen_launch_eoy_dict, brand_surplus_eoy_dict, specialty_surplus_eoy_dict, disp_fee_surplus_eoy_dict,
                                                   client_reimb_column='Price_Reimb_Proj', pharm_reimb_column='Pharm_Price_Reimb_Proj',
                                                   client_TARG_column='TARG_INGCOST_ADJ_PROJ_EOY', pharm_TARG_column='PHARM_TARG_INGCOST_ADJ_PROJ_EOY',)

        new_proj_performance_dict = dict()
        for key in new_proj_performance_eoy_dict:
            new_proj_performance_dict[key] = new_proj_performance_eoy_dict[key] + performance_dict[key]
        lambda_perf_df = dict_to_df(new_proj_performance_dict, perf_dict_col)

        # For Non UNC Clients: (WRITE_TO_BQ = True and unc_flag = False and UNC_ADJUST=False, hence below condition will evaluate to True
        # For UNC Clients:
        #   1st Iteration: (WRITE_TO_BQ = True) and unc_flag = False and UNC_ADJUST=True, hence below condition will evaluate to False
        #   2nd Iteration: (WRITE_TO_BQ = True) and unc_flag = True and UNC_ADJUST=True, hence below condition will evaluate to True
        if p.WRITE_TO_BQ and (unc_flag == True or p.UNC_ADJUST == False):
            uf.write_to_bq(
                lambda_perf_df,
                project_output = p.BQ_OUTPUT_PROJECT_ID,
                dataset_output = p.BQ_OUTPUT_DATASET,
                table_id = "Lambda_Performance",
                client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                timestamp_param = p.TIMESTAMP,
                run_id = p.AT_RUN_ID,
                schema = None # TODO: create schema
            )
        else:
            lambda_perf_df.to_csv(os.path.join(p.FILE_OUTPUT_PATH, p.LAMBDA_PERFORMANCE_OUTPUT), index=False)

        # All prices capped and floored, this should be the real performance of the LP
        if p.INTERCEPTOR_OPT or p.COSTSAVER_CLIENT:
            lp_data_output_df['Price_Effective_Reimb_Proj'] = lp_data_output_df.QTY_PROJ_EOY * (lp_data_output_df.EFF_CAPPED_PRICE_new.round(4) - lp_data_output_df.QTY_ZBD_FRAC * \
                                                     lp_data_output_df[['EFF_CAPPED_PRICE_new', 'VENDOR_PRICE']].min(axis=1) * (1 - lp_data_output_df.ACTUAL_KEEP_SEND)) 

        else:
            lp_data_output_df['Price_Effective_Reimb_Proj'] = lp_data_output_df.QTY_PROJ_EOY * lp_data_output_df.EFF_CAPPED_PRICE_new.round(4)
        
        lp_data_output_df['Pharm_Price_Effective_Reimb_Proj'] = lp_data_output_df.PHARM_QTY_PROJ_EOY * lp_data_output_df.EFF_CAPPED_PRICE_new.round(4)
        
        effective_proj_performance_eoy_dict = calculatePerformance(lp_data_output_df, client_guarantees, pharmacy_guarantees,
                                                   client_list, p.AGREEMENT_PHARMACY_LIST, oc_eoy_pharm_perf, gen_launch_eoy_dict,
                                                   brand_surplus_eoy_dict, specialty_surplus_eoy_dict, disp_fee_surplus_eoy_dict,
                                                   client_reimb_column='Price_Effective_Reimb_Proj', pharm_reimb_column='Pharm_Price_Effective_Reimb_Proj',
                                                   client_TARG_column='TARG_INGCOST_ADJ_PROJ_EOY', pharm_TARG_column='PHARM_TARG_INGCOST_ADJ_PROJ_EOY')
        
        # Replace performance for things that we don't have a guarantee for with spend
        if isinstance(disp_fee_surplus_eoy_dict,pd.DataFrame):
            if len(disp_fee_surplus_eoy_dict) <= 1:
                disp_fee_surplus_dict = disp_fee_surplus_eoy_dict.set_index('ENTITY').squeeze(axis=1).to_dict()
            else:
                disp_fee_surplus_dict = disp_fee_surplus_eoy_dict.set_index('ENTITY').squeeze().to_dict()
        else:
            disp_fee_surplus_dict = disp_fee_surplus_eoy_dict.copy()
            
        effective_proj_performance_dict = dict()
        for key in effective_proj_performance_eoy_dict:
            effective_proj_performance_dict[key] = effective_proj_performance_eoy_dict[key] + act_performance_dict[key]

        model_perf_df = dict_to_df(effective_proj_performance_dict, perf_dict_col)

        # For Non UNC Clients: (WRITE_TO_BQ = True and unc_flag = False and UNC_ADJUST=False, hence below condition will evaluate to True
        # For UNC Clients:
        #   1st Iteration: (WRITE_TO_BQ = True) and unc_flag = False and UNC_ADJUST=True, hence below condition will evaluate to False
        #   2nd Iteration: (WRITE_TO_BQ = True) and unc_flag = True and UNC_ADJUST=True, hence below condition will evaluate to True
        if p.WRITE_TO_BQ and (unc_flag == True or p.UNC_ADJUST == False):
            uf.write_to_bq(
                model_perf_df,
                project_output = p.BQ_OUTPUT_PROJECT_ID,
                dataset_output = p.BQ_OUTPUT_DATASET,
                table_id = "Model_Performance",
                client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                timestamp_param = p.TIMESTAMP,
                run_id = p.AT_RUN_ID,
                schema = None # TODO: create schema
            )
        else:
            model_perf_df.to_csv(os.path.join(p.FILE_OUTPUT_PATH, p.MODEL_PERFORMANCE_OUTPUT), index=False)

        unc_adjust_df = model_perf_df
        unc_adjust_df['PERFORMANCE'] = unc_adjust_df['PERFORMANCE'] - lambda_perf_df['PERFORMANCE']
        unc_adjust_df.rename(columns = {'PERFORMANCE': 'DELTA', 'ENTITY':'BREAKOUT'}, inplace=True)
        # Combines the NON_CAPPED_PHARMACY_LIST and COGS_PHARMACY_LIST into a single list called exclude_pharmacies 
        exclude_pharmacies = all_noncap_cogs
        # Clean the BREAKOUT column using a regex to remove BREAKOUTs containing the specified substrings 
        cleaned_breakout = unc_adjust_df['BREAKOUT'].str.replace(r'(_R90OK|_EXTRL|MCHOICE_)', '', regex=True)
        # Filters the dataframe where the BREAKOUT column matches pharmacies in the exclude_pharmacies set 
        unc_adjust_df = unc_adjust_df[~cleaned_breakout.isin(exclude_pharmacies)]

        # For Non UNC Clients: (WRITE_TO_BQ = True and unc_flag = False and UNC_ADJUST=False, hence below condition will evaluate to True
        # For UNC Clients:
        #   1st Iteration: (WRITE_TO_BQ = True) and unc_flag = False and UNC_ADJUST=True, hence below condition will evaluate to False
        #   2nd Iteration: (WRITE_TO_BQ = True) and unc_flag = True and UNC_ADJUST=True, hence below condition will evaluate to True
        if p.WRITE_TO_BQ and (unc_flag == True or p.UNC_ADJUST == False):
            uf.write_to_bq(
                unc_adjust_df,
                project_output = p.BQ_OUTPUT_PROJECT_ID,
                dataset_output = p.BQ_OUTPUT_DATASET,
                table_id = "UNC_Adjustment",
                client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                timestamp_param = p.TIMESTAMP,
                run_id = p.AT_RUN_ID,
                schema = None # TODO: create schema
            )
        # It is saved by defaut as a CSV file so one can run UNC_ADJUST easily.
        ## modifz for unc_adjust input read
        if unc_flag:
            unc_adjust_df.to_csv(os.path.join(p.FILE_OUTPUT_PATH, p.UNC_ADJUSTMENT), index=False)
        else: #saving unc = False run file to both dynamic i/p and o/p
            unc_adjust_df.to_csv(os.path.join(p.FILE_OUTPUT_PATH, p.UNC_ADJUSTMENT), index=False)
            ## modifz for unc_adjust input read
            unc_adjust_df.to_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.UNC_ADJUSTMENT), index=False)

        # Performance if old prices still in effect with floors and caps
        # lp_data_output_df['Old_Price_Effective_Reimb_Proj_EOY'] = lp_data_output_df.QTY_PROJ_EOY * lp_data_output_df['EFF_UNIT_PRICE_old']
        if p.INTERCEPTOR_OPT or p.COSTSAVER_CLIENT:
            lp_data_output_df['Old_Price_Effective_Reimb_Proj_EOY'] = lp_data_output_df.QTY_PROJ_EOY * (lp_data_output_df[lag_price_col] - lp_data_output_df.QTY_ZBD_FRAC * \
                                                     lp_data_output_df[[lag_price_col, 'VENDOR_PRICE']].min(axis=1) * (1 - lp_data_output_df.ACTUAL_KEEP_SEND))

        else:
            lp_data_output_df['Old_Price_Effective_Reimb_Proj_EOY'] = lp_data_output_df.QTY_PROJ_EOY * lp_data_output_df[lag_price_col]
        
        lp_data_output_df['Pharm_Old_Price_Effective_Reimb_Proj_EOY'] = lp_data_output_df.PHARM_QTY_PROJ_EOY * lp_data_output_df[pharm_lag_price_col]
        
        old_effective_proj_performance_eoy_dict2 = calculatePerformance(lp_data_output_df, client_guarantees, pharmacy_guarantees,
                                                   client_list, p.AGREEMENT_PHARMACY_LIST, oc_eoy_pharm_perf, gen_launch_eoy_dict,
                                                   brand_surplus_eoy_dict, specialty_surplus_eoy_dict, disp_fee_surplus_eoy_dict,
                                                   client_reimb_column='Old_Price_Effective_Reimb_Proj_EOY', pharm_reimb_column='Pharm_Old_Price_Effective_Reimb_Proj_EOY',
                                                   client_TARG_column='TARG_INGCOST_ADJ_PROJ_EOY', pharm_TARG_column='PHARM_TARG_INGCOST_ADJ_PROJ_EOY', )
        if p.UNC_OPT:
            lp_vol_mv_agg_df_actual['Old_Price_Effective_Reimb_Proj_EOY'] = lp_vol_mv_agg_df_actual.QTY_PROJ_EOY * lp_vol_mv_agg_df_actual[lag_price_col]
            lp_vol_mv_agg_df_actual['Pharm_Old_Price_Effective_Reimb_Proj_EOY'] = lp_vol_mv_agg_df_actual.PHARM_QTY_PROJ_EOY * lp_vol_mv_agg_df_actual[pharm_lag_price_col]
            old_effective_proj_performance_eoy_dict2 = calculatePerformance(lp_vol_mv_agg_df_actual, client_guarantees, pharmacy_guarantees,
                                                       client_list, p.AGREEMENT_PHARMACY_LIST, oc_eoy_pharm_perf, gen_launch_eoy_dict,
                                                       brand_surplus_eoy_dict, specialty_surplus_eoy_dict, disp_fee_surplus_eoy_dict,
                                                       client_reimb_column='Old_Price_Effective_Reimb_Proj_EOY', pharm_reimb_column='Pharm_Old_Price_Effective_Reimb_Proj_EOY',
                                                       client_TARG_column='TARG_INGCOST_ADJ_PROJ_EOY', pharm_TARG_column='PHARM_TARG_INGCOST_ADJ_PROJ_EOY')
            lp_vol_mv_agg_df_actual.to_csv(os.path.join(p.FILE_OUTPUT_PATH, 'lp_data_actual_{}.csv'.format(p.DATA_ID)), index=False)
            
        old_effective_proj_performance_dict2 = dict()
        for key in old_effective_proj_performance_eoy_dict2:
            old_effective_proj_performance_dict2[key] = old_effective_proj_performance_eoy_dict2[key] + act_performance_dict[key]

        prexisting_perf_df = dict_to_df(old_effective_proj_performance_dict2, perf_dict_col)

        # For Non UNC Clients: (WRITE_TO_BQ = True and unc_flag = False and UNC_ADJUST=False, hence below condition will evaluate to True
        # For UNC Clients:
        #   1st Iteration: (WRITE_TO_BQ = True) and unc_flag = False and UNC_ADJUST=True, hence below condition will evaluate to False
        #   2nd Iteration: (WRITE_TO_BQ = True) and unc_flag = True and UNC_ADJUST=True, hence below condition will evaluate to True
        if p.WRITE_TO_BQ and (unc_flag == True or p.UNC_ADJUST == False):
            uf.write_to_bq(
                prexisting_perf_df,
                project_output = p.BQ_OUTPUT_PROJECT_ID,
                dataset_output = p.BQ_OUTPUT_DATASET,
                table_id = "Prexisting_Performance",
                client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                timestamp_param = p.TIMESTAMP,
                run_id = p.AT_RUN_ID,
                schema = None # TODO: create schema
            )
        else:
            prexisting_perf_df.to_csv(os.path.join(p.FILE_OUTPUT_PATH, p.PRE_EXISTING_PERFORMANCE_OUTPUT), index=False)


        # Value reporting based on project Interceptor

        # The idea is to compare the pharmacy ger with and without project interceptor live.
        # From the GER difference we can obtain the performance change and that is just normalized by the AWP
        
        #COSTSAVER FLAG: Moving to CPMO_costsaver_functions.py - should be called in CPMO_Reporting_to_IA as per Diego's notes

        #Plan liability calculations
        if p.INCLUDE_PLAN_LIABILITY:
            plan_new_projected_performance_lp, _ = calcPlanCost(lp_data_output_df.loc[(lp_data_output_df.QTY_PROJ_EOY > 0)&(lp_data_output_df.Price_Mutable == 1)], 'Eff_capped_price_new', True)
            #plan_effective_proj_performance_eoy_dict = calcPlanCost(lp_data_output_df.loc[(lp_data_output_df.QTY_PROJ_EOY > 0)], 'Eff_capped_price_new', True)
            #plan_old_effective_proj_performance_eoy_dict = calcPlanCost(lp_data_output_df.loc[(lp_data_output_df.QTY_PROJ_EOY > 0)], 'Eff_capped_price', True)

            plan_new_proj_performance_eoy_dict, df_plan_new_proj_performance_eoy = calcPlanCost(lp_data_output_df.loc[(lp_data_output_df.QTY_PROJ_EOY > 0) & (lp_data_output_df.CURRENT_MAC_PRICE > 0)], 'Eff_capped_price_new', True)
            plan_old_proj_performance_eoy_dict, df_plan_old__proj_performance_eoy = calcPlanCost(lp_data_output_df.loc[(lp_data_output_df.QTY_PROJ_EOY > 0) & (lp_data_output_df.CURRENT_MAC_PRICE > 0)], lag_price_col, True)
            plan_WCold_proj_performance_eoy_dict, _ = calcPlanCost(lp_data_output_df.loc[(lp_data_output_df.QTY_PROJ_EOY > 0) & (lp_data_output_df.CURRENT_MAC_PRICE > 0)], 'MAC_PRICE_UNIT_Adj', True)

            new_plan_liability_df = dict_to_df(plan_new_proj_performance_eoy_dict, perf_dict_col)

            # For Non UNC Clients: (WRITE_TO_BQ = True and unc_flag = False and UNC_ADJUST=False, hence below condition will evaluate to True
            # For UNC Clients:
            #   1st Iteration: (WRITE_TO_BQ = True) and unc_flag = False and UNC_ADJUST=True, hence below condition will evaluate to False
             #   2nd Iteration: (WRITE_TO_BQ = True) and unc_flag = True and UNC_ADJUST=True, hence below condition will evaluate to True
            if p.WRITE_TO_BQ and (unc_flag == True or p.UNC_ADJUST == False):
                uf.write_to_bq(
                    new_plan_liability_df,
                    project_output = p.BQ_OUTPUT_PROJECT_ID,
                    dataset_output = p.BQ_OUTPUT_DATASET,
                    table_id = "New_Plan_Liability",
                    client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                    timestamp_param = p.TIMESTAMP,
                    run_id = p.AT_RUN_ID,
                    schema = None # TODO: create schema
                )
            else:
                new_plan_liability_df.to_csv(p.FILE_OUTPUT_PATH + 'New_Plan_Liability_' + str(month) + '_' + str(p.TIMESTAMP) + '.csv', index=False)
            prexisting_plan_liability_df = dict_to_df(plan_old_proj_performance_eoy_dict, perf_dict_col)

            # For Non UNC Clients: (WRITE_TO_BQ = True and unc_flag = False and UNC_ADJUST=False, hence below condition will evaluate to True
            # For UNC Clients:
            #   1st Iteration: (WRITE_TO_BQ = True) and unc_flag = False and UNC_ADJUST=True, hence below condition will evaluate to False
            #   2nd Iteration: (WRITE_TO_BQ = True) and unc_flag = True and UNC_ADJUST=True, hence below condition will evaluate to True
            if p.WRITE_TO_BQ and (unc_flag == True or p.UNC_ADJUST == False):
                uf.write_to_bq(
                    prexisting_plan_liability_df,
                    project_output = p.BQ_OUTPUT_PROJECT_ID,
                    dataset_output = p.BQ_OUTPUT_DATASET,
                    table_id = "Prexisting_PlanLiability",
                    client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                    timestamp_param = p.TIMESTAMP,
                    run_id = p.AT_RUN_ID,
                    schema = None # TODO: create schema
                )
            else:
                prexisting_plan_liability_df.to_csv(p.FILE_OUTPUT_PATH + 'Prexisting_PlanLiability_' + str(month) + '_' + str(p.TIMESTAMP) + '.csv', index=False)

        ##Last month recast
        # these have zeros for the brand-offset performance
        # These calculations use the same spend/awp for client and pharmacy as we used to use dummy pharmacy approximations
        # We don't ever use these recasts in the code, so this does not currently affect anything downstream
        if p.UNC_CLIENT:
            lp_data_output_df['LAST_MONTH_REIMB_Old'] = lp_data_output_df.LM_QTY_OLDUNC * lp_data_output_df[lag_price_col]
            lp_data_output_df = add_target_ingcost(lp_data_output_df, client_guarantees, client_rate_col = 'RATE', target_cols=['LM_TARG_INGCOST_ADJ_OLDUNC', 'LM_FULLAWP_ADJ'])
            last_month_recast_old = calculatePerformance(lp_data_output_df, client_guarantees, pharmacy_guarantees,
                                                       client_list, p.AGREEMENT_PHARMACY_LIST, oc_pharm_dummy, dummy_perf_dict,
                                                       dummy_perf_dict, dummy_perf_dict, dummy_perf_dict,
                                                       client_reimb_column='LAST_MONTH_REIMB_Old', pharm_reimb_column='LAST_MONTH_REIMB_Old',
                                                       client_TARG_column='LM_TARG_INGCOST_ADJ_OLDUNC', pharm_TARG_column='LM_FULLAWP_ADJ_OLDUNC')
        else:
            lp_data_output_df['LAST_MONTH_REIMB_Old'] = lp_data_output_df.LM_QTY * lp_data_output_df[lag_price_col]
            last_month_recast_old = calculatePerformance(lp_data_output_df, client_guarantees, pharmacy_guarantees,
                                                       client_list, p.AGREEMENT_PHARMACY_LIST, oc_pharm_dummy, dummy_perf_dict,
                                                       dummy_perf_dict, dummy_perf_dict, dummy_perf_dict,
                                                       client_reimb_column='LAST_MONTH_REIMB_Old', pharm_reimb_column='LAST_MONTH_REIMB_Old',
                                                       client_TARG_column='LM_TARG_INGCOST_ADJ', pharm_TARG_column='LM_FULLAWP_ADJ')

        lp_data_output_df['LAST_MONTH_REIMB_New'] = lp_data_output_df.LM_QTY * lp_data_output_df.New_Price.round(4)
        lp_data_output_df = add_target_ingcost(lp_data_output_df, client_guarantees, client_rate_col = 'RATE', target_cols=['LM_TARG_INGCOST_ADJ'])
        last_month_recast_new = calculatePerformance(lp_data_output_df, client_guarantees, pharmacy_guarantees,
                                                   client_list, p.AGREEMENT_PHARMACY_LIST, oc_pharm_dummy, dummy_perf_dict,
                                                   dummy_perf_dict, dummy_perf_dict, dummy_perf_dict,
                                                   client_reimb_column='LAST_MONTH_REIMB_New', pharm_reimb_column='LAST_MONTH_REIMB_New',
                                                   client_TARG_column='LM_TARG_INGCOST_ADJ', pharm_TARG_column='LM_FULLAWP_ADJ')

        lp_data_output_df['Final_Price'] = lp_data_output_df['CURRENT_MAC_PRICE'].where(lp_data_output_df.PRICE_MUTABLE == 0, lp_data_output_df['Rounded_Price'])
        
        ### EVA ADHOC WS CHANGE ###
        # anywhere where is_specialty and is_mac cols are used after this point
        lp_data_output_df['IS_SPECIALTY'] = False
        lp_data_output_df.loc[lp_data_output_df['IMMUTABLE_REASON'] == 'SPECIALTY_EXCLUSION', 'IS_SPECIALTY'] = True
        
        if p.UNC_OPT:
             ### EVA ADHOC WS CHANGE ###
            # anywhere where is_specialty and is_mac cols are used after this point
            lp_vol_mv_agg_df_actual['IS_SPECIALTY'] = False
            lp_vol_mv_agg_df_actual.loc[lp_vol_mv_agg_df_actual['IMMUTABLE_REASON'] == 'SPECIALTY_EXCLUSION', 'IS_SPECIALTY'] = True
        
        comp_score = lp_data_output_df.copy()
        comp_score['CLAIM_PRICE_RATIO'] = comp_score['CLAIMS'] * (comp_score['Final_Price']/comp_score['SOFT_CONST_BENCHMARK_PRICE'])        
        comp_score = (comp_score.groupby(['CLIENT','REGION', 'BREAKOUT', 'CHAIN_GROUP', 'CHAIN_SUBGROUP',
                                          'MEASUREMENT','BG_FLAG','PHARMACY_TYPE', 'IS_MAC', 'IS_SPECIALTY'])[['CLAIMS','Final_Price','SOFT_CONST_BENCHMARK_PRICE','CLAIM_PRICE_RATIO']].sum()).reset_index()
        comp_score['COMPETITIVE_SCORE'] = comp_score['CLAIM_PRICE_RATIO'] / comp_score['CLAIMS']
                                           
        comp_score = comp_score.drop(['Final_Price','SOFT_CONST_BENCHMARK_PRICE'], axis = 1)
        # dispense fees spend performance
        # calculating projections
        lp_data_output_df['DISP_FEE_PROJ_LAG'] = lp_data_output_df['CLAIMS_PROJ_LAG'] * lp_data_output_df['AVG_DISP_FEE']
        lp_data_output_df['DISP_FEE_PROJ_EOY'] = lp_data_output_df['CLAIMS_PROJ_EOY'] * lp_data_output_df['AVG_DISP_FEE']
        lp_data_output_df['TARGET_DISP_FEE_PROJ_LAG'] = lp_data_output_df['CLAIMS_PROJ_LAG'] * lp_data_output_df['AVG_TARGET_DISP_FEE']
        lp_data_output_df['TARGET_DISP_FEE_PROJ_EOY'] = lp_data_output_df['CLAIMS_PROJ_EOY'] * lp_data_output_df['AVG_TARGET_DISP_FEE']
        lp_data_output_df['PHARM_DISP_FEE_PROJ_LAG'] = lp_data_output_df['PHARM_CLAIMS_PROJ_LAG'] * lp_data_output_df['PHARM_AVG_DISP_FEE']
        lp_data_output_df['PHARM_DISP_FEE_PROJ_EOY'] = lp_data_output_df['PHARM_CLAIMS_PROJ_EOY'] * lp_data_output_df['PHARM_AVG_DISP_FEE']
        lp_data_output_df['PHARM_TARGET_DISP_FEE_PROJ_LAG'] = lp_data_output_df['PHARM_CLAIMS_PROJ_LAG'] * lp_data_output_df['PHARM_AVG_TARGET_DISP_FEE']
        lp_data_output_df['PHARM_TARGET_DISP_FEE_PROJ_EOY'] = lp_data_output_df['PHARM_CLAIMS_PROJ_EOY'] * lp_data_output_df['PHARM_AVG_TARGET_DISP_FEE']

        disp_fee_total = (lp_data_output_df.groupby(['CLIENT','REGION', 'BREAKOUT', 'CHAIN_GROUP','CHAIN_SUBGROUP','MEASUREMENT',
                                                                            'BG_FLAG','PHARMACY_TYPE'])[['CLAIMS','CLAIMS_PROJ_LAG','CLAIMS_PROJ_EOY',
                                                                                                            'DISP_FEE','TARGET_DISP_FEE',
                                                                                                           'DISP_FEE_PROJ_LAG','TARGET_DISP_FEE_PROJ_LAG',
                                                                                                           'DISP_FEE_PROJ_EOY','TARGET_DISP_FEE_PROJ_EOY',
                                                                                                            'PHARM_CLAIMS','PHARM_CLAIMS_PROJ_LAG','PHARM_CLAIMS_PROJ_EOY',
                                                                                                           'PHARMACY_DISP_FEE','PHARM_TARGET_DISP_FEE',
                                                                                                           'PHARM_DISP_FEE_PROJ_LAG','PHARM_TARGET_DISP_FEE_PROJ_LAG',
                                                                                                           'PHARM_DISP_FEE_PROJ_EOY','PHARM_TARGET_DISP_FEE_PROJ_EOY'
                                                                                                           ]].sum()).reset_index()

        if p.WRITE_OUTPUT:
            if p.UNC_OPT:
                lp_data_output_df = write_spend_total_df(lp_data_output_df, unc_flag, comp_score, lp_vol_mv_agg_df_actual)
            else:
                lp_data_output_df = write_spend_total_df(lp_data_output_df, unc_flag, comp_score)


            lp_data_output_df['RUN_ID'] = p.AT_RUN_ID
            total_output_columns.extend(['Final_Price'])

            # For Non UNC Clients: (WRITE_TO_BQ = True and unc_flag = False and UNC_ADJUST=False, hence below condition will evaluate to True
            # For UNC Clients:
            #   1st Iteration: (WRITE_TO_BQ = True) and unc_flag = False and UNC_ADJUST=True, hence below condition will evaluate to False
            #   2nd Iteration: (WRITE_TO_BQ = True) and unc_flag = True and UNC_ADJUST=True, hence below condition will evaluate to True
            if False: #if p.WRITE_TO_BQ and (unc_flag == True or p.UNC_ADJUST == False):
                lp_data_output_df.to_csv(os.path.join(p.FILE_OUTPUT_PATH, p.TOTAL_OUTPUT), index=False)
                # df = pd.read_csv(os.path.join(p.FILE_OUTPUT_PATH, p.TOTAL_OUTPUT), dtype = p.VARIABLE_TYPE_DIC, low_memory=False)
                lp_data_output_df = lp_data_output_df.rename(columns={'U&C_EBIT': 'UNC_EBIT', '1026_NDC_PRICE': 'num1026_NDC_PRICE','1026_GPI_PRICE': 'num1026_GPI_PRICE'})
                uf.write_to_bq(
                    lp_data_output_df,
                    project_output = p.BQ_OUTPUT_PROJECT_ID,
                    dataset_output = p.BQ_OUTPUT_DATASET,
                    table_id = "Total_Output_subgroup",
                    client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                    timestamp_param = p.TIMESTAMP,
                    run_id = p.AT_RUN_ID,
                    schema = None # TODO: create schema
                )
            else:
                try:
                    import cap_reasons as cr
                    if p.MAIL_MAC_UNRESTRICTED:
                        lp_data_output_df = cr.apply_cap_reasons(lp_data_output_df, p.MAIL_UNRESTRICTED_CAP, unc_flag, loglevel)
                    else:
                        lp_data_output_df = cr.apply_cap_reasons(lp_data_output_df, 1, unc_flag, loglevel)
                except Exception as e:
                    print(f"Adding capped reason columns to the output file failed with error - {e}")

                    
                lp_data_output_df.to_csv(os.path.join(p.FILE_OUTPUT_PATH, p.TOTAL_OUTPUT), index=False)

        if p.WRITE_OUTPUT:
            ## Create Projections month by month##

            if 'gs://' in p.FILE_OUTPUT_PATH:
                TEMP_WORK_DIR = 'temp_work_dir_' + str(dt.datetime.now())
                os.makedirs(TEMP_WORK_DIR, exist_ok=True)

            #initialize the dataframes
            monthly_proj_new = dict_to_df(performance_dict, ['ENTITY', 'THROUGH_MONTH_' + str(month-1)])
            monthly_proj_old = dict_to_df(performance_dict, ['ENTITY', 'THROUGH_MONTH_' + str(month-1)])

            #Set final month to iterate through
            end_month = 13
            
            subchain_df = lp_data_output_df.groupby(['CLIENT', 'CHAIN_GROUP', 'PHARMACY_PERF_NAME'], as_index=False)['FULLAWP_ADJ_PROJ_EOY'].sum()
            chain_df = lp_data_output_df.groupby(['CLIENT', 'CHAIN_GROUP'], as_index=False)['FULLAWP_ADJ_PROJ_EOY'].sum()
            subchain_df = subchain_df.merge(chain_df, on=['CLIENT', 'CHAIN_GROUP'], suffixes=('_SUBGROUP', '_GROUP'), how='inner')
            subchain_df['MULTIPLIER'] = 0
            subchain_df.loc[subchain_df.FULLAWP_ADJ_PROJ_EOY_GROUP>0, 'MULTIPLIER'] = (subchain_df.loc[subchain_df.FULLAWP_ADJ_PROJ_EOY_GROUP>0, 'FULLAWP_ADJ_PROJ_EOY_SUBGROUP']
                                                                                       /subchain_df.loc[subchain_df.FULLAWP_ADJ_PROJ_EOY_GROUP>0, 'FULLAWP_ADJ_PROJ_EOY_GROUP'])
            subchain_df.drop(columns=['FULLAWP_ADJ_PROJ_EOY_SUBGROUP', 'FULLAWP_ADJ_PROJ_EOY_GROUP'], inplace=True)

            for adj_month in range(month, end_month):

                days_in_month = calendar.monthrange(p.GO_LIVE.year, adj_month)[1]

                #get brand performance for the month
                brand_surplus_month_dict = dict()
                if isinstance(brand_surplus_ytd_dict,pd.DataFrame):
                    if len(brand_surplus_ytd_dict) <= 1:
                        brand_surplus_ytd_dict = brand_surplus_ytd_dict.set_index('ENTITY').squeeze(axis=1).to_dict()
                    else: 
                        brand_surplus_ytd_dict = brand_surplus_ytd_dict.set_index('ENTITY').squeeze().to_dict()
                if isinstance(brand_surplus_lag_dict,pd.DataFrame):
                    if len(brand_surplus_lag_dict) <= 1:
                        brand_surplus_lag_dict = brand_surplus_lag_dict.set_index('ENTITY').squeeze(axis=1).to_dict()
                    else: 
                        brand_surplus_lag_dict = brand_surplus_lag_dict.set_index('ENTITY').squeeze().to_dict()
                if isinstance(brand_surplus_eoy_dict,pd.DataFrame):
                    if len(brand_surplus_eoy_dict) <= 1:
                        brand_surplus_eoy_dict = brand_surplus_eoy_dict.set_index('ENTITY').squeeze(axis=1).to_dict()
                    else:
                        brand_surplus_eoy_dict = brand_surplus_eoy_dict.set_index('ENTITY').squeeze().to_dict()
                for key in brand_surplus_ytd_dict:
                    brand_surplus_month_dict[key] = (brand_surplus_ytd_dict[key] + brand_surplus_lag_dict[key] + brand_surplus_eoy_dict[key]) * (days_in_month)/365
                    
                #get specialty performance for the month
                specialty_surplus_month_dict = dict()
                if isinstance(specialty_surplus_ytd_dict,pd.DataFrame):
                    if len(specialty_surplus_ytd_dict) <= 1:
                        specialty_surplus_ytd_dict = specialty_surplus_ytd_dict.set_index('ENTITY').squeeze(axis=1).to_dict()
                    else:
                        specialty_surplus_ytd_dict = specialty_surplus_ytd_dict.set_index('ENTITY').squeeze().to_dict()
                if isinstance(specialty_surplus_lag_dict,pd.DataFrame):
                    if len(specialty_surplus_lag_dict) <= 1:
                        specialty_surplus_lag_dict = specialty_surplus_lag_dict.set_index('ENTITY').squeeze(axis=1).to_dict()
                    else: 
                        specialty_surplus_lag_dict = specialty_surplus_lag_dict.set_index('ENTITY').squeeze().to_dict()
                if isinstance(specialty_surplus_eoy_dict,pd.DataFrame):
                    if len(specialty_surplus_eoy_dict) <= 1:
                        specialty_surplus_eoy_dict = specialty_surplus_eoy_dict.set_index('ENTITY').squeeze(axis=1).to_dict()
                    else: 
                        specialty_surplus_lag_dict = specialty_surplus_lag_dict.set_index('ENTITY').squeeze().to_dict()
                for key in specialty_surplus_ytd_dict:
                    specialty_surplus_month_dict[key] = (specialty_surplus_ytd_dict[key] + specialty_surplus_lag_dict[key] + specialty_surplus_eoy_dict[key]) * (days_in_month)/365
                    
                #get dispensing fee performance for the month
                disp_fee_surplus_month_dict = dict()
                if isinstance(disp_fee_surplus_ytd_dict,pd.DataFrame):
                    if len(disp_fee_surplus_ytd_dict) <= 1:
                        disp_fee_surplus_ytd_dict = disp_fee_surplus_ytd_dict.set_index('ENTITY').squeeze(axis=1).to_dict()
                    else:
                        disp_fee_surplus_ytd_dict = disp_fee_surplus_ytd_dict.set_index('ENTITY').squeeze().to_dict()
                if isinstance(disp_fee_surplus_lag_dict,pd.DataFrame):
                    if len(disp_fee_surplus_lag_dict) <= 1:
                        disp_fee_surplus_lag_dict = disp_fee_surplus_lag_dict.set_index('ENTITY').squeeze(axis=1).to_dict()
                    else:
                        disp_fee_surplus_lag_dict = disp_fee_surplus_lag_dict.set_index('ENTITY').squeeze().to_dict()
                if isinstance(disp_fee_surplus_eoy_dict,pd.DataFrame):
                    if len(disp_fee_surplus_eoy_dict) <= 1:
                        disp_fee_surplus_eoy_dict = disp_fee_surplus_eoy_dict.set_index('ENTITY').squeeze(axis=1).to_dict()
                    else:
                        disp_fee_surplus_eoy_dict = disp_fee_surplus_eoy_dict.set_index('ENTITY').squeeze().to_dict()
                for key in disp_fee_surplus_ytd_dict:
                    disp_fee_surplus_month_dict[key] = (disp_fee_surplus_ytd_dict[key] + disp_fee_surplus_lag_dict[key] + disp_fee_surplus_eoy_dict[key]) * (days_in_month)/365
                
                ####LP mod change- added CHain_subgroup to the group by for calc performance 
                #get generic launch performance for that month
                gen_launch_month = generic_launch_df.loc[generic_launch_df.MONTH == adj_month].groupby(['CLIENT',
                                                                                                        'BREAKOUT',
                                                                                                        'REGION',
                                                                                                        'MEASUREMENT',
                                                                                                        'BG_FLAG',
                                                                                                        'CHAIN_GROUP',
                                                                                                        'CHAIN_SUBGROUP'])[['FULLAWP','ING_COST']].agg('sum').reset_index()
                gen_launch_month['PHARMACY_TYPE'] = gen_launch_month.apply(pharmacy_type_new, args=tuple([pref_pharm_list]), axis=1)
                gen_launch_month_dict = calculatePerformance(gen_launch_month, client_guarantees, pharmacy_guarantees,
                                                              client_list, p.AGREEMENT_PHARMACY_LIST, oc_pharm_dummy,dummy_perf_dict, dummy_perf_dict, dummy_perf_dict,
                                                               dummy_perf_dict, client_reimb_column='ING_COST', pharm_reimb_column='ING_COST',
                                                              client_TARG_column='FULLAWP', pharm_TARG_column='FULLAWP', other=False, subchain_df=subchain_df)
                # Cover for channels not present in generic launch data, Not originaly on PRO
                for breakout in breakout_df['Combined'].tolist():
                    if breakout not in gen_launch_month_dict:
                        gen_launch_month_dict[breakout] = 0

                #Get other client performance for that month
                oc_pharm_surplus['Month_' + str(adj_month)] = oc_pharm_surplus.SURPLUS * (days_in_month/proj_days)
                oc_month_pharm_perf = df_to_dict(oc_pharm_surplus, ['CHAIN_GROUP', 'Month_' + str(adj_month)])

                lp_data_output_df['QTY_MONTH_' + str(adj_month)] = lp_data_output_df.QTY_PROJ_EOY * (days_in_month/eoy_days)
                lp_data_output_df['FULLAWP_ADJ_MONTH_' + str(adj_month)] = lp_data_output_df.FULLAWP_ADJ_PROJ_EOY * (days_in_month/eoy_days)
                lp_data_output_df['TARG_INGCOST_ADJ_MONTH_' + str(adj_month)] = lp_data_output_df.TARG_INGCOST_ADJ_PROJ_EOY * (days_in_month/eoy_days)
                lp_data_output_df['NEW_REIMB_' + str(adj_month)] = lp_data_output_df['QTY_MONTH_' + str(adj_month)] * lp_data_output_df.EFF_CAPPED_PRICE_new.round(4)
                lp_data_output_df['PHARM_QTY_MONTH_' + str(adj_month)] = lp_data_output_df.QTY_PROJ_EOY * (days_in_month/eoy_days)
                lp_data_output_df['PHARM_FULLAWP_ADJ_MONTH_' + str(adj_month)] = lp_data_output_df.PHARM_FULLAWP_ADJ_PROJ_EOY * (days_in_month/eoy_days)
                lp_data_output_df['PHARM_TARG_INGCOST_ADJ_MONTH_' + str(adj_month)] = lp_data_output_df.PHARM_TARG_INGCOST_ADJ_PROJ_EOY * (days_in_month/eoy_days)
                lp_data_output_df['PHARM_NEW_REIMB_' + str(adj_month)] = lp_data_output_df['PHARM_QTY_MONTH_' + str(adj_month)] * lp_data_output_df.EFF_CAPPED_PRICE_new.round(4)


                if p.UNC_CLIENT:
                    lp_data_output_df = add_target_ingcost(lp_data_output_df, client_guarantees, client_rate_col = 'RATE', target_cols=['TARG_INGCOST_ADJ_PROJ_EOY_OLDUNC'])
                    lp_data_output_df['QTY_MONTH_OLDUNC' + str(adj_month)] = lp_data_output_df.QTY_PROJ_EOY_OLDUNC * (days_in_month / eoy_days)
                    lp_data_output_df['OLD_REIMB_' + str(adj_month)] = lp_data_output_df['QTY_MONTH_OLDUNC' + str(adj_month)] * lp_data_output_df[lag_price_col].round(4)
                    lp_data_output_df['FULLAWP_ADJ_MONTH_OLDUNC' + str(adj_month)] = lp_data_output_df.FULLAWP_ADJ_PROJ_EOY_OLDUNC * (days_in_month / eoy_days)
                    lp_data_output_df['TARG_INGCOST_ADJ_MONTH_OLDUNC' + str(adj_month)] = lp_data_output_df.TARG_INGCOST_ADJ_PROJ_EOY_OLDUNC * (days_in_month/eoy_days)

                    # Since all UNC claims count towards pharmacy guarantees (except at WAG), don't use the projections modified based on UNC logic
                    # This logic will need to be updated when the WAG UNC code changes are merged
                    lp_data_output_df['PHARM_QTY_MONTH_' + str(adj_month)] = lp_data_output_df.QTY_PROJ_EOY * (days_in_month / eoy_days)
                    lp_data_output_df['PHARM_OLD_REIMB_' + str(adj_month)] = lp_data_output_df['PHARM_QTY_MONTH_' + str(adj_month)] * lp_data_output_df[pharm_lag_price_col].round(4)
                    lp_data_output_df['PHARM_FULLAWP_ADJ_MONTH_' + str(adj_month)] = lp_data_output_df.PHARM_FULLAWP_ADJ_PROJ_EOY * (days_in_month / eoy_days)
                    lp_data_output_df['PHARM_TARG_INGCOST_ADJ_MONTH_' + str(adj_month)] = lp_data_output_df.PHARM_TARG_INGCOST_ADJ_PROJ_EOY * (days_in_month / eoy_days)
                else:
                    lp_data_output_df['OLD_REIMB_' + str(adj_month)] = lp_data_output_df['QTY_MONTH_' + str(adj_month)] * \
                                                                       lp_data_output_df[lag_price_col].round(4)
                    lp_data_output_df['PHARM_OLD_REIMB_' + str(adj_month)] = lp_data_output_df['PHARM_QTY_MONTH_' + str(adj_month)] * \
                                                                       lp_data_output_df[pharm_lag_price_col].round(4)

                effective_proj_month_new_performance_dict = calculatePerformance(lp_data_output_df, client_guarantees, pharmacy_guarantees,
                                                   client_list, p.AGREEMENT_PHARMACY_LIST, oc_month_pharm_perf, gen_launch_month_dict,
                                                   brand_surplus_month_dict, specialty_surplus_month_dict, disp_fee_surplus_month_dict,
                                                   client_reimb_column='NEW_REIMB_' + str(adj_month), pharm_reimb_column='PHARM_NEW_REIMB_' + str(adj_month),
                                                   client_TARG_column='TARG_INGCOST_ADJ_MONTH_' + str(adj_month), pharm_TARG_column='PHARM_TARG_INGCOST_ADJ_MONTH_' + str(adj_month))

                if p.UNC_CLIENT:
                    effective_proj_month_old_performance_dict = calculatePerformance(lp_data_output_df, client_guarantees, pharmacy_guarantees,
                                                   client_list, p.AGREEMENT_PHARMACY_LIST, oc_month_pharm_perf, gen_launch_month_dict,
                                                   brand_surplus_month_dict, specialty_surplus_month_dict, disp_fee_surplus_month_dict,
                                                   client_reimb_column='OLD_REIMB_' + str(adj_month), pharm_reimb_column='PHARM_OLD_REIMB_' + str(adj_month),
                                                   client_TARG_column='TARG_INGCOST_ADJ_MONTH_OLDUNC' + str(adj_month), pharm_TARG_column='PHARM_TARG_INGCOST_ADJ_MONTH_' + str(adj_month))
                    
                else:
                    effective_proj_month_old_performance_dict = calculatePerformance(lp_data_output_df, client_guarantees, pharmacy_guarantees,
                                                   client_list, p.AGREEMENT_PHARMACY_LIST, oc_month_pharm_perf, gen_launch_month_dict,
                                                   brand_surplus_month_dict, specialty_surplus_month_dict, disp_fee_surplus_month_dict,
                                                   client_reimb_column='OLD_REIMB_' + str(adj_month), pharm_reimb_column='PHARM_OLD_REIMB_' + str(adj_month),
                                                   client_TARG_column='TARG_INGCOST_ADJ_MONTH_' + str(adj_month), pharm_TARG_column='PHARM_TARG_INGCOST_ADJ_MONTH_' + str(adj_month))

                # Creat the dictionaries to output
                monthly_proj_columns = ['ENTITY', 'MONTH_' + str(adj_month)]

                monthly_proj_new = pd.merge(monthly_proj_new, dict_to_df(effective_proj_month_new_performance_dict, monthly_proj_columns), how='left', on=['ENTITY'])
                monthly_proj_old = pd.merge(monthly_proj_old, dict_to_df(effective_proj_month_old_performance_dict, monthly_proj_columns), how='left', on=['ENTITY'])

            # Output the monthly projections

            # For Non UNC Clients: (WRITE_TO_BQ = True and unc_flag = False and UNC_ADJUST=False, hence below condition will evaluate to True
            # For UNC Clients:
            #   1st Iteration: (WRITE_TO_BQ = True) and unc_flag = False and UNC_ADJUST=True, hence below condition will evaluate to False
            #   2nd Iteration: (WRITE_TO_BQ = True) and unc_flag = True and UNC_ADJUST=True, hence below condition will evaluate to True
            if p.WRITE_TO_BQ and (unc_flag == True or p.UNC_ADJUST == False):
                uf.write_to_bq(
                    monthly_proj_new,
                    project_output = p.BQ_OUTPUT_PROJECT_ID,
                    dataset_output = p.BQ_OUTPUT_DATASET,
                    table_id = "NEW_PRICES_MONTHLY_PROJECTIONS",
                    client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                    timestamp_param = p.TIMESTAMP,
                    run_id = p.AT_RUN_ID,
                    schema = None # TODO: create schema
                )
            #NOTE: this change is temporary. Until an iteration independent schema can be utilized for <new_prices_monthly_projections> table in BQ
            #else:
                #monthly_proj_new.to_csv(os.path.join(p.FILE_OUTPUT_PATH, 'NEW_PRICES_MONTHLY_PROJECTIONS_MONTHS_{}_THROUGH_{}'.format(month, end_month) + str(p.TIMESTAMP) + '.csv'), index=False)
            monthly_proj_new.to_csv(os.path.join(p.FILE_OUTPUT_PATH, 'NEW_PRICES_MONTHLY_PROJECTIONS_MONTHS_{}_THROUGH_{}'.format(month, end_month) + str(p.TIMESTAMP) + '.csv'), index=False)
            if p.WRITE_TO_BQ and (unc_flag == True or p.UNC_ADJUST == False):
                uf.write_to_bq(
                    monthly_proj_old,
                    project_output = p.BQ_OUTPUT_PROJECT_ID,
                    dataset_output = p.BQ_OUTPUT_DATASET,
                    table_id = "OLD_PRICES_MONTHLY_PROJECTIONS",
                    client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                    timestamp_param = p.TIMESTAMP,
                    run_id = p.AT_RUN_ID,
                    schema = None # TODO: create schema
                )
            #NOTE: this change is temporary. Until an iteration independent schema can be utilized for <old_prices_monthly_projections> table in BQ
            #else:
                #monthly_proj_old.to_csv(os.path.join(p.FILE_OUTPUT_PATH, 'OLD_PRICES_MONTHLY_PROJECTIONS_MONTHS_{}_THROUGH_{}'.format(month, end_month) + str(p.TIMESTAMP) + '.csv'), index=False)
            monthly_proj_old.to_csv(os.path.join(p.FILE_OUTPUT_PATH, 'OLD_PRICES_MONTHLY_PROJECTIONS_MONTHS_{}_THROUGH_{}'.format(month, end_month) + str(p.TIMESTAMP) + '.csv'), index=False)


            ## Output SPEND, AWP, and CLAIMS for YTD, LAG, and Implementation Period
            if month == p.LP_RUN[0]:
                groupby_columns = ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT','BG_FLAG','CHAIN_GROUP', 'CHAIN_SUBGROUP']

                ytd_data = lp_data_output_df.groupby(groupby_columns)[['PRICE_REIMB', 'FULLAWP_ADJ', 'CLAIMS', 'QTY']].agg(sum).reset_index()
                ytd_data.rename(columns = {'FULLAWP_ADJ': 'AWP'}, inplace=True)
                ytd_data['PERIOD'] = 'YTD'

                lag_data = lp_data_output_df.groupby(groupby_columns)[['LAG_REIMB', 'FULLAWP_ADJ_PROJ_LAG', 'CLAIMS_PROJ_LAG', 'QTY_PROJ_LAG']].agg(sum).reset_index()
                lag_data.rename(columns = {'LAG_REIMB': 'PRICE_REIMB',
                                           'FULLAWP_ADJ_PROJ_LAG': 'AWP',
                                           'CLAIMS_PROJ_LAG': 'CLAIMS',
                                           'QTY_PROJ_LAG': 'QTY'}, inplace=True)
                lag_data['PERIOD'] = 'LAG'

                imp_data = lp_data_output_df.groupby(groupby_columns)[['Price_Effective_Reimb_Proj', 'FULLAWP_ADJ_PROJ_EOY', 'CLAIMS_PROJ_EOY', 'QTY_PROJ_EOY']].agg(sum).reset_index()
                imp_data.rename(columns = {'Price_Effective_Reimb_Proj': 'PRICE_REIMB',
                                           'FULLAWP_ADJ_PROJ_EOY': 'AWP',
                                           'CLAIMS_PROJ_EOY': 'CLAIMS',
                                           'QTY_PROJ_EOY': 'QTY'}, inplace=True)
                imp_data['PERIOD'] = 'IMP_NEW'

                imp_old_data = lp_data_output_df.groupby(groupby_columns)[['Old_Price_Effective_Reimb_Proj_EOY', 'FULLAWP_ADJ_PROJ_EOY', 'CLAIMS_PROJ_EOY', 'QTY_PROJ_EOY']].agg(sum).reset_index()
                if p.UNC_OPT:
                    del imp_old_data['Old_Price_Effective_Reimb_Proj_EOY']
                    imp_old_data_actual = lp_vol_mv_agg_df_actual.groupby(groupby_columns)[['Old_Price_Effective_Reimb_Proj_EOY']].agg(sum).reset_index()
                    imp_old_data = pd.merge(imp_old_data,imp_old_data_actual, on = groupby_columns )
                    
                imp_old_data.rename(columns = {'Old_Price_Effective_Reimb_Proj_EOY': 'PRICE_REIMB',
                                           'FULLAWP_ADJ_PROJ_EOY': 'AWP',
                                           'CLAIMS_PROJ_EOY': 'CLAIMS',
                                           'QTY_PROJ_EOY': 'QTY'}, inplace=True)
                imp_old_data['PERIOD'] = 'IMP_ORIGINAL'

                sorting_bools = [True] * len(groupby_columns) + [False]
                full_spend_data = pd.concat([ytd_data, lag_data, imp_data, imp_old_data]).sort_values(by=(groupby_columns + ['PERIOD']), ascending=sorting_bools).reset_index(drop=True)
                if p.WRITE_TO_BQ and (unc_flag == True or p.UNC_ADJUST == False):
                    uf.write_to_bq(
                        full_spend_data,
                        project_output = p.BQ_OUTPUT_PROJECT_ID,
                        dataset_output = p.BQ_OUTPUT_DATASET,
                        table_id = "Spend_data_subgroup",
                        client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                        timestamp_param = p.TIMESTAMP,
                        run_id = p.AT_RUN_ID,
                        schema = None # TODO: create schema
                    )
                else:
                    full_spend_data.to_csv(os.path.join(p.FILE_OUTPUT_PATH, 'Spend_data_' + str(p.TIMESTAMP) + str(month) + '.csv'), index=False)

            ##########################################################################################################################
            # Create output to internally check constraints
            ##########################################################################################################################

            if p.WRITE_TO_BQ and (unc_flag == True or p.UNC_ADJUST == False):
                uf.write_to_bq(
                    lambda_output_df,
                    project_output = p.BQ_OUTPUT_PROJECT_ID,
                    dataset_output = p.BQ_OUTPUT_DATASET,
                    table_id = "Model_2_Performance",
                    client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                    timestamp_param = p.TIMESTAMP,
                    run_id = p.AT_RUN_ID,
                    schema = None # TODO: create schema
                )
            else:
                lambda_output_df.to_csv(os.path.join(p.FILE_OUTPUT_PATH, p.MODEL_02_PERFORMANCE_OUTPUT), index=False)

            columns_to_include = ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'MAC_LIST', 'GPI_NDC', 'GPI', 'NDC', 
                                  'BG_FLAG', 'OLD_MAC_PRICE', 'New_Price', 'Final_Price', 'PKG_SZ', 'QTY_PROJ_EOY', 'GPI_CHANGE_EXCEPT', 
                                  'FULLAWP_ADJ_PROJ_EOY','CLAIMS_PROJ_EOY', 'PRICE_MUTABLE', 'MAC1026_UNIT_PRICE']
            if p.WRITE_TO_BQ and (unc_flag == True or p.UNC_ADJUST == False):

                temp = lp_data_output_df.loc[lp_data_output_df.CURRENT_MAC_PRICE > 0, columns_to_include]
                uf.write_to_bq(
                    temp,
                    project_output = p.BQ_OUTPUT_PROJECT_ID,
                    dataset_output = p.BQ_OUTPUT_DATASET,
                    table_id = "Price_Check_Output_subgroup",
                    client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                    timestamp_param = p.TIMESTAMP,
                    run_id = p.AT_RUN_ID,
                    schema = None # TODO: create schema
                )
            else:
                lp_data_output_df.loc[
                    lp_data_output_df.CURRENT_MAC_PRICE > 0,
                    columns_to_include
                ].to_csv(os.path.join(p.FILE_OUTPUT_PATH, p.PRICE_CHECK_OUTPUT), index=False)


            ##########################################################################################################################
            # Creat output for pharmacy team
            ##########################################################################################################################

            columns_to_include = ['GPI_NDC', 'GPI', 'NDC', 'BG_FLAG', 'PKG_SZ', 'CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT',
                                  'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'MAC_LIST', 'PRICE_MUTABLE',
                                  'CLAIMS_PROJ_EOY', 'QTY_PROJ_EOY', 'FULLAWP_ADJ_PROJ_EOY', 'OLD_MAC_PRICE',
                                  'MAC1026_UNIT_PRICE', 'GPI_Strength', 'New_Price', 'lb', 'ub', 'LM_CLAIMS', 'LM_QTY',
                                  'LM_FULLAWP_ADJ', 'LM_PRICE_REIMB', 'PRICE_REIMB_CLAIM']
            if p.WRITE_TO_BQ and (unc_flag == True or p.UNC_ADJUST == False):
                temp = lp_data_output_df.loc[(lp_data_output_df.CURRENT_MAC_PRICE > 0), columns_to_include]
                uf.write_to_bq(
                    temp,
                    project_output = p.BQ_OUTPUT_PROJECT_ID,
                    dataset_output = p.BQ_OUTPUT_DATASET,
                    table_id = "MedD_LP_Algorithm_Pharmacy_Output_Month_subgroup",
                    client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                    timestamp_param = p.TIMESTAMP,
                    run_id = p.AT_RUN_ID,
                    schema = None # TODO: create schema
                )
            else:
                lp_data_output_df.loc[
                    (lp_data_output_df.CURRENT_MAC_PRICE > 0),
                    columns_to_include
                ].to_csv(os.path.join(p.FILE_OUTPUT_PATH, p.PHARMACY_OUTPUT), index=False)


            ##########################################################################################################################
            # Create formal output
            ##########################################################################################################################

            ## Output for Formal Purposes
            columns_to_include = ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'MAC_LIST', 'GPI_NDC', 'GPI', 'NDC',
                                  'BG_FLAG', 'OLD_MAC_PRICE', 'CURRENT_MAC_PRICE', 'PKG_SZ', 'PHARMACY_TYPE', 'Final_Price', 'QTY_PROJ_EOY']

            if p.OUTPUT_FULL_MAC:
                output_df = lp_data_output_df.loc[(lp_data_output_df.CURRENT_MAC_PRICE > 0), columns_to_include]
            
                if p.CLIENT_TYPE == 'MEDD' and p.FULL_YEAR and not p.EGWP:
                    output_df_NY = lp_data_output_df.loc[(lp_data_output_df.CURRENT_MAC_PRICE > 0)].copy(deep = True)
                    output_df = output_df_NY.loc[output_df_NY.NDC == '***********', columns_to_include]
            else:
                if p.FLOOR_PRICE:
                    floor_gpi = standardize_df(pd.read_csv(p.FILE_INPUT_PATH + p.FLOOR_GPI_LIST, dtype = p.VARIABLE_TYPE_DIC))
                    output_df = lp_data_output_df.loc[(lp_data_output_df.PRICE_MUTABLE == 1) | (lp_data_output_df.GPI.isin(floor_gpi.GPI)), columns_to_include]
                else:
                    output_df = lp_data_output_df.loc[(lp_data_output_df.PRICE_MUTABLE == 1), columns_to_include]

                output_df = pd.concat([output_df, lp_data_output_df.loc[(lp_data_output_df.CURRENT_MAC_PRICE != lp_data_output_df.OLD_MAC_PRICE) &
                                                                        (lp_data_output_df.CURRENT_MAC_PRICE > 0), columns_to_include]]).reset_index(drop=True)
                
            output_df.rename(columns={'PKG_SZ':'GPPC', 'NDC':'NDC11'}, inplace=True)

            output_df['MAC_LIST'] = p.APPLY_VCML_PREFIX + output_df['MAC_LIST'].astype(str)
            
            #Insert customer loop here
            for client in output_df.CLIENT.unique():
                excel_name = p.PRICE_CHANGES_OUTPUT_FILE
                if 'gs://' in p.FILE_OUTPUT_PATH:
                    writer = pd.ExcelWriter(os.path.join(TEMP_WORK_DIR, excel_name), engine='xlsxwriter')
                else:
                    writer = pd.ExcelWriter(os.path.join(p.FILE_OUTPUT_PATH, excel_name), engine='xlsxwriter')
                #Get the client specific output for old RxClaims upload and new TMAC upload
                client_output_df = output_df.loc[(output_df.CLIENT==client)].copy()
                client_tmac_output_df = output_df.loc[(output_df.CLIENT==client) & (output_df.MAC_LIST.isin(p.NDC_MAC_LISTS))].copy()

                #format package sizes for reading
                client_output_df['GPPC'] = client_output_df['GPPC'].astype(str)
                client_output_df.loc[client_output_df.GPPC=='0.0', 'GPPC'] = '********'

                # Create first page of output for RxUpload
                rx_upload = client_output_df.loc[~client_output_df.MAC_LIST.isin(p.NDC_MAC_LISTS), ['MAC_LIST', 'GPI', 'GPPC', 'NDC11',
                                                                                                    'OLD_MAC_PRICE', 'Final_Price']]
                rx_upload = rx_upload.groupby(['MAC_LIST', 'GPI', 'GPPC', 'NDC11', 'OLD_MAC_PRICE'])[['Final_Price']].agg(np.nanmin).reset_index()
                if p.TRUECOST_CLIENT or p.UCL_CLIENT:
                    if not p.GNRC_PRICE_FREEZE:
                        gnrc_rx_upload = client_output_df.loc[(~client_output_df.MAC_LIST.isin(p.NDC_MAC_LISTS)) & (client_output_df.BG_FLAG == 'G'),
                                                              ['MAC_LIST', 'GPI', 'GPPC', 'NDC11','OLD_MAC_PRICE', 'Final_Price']]
                        gnrc_rx_upload = gnrc_rx_upload.groupby(['MAC_LIST', 'GPI', 'GPPC', 'NDC11', 'OLD_MAC_PRICE'])[['Final_Price']].agg(np.nanmin).reset_index()
                    else:
                        schema = {'MAC_LIST': 'str', 'GPI': 'str', 'GPPC': 'str', 'NDC11': 'str', 'OLD_MAC_PRICE': 'float64', 'Final_Price': 'float64'}
                        gnrc_rx_upload = pd.DataFrame(columns=schema.keys()).astype(schema)
                    
                    if not p.BRND_PRICE_FREEZE:
                        brnd_rx_upload = client_output_df.loc[(~client_output_df.MAC_LIST.isin(p.NDC_MAC_LISTS)) & (client_output_df.BG_FLAG == 'B'),
                                                              ['MAC_LIST', 'GPI', 'GPPC', 'NDC11','OLD_MAC_PRICE', 'Final_Price']]
                        brnd_rx_upload = brnd_rx_upload.groupby(['MAC_LIST', 'GPI', 'GPPC', 'NDC11', 'OLD_MAC_PRICE'])[['Final_Price']].agg(np.nanmin).reset_index()
                        brnd_rx_upload = brnd_rx_upload.rename(columns={'OLD_MAC_PRICE':'OLD_BRND_MAC_PRICE','Final_Price':'Final_Brand_Price'})
                    else:
                        schema = {'MAC_LIST': 'str', 'GPI': 'str', 'GPPC': 'str', 'NDC11': 'str', 'OLD_BRND_MAC_PRICE': 'float64', 'Final_Brand_Price': 'float64'}
                        brnd_rx_upload = pd.DataFrame(columns=schema.keys()).astype(schema)
                    
                    rx_upload = pd.merge(gnrc_rx_upload,brnd_rx_upload,on=['MAC_LIST', 'GPI', 'GPPC', 'NDC11'],how='outer')
                    
                # RxClaim Upload specific column names and formatting
                if p.TRUECOST_CLIENT or p.UCL_CLIENT:
                    rx_upload.rename(columns={'MAC_LIST': 'MACLIST',
                              'Final_Price': 'GNRC_MACPRC',
                              'Final_Brand_Price':'BRND_MACPRC',
                              'OLD_MAC_PRICE': 'GNRC_CurrentMAC',
                              'OLD_BRND_MAC_PRICE':'BRND_CurrentMAC'
                                }, inplace=True)
                else:
                    # RxClaim Upload specific column names and formatting
                    rx_upload.rename(columns={'MAC_LIST': 'MACLIST',
                                          'Final_Price': 'MACPRC',
                                          'OLD_MAC_PRICE': 'Current MAC'}, inplace=True)

                ## Fix added at the request of Elena.  She only wants to see the one that change.
                if not p.OUTPUT_FULL_MAC:
                    if p.TRUECOST_CLIENT or p.UCL_CLIENT:
                        if p.GNRC_PRICE_FREEZE and not p.BRND_PRICE_FREEZE:
                            rx_upload = rx_upload.loc[(np.abs(rx_upload['BRND_MACPRC'] - rx_upload['BRND_CurrentMAC']) > 0.00009)]
                        elif not p.GNRC_PRICE_FREEZE and p.BRND_PRICE_FREEZE:
                            rx_upload = rx_upload.loc[(np.abs(rx_upload['GNRC_MACPRC'] - rx_upload['GNRC_CurrentMAC']) > 0.00009)]
                        else:
                            rx_upload = rx_upload.loc[(np.abs(rx_upload['GNRC_MACPRC'] - rx_upload['GNRC_CurrentMAC']) > 0.00009) | (np.abs(rx_upload['BRND_MACPRC'] - rx_upload['BRND_CurrentMAC']) > 0.00009)]                           
                    else:
                        rx_upload = rx_upload[np.abs(rx_upload['MACPRC'] - rx_upload['Current MAC']) > 0.00009]
                    
                ## Added to fit the format desired by Elena
                rx_upload['EFFDATE'] = p.GO_LIVE.strftime("%Y%m%d")   # Go live date
                rx_upload['TERMDATE'] = '20391231'
                
                if p.TRUECOST_CLIENT or p.UCL_CLIENT:
                    rx_upload = rx_upload[['MACLIST', 'GPI', 'GPPC', 'NDC11', 'EFFDATE', 'TERMDATE', 'GNRC_MACPRC', 'BRND_MACPRC',
                                           'GNRC_CurrentMAC','BRND_CurrentMAC']].sort_values(by=['MACLIST', 'GPI', 'GPPC', 'NDC11'])
                else:
                    rx_upload = rx_upload[['MACLIST', 'GPI', 'GPPC', 'NDC11', 'EFFDATE', 'TERMDATE', 'MACPRC', 'Current MAC']].sort_values(
                    by=['MACLIST', 'GPI', 'GPPC', 'NDC11'])
                
                # Feb-2025 runs - Loading the prices ensuring the respective VCML structure is preserved  
                if (p.TRUECOST_CLIENT or p.UCL_CLIENT) and not p.GNRC_PRICE_FREEZE:
                    vcml_df = uf.read_BQ_data(
                    BQ.vcml_reference,
                    project_id=p.BQ_INPUT_PROJECT_ID,
                    dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                    table_id = 'vcml_reference'+p.WS_SUFFIX,
                    customer = ', '.join(sorted(p.CUSTOMER_ID))
                     )
                    TRUs = vcml_df[vcml_df['VCML_ID'].str.contains(p.APPLY_VCML_PREFIX)]['VCML_ID'].str[3:].unique().tolist()
                    MACs = vcml_df[vcml_df['VCML_ID'].str.contains('MAC')]['VCML_ID'].str[3:].unique().tolist()
                    
                    rx_upload_TRU = rx_upload[rx_upload['MACLIST'].str[3:].isin(TRUs)]
                    
                    rx_upload_MAC = rx_upload[rx_upload['MACLIST'].str[3:].isin(MACs)]
                    rx_upload_MAC['MACLIST'] = 'MAC' + rx_upload['MACLIST'].str[3:]
                    
                    mac_list_df_fill = uf.read_BQ_data(BQ.mac_list_ADHOC_custom.format(_customer_id = uf.get_formatted_string(p.CUSTOMER_ID),
                                                  _project = p.BQ_INPUT_PROJECT_ID, 
                                                  _dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP, 
                                                  _table_id = 'mac_list',
                                                  _vcml_ref_table_id='vcml_reference'+p.WS_SUFFIX,
                                                  ), 
                                                  project_id = p.BQ_INPUT_PROJECT_ID, 
                                                  dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP, 
                                                  table_id = 'mac_list',
                                                  customer = ', '.join(sorted(p.CUSTOMER_ID)), 
                                                  vcml_ref_table_id='vcml_reference'+p.WS_SUFFIX,
                                                    custom=True)
                    mac_list_df_fill = standardize_df(mac_list_df_fill)
                    tru_price_list_fill = uf.read_BQ_data(
                        BQ.mac_list_tru, 
                        project_id = p.BQ_INPUT_PROJECT_ID, 
                        dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP, 
                        table_id = 'tru_price_list'+p.WS_SUFFIX, 
                        customer = ', '.join(sorted(p.CUSTOMER_ID)),  
                        mac = True,
                        vcml_ref_table_id='vcml_reference'+p.WS_SUFFIX)
                    tru_price_list_fill = standardize_df(tru_price_list_fill)
                    
                    mac_list_df_fill_temp = mac_list_df_fill.copy()
                    mac_list_df_fill_temp = mac_list_df_fill_temp.loc[mac_list_df_fill['MAC_LIST'].isin(MACs),
                                        ['MAC', 'GPI', 'PRICE', 'NDC', 'GPPC']].rename(columns={'MAC':'MACLIST', 
                                                                                                'PRICE':'MAC_PRICE', 'NDC': 'NDC11'})
                    mac_list_df_fill_temp = mac_list_df_fill_temp.merge(rx_upload_MAC[['MACLIST', 'GPI', 'GNRC_MACPRC', 'BRND_MACPRC', 'GNRC_CurrentMAC', 'BRND_CurrentMAC']], on=['MACLIST', 'GPI'], how='left')

                    # setting to old MAC price
                    mac_list_df_fill_temp['GNRC_CurrentMAC'] = mac_list_df_fill_temp['MAC_PRICE'].copy()
                    mac_list_df_fill_temp['EFFDATE'] = p.GO_LIVE.strftime("%Y%m%d")   # Go live date
                    mac_list_df_fill_temp['TERMDATE'] = '20391231'
                    
                    # check if any duplicate rows were created
                    assert len(mac_list_df_fill_temp) == len(mac_list_df_fill), "Duplicate rows being created in MAC price lists"
                    
                    # check if MAC VCML structure is preserved
                    assert mac_list_df_fill_temp[['MACLIST','GPI','NDC11','GPPC']].rename(columns = {'MACLIST':'MAC','NDC11':'NDC'}).equals(mac_list_df_fill[['MAC','GPI','NDC','GPPC']]), "MAC VCML structure changed. Please check the code"
                    
                    mac_list_df_fill = mac_list_df_fill_temp.loc[~mac_list_df_fill_temp['GNRC_MACPRC'].isna(), 
                                        ['MACLIST', 'GPI', 'GPPC', 'NDC11', 'EFFDATE', 'TERMDATE',
                                         'GNRC_MACPRC', 'BRND_MACPRC', 'GNRC_CurrentMAC', 'BRND_CurrentMAC']].copy()
                    
                    mac_list_df_fill['BRND_MACPRC'] = None
                    mac_list_df_fill['BRND_CurrentMAC'] = None
                    
                    # check if the same price is being loaded for common VCMLs
                    tru_prices = rx_upload_TRU[rx_upload_TRU['MACLIST'].str[3:].isin(MACs)]
                    tru_prices['MACLIST'] = tru_prices['MACLIST'].str[3:]
                    tru_prices = tru_prices.groupby(['MACLIST','GPI'])['GNRC_MACPRC'].mean().reset_index()
                    mac_prices = mac_list_df_fill[mac_list_df_fill['MACLIST'].str[3:].isin(TRUs)]
                    mac_prices['MACLIST'] = mac_prices['MACLIST'].str[3:]
                    mac_prices = mac_prices.groupby(['MACLIST','GPI'])['GNRC_MACPRC'].mean().reset_index()
                    mac_prices['GNRC_MACPRC'] = round(mac_prices['GNRC_MACPRC'].astype(float),4)
                    tru_prices['GNRC_MACPRC'] = round(tru_prices['GNRC_MACPRC'].astype(float),4)
                    
                    assert len(mac_prices.merge(tru_prices, how='inner',on=['MACLIST','GPI', 'GNRC_MACPRC'])) == len(mac_prices), "Same prices are not being loaded to common MAC and TRU VCMLS"
                    
                    rx_upload = pd.concat([rx_upload_TRU, mac_list_df_fill])
                    
                    ## jun-2025 price SX vcml (non-specialty)
                    if p.TRUECOST_CLIENT:
                        tru_price_list_fill_sx = tru_price_list_fill.copy()
                        tru_price_list_fill_sx = tru_price_list_fill_sx[tru_price_list_fill_sx['MAC_LIST'].isin(
                            [p.CUSTOMER_ID[0]+'SX'])].rename(
                            columns={'MAC':'MACLIST', 'GENERIC_PRICE':'GNRC_CurrentMAC', 'BRAND_PRICE':'BRND_CurrentMAC', 'NDC': 'NDC11'})

                        tru_price_list_fill_sx = tru_price_list_fill_sx.merge(
                            rx_upload_TRU.loc[rx_upload_TRU['MACLIST'].str[3:] == p.CUSTOMER_ID[0]+'1', 
                                              ['GPI', 'GPPC', 'GNRC_MACPRC', 'BRND_MACPRC']], on=['GPI'], how='left')

                        tru_price_list_fill_sx['EFFDATE'] = p.GO_LIVE.strftime("%Y%m%d")   # Go live date
                        tru_price_list_fill_sx['TERMDATE'] = '20391231'

                        tru_price_list_fill_sx = tru_price_list_fill_sx.loc[
                            (tru_price_list_fill_sx['GNRC_MACPRC'].notna()) | (tru_price_list_fill_sx['BRND_MACPRC'].notna()), ['MACLIST', 'GPI', 'GPPC', 'NDC11', 'EFFDATE', 'TERMDATE',
                                                                 'GNRC_MACPRC', 'BRND_MACPRC', 'GNRC_CurrentMAC', 'BRND_CurrentMAC']]
                        
                        rx_upload = pd.concat([rx_upload, tru_price_list_fill_sx])
                    
                    
                #Copy the prices from older VCMLs to newly created VCMLs based on the vcml crosswalk 
                if p.OUTPUT_FULL_MAC and p.CLIENT_TYPE == 'MEDD' and p.FULL_YEAR and not p.EGWP and p.GO_LIVE.year == 2025:
                    
                    vcml_crosswalk = standardize_df(pd.read_csv(p.FILE_INPUT_PATH + p.VCML_CROSSWALK_MEDD_2024, dtype = p.VARIABLE_TYPE_DIC))
                    
                    if p.CUSTOMER_ID[0] in list(vcml_crosswalk.CLIENT.unique()):
                        vcml_crosswalk = vcml_crosswalk[vcml_crosswalk.CLIENT == p.CUSTOMER_ID[0]]
                        vcml_crosswalk['OLD_SUFFIX'] = vcml_crosswalk['MAC_LIST'].str[len(p.CUSTOMER_ID[0]) + 3:]

                        assert len(vcml_crosswalk[vcml_crosswalk['NEW_MACLIST'].duplicated()]) == 0,\
                        'Duplicate mapping found for new maclists. Validate the crosswalk'

                        vcml_crosswalk  = vcml_crosswalk.groupby('OLD_SUFFIX')['NEW_MACLIST'].apply(list).reset_index()

                        rx_upload['OLD_SUFFIX'] = rx_upload['MACLIST'].str[len(p.CUSTOMER_ID[0]) + 3:]
                        rx_upload = pd.merge(rx_upload, vcml_crosswalk, on = ['OLD_SUFFIX'], how = 'left')
                        rx_upload = rx_upload.explode('NEW_MACLIST')
                        rx_upload['MACLIST'] = rx_upload['NEW_MACLIST']
                        rx_upload.drop(columns = ['NEW_MACLIST','OLD_SUFFIX'], inplace = True)
                    else:
                        print("WARNING: VCML Crosswalk not available for this client. Submitting prices only for the original vcmls")
                        
                rx_upload.to_excel(writer, sheet_name='RXC_MACLISTS', index=False)
                rx_upload['NAME'] = ''
                if p.TRUECOST_CLIENT or p.UCL_CLIENT:
                    rx_upload = rx_upload[['MACLIST', 'GPI', 'GPPC', 'NDC11', 'NAME', 'EFFDATE', 'TERMDATE', 
                                              'GNRC_MACPRC', 'BRND_MACPRC', 'GNRC_CurrentMAC','BRND_CurrentMAC']]
                else:
                    rx_upload = rx_upload[['MACLIST', 'GPI', 'GPPC', 'NDC11', 'NAME', 'EFFDATE', 'TERMDATE', 'MACPRC', 'Current MAC']]
                
                rx_upload.to_csv(p.FILE_OUTPUT_PATH + p.PRICE_CHANGE_FILE, sep ='\t', index = False)

                # Writing to BQ should always be the default. The try statement with in write_to_bq, will make it also work on prem.
                if p.WRITE_TO_BQ and (unc_flag == True or p.UNC_ADJUST == False):
                    if p.TRUECOST_CLIENT or p.UCL_CLIENT:
                        uf.write_to_bq(
                        rx_upload,
                        project_output = p.BQ_OUTPUT_PROJECT_ID,
                        dataset_output = p.BQ_OUTPUT_DATASET,
                        table_id = "LP_Price_Recomendations_UCL",
                        client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                        timestamp_param = p.TIMESTAMP,
                        run_id = p.AT_RUN_ID,
                        schema = None
                        )
                    else:
                        uf.write_to_bq(
                            rx_upload,
                            project_output = p.BQ_OUTPUT_PROJECT_ID,
                            dataset_output = p.BQ_OUTPUT_DATASET,
                            table_id = "LP_Price_Recomendations",
                            client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                            timestamp_param = p.TIMESTAMP,
                            run_id = p.AT_RUN_ID,
                            schema = None
                        )


                # Create TMAC upload if those files exist
                if len(client_tmac_output_df) > 0:
                    if p.READ_FROM_BQ:
                        new_package_sizes = uf.read_BQ_data(BQ.package_size_to_ndc, project_id = p.BQ_INPUT_PROJECT_ID, dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP, table_id = 'package_size_to_ndc' + p.WS_SUFFIX)
                    else:
                        new_package_sizes = pd.read_csv(os.path.join(p.FILE_INPUT_PATH, '20190530_NDC_TO_PKGSZ_MAPPING.csv'), dtype = p.VARIABLE_TYPE_DIC)
                    new_package_sizes= standardize_df(new_package_sizes.fillna(0))
                    qa_dataframe(new_package_sizes, dataset = '20190530_NDC_TO_PKGSZ_MAPPING_AT_{}'.format(os.path.basename(__file__)))
                    new_package_sizes.rename(columns={'NDC':'NDC11'}, inplace=True)
                    df_len = len(client_tmac_output_df)
                    client_tmac_output_df = pd.merge(client_tmac_output_df, new_package_sizes, how='left', on=['GPI', 'NDC11'])
                    client_tmac_output_df['GPPC'] = client_tmac_output_df.PACKSIZE.fillna(0)
                    assert(len(client_tmac_output_df) == df_len), "len(client_tmac_output_df) == df_len"
                    
                    client_tmac_output_df['PKG SIZE'] = client_tmac_output_df[['GPPC']].applymap(lambda x: '{0:09.2f}'.format(x))
                    client_tmac_output_df.loc[client_tmac_output_df['PKG SIZE'] == '000000.00', 'PKG SIZE'] = '999999.00'

                    client_tmac_output_df.loc[client_tmac_output_df['PKG SIZE'] != '999999.00', 'Rounded_Price'] = client_tmac_output_df.loc[client_tmac_output_df['PKG SIZE'] != '999999.00', 'CURRENT_MAC_PRICE'].apply(round_to)

                    groupby_columns = ['MAC_LIST', 'GPI', 'PKG SIZE']
                    client_tmac_grouped_df = client_tmac_output_df.groupby(by=groupby_columns)[['Rounded_Price']].agg(np.mean).reset_index()
                    client_tmac_grouped_df.Rounded_Price = client_tmac_grouped_df.Rounded_Price.apply(round_to)

                    # read in and merge on TMAC MAC info
                    tmac_mac_mapping = pd.read_csv(os.path.join(p.FILE_INPUT_PATH, p.TMAC_MAC_MAP_FILE), dtype = p.VARIABLE_TYPE_DIC) #MARCEL, OK
                    tmac_mac_mapping['RxClaim MAC List'] = tmac_mac_mapping['RxClaim MAC List'].str.strip()
                    client_tmac_mac_grouped = pd.merge(client_tmac_grouped_df, tmac_mac_mapping, how='left', left_on='MAC_LIST', right_on='RxClaim MAC List')
                    assert len(client_tmac_grouped_df) == len(client_tmac_mac_grouped), "len(client_tmac_grouped_df) == len(client_tmac_mac_grouped)"
                    assert len(client_tmac_mac_grouped.loc[client_tmac_mac_grouped['MAC List'].isna()]) == 0, "len(client_tmac_mac_grouped.loc[client_tmac_mac_grouped['MAC List'].isna()]) == 0"

                    # read in and format TMAC Drug info
                    tmac_drug_info = pd.read_csv(os.path.join(p.FILE_INPUT_PATH, p.TMAC_DRUG_FILE), dtype = p.VARIABLE_TYPE_DIC) #MARCEL, OK
                    tmac_drug_info.rename(columns={'MAC_GPI_CD':'GPI'}, inplace=True)
                    tmac_drug_info = standardize_df(tmac_drug_info)
                    qa_dataframe(tmac_drug_info, dataset = 'TMAC_DRUG_FILE_AT_{}'.format(os.path.basename(__file__)))
                    tmac_drug_info['PKG SIZE'] = tmac_drug_info[['MAC_PKG_SZ']].applymap(lambda x: '{0:09.2f}'.format(x))
                    tmac_drug_info.loc[tmac_drug_info['PKG SIZE']=='000999.00', 'PKG SIZE'] = '999999.00'

                    #merg on TMAC Drug info
                    tmac_drug_cols = ['MAC_GENERIC_NM', 'MAC_DOSAGE_FORM', 'MAC_STRENGTH', 'GPI', 'PKG SIZE', 'MAC_PSU']
                    full_tmac_info = pd.merge(client_tmac_mac_grouped, tmac_drug_info[tmac_drug_cols], how='left', on=['GPI', 'PKG SIZE'])
                    assert len(client_tmac_mac_grouped) == len(full_tmac_info), "len(client_tmac_mac_grouped) == len(full_tmac_info)"
                    assert len(full_tmac_info.loc[full_tmac_info.MAC_GENERIC_NM.isna()]) == 0, "len(full_tmac_info.loc[full_tmac_info.MAC_GENERIC_NM.isna()]) == 0"

                    full_tmac_info['Effective Date'] = p.GO_LIVE
                    full_tmac_info['Expiration Date'] = '9999-12-31'
                    full_tmac_info['DGC'] = 1
                    full_tmac_info['Rounded_Price'] = full_tmac_info.Rounded_Price.apply(round_to)
                    full_tmac_info['New MAC'] = full_tmac_info[['Rounded_Price']].applymap(lambda x: '{0:08.4f}'.format(x))

                    formatted_tmac = full_tmac_info[['MAC_GENERIC_NM', 'MAC_DOSAGE_FORM', 'MAC_STRENGTH',
                                                     'GPI', 'Price Source', 'Price Type', 'MAC List', 'PKG SIZE',
                                                     'MAC_PSU', 'DGC', 'New MAC', 'Effective Date', 'Expiration Date']]

                    formatted_tmac.rename(columns={'MAC_GENERIC_NM': 'Drug Name',
                                                   'MAC_DOSAGE_FORM': 'Dosage Form',
                                                   'MAC_STRENGTH': 'Strength'}, inplace=True)

                    formatted_tmac.to_excel(writer, sheet_name=str(p.CUSTOMER_ID) + 'Upload {}.{}.{}'.format(dt.date.today().day,
                                                                                                            dt.date.today().month,
                                                                                                            str(dt.date.today().year)[-2:]), index=False)

                # Create individual tabs for each region to see price changes
                reg_columns = ['REGION', 'OLD_MAC_PRICE', 'Final_Price', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'GPI', 'GPPC', 'NDC11', 'PHARMACY_TYPE']
                if p.TRUECOST_CLIENT or p.UCL_CLIENT:
                    reg_columns = ['REGION', 'OLD_MAC_PRICE', 'Final_Price', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'GPI', 'GPPC', 'NDC11', 'PHARMACY_TYPE']
                    if not p.GNRC_PRICE_FREEZE:
                        gnrc_client_output_df = client_output_df[client_output_df.BG_FLAG == 'G'][['REGION', 'OLD_MAC_PRICE', 'Final_Price', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'GPI', 'GPPC', 'NDC11', 'PHARMACY_TYPE']]
                        gnrc_client_output_df = gnrc_client_output_df.rename(columns={'OLD_MAC_PRICE':'Generic Current Price','Final_Price':'Generic New Price'})
                    else:
                        gnrc_client_output_df = pd.DataFrame(columns = ['REGION', 'Generic Current Price', 'Generic New Price', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'GPI', 'GPPC', 'NDC11', 'PHARMACY_TYPE'])
                    
                    if not p.BRND_PRICE_FREEZE:
                        brnd_client_output_df = client_output_df[client_output_df.BG_FLAG == 'B'][['REGION', 'OLD_MAC_PRICE', 'Final_Price', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'GPI', 'GPPC', 'NDC11', 'PHARMACY_TYPE']]
                        brnd_client_output_df = brnd_client_output_df.rename(columns={'OLD_MAC_PRICE':'Brand Current Price','Final_Price':'Brand New Price'})
                    else:
                        brnd_client_output_df = pd.DataFrame(columns = ['REGION', 'Brand Current Price', 'Brand New Price', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'GPI', 'GPPC', 'NDC11', 'PHARMACY_TYPE'])
                    
                    total_output_df = pd.merge(gnrc_client_output_df,brnd_client_output_df,on=['REGION', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'GPI', 'GPPC', 'NDC11', 'PHARMACY_TYPE'],how='outer')
                    for region in client_output_df.REGION.unique():
                        reg_df = total_output_df.loc[total_output_df.REGION==region].drop(columns=['REGION'])
                        reg_output = pd.pivot_table(reg_df, index=['GPI', 'GPPC', 'NDC11'], columns=['PHARMACY_TYPE', 'CHAIN_GROUP', 'CHAIN_SUBGROUP'], 
                                                    values=['Generic Current Price', 'Generic New Price','Brand Current Price','Brand New Price'], aggfunc=np.nanmax)

                        reg_output.to_excel(writer, sheet_name=region, index=True)
                    writer.save()
                else:
                    reg_full_df = client_output_df[reg_columns]
                    reg_full_df.rename(columns = {'OLD_MAC_PRICE': 'Current Price', 'Final_Price': 'New Price'}, inplace=True)
                    for region in client_output_df.REGION.unique():
                        reg_df = reg_full_df.loc[reg_full_df.REGION==region].drop(columns=['REGION'])
                        reg_output = pd.pivot_table(reg_df, index=['GPI', 'GPPC', 'NDC11'], columns=['PHARMACY_TYPE', 'CHAIN_GROUP', 'CHAIN_SUBGROUP'], values=['Current Price', 'New Price'], aggfunc=np.nanmax)

                        reg_output.to_excel(writer, sheet_name=region, index=True)

                    writer.save()
                # write file to cloud storage
                if 'gs://' in p.FILE_OUTPUT_PATH:
                    bucket = p.FILE_OUTPUT_PATH[5:].split('/', 1)[0]
                    local_fpath = os.path.join(TEMP_WORK_DIR, excel_name)
                    cloud_path = os.path.join(p.FILE_OUTPUT_PATH, excel_name)
                    assert os.path.exists(local_fpath), f'Path not found: {local_fpath}'
                    logger.info(f'Uploading file {excel_name} to cloud path: {cloud_path}')
                    uf.upload_blob(bucket, local_fpath, cloud_path)

                #Create plan liaiblity full output
                if p.INCLUDE_PLAN_LIABILITY:
                    # Create data frame with both the new and old plan costs (ICL + GAP +CAT)
                    new_old_price = pd.merge(df_plan_new_proj_performance_eoy, df_plan_old__proj_performance_eoy, how='inner', on=None, left_index=True, right_index=True)
                    new_old_price['PLAN_COST_NEW'] = new_old_price['ICL_Cost_x'] + new_old_price['GAP_Cost_x'] + new_old_price['CAT_Cost_x']
                    new_old_price['PLAN_COST_OLD'] = new_old_price['ICL_Cost_y'] + new_old_price['GAP_Cost_y'] + new_old_price['CAT_Cost_y']

                    # Create new ingredient costs
                    lp_data_output_df['OLD_INGREDIENT_COST'] = lp_data_output_df['QTY_PROJ_EOY'] * lp_data_output_df[lag_price_col]
                    lp_data_output_df['NEW_INGREDIENT_COST'] = lp_data_output_df['QTY_PROJ_EOY'] * lp_data_output_df['Eff_capped_price_new']

                    # Filter on QTY_PROJ_EOY > 0 and MEASUREMENT not equal to MAIL30
                    new_old_price = new_old_price.loc[(new_old_price.QTY_PROJ_EOY_x > 0) & (new_old_price.MEASUREMENT_x != 'M30')]

                    # Function that merges all final output data and writes to CSV in p.FILE_OUTPUT_PATH
                    _ = generatePlanLiabilityOutput(lp_data_output_df, new_old_price, lag_price_col, temp_work_dir=TEMP_WORK_DIR)

                # Create U&C pricing output if used
                columns_to_include = ['CLIENT', 'MEASUREMENT', 'CHAIN_GROUP', 'GPI', 'NDC', 'BG_FLAG','RAISED_PRICE_UC', 'IS_TWOSTEP_UNC', 'IS_MAINTENANCE_UC', 'MATCH_VCML', 'UNC_FRAC_OLD', 'UNC_FRAC', 'QTY', 'UNC_VALUE', 'UNC_VALUE_CLIENT', 'UNC_VALUE_PHARM']
                if p.UNC_OPT:
                    unc_upload = lp_data_output_df.loc[lp_data_output_df.PRICE_CHANGED_UC, columns_to_include]

                    unc_upload['UNC_QTY_FRAC'] = unc_upload['QTY']*unc_upload['UNC_FRAC']
                    unc_upload['UNC_QTY_FRAC_OLD'] = unc_upload['QTY']*unc_upload['UNC_FRAC_OLD']
                    unc_upload = unc_upload.groupby(['CLIENT', 'MEASUREMENT', 'CHAIN_GROUP', 'GPI', 'NDC', 'BG_FLAG', 'RAISED_PRICE_UC', 'IS_TWOSTEP_UNC'], as_index=False).sum()
                    unc_upload['UNC_FRAC'] = unc_upload['UNC_QTY_FRAC']/unc_upload['QTY']
                    unc_upload['UNC_FRAC_OLD'] = unc_upload['UNC_QTY_FRAC_OLD']/unc_upload['QTY']
                    # if p.WRITE_TO_BQ and len(unc_upload)>0:
                    #     uf.write_to_bq(
                    #         unc_upload[['CLIENT', 'MEASUREMENT', 'CHAIN_GROUP', 'GPI', 'NDC', 'RAISED_PRICE_UC', 'IS_TWOSTEP_UNC', 'IS_MAINTENANCE_UC', 'MATCH_VCML', 'UNC_FRAC_OLD', 'UNC_FRAC', 'UNC_VALUE', 'UNC_VALUE_CLIENT', 'UNC_VALUE_PHARM']],
                    #         project_output = p.BQ_OUTPUT_PROJECT_ID,
                    #         dataset_output = p.BQ_OUTPUT_DATASET,
                    #         table_id = "UNC_Price_Targets",
                    #         client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                    #         timestamp_param = p.TIMESTAMP,
                    #         run_id = p.AT_RUN_ID,
                    #         schema = None
                    #     )

            ########################################################################################################################
            # Create the Leakage Report for ZBD_OPT and LEAKAGE_OPT runs.
            # A positive number in LEAKAGE_AVOID is good; a negative number is bad, suggesting increased leakage after a run.
            ########################################################################################################################
            
            if p.LOCKED_CLIENT:
                # Read in list of GPIs that were handled and filter for only those GPIs at in-scope pharmacies
                if p.LEAKAGE_LIST != 'All':
                    gpi_list = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + 'LEAKAGE_GPI_DF_{}.csv'.format(p.DATA_ID), dtype=p.VARIABLE_TYPE_DIC)
                    gpi_list_combined = gpi_list[['GPI', 'BG_FLAG']].apply(tuple, axis=1).tolist()
                    leakage_df = lp_data_output_df.copy()
                    lscope_mask = ((leakage_df['CHAIN_GROUP'].isin(gnrc_noncap_cogs) & 
                                   (leakage_df[['GPI', 'BG_FLAG']].apply(tuple, axis=1).isin(gpi_list_combined)) & 
                                   (leakage_df['GPI_ONLY'] == 1)))
                                   
                else:
                    leakage_df = lp_data_output_df.copy()
                    lscope_mask = (((leakage_df['CHAIN_GROUP'].isin(gnrc_noncap_cogs)) & (leakage_df['GPI_ONLY'] == 1)))                
                
                # Calculate pre-existing leakage using old prices. Due to aggregation, we need to weight the calculations by the different plan design weights.
                leakage_df.loc[lscope_mask, 'LEAKAGE_PRE_COPAY_ONLY'] = (leakage_df['COPAY_ONLY_QTY_WT'][lscope_mask] * 
                                                                         ((leakage_df[['AVG_COPAY_UNIT','OLD_MAC_PRICE']][lscope_mask].min(axis = 1) - 
                                                                           leakage_df['MAC1026_UNIT_PRICE'][lscope_mask]).clip(lower = 0) * 
                                                                          leakage_df['QTY_PROJ_EOY'][lscope_mask]))
                
                leakage_df.loc[lscope_mask, 'LEAKAGE_PRE_COINS_ONLY'] = (leakage_df['COINS_ONLY_QTY_WT'][lscope_mask] * 
                                                                         ((leakage_df['OLD_MAC_PRICE'][lscope_mask] *
                                                                           leakage_df['AVG_COINS'][lscope_mask] - 
                                                                           leakage_df['MAC1026_UNIT_PRICE'][lscope_mask]).clip(lower = 0) * 
                                                                          leakage_df['QTY_PROJ_EOY'][lscope_mask]))
                
                leakage_df.loc[lscope_mask, 'LEAKAGE_PRE_COPAY_COINS'] = (leakage_df['COPAY_COINS_QTY_WT'][lscope_mask] * 
                                                                         (((leakage_df[['AVG_COPAY_UNIT','OLD_MAC_PRICE']][lscope_mask].min(axis = 1) - 
                                                                           leakage_df['MAC1026_UNIT_PRICE'][lscope_mask]).clip(lower = 0) * 
                                                                          leakage_df['QTY_PROJ_EOY'][lscope_mask]) +   
                                                                         (((leakage_df['OLD_MAC_PRICE'][lscope_mask] -
                                                                           leakage_df['AVG_COPAY_UNIT'][lscope_mask]).clip(lower = 0) *
                                                                          leakage_df['QTY_PROJ_EOY'][lscope_mask]) *
                                                                          leakage_df['AVG_COINS'][lscope_mask])))
                                                                         
                leakage_df.loc[lscope_mask,'LEAKAGE_PRE'] = (leakage_df['LEAKAGE_PRE_COPAY_ONLY'][lscope_mask] +
                                                             leakage_df['LEAKAGE_PRE_COINS_ONLY'][lscope_mask] +
                                                             leakage_df['LEAKAGE_PRE_COPAY_COINS'][lscope_mask])
                
                # Calculate post-optimization leakage using new prices. Due to aggregation, we need to weight the calculations by the different plan design weights.
                leakage_df.loc[lscope_mask, 'LEAKAGE_POST_COPAY_ONLY'] = (leakage_df['COPAY_ONLY_QTY_WT'][lscope_mask] * 
                                                                          ((leakage_df[['AVG_COPAY_UNIT','Final_Price']][lscope_mask].min(axis = 1) - 
                                                                            leakage_df['MAC1026_UNIT_PRICE'][lscope_mask]).clip(lower = 0) * 
                                                                           leakage_df['QTY_PROJ_EOY'][lscope_mask]))
                
                leakage_df.loc[lscope_mask, 'LEAKAGE_POST_COINS_ONLY'] = (leakage_df['COINS_ONLY_QTY_WT'][lscope_mask] * 
                                                                          ((leakage_df['Final_Price'][lscope_mask] *
                                                                            leakage_df['AVG_COINS'][lscope_mask] - 
                                                                            leakage_df['MAC1026_UNIT_PRICE'][lscope_mask]).clip(lower = 0) * 
                                                                           leakage_df['QTY_PROJ_EOY'][lscope_mask]))
                
                leakage_df.loc[lscope_mask, 'LEAKAGE_POST_COPAY_COINS'] = (leakage_df['COPAY_COINS_QTY_WT'][lscope_mask] * 
                                                                          (((leakage_df[['AVG_COPAY_UNIT','Final_Price']][lscope_mask].min(axis = 1) - 
                                                                            leakage_df['MAC1026_UNIT_PRICE'][lscope_mask]).clip(lower = 0) * 
                                                                           leakage_df['QTY_PROJ_EOY'][lscope_mask]) +   
                                                                          (((leakage_df['Final_Price'][lscope_mask] -
                                                                            leakage_df['AVG_COPAY_UNIT'][lscope_mask]).clip(lower = 0) *
                                                                           leakage_df['QTY_PROJ_EOY'][lscope_mask]) *
                                                                           leakage_df['AVG_COINS'][lscope_mask])))
                                                                         
                leakage_df.loc[lscope_mask,'LEAKAGE_POST'] = (leakage_df['LEAKAGE_POST_COPAY_ONLY'][lscope_mask] +
                                                              leakage_df['LEAKAGE_POST_COINS_ONLY'][lscope_mask] +
                                                              leakage_df['LEAKAGE_POST_COPAY_COINS'][lscope_mask])
                
                # Calculate amount of leakage we avoided. Positive values are good, negative values mean we've added leakage post-LP.
                leakage_df.loc[lscope_mask,'LEAKAGE_AVOID'] = leakage_df['LEAKAGE_PRE'][lscope_mask] - leakage_df['LEAKAGE_POST'][lscope_mask]
                
                # Sum up how complete plan design data was in order to inform value reporting
                leakage_df.loc[lscope_mask,'PLAN_DESIGN_AVAIL_AWP'] = (leakage_df['FULLAWP_ADJ_PROJ_EOY'][lscope_mask] * leakage_df['COPAY_ONLY_AWP_WT'][lscope_mask] +
                                                                       leakage_df['FULLAWP_ADJ_PROJ_EOY'][lscope_mask] * leakage_df['COINS_ONLY_AWP_WT'][lscope_mask] +
                                                                       leakage_df['FULLAWP_ADJ_PROJ_EOY'][lscope_mask] * leakage_df['COPAY_COINS_AWP_WT'][lscope_mask])

                # Save reports to file
                leakage_df.loc[lscope_mask].to_csv(os.path.join(p.FILE_OUTPUT_PATH, 'LEAKAGE_DATASET_{}.csv'.format(p.DATA_ID)), index=False)
                
                leakage_report = leakage_df.loc[lscope_mask].groupby(['CLIENT','MEASUREMENT','BREAKOUT','CHAIN_GROUP','BG_FLAG']).agg({'LEAKAGE_PRE':'sum',
                                                                                                                             'LEAKAGE_POST':'sum',
                                                                                                                             'LEAKAGE_AVOID':'sum',
                                                                                                                             'PLAN_DESIGN_AVAIL_AWP':'sum',
                                                                                                                             'FULLAWP_ADJ_PROJ_EOY':'sum'}).reset_index()
                leakage_report.to_csv(os.path.join(p.FILE_OUTPUT_PATH, 'LEAKAGE_REPORT_{}.csv'.format(p.DATA_ID)), index=False)
                      
            if 'gs://' in p.FILE_OUTPUT_PATH:
                shutil.rmtree(TEMP_WORK_DIR)  # cleanup

        logger.info('*******END LP*******')
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'LP Output', repr(e), error_loc)
        raise e


if __name__ == '__main__':
    import os
    import pickle
    import logging
    import shutil
    import importlib
    import util_funcs as uf
    import BQ
    from datetime import datetime
    import CPMO_shared_functions as sf
        
    # get args (including parameters) from user
    cli_args = arguments()
    # directory support
    tstamper = datetime.now().strftime('%Y%m%d_%H%M')
    temp_workdir = 'temp_workdir_' + tstamper
    os.makedirs(temp_workdir, exist_ok=True)

    # import parameters
    pfilename = os.path.join(os.path.dirname(__file__), f'CPMO_parameters.py')
    with open(pfilename, 'w') as pfile:
        pfile.write(cli_args.params)
    import CPMO_parameters as p
    

    sf.check_and_create_folder(p.FILE_OUTPUT_PATH)
    sf.check_and_create_folder(p.FILE_DYNAMIC_INPUT_PATH)
    sf.check_and_create_folder(p.FILE_LP_PATH)

    # setup logger (for this notebook pipeline runner)
    log_path = os.path.join(p.FILE_LOG_PATH, f'ClientPharmacyMacOptimization_{tstamper}.log')
    logger = uf.log_setup(log_file_path=log_path, loglevel = cli_args.loglevel)    
    
    logger.info("Log level: {}".format(cli_args.loglevel))
    logger.info("Log path: {}".format(log_path))
    
    ## Setting a flag to account for local notebook UNC Adjustment run
    unc_flag=False
    
    ## If UNC Adjustment is required, CPMO needs to run twice
    if p.UNC_ADJUST:
        n=2
        print("This Client requires UNC Adjustment")
    else:
        n=1
    
    ## Looping over CPMO accordingly if UNC Adjustment is required or not
    for i in range(n): 
        for m in range(len(p.LP_RUN)):
            month = p.LP_RUN[m]
            prep = opt_preprocessing(
                m, '',
                unc_flag,
                eoy_days_out = os.path.join(temp_workdir, 'eoy_days.pkl'),
                proj_days_out = os.path.join(temp_workdir, 'proj_days.pkl'),
                lp_vol_mv_agg_df_out = os.path.join(temp_workdir, 'lp_vol_mv_agg_df.pkl'),
                mac_list_df_out = os.path.join(temp_workdir, 'mac_list_df.pkl'),
                chain_region_mac_mapping_out = os.path.join(temp_workdir, 'chain_region_mac_mapping.pkl'),
                lp_vol_mv_agg_df_actual_out = os.path.join(temp_workdir, 'lp_vol_mv_agg_df_actual.pkl'),
                client_list_out = os.path.join(temp_workdir, 'client_list.pkl'),
                breakout_df_out = os.path.join(temp_workdir, 'breakout_df.pkl'),
                client_guarantees_out = os.path.join(temp_workdir, 'client_guarantees.pkl'),
                pharmacy_guarantees_out = os.path.join(temp_workdir, 'pharmacy_guarantees.pkl'),
                pref_pharm_list_out = os.path.join(temp_workdir, 'pref_pharm_list.pkl'),
                # non_capped_pharmacy_list_out = os.path.join(temp_workdir, 'non_capped_pharmacy_list.pkl'),
                # agreement_pharmacy_list_out = os.path.join(temp_workdir, 'agreement_pharmacy_list.pkl'),
                oc_pharm_surplus_out = os.path.join(temp_workdir, 'oc_pharm_surplus.pkl'),
                other_client_pharm_lageoy_out = os.path.join(temp_workdir, 'other_client_pharm_lageoy.pkl'),
                oc_eoy_pharm_perf_out = os.path.join(temp_workdir, 'oc_eoy_pharm_perf.pkl'),
                generic_launch_df_out = os.path.join(temp_workdir, 'generic_launch_df.pkl'),
                oc_pharm_dummy_out = os.path.join(temp_workdir, 'oc_pharm_dummy.pkl'),
                dummy_perf_dict_out = os.path.join(temp_workdir, 'dummy_perf_dict.pkl'),
                perf_dict_col_out = os.path.join(temp_workdir, 'perf_dict_col.pkl'),
                gen_launch_eoy_dict_out = os.path.join(temp_workdir, 'gen_launch_eoy_dict.pkl'),
                gen_launch_lageoy_dict_out = os.path.join(temp_workdir, 'gen_launch_lageoy_dict.pkl'),
                ytd_perf_pharm_actuals_dict_out = os.path.join(temp_workdir, 'ytd_perf_pharm_actuals_dict.pkl'),
                performance_dict_out = os.path.join(temp_workdir, 'performance_dict.pkl'),
                act_performance_dict_out = os.path.join(temp_workdir, 'act_performance_dict.pkl'),
                # total_pharm_list_out = os.path.join(temp_workdir, 'total_pharm_list.pkl'),
                lp_data_df_out = os.path.join(temp_workdir, 'lp_data_df.pkl'),
                price_lambdas_out = os.path.join(temp_workdir, 'price_lambdas.pkl'),
                # brand-surplus files
                brand_surplus_ytd_out = os.path.join(temp_workdir, 'brand_surplus_ytd.pkl'),
                brand_surplus_lag_out = os.path.join(temp_workdir, 'brand_surplus_lag.pkl'),
                brand_surplus_eoy_out = os.path.join(temp_workdir, 'brand_surplus_eoy.pkl'),
                # specialty surplus files
                specialty_surplus_ytd_out = os.path.join(temp_workdir, 'specialty_surplus_ytd.pkl'),
                specialty_surplus_lag_out = os.path.join(temp_workdir, 'specialty_surplus_lag.pkl'),
                specialty_surplus_eoy_out = os.path.join(temp_workdir, 'specialty_surplus_eoy.pkl'),
                # dispensing fee surplus files
                disp_fee_surplus_ytd_out = os.path.join(temp_workdir, 'disp_fee_surplus_ytd.pkl'),
                disp_fee_surplus_lag_out = os.path.join(temp_workdir, 'disp_fee_surplus_lag.pkl'),
                disp_fee_surplus_eoy_out = os.path.join(temp_workdir, 'disp_fee_surplus_eoy.pkl'),
                loglevel = cli_args.loglevel
            )
            if month in p.LP_RUN:
                # t_cost, cons_strength_cons = 
                consistent_strength_pricing_constraints(
                    '',
                    unc_flag,
                    # lp_data_df, price_lambdas
                    lp_data_df_in = os.path.join(temp_workdir, 'lp_data_df.pkl'),
                    price_lambdas_in = os.path.join(temp_workdir, 'price_lambdas.pkl'),
                    lp_data_df_out = os.path.join(temp_workdir, 'lp_data_df.pkl'),
                    t_cost_out = os.path.join(temp_workdir, 't_cost.pkl'),
                    cons_strength_cons_out = os.path.join(temp_workdir, 'cons_strength_cons.pkl'),
                    loglevel = cli_args.loglevel
                )
                # lambda_df, client_constraint_list, client_constraint_target, lambda_decision_var = 
                client_level_constraints(
                    '',
                    unc_flag,
                    # lp_data_df, client_guarantees, pharmacy_guarantees, performance_dict, breakout_df, client_list, pharmacy_approx, 
                    # oc_eoy_pharm_perf, gen_launch_eoy_dict, price_lambdas
                    lp_data_df_in = os.path.join(temp_workdir, 'lp_data_df.pkl'),
                    client_guarantees_in = os.path.join(temp_workdir, 'client_guarantees.pkl'),
                    pharmacy_guarantees_in = os.path.join(temp_workdir, 'pharmacy_guarantees.pkl'),
                    performance_dict_in = os.path.join(temp_workdir, 'performance_dict.pkl'),
                    breakout_df_in = os.path.join(temp_workdir, 'breakout_df.pkl'),
                    client_list_in = os.path.join(temp_workdir, 'client_list.pkl'),
                    oc_eoy_pharm_perf_in = os.path.join(temp_workdir, 'oc_eoy_pharm_perf.pkl'),
                    gen_launch_eoy_dict_in = os.path.join(temp_workdir, 'gen_launch_eoy_dict.pkl'),
                    brand_surplus_eoy_in = os.path.join(temp_workdir, 'brand_surplus_eoy.pkl'),
                    specialty_surplus_eoy_in = os.path.join(temp_workdir, 'specialty_surplus_eoy.pkl'),
                    disp_fee_surplus_eoy_in = os.path.join(temp_workdir, 'disp_fee_surplus_eoy.pkl'),
                    price_lambdas_in = os.path.join(temp_workdir, 'price_lambdas.pkl'),
                    # total_pharm_list_in = os.path.join(temp_workdir, 'total_pharm_list.pkl'),
                    # agreement_pharmacy_list_in = os.path.join(temp_workdir, 'agreement_pharmacy_list.pkl'),
                    lambda_df_out = os.path.join(temp_workdir, 'lambda_df.pkl'),
                    client_constraint_list_out = os.path.join(temp_workdir, 'client_constraint_list.pkl'),
                    client_constraint_target_out = os.path.join(temp_workdir, 'client_constraint_target.pkl'),
                    loglevel = cli_args.loglevel
                )
                # pref_lt_non_pref_cons_list = 
                preferred_pricing_less_than_non_preferred_pricing_constraints(
                    '',
                    unc_flag,
                    # lp_data_df
                    lp_data_df_in = os.path.join(temp_workdir, 'lp_data_df.pkl'),
                    pref_pharm_list_in = os.path.join(temp_workdir, 'pref_pharm_list.pkl'),
                    # total_pharm_list_in = os.path.join(temp_workdir, 'total_pharm_list.pkl'),
                    pref_lt_non_pref_cons_list_out = os.path.join(temp_workdir, 'pref_lt_non_pref_cons_list.pkl'),
                    anomaly_gpi_out = os.path.join(temp_workdir, 'anomaly_gpi.pkl'),
                    loglevel = cli_args.loglevel
                )
                # meas_specific_price_cons_list = 
                specific_pricing_constraints(
                    '',
                    unc_flag,
                    # lp_data_df, total_pharm_list
                    lp_data_df_in = os.path.join(temp_workdir, 'lp_data_df.pkl'),
                    # total_pharm_list_in = os.path.join(temp_workdir, 'total_pharm_list.pkl'),
                    meas_specific_price_cons_list_out = os.path.join(temp_workdir, 'meas_specific_price_cons_list.pkl'),
                    anomaly_mes_gpi_out = os.path.join(temp_workdir, 'anomaly_mes_gpi.pkl'),
                    loglevel = cli_args.loglevel
                )
                # brnd_gnrc_price_cons_list = 
                brand_generic_pricing_constraints(
                    '',
                    unc_flag,
                    # lp_data_df, total_pharm_list
                    lp_data_df_in = os.path.join(temp_workdir, 'lp_data_df.pkl'),
                    # total_pharm_list_in = os.path.join(temp_workdir, 'total_pharm_list.pkl'),
                    brnd_gnrc_price_cons_list_out = os.path.join(temp_workdir, 'brnd_gnrc_price_cons_list.pkl'),
                    anomaly_bg_gpi_out = os.path.join(temp_workdir, 'anomaly_bg_gpi.pkl'),
                    loglevel = cli_args.loglevel
                )
                # adj_cap_price_cons_list_out = 
                adj_cap_constraints(
                    '',
                    unc_flag,
                    # lp_data_df, total_pharm_list
                    lp_data_df_in = os.path.join(temp_workdir, 'lp_data_df.pkl'),
                    # total_pharm_list_in = os.path.join(temp_workdir, 'total_pharm_list.pkl'),
                    adj_cap_price_cons_list_out = os.path.join(temp_workdir, 'adj_cap_price_cons_list.pkl'),
                    anomaly_adj_cap_gpi_out = os.path.join(temp_workdir, 'anomaly_adj_cap_gpi.pkl'),
                    loglevel = cli_args.loglevel
                )
                # pref_other_price_cons_list = 
                cvs_parity_price_constraint(
                    '',
                    unc_flag,
                    # lp_data_df, pref_pharm_list
                    lp_data_df_in = os.path.join(temp_workdir, 'lp_data_df.pkl'),
                    pref_pharm_list_in = os.path.join(temp_workdir, 'pref_pharm_list.pkl'),
                    pref_other_price_cons_list_out = os.path.join(temp_workdir, 'pref_other_price_cons_list.pkl'),
                    anomaly_pref_gpi_out = os.path.join(temp_workdir, 'anomaly_pref_gpi.pkl'),
                    loglevel = cli_args.loglevel
                )
                # parity_price_cons_list_out = 
                state_parity_constraints(
                    '',
                    unc_flag,
                    lp_data_df_in = os.path.join(temp_workdir, 'lp_data_df.pkl'), 
                    parity_price_cons_list_out = os.path.join(temp_workdir, 'parity_price_cons_list.pkl'),
                    anomaly_state_parity_gpi_out = os.path.join(temp_workdir, 'anomaly_state_parity_gpi.pkl'),
                    loglevel = cli_args.loglevel
                )
                # mac_cons_list = 
                consistent_mac_constraints(
                    '',
                    unc_flag,
                    # lp_data_df
                    lp_data_df_in = os.path.join(temp_workdir, 'lp_data_df.pkl'),
                    mac_cons_list_out = os.path.join(temp_workdir, 'mac_cons_list.pkl'),
                    anomaly_const_mac_out = os.path.join(temp_workdir, 'anomaly_const_mac.pkl'),
                    loglevel = cli_args.loglevel
                )
                # agg_mac_cons_list = 
                agg_mac_constraints(
                    '',
                    unc_flag,
                    month, 
                    # lp_data_df
                    lp_data_df_in = os.path.join(temp_workdir, 'lp_data_df.pkl'),
                    agg_mac_cons_list_out = os.path.join(temp_workdir, 'agg_mac_cons_list.pkl'),
                    loglevel = cli_args.loglevel
                )
                # eq_pkg_sz_cons_list = 
                equal_package_size_constraints(
                    '',
                    unc_flag,
                    # lp_data_df
                    lp_data_df_in = os.path.join(temp_workdir, 'lp_data_df.pkl'),
                    eq_pkg_sz_cons_list_out = os.path.join(temp_workdir, 'eq_pkg_sz_cons_list.pkl'),
                    anomaly_const_pkg_sz_out = os.path.join(temp_workdir, 'anomaly_const_pkg_sz.pkl'),
                    loglevel = cli_args.loglevel
                )
                # sm_diff_pkg_sz_cons_list = 
                same_difference_package_size_constraints(
                    '',
                    unc_flag,
                    # lp_data_df
                    lp_data_df_in = os.path.join(temp_workdir, 'lp_data_df.pkl'),
                    sm_diff_pkg_sz_cons_list_out = os.path.join(temp_workdir, 'sm_diff_pkg_sz_cons_list.pkl'),
                    loglevel = cli_args.loglevel
                )
                # sm_thera_class_cons_list = 
                same_therapeutic_constraints(
                    '',
                    unc_flag,
                    # lp_data_df
                    lp_data_df_in = os.path.join(temp_workdir, 'lp_data_df.pkl'),
                    sm_thera_class_cons_list_out = os.path.join(temp_workdir, 'sm_thera_class_cons_list.pkl'),
                    anomaly_sm_thera_gpi_out = os.path.join(temp_workdir, 'anomaly_sm_thera_gpi.pkl'),
                    loglevel = cli_args.loglevel
                )
                # leakage optimization
                if p.LEAKAGE_OPT and p.LOCKED_CLIENT:               
                    leakage_opt(
                        '',
                        lp_data_df_in = os.path.join(temp_workdir, 'lp_data_df.pkl'),
                        leakage_cost_list_out = os.path.join(temp_workdir, 'leakage_cost_list.pkl'),
                        leakage_const_list_out = os.path.join(temp_workdir, 'leakage_const_list.pkl'),
                        loglevel = cli_args.loglevel
                    )
                    
                #generate conflicting GPIs
                generate_conflict_gpi(
                    '', month,
                    unc_flag,
                    # t_cost, cons_strength_cons, client_constraint_list, client_constraint_target, pref_lt_non_pref_cons_list,
                    # meas_specific_price_cons_list, pref_other_price_cons_list, mac_cons_list, agg_mac_cons_list, eq_pkg_sz_cons_list,
                    # sm_diff_pkg_sz_cons_list, lambda_df, lp_data_df
                    t_cost_in = os.path.join(temp_workdir, 't_cost.pkl'),
                    cons_strength_cons_in = os.path.join(temp_workdir, 'cons_strength_cons.pkl'),
                    client_constraint_list_in = os.path.join(temp_workdir, 'client_constraint_list.pkl'),
                    client_constraint_target_in = os.path.join(temp_workdir, 'client_constraint_target.pkl'),
                    pref_lt_non_pref_cons_list_in = os.path.join(temp_workdir, 'pref_lt_non_pref_cons_list.pkl'),
                    meas_specific_price_cons_list_in = os.path.join(temp_workdir, 'meas_specific_price_cons_list.pkl'),
                    brnd_gnrc_price_cons_list_in = os.path.join(temp_workdir, 'brnd_gnrc_price_cons_list.pkl'),
                    adj_cap_price_cons_list_in = os.path.join(temp_workdir, 'adj_cap_price_cons_list.pkl'),
                    pref_other_price_cons_list_in = os.path.join(temp_workdir, 'pref_other_price_cons_list.pkl'),
                    parity_price_cons_list_in = os.path.join(temp_workdir, 'parity_price_cons_list.pkl'),
                    mac_cons_list_in = os.path.join(temp_workdir, 'mac_cons_list.pkl'),
                    agg_mac_cons_list_in = os.path.join(temp_workdir, 'agg_mac_cons_list.pkl'),
                    eq_pkg_sz_cons_list_in = os.path.join(temp_workdir, 'eq_pkg_sz_cons_list.pkl'),
                    sm_diff_pkg_sz_cons_list_in = os.path.join(temp_workdir, 'sm_diff_pkg_sz_cons_list.pkl'),
                    sm_thera_class_cons_list_in = os.path.join(temp_workdir, 'sm_thera_class_cons_list.pkl'),
                    leakage_cost_list_in = os.path.join(temp_workdir, 'leakage_cost_list.pkl'), 
                    leakage_const_list_in = os.path.join(temp_workdir, 'leakage_const_list.pkl'),  
                    lambda_df_in = os.path.join(temp_workdir, 'lambda_df.pkl'),
                    breakout_df_in = os.path.join(temp_workdir, 'breakout_df.pkl'),
                    # total_pharm_list_in = os.path.join(temp_workdir, 'total_pharm_list.pkl'),
                    lp_data_df_in = os.path.join(temp_workdir, 'lp_data_df.pkl'),
                    #lp_data_output_df_out = os.path.join(temp_workdir, 'lp_data_output_df.pkl'),
                    # pilot_output_columns_out = os.path.join(temp_workdir, 'pilot_output_columns.pkl'),
                    #lambda_output_df_out = os.path.join(temp_workdir, 'lambda_output_df.pkl'),
                    #total_output_columns_out = os.path.join(temp_workdir, 'total_output_columns.pkl'),
                    conflict_gpi_out = os.path.join(temp_workdir, 'conflict_gpi_out.pkl'),
                    lp_vol_mv_agg_df_in = os.path.join(temp_workdir, 'lp_vol_mv_agg_df.pkl'),
                    anomaly_gpi_in = os.path.join(temp_workdir, 'anomaly_gpi.pkl'),
                    anomaly_mes_gpi_in = os.path.join(temp_workdir, 'anomaly_mes_gpi.pkl'),
                    anomaly_bg_gpi_in = os.path.join(temp_workdir, 'anomaly_bg_gpi.pkl'),
                    anomaly_adj_cap_gpi_in = os.path.join(temp_workdir, 'anomaly_adj_cap_gpi.pkl'),
                    anomaly_pref_gpi_in = os.path.join(temp_workdir, 'anomaly_pref_gpi.pkl'),
                    anomaly_const_pkg_sz_in = os.path.join(temp_workdir, 'anomaly_const_pkg_sz.pkl'),
                    anomaly_state_parity_gpi_in = os.path.join(temp_workdir, 'anomaly_state_parity_gpi.pkl'),
                    anomaly_const_mac_in = os.path.join(temp_workdir, 'anomaly_const_mac.pkl'),
                    anomaly_sm_thera_gpi_in = os.path.join(temp_workdir, 'anomaly_sm_thera_gpi.pkl'),
                    loglevel = cli_args.loglevel
                )
                
                # lp_data_output_df, pilot_output_columns, lambda_output_df = 
                run_solver(
                    '', month,
                    unc_flag,
                    # t_cost, cons_strength_cons, client_constraint_list, client_constraint_target, pref_lt_non_pref_cons_list,
                    # meas_specific_price_cons_list, pref_other_price_cons_list, mac_cons_list, agg_mac_cons_list, eq_pkg_sz_cons_list,
                    # sm_diff_pkg_sz_cons_list, lambda_df, lp_data_df
                    t_cost_in = os.path.join(temp_workdir, 't_cost.pkl'),
                    cons_strength_cons_in = os.path.join(temp_workdir, 'cons_strength_cons.pkl'),
                    client_constraint_list_in = os.path.join(temp_workdir, 'client_constraint_list.pkl'),
                    client_constraint_target_in = os.path.join(temp_workdir, 'client_constraint_target.pkl'),
                    pref_lt_non_pref_cons_list_in = os.path.join(temp_workdir, 'pref_lt_non_pref_cons_list.pkl'),
                    meas_specific_price_cons_list_in = os.path.join(temp_workdir, 'meas_specific_price_cons_list.pkl'),
                    brnd_gnrc_price_cons_list_in = os.path.join(temp_workdir, 'brnd_gnrc_price_cons_list.pkl'),
                    adj_cap_price_cons_list_in = os.path.join(temp_workdir, 'adj_cap_price_cons_list.pkl'),
                    pref_other_price_cons_list_in = os.path.join(temp_workdir, 'pref_other_price_cons_list.pkl'),
                    parity_price_cons_list_in = os.path.join(temp_workdir, 'parity_price_cons_list.pkl'),
                    mac_cons_list_in = os.path.join(temp_workdir, 'mac_cons_list.pkl'),
                    agg_mac_cons_list_in = os.path.join(temp_workdir, 'agg_mac_cons_list.pkl'),
                    eq_pkg_sz_cons_list_in = os.path.join(temp_workdir, 'eq_pkg_sz_cons_list.pkl'),
                    sm_diff_pkg_sz_cons_list_in = os.path.join(temp_workdir, 'sm_diff_pkg_sz_cons_list.pkl'),
                    sm_thera_class_cons_list_in = os.path.join(temp_workdir, 'sm_thera_class_cons_list.pkl'),
                    leakage_cost_list_in = os.path.join(temp_workdir, 'leakage_cost_list.pkl'), 
                    leakage_const_list_in = os.path.join(temp_workdir, 'leakage_const_list.pkl'),  
                    lambda_df_in = os.path.join(temp_workdir, 'lambda_df.pkl'),
                    breakout_df_in = os.path.join(temp_workdir, 'breakout_df.pkl'),
                    # total_pharm_list_in = os.path.join(temp_workdir, 'total_pharm_list.pkl'),
                    lp_data_df_in = os.path.join(temp_workdir, 'lp_data_df.pkl'),
                    lp_data_output_df_out = os.path.join(temp_workdir, 'lp_data_output_df.pkl'),
                    # pilot_output_columns_out = os.path.join(temp_workdir, 'pilot_output_columns.pkl'),
                    lambda_output_df_out = os.path.join(temp_workdir, 'lambda_output_df.pkl'),
                    total_output_columns_out = os.path.join(temp_workdir, 'total_output_columns.pkl'),
                    lp_vol_mv_agg_df_in = os.path.join(temp_workdir, 'lp_vol_mv_agg_df.pkl'),
                    anomaly_gpi_in = os.path.join(temp_workdir, 'anomaly_gpi.pkl'),
                    anomaly_mes_gpi_in = os.path.join(temp_workdir, 'anomaly_mes_gpi.pkl'),
                    anomaly_bg_gpi_in = os.path.join(temp_workdir, 'anomaly_bg_gpi.pkl'),
                    anomaly_adj_cap_gpi_in = os.path.join(temp_workdir, 'anomaly_adj_cap_gpi.pkl'),
                    anomaly_pref_gpi_in = os.path.join(temp_workdir, 'anomaly_pref_gpi.pkl'),
                    anomaly_const_pkg_sz_in = os.path.join(temp_workdir, 'anomaly_const_pkg_sz.pkl'),
                    anomaly_state_parity_gpi_in = os.path.join(temp_workdir, 'anomaly_state_parity_gpi.pkl'),
                    anomaly_const_mac_in = os.path.join(temp_workdir, 'anomaly_const_mac.pkl'),
                    anomaly_sm_thera_gpi_in = os.path.join(temp_workdir, 'anomaly_sm_thera_gpi.pkl'),
                    loglevel = cli_args.loglevel
                )
            else:
                no_lp_run(
                    # lp_vol_mv_agg_df
                    lp_vol_mv_agg_df_in = os.path.join(temp_workdir, 'lp_vol_mv_agg_df.pkl'),
                    lp_data_output_df_out = os.path.join(temp_workdir, 'lp_data_output_df.pkl'),
                    loglevel = cli_args.loglevel
                )
            lp_output(
                '',
                # m,
                unc_flag,
                month,
                lag_price_col=prep[0],
                pharm_lag_price_col=prep[1],
                # lp_data_output_df, performance_dict, act_performance_dict,
                # ytd_perf_pharm_actuals_dict, client_list, client_guarantees, pharmacy_guarantees, oc_eoy_pharm_perf, 
                # gen_launch_eoy_dict, pharmacy_approx, eoy_days, perf_dict_col, mac_list_df, lp_vol_mv_agg_df_actual, 
                # oc_pharm_dummy, dummy_perf_dict, pharmacy_approx_dummy, pilot_output_columns, generic_launch_df, 
                # pref_pharm_list, breakout_df, oc_pharm_surplus, proj_days, lambda_output_df, chain_region_mac_mapping
                lp_data_output_df_in = os.path.join(temp_workdir, 'lp_data_output_df.pkl'),
                performance_dict_in = os.path.join(temp_workdir, 'performance_dict.pkl'),
                act_performance_dict_in = os.path.join(temp_workdir, 'act_performance_dict.pkl'),
                ytd_perf_pharm_actuals_dict_in = os.path.join(temp_workdir, 'ytd_perf_pharm_actuals_dict.pkl'),
                client_list_in = os.path.join(temp_workdir, 'client_list.pkl'),
                client_guarantees_in = os.path.join(temp_workdir, 'client_guarantees.pkl'),
                pharmacy_guarantees_in = os.path.join(temp_workdir, 'pharmacy_guarantees.pkl'),
                oc_eoy_pharm_perf_in = os.path.join(temp_workdir, 'oc_eoy_pharm_perf.pkl'),
                gen_launch_eoy_dict_in = os.path.join(temp_workdir, 'gen_launch_eoy_dict.pkl'),
                eoy_days_in = os.path.join(temp_workdir, 'eoy_days.pkl'),
                perf_dict_col_in = os.path.join(temp_workdir, 'perf_dict_col.pkl'),
                mac_list_df_in = os.path.join(temp_workdir, 'mac_list_df.pkl'),
                lp_vol_mv_agg_df_actual_in = os.path.join(temp_workdir, 'lp_vol_mv_agg_df_actual.pkl'),
                oc_pharm_dummy_in = os.path.join(temp_workdir, 'oc_pharm_dummy.pkl'),
                dummy_perf_dict_in = os.path.join(temp_workdir, 'dummy_perf_dict.pkl'),
                # pilot_output_columns_in = os.path.join(temp_workdir, 'pilot_output_columns.pkl'),
                generic_launch_df_in = os.path.join(temp_workdir, 'generic_launch_df.pkl'),
                pref_pharm_list_in = os.path.join(temp_workdir, 'pref_pharm_list.pkl'),
                breakout_df_in = os.path.join(temp_workdir, 'breakout_df.pkl'),
                oc_pharm_surplus_in = os.path.join(temp_workdir, 'oc_pharm_surplus.pkl'),
                proj_days_in = os.path.join(temp_workdir, 'proj_days.pkl'),
                lambda_output_df_in = os.path.join(temp_workdir, 'lambda_output_df.pkl'),
                chain_region_mac_mapping_in = os.path.join(temp_workdir, 'chain_region_mac_mapping.pkl'),
                total_output_columns_in = os.path.join(temp_workdir, 'total_output_columns.pkl'),
                # agreement_pharmacy_list_in = os.path.join(temp_workdir, 'agreement_pharmacy_list.pkl'),
                # non_capped_pharmacy_list_in = os.path.join(temp_workdir, 'non_capped_pharmacy_list.pkl'),
                brand_surplus_ytd_in = os.path.join(temp_workdir, 'brand_surplus_ytd.pkl'),
                brand_surplus_lag_in = os.path.join(temp_workdir, 'brand_surplus_lag.pkl'),
                brand_surplus_eoy_in = os.path.join(temp_workdir, 'brand_surplus_eoy.pkl'),
                specialty_surplus_ytd_in = os.path.join(temp_workdir, 'specialty_surplus_ytd.pkl'),
                specialty_surplus_lag_in = os.path.join(temp_workdir, 'specialty_surplus_lag.pkl'),
                specialty_surplus_eoy_in = os.path.join(temp_workdir, 'specialty_surplus_eoy.pkl'),
                disp_fee_surplus_ytd_in = os.path.join(temp_workdir, 'disp_fee_surplus_ytd.pkl'),
                disp_fee_surplus_lag_in = os.path.join(temp_workdir, 'disp_fee_surplus_lag.pkl'),
                disp_fee_surplus_eoy_in = os.path.join(temp_workdir, 'disp_fee_surplus_eoy.pkl'),
                loglevel = cli_args.loglevel
            )
        
        ## Setting unc flag to True to run CPMO again
        if p.UNC_ADJUST:
            unc_flag=True
            print("Starting UNC Adjustment")
        else: 
            print("This Client does not require UNC Adjustment")
        
    shutil.rmtree(temp_workdir)
    logger.info("Done.")
