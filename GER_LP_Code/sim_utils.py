import os
import importlib
import jinja2 as jj2
import datetime as dt
import random
import numpy as np
import socket
from pytz import timezone
from typing import NamedTuple, List
import json
import pandas as pd
from google.cloud import bigquery

def kubeflow_query_run(client, run_details, timeout, params_run_id) -> NamedTuple('Outputs', [('FLAG', bool), ('STATUS', str)]):
    '''
    Input: Kubeflow run client, details of the run, the number of seconds to go on before timing out, and the run_id as defined in the custom_params for the run
    This function queries the status of a kubeflow run. Returning True if the run was a success and False if it fails for any reason.
    It also uploads the status of each run to a BQ table so that the historical record can be preserved.
    Output: Boolean flag for success of run. Text status of run.
    '''
    import time
    from google.cloud import bigquery
    ##get pipeline run id
    run_id = run_details.run_id
    flag = False
    status = None
    try:
        #wait for run completion (or timeout) to generate run metrics
        resp = client.wait_for_run_completion(run_id, timeout = timeout)
        iter_status = resp.run.status
        run_details_latest = client.get_run(run_details.run_id )

        keys = ['id','name','created_at','scheduled_at','finished_at','status']
        status_dic = {k: run_details_latest.to_dict()['run'][k] for k in keys}
        status_dic['AT_RUN_ID'] = params_run_id#p.AT_RUN_ID


        status_df = pd.Series(status_dic).to_frame().T
        status_df.rename(columns={"id": "KUBERNETES_ID", "name": "KUBERNETES_RUN_NAME",
                                  'scheduled_at':'SCHEDULED_AT','finished_at':'FINISHED_AT',
                                  'status':'STATUS', 'created_at':'CREATED_AT'}, inplace = True)
        print(status_df)
        table_id = f"pbm-mac-lp-prod-ai.pricing_management.Kubeflow_run_status "
        bqclient = bigquery.Client()
        job_config = bigquery.LoadJobConfig(
                autodetect = True,
                write_disposition = bigquery.WriteDisposition.WRITE_APPEND
        )
        job = bqclient.load_table_from_dataframe(status_df, table_id, job_config = job_config)
        job.result()
    
    except TimeoutError as e:
        iter_status = 'Timeout'
    if iter_status == 'Succeeded':
        flag = True
    return(flag, iter_status)

def seq_sim_mac_list_transfer(mac_list_df, lp_output_df):
    import pandas as pd
    import numpy as np

    #MAC list transformation
    #NOTE: below, it is assumed that in lp_output_df, final_price is similar across GPI_NDC and MAC_LIST for different pharmacies
    temp_mac_price_df = lp_output_df[['GPI_NDC', 'MAC_LIST', 'Final_Price']].drop_duplicates()
    mac_list_df = mac_list_df.merge(temp_mac_price_df, on=['GPI_NDC', 'MAC_LIST'], how='left')
    mac_list_df['PRICE'] = np.where(mac_list_df['PRICE'] != mac_list_df['Final_Price'], mac_list_df['Final_Price'], mac_list_df['PRICE'])
    mac_list_df.drop('Final_Price', inplace = True, axis = 1)

    return mac_list_df

def seq_sim_lp_data_transfer(lp_input_df, lp_output_df, pre_ytd_date, pre_golive_date, new_ytd_date, new_golive_date):
    '''
    Input: LP_Input data of previous iteration, LP_Output data of previous iteration, previous Iteration Go-Live Date, next iteration YTD date, next iteration GO-Live date.
    Transforms lag data of an iteration's lp_input_df to ytd data of the next iteration. This is done to move from one go-live date to another in simulation mode. It uses some columns from the input_df of the previous iteration and some from the output_df of the previous iteration depending on the transformations taking place. Columns for whom replacements are created by the LP, such as EFF_Capped_price_new, are taken from the output_df of the previous iteration. Other columns are taken from the input_df of the previous iteration.
    Output: A new lp_input_df
    '''
    import os
    import sys
    sys.path.append('/home/jupyter/clientpharmacymacoptimization/GER_LP_Code')
    from GER_LP_Code.CPMO_shared_functions import unc_optimization, unc_ebit, determine_effective_price
    
    import GER_LP_Code.CPMO_parameters as p

    import pandas as pd
    import numpy as np

    #interval calculation
    eoy_date = dt.datetime.strptime('12/31/' + str(p.LAST_DATA.year), '%m/%d/%Y')
    pre_lag_days = (pre_golive_date - pre_ytd_date).days - 1
    pre_eoy_days = (eoy_date - pre_golive_date).days + 1
    new_lag_days = (new_golive_date - new_ytd_date).days - 1
    new_eoy_days = (eoy_date - new_golive_date).days + 1

    #define the new input lp_data
    #lp_input_df_new = pd.DataFrame(columns = lp_input_df.columns)
    lp_input_df_new = lp_input_df.copy(deep = True)

    #based on the input data, regular lp_data, lp_data_nounc, or lp_data_unc,
    #intersection of the columns in the <all_cols_to_copy> list and the input data column list is copied to new output
    potential_cols_to_copy = ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'GPI', 'CHAIN_GROUP', 'GO_LIVE', 'MAC_LIST', 'GPI_ONLY', 'NDC',
                              'GPI_NDC', 'PKG_SZ', 'PHARMACY_TYPE', 'PRICE_MUTABLE', 'CLIENT_RATE', 'PHARMACY_RATE', 'CUSTOMER_ID', 'VCML_ID',
                              'MLOR_CD', 'GNRCIND', 'GOODRX_UPPER_LIMIT']
    input_cols = lp_input_df.columns
    cols_to_copy = list(set(input_cols).intersection(set(potential_cols_to_copy)))
    lp_input_df_new[cols_to_copy] = lp_input_df[cols_to_copy]

    #claim transfer
    lp_input_df_new['CLAIMS'] = lp_input_df['CLAIMS'] + lp_output_df['CLAIMS_PROJ_LAG'] #use output because GPI backout zeros input claim projections in pre-processsing
    lp_input_df_new['CLAIMS_PROJ_LAG'] = lp_input_df['CLAIMS_PROJ_EOY'] * (new_lag_days / pre_eoy_days)
    lp_input_df_new['CLAIMS_PROJ_EOY'] = lp_input_df['CLAIMS_PROJ_EOY'] * (new_eoy_days / pre_eoy_days)
    lp_input_df_new['PHARM_CLAIMS'] = lp_input_df['PHARM_CLAIMS'] + lp_output_df['PHARM_CLAIMS_PROJ_LAG']
    lp_input_df_new['PHARM_CLAIMS_PROJ_LAG'] = lp_input_df['PHARM_CLAIMS_PROJ_EOY'] * (new_lag_days / pre_eoy_days)
    lp_input_df_new['PHARM_CLAIMS_PROJ_EOY'] = lp_input_df['PHARM_CLAIMS_PROJ_EOY'] * (new_eoy_days / pre_eoy_days)

    #quantity transfer
    lp_input_df_new['QTY'] = lp_input_df['QTY'] + lp_output_df['QTY_PROJ_LAG']
    lp_input_df_new['QTY_PROJ_LAG'] = lp_input_df['QTY_PROJ_EOY'] * (new_lag_days / pre_eoy_days)
    lp_input_df_new['QTY_PROJ_EOY'] = lp_input_df['QTY_PROJ_EOY'] * (new_eoy_days / pre_eoy_days)
    lp_input_df_new['PHARM_QTY'] = lp_input_df['PHARM_QTY'] + lp_output_df['PHARM_QTY_PROJ_LAG']
    lp_input_df_new['PHARM_QTY_PROJ_LAG'] = lp_input_df['PHARM_QTY_PROJ_EOY'] * (new_lag_days / pre_eoy_days)
    lp_input_df_new['PHARM_QTY_PROJ_EOY'] = lp_input_df['PHARM_QTY_PROJ_EOY'] * (new_eoy_days / pre_eoy_days)

    #full AWP transfer
    #nounc_ytd_new = nounc_ytd_old + nounc_lag_old
    lp_input_df_new['FULLAWP_ADJ'] = lp_input_df['FULLAWP_ADJ'] + lp_output_df['FULLAWP_ADJ_PROJ_LAG']
    lp_input_df_new['FULLAWP_ADJ_PROJ_LAG'] = lp_input_df['FULLAWP_ADJ_PROJ_EOY'] * (new_lag_days / pre_eoy_days)
    lp_input_df_new['FULLAWP_ADJ_PROJ_EOY'] = lp_input_df['FULLAWP_ADJ_PROJ_EOY'] * (new_eoy_days / pre_eoy_days)
    lp_input_df_new['PHARM_FULLAWP_ADJ'] = lp_input_df['PHARM_FULLAWP_ADJ'] + lp_output_df['PHARM_FULLAWP_ADJ_PROJ_LAG']
    lp_input_df_new['PHARM_FULLAWP_ADJ_PROJ_LAG'] = lp_input_df['PHARM_FULLAWP_ADJ_PROJ_EOY'] * (new_lag_days / pre_eoy_days)
    lp_input_df_new['PHARM_FULLAWP_ADJ_PROJ_EOY'] = lp_input_df['PHARM_FULLAWP_ADJ_PROJ_EOY'] * (new_eoy_days / pre_eoy_days)
    
    #price reimbursement transfer
    #NOTE: PRICE_REIMB_ADJ does not replicate the logic in Daily_Input_Read or CPMO.opt_preprocessing
    #try with a nonunc first
    lp_input_df_new['PRICE_REIMB'] = lp_input_df['PRICE_REIMB'] + lp_output_df['QTY_PROJ_LAG'] * lp_input_df['EFF_CAPPED_PRICE'].round(4)
    lp_input_df_new['PRICE_REIMB_ADJ'] = lp_input_df_new['QTY'] * lp_input_df['CURRENT_MAC_PRICE']
    lp_input_df_new['PRICE_REIMB_UNIT'] = np.where(lp_input_df_new['QTY'] != 0, lp_input_df_new['PRICE_REIMB'] / lp_input_df_new['QTY'], 0)
    lp_input_df_new['PHARM_PRICE_REIMB'] = lp_input_df['PHARM_PRICE_REIMB'] + lp_output_df['PHARM_QTY_PROJ_LAG'] * lp_input_df['EFF_CAPPED_PRICE'].round(4)
    lp_input_df_new['PHARM_PRICE_REIMB_ADJ'] = lp_input_df_new['PHARM_QTY'] * lp_input_df['CURRENT_MAC_PRICE']
    lp_input_df_new['PHARM_PRICE_REIMB_UNIT'] = np.where(lp_input_df_new['PHARM_QTY'] != 0, lp_input_df_new['PHARM_PRICE_REIMB'] / lp_input_df_new['PHARM_QTY'], 0)
    
    #last month data transfer
    #NOTE: currently, last month is equal to number of days in the month of go_live date, this might change to consider the last 30 (31, ...) days
    #if pre_golive_date.month != pre_ytd_date.month and pre_lag_days != 0:
    if pre_golive_date.month != pre_ytd_date.month:
        #if previous go_live date and last_data date are in different months, then new lm period consists of all the days in the last month of last_data date
        new_last_month_days = new_ytd_date.day
        lp_input_df_new['LM_CLAIMS'] = lp_output_df['CLAIMS_PROJ_LAG'] * (new_last_month_days / pre_lag_days) #for unc.opt this is different
        lp_input_df_new['LM_FULLAWP_ADJ'] = lp_output_df['FULLAWP_ADJ_PROJ_LAG'] * (new_last_month_days / pre_lag_days)
        lp_input_df_new['LM_QTY'] = lp_output_df['QTY_PROJ_LAG'] * (new_last_month_days / pre_lag_days)
        lp_input_df_new['LM_PRICE_REIMB'] = lp_input_df_new['LM_QTY'] * lp_input_df['CURRENT_MAC_PRICE']
        lp_input_df_new['LM_PHARM_CLAIMS'] = lp_output_df['PHARM_CLAIMS_PROJ_LAG'] * (new_last_month_days / pre_lag_days) #for unc.opt this is different
        lp_input_df_new['LM_PHARM_FULLAWP_ADJ'] = lp_output_df['PHARM_FULLAWP_ADJ_PROJ_LAG'] * (new_last_month_days / pre_lag_days)
        lp_input_df_new['LM_PHARM_QTY'] = lp_output_df['PHARM_QTY_PROJ_LAG'] * (new_last_month_days / pre_lag_days)
        lp_input_df_new['LM_PHARM_PRICE_REIMB'] = lp_input_df_new['LM_PHARM_QTY'] * lp_input_df['CURRENT_MAC_PRICE']
    #elif pre_lag_days == 0:
    #    lp_input_df_new['LM_CLAIMS'] = lp_input_df['LM_CLAIMS']
    #    lp_input_df_new['LM_FULLAWP_ADJ'] = lp_input_df['LM_FULLAWP_ADJ']
    #    lp_input_df_new['LM_QTY'] = lp_input_df['LM_QTY']
    #    lp_input_df_new['LM_PRICE_REIMB'] = lp_input_df['LM_PRICE_REIMB']
    else:
        #if previous go_live data and last_data date are in the same month, the new lm period consist of the old lm period + all the days in the old lag period
        lp_input_df_new['LM_CLAIMS'] = lp_input_df['LM_CLAIMS'] + lp_output_df['CLAIMS_PROJ_LAG'] #for unc.opt this is different
        lp_input_df_new['LM_FULLAWP_ADJ'] = lp_input_df['LM_FULLAWP_ADJ'] + lp_output_df['FULLAWP_ADJ_PROJ_LAG']
        lp_input_df_new['LM_QTY'] = lp_input_df['LM_QTY'] + lp_output_df['QTY_PROJ_LAG']
        lp_input_df_new['LM_PRICE_REIMB'] = lp_input_df['LM_PRICE_REIMB'] + lp_output_df['QTY_PROJ_LAG'] * lp_input_df['CURRENT_MAC_PRICE']
        lp_input_df_new['PHARM_LM_CLAIMS'] = lp_input_df['LM_PHARM_CLAIMS'] + lp_output_df['PHARM_CLAIMS_PROJ_LAG'] #for unc.opt this is different
        lp_input_df_new['PHARM_LM_FULLAWP_ADJ'] = lp_input_df['LM_PHARM_FULLAWP_ADJ'] + lp_output_df['PHARM_FULLAWP_ADJ_PROJ_LAG']
        lp_input_df_new['PHARM_LM_QTY'] = lp_input_df['LM_PHARM_QTY'] + lp_output_df['PHARM_QTY_PROJ_LAG']
        lp_input_df_new['PHARM_LM_PRICE_REIMB'] = lp_input_df['LM_PHARM_PRICE_REIMB'] + lp_output_df['PHARM_QTY_PROJ_LAG'] * lp_input_df['CURRENT_MAC_PRICE']
        
    #UC unit data transfer
    #NOTE: copying from input and not from output to prevent UC_UNIT25 multiplying by 500 everytime
    cols_to_copy = ['UC_UNIT', 'UC_UNIT25']
    lp_input_df_new[cols_to_copy] = lp_input_df[cols_to_copy]

    #AWP data transfer
    cols_to_copy = list(set(['CURR_AWP', 'CURR_AWP_MAX', 'CURR_AWP_MIN', 'BREAKOUT_AWP_MAX']).intersection(set(input_cols)))
    lp_input_df_new[cols_to_copy] = lp_input_df[cols_to_copy]
    lp_input_df_new['AVG_AWP'] = np.where(lp_input_df_new['QTY'] != 0, lp_input_df_new['FULLAWP_ADJ'] / lp_input_df_new['QTY'], 0)


    #1026 data transfer
    cols_to_copy = ['1026_GPI_PRICE', '1026_NDC_PRICE', 'MAC1026_GPI_FLAG', 'MAC1026_UNIT_PRICE']
    lp_input_df_new[cols_to_copy] = lp_input_df[cols_to_copy]

    #new price transfer
    lp_input_df_new['CURRENT_MAC_PRICE'] = lp_output_df['Final_Price']

    #effective price data transfer; replicate the process in Daily_Input_Read.py
    #lp_input_df_new['EFF_CAPPED_PRICE'] = lp_input_df_new.apply(determine_effective_price, args=tuple(['CURRENT_MAC_PRICE', 'UC_UNIT25', True]), axis=1)
    #lp_input_df_new['EFF_CAPPED_PRICE'] = np.where(lp_input_df_new['EFF_CAPPED_PRICE'] > 0, lp_input_df_new['EFF_CAPPED_PRICE'], lp_input_df_new['PRICE_REIMB_UNIT'])
    lp_input_df_new['EFF_CAPPED_PRICE'] = lp_output_df['EFF_CAPPED_PRICE_new']

    #lp_input_df_new['EFF_UNIT_PRICE'] = lp_input_df_new.apply(determine_effective_price, args=tuple(['CURRENT_MAC_PRICE']), axis=1)
    #lp_input_df_new['EFF_UNIT_PRICE'] = np.where(lp_input_df_new['EFF_UNIT_PRICE'] > 0, lp_input_df_new['EFF_UNIT_PRICE'], lp_input_df_new['PRICE_REIMB_UNIT'])
    lp_input_df_new['EFF_UNIT_PRICE'] = lp_output_df['EFF_UNIT_PRICE_new']

    #mac price data transfer; replicate the process in Daily_Input_Read.py
    lp_input_df_new['MAC_PRICE_UNIT_ADJ'] = np.where(lp_input_df_new['CURRENT_MAC_PRICE'] > 0, lp_input_df_new['CURRENT_MAC_PRICE'], lp_input_df_new['PRICE_REIMB_UNIT'])
    
    lp_input_df_new.fillna(value = 0, inplace = True)

    return lp_input_df_new

def seq_sim_unc_lp_data_transfer(lp_input_df, lp_output_df, pre_ytd_date, pre_golive_date, new_ytd_date, new_golive_date,
                                 iteration = 0, unc_flag = False, lp_vol_mv_agg_df_nounc = pd.DataFrame(), lp_input_df_unc = pd.DataFrame()):
    '''
    Input: LP_Input data of previous iteration, LP_Output data of previous iteration, previous iteration YTD date, previous Iteration Go-Live Date, next iteration YTD date, next iteration GO-Live date, iteration number, unc_flag, nounc_lp_data, unc_lp_data.
    Transforms lag data of an iteration's lp_input_df to ytd data of the next iteration. This is done to move from one go-live date to another in simulation mode. It uses some columns from the input_df of the previous iteration and some from the output_df of the previous iteration depending on the transformations taking place. Columns for whom replacements are created by the LP, such as EFF_Capped_price_new, are taken from the output_df of the previous iteration. Other columns are taken from the input_df of the previous iteration.
    This version is for when UNC_OPT is True
    Output: A new lp_input_df
    '''
    import os
    import sys
    sys.path.append('/home/jupyter/clientpharmacymacoptimization/GER_LP_Code')
    from GER_LP_Code.CPMO_shared_functions import unc_optimization, unc_ebit, determine_effective_price
    
    import GER_LP_Code.CPMO_parameters as p

    import pandas as pd
    import numpy as np

    #interval calculation
    eoy_date = dt.datetime.strptime('12/31/' + str(p.LAST_DATA.year), '%m/%d/%Y')
    pre_lag_days = (pre_golive_date - pre_ytd_date).days - 1
    pre_eoy_days = (eoy_date - pre_golive_date).days + 1
    new_lag_days = (new_golive_date - new_ytd_date).days - 1
    new_eoy_days = (eoy_date - new_golive_date).days + 1

    #define the new input lp_data
    #lp_input_df_new = pd.DataFrame(columns = lp_input_df.columns)
    lp_input_df_new = lp_input_df.copy(deep = True)

    #based on the input data, regular lp_data, lp_data_nounc, or lp_data_unc,
    #intersection of the columns in the <all_cols_to_copy> list and the input data column list is copied to new output
    potential_cols_to_copy = ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'GPI', 'CHAIN_GROUP', 'GO_LIVE', 'MAC_LIST', 'GPI_ONLY', 'NDC',
                              'GPI_NDC', 'PKG_SZ', 'PHARMACY_TYPE', 'PRICE_MUTABLE', 'CLIENT_RATE', 'PHARMACY_RATE', 'CUSTOMER_ID', 'VCML_ID',
                              'MLOR_CD', 'GNRCIND', 'GOODRX_UPPER_LIMIT']

    input_cols = lp_input_df.columns
    cols_to_copy = list(set(input_cols).intersection(set(potential_cols_to_copy)))
    lp_input_df_new[cols_to_copy] = lp_input_df[cols_to_copy]

    #NOTE: in the first iteration, and for transforming lp_data_nounc (new ytd = old ytd + lag), the lag data comes from the nounc output (or lp_vol_mv_agg_nounc)
    if unc_flag == False and iteration == 0:
        lp_input_df_new['CLAIMS'] = lp_input_df['CLAIMS'] + lp_vol_mv_agg_df_nounc['CLAIMS_PROJ_LAG']
        lp_input_df_new['QTY'] = lp_input_df['QTY'] + lp_vol_mv_agg_df_nounc['QTY_PROJ_LAG']
        lp_input_df_new['FULLAWP_ADJ'] = lp_input_df['FULLAWP_ADJ'] + lp_vol_mv_agg_df_nounc['FULLAWP_ADJ_PROJ_LAG']
        lp_input_df_new['PRICE_REIMB'] = lp_input_df['PRICE_REIMB'] + lp_vol_mv_agg_df_nounc['QTY_PROJ_LAG'] * lp_input_df['EFF_CAPPED_PRICE'].round(4)
        lp_input_df_new['PHARM_CLAIMS'] = lp_input_df['PHARM_CLAIMS'] + lp_vol_mv_agg_df_nounc['PHARM_CLAIMS_PROJ_LAG']
        lp_input_df_new['PHARM_QTY'] = lp_input_df['PHARM_QTY'] + lp_vol_mv_agg_df_nounc['PHARM_QTY_PROJ_LAG']
        lp_input_df_new['PHARM_FULLAWP_ADJ'] = lp_input_df['PHARM_FULLAWP_ADJ'] + lp_vol_mv_agg_df_nounc['PHARM_FULLAWP_ADJ_PROJ_LAG']
        lp_input_df_new['PHARM_PRICE_REIMB'] = lp_input_df['PHARM_PRICE_REIMB'] + lp_vol_mv_agg_df_nounc['PHARM_QTY_PROJ_LAG'] * lp_input_df['EFF_CAPPED_PRICE'].round(4)
        if pre_golive_date.month != pre_ytd_date.month and pre_lag_days != 0:
            #if pre_golive_date.month != pre_ytd_date.month:
            #if previous go_live date and last_data date are in different months, and lag <> 0, then new lm period consists of all the days in the last month of last_data date
            new_last_month_days = new_ytd_date.day
            lp_input_df_new['LM_CLAIMS'] = lp_vol_mv_agg_df_nounc['CLAIMS_PROJ_LAG'] * (new_last_month_days / pre_lag_days)
            lp_input_df_new['LM_FULLAWP_ADJ'] = lp_vol_mv_agg_df_nounc['FULLAWP_ADJ_PROJ_LAG'] * (new_last_month_days / pre_lag_days)
            lp_input_df_new['LM_QTY'] = lp_vol_mv_agg_df_nounc['QTY_PROJ_LAG'] * (new_last_month_days / pre_lag_days)
            lp_input_df_new['LM_PRICE_REIMB'] = lp_input_df_new['LM_QTY'] * lp_input_df['CURRENT_MAC_PRICE']
            lp_input_df_new['LM_PHARM_CLAIMS'] = lp_vol_mv_agg_df_nounc['PHARM_CLAIMS_PROJ_LAG'] * (new_last_month_days / pre_lag_days)
            lp_input_df_new['LM_PHARM_FULLAWP_ADJ'] = lp_vol_mv_agg_df_nounc['PHARM_FULLAWP_ADJ_PROJ_LAG'] * (new_last_month_days / pre_lag_days)
            lp_input_df_new['LM_PHARM_QTY'] = lp_vol_mv_agg_df_nounc['PHARM_QTY_PROJ_LAG'] * (new_last_month_days / pre_lag_days)
            lp_input_df_new['LM_PHARM_PRICE_REIMB'] = lp_input_df_new['LM_PHARM_QTY'] * lp_input_df['CURRENT_MAC_PRICE']
        elif pre_lag_days == 0:
            #if lag == 0, then, last month data stays the same because last data date does not change.
            lp_input_df_new['LM_CLAIMS'] = lp_input_df['LM_CLAIMS']
            lp_input_df_new['LM_FULLAWP_ADJ'] = lp_input_df['LM_FULLAWP_ADJ']
            lp_input_df_new['LM_QTY'] = lp_input_df['LM_QTY']
            lp_input_df_new['LM_PRICE_REIMB'] = lp_input_df['LM_PRICE_REIMB']
            lp_input_df_new['LM_PHARM_CLAIMS'] = lp_input_df['LM_PHARM_CLAIMS']
            lp_input_df_new['LM_PHARM_FULLAWP_ADJ'] = lp_input_df['LM_PHARM_FULLAWP_ADJ']
            lp_input_df_new['LM_PHARM_QTY'] = lp_input_df['LM_PHARM_QTY']
            lp_input_df_new['LM_PHARM_PRICE_REIMB'] = lp_input_df['LM_PHARM_PRICE_REIMB']
        else:
            #if previous go_live data and last_data date are in the same month, the new lm period consist of the old lm period + all the days in the old lag period
            lp_input_df_new['LM_CLAIMS'] = lp_input_df['LM_CLAIMS'] + lp_vol_mv_agg_df_nounc['CLAIMS_PROJ_LAG']
            lp_input_df_new['LM_FULLAWP_ADJ'] = lp_input_df['LM_FULLAWP_ADJ'] + lp_vol_mv_agg_df_nounc['FULLAWP_ADJ_PROJ_LAG']
            lp_input_df_new['LM_QTY'] = lp_input_df['LM_QTY'] + lp_vol_mv_agg_df_nounc['QTY_PROJ_LAG']
            lp_input_df_new['LM_PRICE_REIMB'] = lp_input_df['LM_PRICE_REIMB'] + lp_vol_mv_agg_df_nounc['QTY_PROJ_LAG'] * lp_input_df['CURRENT_MAC_PRICE']
            lp_input_df_new['LM_PHARM_CLAIMS'] = lp_input_df['LM_PHARM_CLAIMS'] + lp_vol_mv_agg_df_nounc['CLAIMS_PROJ_LAG']
            lp_input_df_new['LM_PHARM_FULLAWP_ADJ'] = lp_input_df['LM_PHARM_FULLAWP_ADJ'] + lp_vol_mv_agg_df_nounc['PHARM_FULLAWP_ADJ_PROJ_LAG']
            lp_input_df_new['LM_PHARM_QTY'] = lp_input_df['LM_PHARM_QTY'] + lp_vol_mv_agg_df_nounc['PHARM_QTY_PROJ_LAG']
            lp_input_df_new['LM_PHARM_PRICE_REIMB'] = lp_input_df['LM_PHARM_PRICE_REIMB'] + lp_vol_mv_agg_df_nounc['PHARM_QTY_PROJ_LAG'] * lp_input_df['CURRENT_MAC_PRICE']
    else:
        lp_input_df_new['CLAIMS'] = lp_input_df['CLAIMS'] + lp_output_df['CLAIMS_PROJ_LAG']
        lp_input_df_new['QTY'] = lp_input_df['QTY'] + lp_output_df['QTY_PROJ_LAG']
        lp_input_df_new['FULLAWP_ADJ'] = lp_input_df['FULLAWP_ADJ'] + lp_output_df['FULLAWP_ADJ_PROJ_LAG']
        lp_input_df_new['PRICE_REIMB'] = lp_input_df['PRICE_REIMB'] + lp_output_df['QTY_PROJ_LAG'] * lp_input_df['EFF_CAPPED_PRICE'].round(4)
        lp_input_df_new['PHARM_CLAIMS'] = lp_input_df['PHARM_CLAIMS'] + lp_output_df['PHARM_CLAIMS_PROJ_LAG']
        lp_input_df_new['PHARM_QTY'] = lp_input_df['PHARM_QTY'] + lp_output_df['PHARM_QTY_PROJ_LAG']
        lp_input_df_new['PHARM_FULLAWP_ADJ'] = lp_input_df['PHARM_FULLAWP_ADJ'] + lp_output_df['PHARM_FULLAWP_ADJ_PROJ_LAG']
        lp_input_df_new['PHARM_PRICE_REIMB'] = lp_input_df['PHARM_PRICE_REIMB'] + lp_output_df['PHARM_QTY_PROJ_LAG'] * lp_input_df['EFF_CAPPED_PRICE'].round(4)
        if pre_golive_date.month != pre_ytd_date.month and pre_lag_days != 0:
            new_last_month_days = new_ytd_date.day
            lp_input_df_new['LM_CLAIMS'] = lp_output_df['CLAIMS_PROJ_LAG'] * (new_last_month_days / pre_lag_days)
            lp_input_df_new['LM_FULLAWP_ADJ'] = lp_output_df['FULLAWP_ADJ_PROJ_LAG'] * (new_last_month_days / pre_lag_days)
            lp_input_df_new['LM_QTY'] = lp_output_df['QTY_PROJ_LAG'] * (new_last_month_days / pre_lag_days)
            lp_input_df_new['LM_PRICE_REIMB'] = lp_input_df_new['LM_QTY'] * lp_input_df['CURRENT_MAC_PRICE']
            lp_input_df_new['PHARM_LM_CLAIMS'] = lp_output_df['PHARM_CLAIMS_PROJ_LAG'] * (new_last_month_days / pre_lag_days)
            lp_input_df_new['PHARM_LM_FULLAWP_ADJ'] = lp_output_df['PHARM_FULLAWP_ADJ_PROJ_LAG'] * (new_last_month_days / pre_lag_days)
            lp_input_df_new['PHARM_LM_QTY'] = lp_output_df['PHARM_QTY_PROJ_LAG'] * (new_last_month_days / pre_lag_days)
            lp_input_df_new['PHARM_LM_PRICE_REIMB'] = lp_input_df_new['LM_PHARM_QTY'] * lp_input_df['CURRENT_MAC_PRICE']
        elif pre_lag_days == 0:
            #if lag == 0, then, last month data stays the same because last data date does not change.
            lp_input_df_new['LM_CLAIMS'] = lp_input_df['LM_CLAIMS']
            lp_input_df_new['LM_FULLAWP_ADJ'] = lp_input_df['LM_FULLAWP_ADJ']
            lp_input_df_new['LM_QTY'] = lp_input_df['LM_QTY']
            lp_input_df_new['LM_PRICE_REIMB'] = lp_input_df['LM_PRICE_REIMB']
            lp_input_df_new['LM_PHARM_CLAIMS'] = lp_input_df['LM_PHARM_CLAIMS']
            lp_input_df_new['LM_PHARM_FULLAWP_ADJ'] = lp_input_df['LM_PHARM_FULLAWP_ADJ']
            lp_input_df_new['LM_PHARM_QTY'] = lp_input_df['LM_PHARM_QTY']
            lp_input_df_new['LM_PHARM_PRICE_REIMB'] = lp_input_df['LM_PHARM_PRICE_REIMB']
        else:
            lp_input_df_new['LM_CLAIMS'] = lp_input_df['LM_CLAIMS'] + lp_output_df['CLAIMS_PROJ_LAG']
            lp_input_df_new['LM_FULLAWP_ADJ'] = lp_input_df['LM_FULLAWP_ADJ'] + lp_output_df['FULLAWP_ADJ_PROJ_LAG']
            lp_input_df_new['LM_QTY'] = lp_input_df['LM_QTY'] + lp_output_df['QTY_PROJ_LAG']
            lp_input_df_new['LM_PRICE_REIMB'] = lp_input_df['LM_PRICE_REIMB'] + lp_output_df['QTY_PROJ_LAG'] * lp_input_df['CURRENT_MAC_PRICE']
            lp_input_df_new['LM_PHARM_CLAIMS'] = lp_input_df['LM_PHARM_CLAIMS'] + lp_output_df['PHARM_CLAIMS_PROJ_LAG']
            lp_input_df_new['LM_PHARM_FULLAWP_ADJ'] = lp_input_df['LM_PHARM_FULLAWP_ADJ'] + lp_output_df['PHARM_FULLAWP_ADJ_PROJ_LAG']
            lp_input_df_new['LM_PHARM_QTY'] = lp_input_df['LM_PHARM_QTY'] + lp_output_df['PHARM_QTY_PROJ_LAG']
            lp_input_df_new['LM_PHARM_PRICE_REIMB'] = lp_input_df['LM_PHARM_PRICE_REIMB'] + lp_output_df['PHARM_QTY_PROJ_LAG'] * lp_input_df['CURRENT_MAC_PRICE']

    lp_input_df_new['PRICE_REIMB_ADJ'] = lp_input_df_new['QTY'] * lp_input_df['CURRENT_MAC_PRICE']
    lp_input_df_new['PRICE_REIMB_UNIT'] = np.where(lp_input_df_new['QTY'] != 0, lp_input_df_new['PRICE_REIMB'] / lp_input_df_new['QTY'], 0)
    lp_input_df_new['PHARM_PRICE_REIMB_ADJ'] = lp_input_df_new['PHARM_QTY'] * lp_input_df['CURRENT_MAC_PRICE']
    lp_input_df_new['PHARM_PRICE_REIMB_UNIT'] = np.where(lp_input_df_new['PHARM_QTY'] != 0, lp_input_df_new['PHARM_PRICE_REIMB'] / lp_input_df_new['PHARM_QTY'], 0)
    #UC unit data transfer
    #NOTE: copying from input and not from output to prevent UC_UNIT25 multiplying by 500 everytime
    cols_to_copy = ['UC_UNIT', 'UC_UNIT25']
    lp_input_df_new[cols_to_copy] = lp_input_df[cols_to_copy]

    #AWP data transfer
    cols_to_copy = list(set(['CURR_AWP', 'CURR_AWP_MAX', 'CURR_AWP_MIN', 'BREAKOUT_AWP_MAX']).intersection(set(input_cols)))
    lp_input_df_new[cols_to_copy] = lp_input_df[cols_to_copy]
    lp_input_df_new['AVG_AWP'] = np.where(lp_input_df_new['QTY'] != 0, lp_input_df_new['FULLAWP_ADJ'] / lp_input_df_new['QTY'], 0)

    #1026 data transfer
    cols_to_copy = ['1026_GPI_PRICE', '1026_NDC_PRICE', 'MAC1026_GPI_FLAG', 'MAC1026_UNIT_PRICE']
    lp_input_df_new[cols_to_copy] = lp_input_df[cols_to_copy]

    #adding unc claim rows to lp_data_nounc happens only when lp_input_df is of the nounc type (unc_flag = False) and the simulation transfers data between first and second iteration
    #for later iterations, there is no need to add further unc claim rows since all of them are added in the first iteration
    #in lp_input_df of unc type (unc_flag = True), those claim already exist
    if unc_flag == False and iteration == 0:
        #find the claim rows that are not in lp_input_df_nounc
        unc_nounc_id_cols = ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'GPI_NDC', 'CHAIN_GROUP']
        lp_input_diff_unc_nounc = pd.concat([lp_input_df[unc_nounc_id_cols], lp_input_df_unc[unc_nounc_id_cols]], axis = 0, ignore_index = True)
        lp_input_diff_unc_nounc.drop_duplicates(keep = False, ignore_index = True, inplace = True)

        #filter input data for all columns of the <lp_input_diff_unc_nounc> dataframe
        lp_input_diff_unc_nounc = pd.merge(lp_input_diff_unc_nounc, lp_input_df_unc[input_cols], on = unc_nounc_id_cols, how = 'left')

        #zero out ytd (and lag?) information of the <lp_input_diff_unc_nounc> dataframe
        #this is because unc claims should only show up in performance calculation for end of year projections
        #lp_input_diff_unc_nounc[['CLAIMS', 'CLAIMS_PROJ_LAG', 'QTY', 'QTY_PROJ_LAG', 'FULLAWP_ADJ', 'FULLAWP_ADJ_PROJ_LAG', 'PRICE_REIMB',
        #                         'PRICE_REIMB_ADJ', 'PRICE_REIMB_UNIT', 'LM_CLAIMS', 'LM_FULLAWP_ADJ', 'LM_QTY', 'LM_PRICE_REIMB']] = 0
        lp_input_diff_unc_nounc[['CLAIMS', 'CLAIMS_PROJ_LAG', 'QTY_PROJ_LAG', 'FULLAWP_ADJ', 'FULLAWP_ADJ_PROJ_LAG', 'PRICE_REIMB',
                                 'PRICE_REIMB_ADJ', 'PRICE_REIMB_UNIT', 'LM_CLAIMS', 'LM_FULLAWP_ADJ', 'LM_QTY', 'LM_PRICE_REIMB',
                                 'PHARM_CLAIMS', 'PHARM_CLAIMS_PROJ_LAG', 'PHARM_QTY_PROJ_LAG', 'PHARM_FULLAWP_ADJ', 'PHARM_FULLAWP_ADJ_PROJ_LAG', 'PHARM_PRICE_REIMB',
                                 'PHARM_PRICE_REIMB_ADJ', 'PHARM_PRICE_REIMB_UNIT', 'PHARM_LM_CLAIMS', 'PHARM_LM_FULLAWP_ADJ', 'PHARM_LM_QTY', 'PHARM_LM_PRICE_REIMB']] = 0

        #add the <lp_input_diff_unc_nounc> dataframe to <lp_input_df> complete the lp_data_nounc with missing unc claims
        lp_input_df_new = pd.concat([lp_input_df_new, lp_input_diff_unc_nounc], axis = 0, ignore_index = True)

        #sort dataframes to line up
        lp_input_df_unc.sort_values(['CLIENT', 'REGION', 'BREAKOUT', 'MEASUREMENT', 'CHAIN_GROUP', 'GPI_NDC'], inplace = True, ignore_index = True)

    #sort every dataframe to line up
    lp_input_df_new.sort_values(['CLIENT', 'REGION', 'BREAKOUT', 'MEASUREMENT', 'CHAIN_GROUP', 'GPI_NDC'], inplace = True, ignore_index = True)
    lp_input_df.sort_values(['CLIENT', 'REGION', 'BREAKOUT', 'MEASUREMENT', 'CHAIN_GROUP', 'GPI_NDC'], inplace = True, ignore_index = True)
    lp_output_df.sort_values(['CLIENT', 'REGION', 'BREAKOUT', 'MEASUREMENT', 'CHAIN_GROUP', 'GPI_NDC'], inplace = True, ignore_index = True)

    if unc_flag == False and iteration == 0:
        lp_input_df_new['CLAIMS_PROJ_LAG'] = lp_input_df_unc['CLAIMS_PROJ_EOY'] * (new_lag_days / pre_eoy_days)
        lp_input_df_new['CLAIMS_PROJ_EOY'] = lp_input_df_unc['CLAIMS_PROJ_EOY'] * (new_eoy_days / pre_eoy_days)
        lp_input_df_new['QTY_PROJ_LAG'] = lp_input_df_unc['QTY_PROJ_EOY'] * (new_lag_days / pre_eoy_days)
        lp_input_df_new['QTY_PROJ_EOY'] = lp_input_df_unc['QTY_PROJ_EOY'] * (new_eoy_days / pre_eoy_days)
        lp_input_df_new['FULLAWP_ADJ_PROJ_LAG'] = lp_input_df_unc['FULLAWP_ADJ_PROJ_EOY'] * (new_lag_days / pre_eoy_days)
        lp_input_df_new['FULLAWP_ADJ_PROJ_EOY'] = lp_input_df_unc['FULLAWP_ADJ_PROJ_EOY'] * (new_eoy_days / pre_eoy_days)
        lp_input_df_new['PHARM_CLAIMS_PROJ_LAG'] = lp_input_df_unc['PHARM_CLAIMS_PROJ_EOY'] * (new_lag_days / pre_eoy_days)
        lp_input_df_new['PHARM_CLAIMS_PROJ_EOY'] = lp_input_df_unc['PHARM_CLAIMS_PROJ_EOY'] * (new_eoy_days / pre_eoy_days)
        lp_input_df_new['PHARM_QTY_PROJ_LAG'] = lp_input_df_unc['PHARM_QTY_PROJ_EOY'] * (new_lag_days / pre_eoy_days)
        lp_input_df_new['PHARM_QTY_PROJ_EOY'] = lp_input_df_unc['PHARM_QTY_PROJ_EOY'] * (new_eoy_days / pre_eoy_days)
        lp_input_df_new['PHARM_FULLAWP_ADJ_PROJ_LAG'] = lp_input_df_unc['PHARM_FULLAWP_ADJ_PROJ_EOY'] * (new_lag_days / pre_eoy_days)
        lp_input_df_new['PHARM_FULLAWP_ADJ_PROJ_EOY'] = lp_input_df_unc['PHARM_FULLAWP_ADJ_PROJ_EOY'] * (new_eoy_days / pre_eoy_days)
    else:
        lp_input_df_new['CLAIMS_PROJ_LAG'] = lp_input_df['CLAIMS_PROJ_EOY'] * (new_lag_days / pre_eoy_days)
        lp_input_df_new['CLAIMS_PROJ_EOY'] = lp_input_df['CLAIMS_PROJ_EOY'] * (new_eoy_days / pre_eoy_days)
        lp_input_df_new['QTY_PROJ_LAG'] = lp_input_df['QTY_PROJ_EOY'] * (new_lag_days / pre_eoy_days)
        lp_input_df_new['QTY_PROJ_EOY'] = lp_input_df['QTY_PROJ_EOY'] * (new_eoy_days / pre_eoy_days)
        lp_input_df_new['FULLAWP_ADJ_PROJ_LAG'] = lp_input_df['FULLAWP_ADJ_PROJ_EOY'] * (new_lag_days / pre_eoy_days)
        lp_input_df_new['FULLAWP_ADJ_PROJ_EOY'] = lp_input_df['FULLAWP_ADJ_PROJ_EOY'] * (new_eoy_days / pre_eoy_days)
        lp_input_df_new['PHARM_CLAIMS_PROJ_LAG'] = lp_input_df['PHARM_CLAIMS_PROJ_EOY'] * (new_lag_days / pre_eoy_days)
        lp_input_df_new['PHARM_CLAIMS_PROJ_EOY'] = lp_input_df['PHARM_CLAIMS_PROJ_EOY'] * (new_eoy_days / pre_eoy_days)
        lp_input_df_new['PHARM_QTY_PROJ_LAG'] = lp_input_df['PHARM_QTY_PROJ_EOY'] * (new_lag_days / pre_eoy_days)
        lp_input_df_new['PHARM_QTY_PROJ_EOY'] = lp_input_df['PHARM_QTY_PROJ_EOY'] * (new_eoy_days / pre_eoy_days)
        lp_input_df_new['PHARM_FULLAWP_ADJ_PROJ_LAG'] = lp_input_df['PHARM_FULLAWP_ADJ_PROJ_EOY'] * (new_lag_days / pre_eoy_days)
        lp_input_df_new['PHARM_FULLAWP_ADJ_PROJ_EOY'] = lp_input_df['PHARM_FULLAWP_ADJ_PROJ_EOY'] * (new_eoy_days / pre_eoy_days)

    #new price transfer
    lp_input_df_new['CURRENT_MAC_PRICE'] = lp_output_df['Final_Price']

    #effective price data transfer; replicate the process in Daily_Input_Read.py
    #lp_input_df_new['EFF_CAPPED_PRICE'] = lp_input_df_new.apply(determine_effective_price, args=tuple(['CURRENT_MAC_PRICE', 'UC_UNIT25', True]), axis=1)
    #lp_input_df_new['EFF_CAPPED_PRICE'] = np.where(lp_input_df_new['EFF_CAPPED_PRICE'] > 0, lp_input_df_new['EFF_CAPPED_PRICE'], lp_input_df_new['PRICE_REIMB_UNIT'])
    lp_input_df_new['EFF_CAPPED_PRICE'] = lp_output_df['EFF_CAPPED_PRICE_new']

    #lp_input_df_new['EFF_UNIT_PRICE'] = lp_input_df_new.apply(determine_effective_price, args=tuple(['CURRENT_MAC_PRICE']), axis=1)
    #lp_input_df_new['EFF_UNIT_PRICE'] = np.where(lp_input_df_new['EFF_UNIT_PRICE'] > 0, lp_input_df_new['EFF_UNIT_PRICE'], lp_input_df_new['PRICE_REIMB_UNIT'])
    lp_input_df_new['EFF_UNIT_PRICE'] = lp_output_df['EFF_UNIT_PRICE_new']

    #mac price data transfer; replicate the process in Daily_Input_Read.py
    lp_input_df_new['MAC_PRICE_UNIT_ADJ'] = np.where(lp_input_df_new['CURRENT_MAC_PRICE'] > 0, lp_input_df_new['CURRENT_MAC_PRICE'], lp_input_df_new['PRICE_REIMB_UNIT'])
    #######################################################################################################################################################
    #######################################################################################################################################################
    #UNC specific columns transfer

    #unc_cols_to_copy = ["CURR_AWP", "PRICE_MUTABLE", "GOODRX_UPPER_LIMIT"]
    #lp_input_df_new[unc_cols_to_copy] = lp_input_df_unc[unc_cols_to_copy]

    #list all unc columns to for later check to determine whether <input_lp_df> is of the type lp_data_nounce or lp_data_unc
    unc_cols = ['UC_CLAIMS', 'UNC_FRAC_OLD', 'CLAIMS_PROJ_EOY_OLDUNC', 'CLAIMS_IN_CONSTRAINTS', 'UC_PERCENTILE_CLAIMS', 'CLAIMS_GT_PCT01', 'CLAIMS_GT_PCT25',
                'CLAIMS_GT_PCT50', 'CLAIMS_GT_PCT90', 'VCML_AVG_CLAIMS_QTY', 'LM_CLAIMS_OLDUNC', 'LM_UC_CLAIMS', 'LM_QTY_OLDUNC', 'QTY_PROJ_EOY_OLDUNC',
                'LM_FULLAWP_ADJ_OLDUNC', 'FULLAWP_ADJ_PROJ_EOY_OLDUNC', 'DISTINCT_UC_PRICES', 'PRICE_CHANGED_UC', 'MAC_PRICE_UPPER_LIMIT_UC', 'PRE_UC_MAC_PRICE',
                'RAISED_PRICE_UC', 'UNC_FRAC', 'LM_UNC_FRAC_OLD', 'U&C_EBIT', 'VCML_AVG_AWP', 'MIN_UCAMT_QUANTITY', 'MAX_UCAMT_QUANTITY', 'PCT01_UCAMT_UNIT',
                'PCT25_UCAMT_UNIT', 'PCT50_UCAMT_UNIT', 'PCT90_UCAMT_UNIT']

    #this condition separates lp_data_nounc and lp_data_unc when <UNC_OPT> is True
    #it only goes through for lp_data_unc and <UNC_OPT> set to True
    if unc_flag == True and len(set(input_cols).intersection(set(unc_cols))) > 0:
        #unc claim data transfer
        #lp_input_df_new['UC_CLAIMS'] = lp_input_df['UC_CLAIMS'] + lp_output_df['CLAIMS_PROJ_LAG'] * (1 - lp_output_df['UNC_FRAC_OLD'])
        lp_input_df_new['UC_CLAIMS'] = lp_input_df['UC_CLAIMS'] + lp_output_df['CLAIMS_PROJ_LAG'] * (1 - lp_output_df['UNC_FRAC'])
        lp_input_df_new['UNC_FRAC_OLD'] = 1.0 * lp_input_df_new['UC_CLAIMS'] / lp_input_df_new['CLAIMS']
        lp_input_df_new['CLAIMS_PROJ_EOY_OLDUNC'] = lp_input_df_new['CLAIMS_PROJ_EOY'] * (1 - lp_input_df_new['UNC_FRAC_OLD'])
        #confirm whether the columns below should be updated from one iteration to the other
        #confirmed with Diego through email
        lp_input_df_new['CLAIMS_IN_CONSTRAINTS'] = lp_input_df['CLAIMS_IN_CONSTRAINTS']
        lp_input_df_new['UC_PERCENTILE_CLAIMS'] = lp_input_df['UC_PERCENTILE_CLAIMS']
        lp_input_df_new['CLAIMS_GT_PCT01'] = lp_input_df['CLAIMS_GT_PCT01']
        lp_input_df_new['CLAIMS_GT_PCT25'] = lp_input_df['CLAIMS_GT_PCT25']
        lp_input_df_new['CLAIMS_GT_PCT50'] = lp_input_df['CLAIMS_GT_PCT50']
        lp_input_df_new['CLAIMS_GT_PCT90'] = lp_input_df['CLAIMS_GT_PCT90']

        #unc quantity transfer
        lp_input_df_new['QTY_PROJ_EOY_OLDUNC'] = lp_input_df_new['CLAIMS_PROJ_EOY'] * (1 - lp_input_df_new['UNC_FRAC_OLD'])

        #unc full AWP transfer
        lp_input_df_new['FULLAWP_ADJ_PROJ_EOY_OLDUNC'] = lp_input_df_new['FULLAWP_ADJ_PROJ_EOY'] * (1 - lp_input_df_new['UNC_FRAC_OLD'])

        #unc last month data transfer
        if pre_golive_date.month != pre_ytd_date.month:
            #if previous go_live date and last_data date are in different months, then new lm period consists of all the days in the last month of last_data date
            new_last_month_days = new_ytd_date.day
            lp_input_df_new['LM_UC_CLAIMS'] = lp_output_df['CLAIMS_PROJ_LAG'] * (1 - lp_output_df['UNC_FRAC']) * (new_last_month_days / pre_lag_days)
        else:
            #if previous go_live data and last_data date are in the same month, the new lm period consist of the old lm period + all the days in the old lag period
            lp_input_df_new['LM_UC_CLAIMS'] = lp_input_df['LM_UC_CLAIMS'] + lp_output_df['CLAIMS_PROJ_LAG'] * (1 - lp_output_df['UNC_FRAC'])
        lp_input_df_new['LM_UNC_FRAC_OLD'] = 1.0 * lp_input_df_new['LM_UC_CLAIMS'] / lp_input_df_new['LM_CLAIMS']
        lp_input_df_new['LM_CLAIMS_OLDUNC'] = lp_input_df_new['LM_CLAIMS'] * (1 - lp_input_df_new['LM_UNC_FRAC_OLD'])
        lp_input_df_new['LM_QTY_OLDUNC'] = lp_input_df_new['LM_QTY'] * (1 - lp_input_df_new['LM_UNC_FRAC_OLD'])
        lp_input_df_new['LM_FULLAWP_ADJ_OLDUNC'] = lp_input_df_new['LM_FULLAWP_ADJ'] * (1 - lp_input_df_new['LM_UNC_FRAC_OLD'])

        #confirm copying below columns
        #confirmed with Diego through email
        lp_input_df_new['DISTINCT_UC_PRICES'] = lp_input_df['DISTINCT_UC_PRICES']
        lp_input_df_new['MIN_UCAMT_QUANTITY'] = lp_input_df['MIN_UCAMT_QUANTITY']
        lp_input_df_new['MAX_UCAMT_QUANTITY'] = lp_input_df['MAX_UCAMT_QUANTITY']
        lp_input_df_new['PCT01_UCAMT_UNIT'] = lp_input_df['PCT01_UCAMT_UNIT']
        lp_input_df_new['PCT25_UCAMT_UNIT'] = lp_input_df['PCT25_UCAMT_UNIT']
        lp_input_df_new['PCT50_UCAMT_UNIT'] = lp_input_df['PCT50_UCAMT_UNIT']
        lp_input_df_new['PCT90_UCAMT_UNIT'] = lp_input_df['PCT90_UCAMT_UNIT']

        #columns ['VCML_AVG_CLAIMS_QTY', 'VCML_AVG_AWP', 'UNC_FRAC', 'U&C_EBIT', 'MAC_PRICE_UPPER_LIMIT_UC', 'PRE_UC_MAC_PRICE', 'RAISED_PRICE_UC', 'PRICE_CHANGED_UC']
        #are updated using <unc_optimization> and <unc_ebit> functions in the CPMO_shared_fun.py
        #lp_input_df_new.drop(['VCML_AVG_AWP', 'VCML_AVG_CALIM_QTY', 'PRICE_CHANGED_UC', 'RAISED_PRICE_UC'], axis=1, inplace=True)
        #lp_input_df_new = unc_optimization(lp_input_df_new,
        #                                   awp_discount_percent = 0.75,
        #                                   unc_low = 'PCT01_UCAMT_UNIT',
        #                                   unc_high = 'PCT90_UCAMT_UNIT')
        cols_to_copy = ['VCML_AVG_CALIM_QTY', 'VCML_AVG_AWP', 'UNC_FRAC', 'U&C_EBIT', 'MAC_PRICE_UPPER_LIMIT_UC', 'PRE_UC_MAC_PRICE', 'RAISED_PRICE_UC', 'PRICE_CHANGED_UC']
        lp_input_df_new[cols_to_copy] = lp_input_df[cols_to_copy]

        #columns below are updated after UNC_FRAC is updated
        #NOTE: this only happens to lp_data_unc when UNC_OPT is True. lp_data_nounce does not go through the same adjustment
        #lp_input_df_new['CLAIMS_PROJ_EOY'] = lp_input_df_new['CLAIMS_PROJ_EOY'] * (1 - lp_input_df_new['UNC_FRAC'])
        #lp_input_df_new['QTY_PROJ_EOY'] = lp_input_df_new['QTY_PROJ_EOY'] * (1 - lp_input_df_new['UNC_FRAC'])
        #lp_input_df_new['FULLAWP_ADJ_PROJ_EOY'] = lp_input_df_new['FULLAWP_ADJ_PROJ_EOY'] * (1 - lp_input_df_new['UNC_FRAC'])
        #lp_input_df_new['LM_CLAIMS'] = lp_input_df_new['LM_CLAIMS'] * (1 - lp_input_df_new['UNC_FRAC'])
        #lp_input_df_new['LM_QTY'] = lp_input_df_new['LM_QTY'] * (1 - lp_input_df_new['UNC_FRAC'])
        #lp_input_df_new['LM_FULLAWP_ADJ'] = lp_input_df_new['LM_FULLAWP_ADJ'] * (1 - lp_input_df_new['UNC_FRAC'])

        #NOTE: without the following line, some <UNC_FRAC> are NaN which results in <FULLAWP_ADJ_PROJ_EOY> being NaN which results in <Consistent Strength Pricing Constraints> component to fail
    lp_input_df_new.fillna(value = 0, inplace = True)

    return lp_input_df_new

def seq_sim_brand_generic_transfer(brand_generic_new,pre_ytd_date, pre_golive_date, new_ytd_date, new_golive_date):
    '''
    Input: previous iteration brand_generic surplus dataframe, previous iteration YTD date, previous Iteration Go-Live Date, next iteration YTD date, next iteration GO-Live date
    Transforms the brand_generic data to include projected surplus from the previous iterations lag period. This is done so the ytd brand_generic_surplus is correct for the next iteration.
    Output: a new brand_generic_surplus_df
    '''
    import os
    import sys
    sys.path.append('/home/jupyter/clientpharmacymacoptimization/GER_LP_Code')
    from GER_LP_Code.CPMO_shared_functions import unc_optimization, unc_ebit, determine_effective_price
    
    import GER_LP_Code.CPMO_parameters as p

    import pandas as pd
    import numpy as np
    
    #interval calculation
    eoy_date = dt.datetime.strptime('12/31/' + str(p.LAST_DATA.year), '%m/%d/%Y')
    pre_lag_days = (pre_golive_date - pre_ytd_date).days - 1
    pre_eoy_days = (eoy_date - pre_golive_date).days + 1
    new_lag_days = (new_golive_date - new_ytd_date).days - 1
    new_eoy_days = (eoy_date - new_golive_date).days + 1
    pre_ytd_days = (pre_ytd_date - dt.datetime.strptime('1/1/' + str(p.LAST_DATA.year), '%m/%d/%Y')).days
    
    #Add previous lag to current ytd to correct mismatch
    brand_generic_new['SURPLUS_DAY'] = brand_generic_new['SURPLUS'] / (pre_ytd_days)
    brand_generic_new['SURPLUS_PRE_LAG'] = brand_generic_new['SURPLUS_DAY'] * pre_lag_days
    brand_generic_new['SURPLUS'] = brand_generic_new['SURPLUS'] + brand_generic_new['SURPLUS_PRE_LAG']
    
    return brand_generic_new

def seq_sim_perf_override_transfer(perf_over_df, lag_perf_df):
    #sys.path.append('C:/Users/C255085/Downloads/Sim_dev')
    '''
    Input: previous iteration performance override dataframe, previous iteration lag performance dataframe
    This function updates the performance_override file with the correct lag_ytd values for the next iteration.
    Output: A new perf_over_df
    '''
    from GER_LP_Code.qa_checks import qa_dataframe
    import GER_LP_Code.CPMO_shared_functions as sf

    perf_over_df = sf.standardize_df(perf_over_df)
    qa_dataframe(perf_over_df, dataset = 'LAG_YTD_Override_File_AT_{}'.format(os.path.basename(__file__)))
    perf_over_dict = sf.df_to_dict(perf_over_df, ['BREAKOUT', 'SURPLUS'])
    lag_perf_dict = sf.df_to_dict(lag_perf_df, ['BREAKOUT', 'SURPLUS'])

    for key in perf_over_dict:
        perf_over_dict[key] += lag_perf_dict[key]

    return sf.dict_to_df(perf_over_dict, ['BREAKOUT', 'SURPLUS'])

def seq_sim_data_transfer(input_path, new_output_path, pre_ytd_date, pre_golive_date, new_ytd_date, new_golive_date,custom_params, iteration) -> NamedTuple('Outputs', [('FLAG', bool)]):
    '''
    Input: Path to previous iteration data storage, new path to next iteration Dynamic Input folder, previous iteration YTD date, previous Iteration Go-Live Date, next iteration YTD date, next iteration GO-Live date, custom_params dictionary, iteration number.
    Calls helper functions from sim_utils to create a new Dynamic_Input folder for the next iteration in a simulation from the outputs of the previous iteration.
    Transforms lp_input_data, pharm_spend data, brand_generic_surplus data, performance override data, and mac list data.
    Handles UNC_OPT = True cases as well.
    Output: New Dynamic Input folder for the next iteration.
    '''
    #setup
    import os
    import sys
    sys.path.append('/home/jupyter/clientpharmacymacoptimization/GER_LP_Code')

    import pandas as pd
    import numpy as np
    import datetime as dt

    import GER_LP_Code.util_funcs as uf
    uf.write_params(os.path.join(custom_params['PROGRAM_OUTPUT_PATH'], 'CPMO_parameters.py'))
    import GER_LP_Code.CPMO_parameters as p
    import GER_LP_Code.BQ as BQ

    #######################################################################################################################################################
    #######################################################################################################################################################
    #reading data of the current iteration
    print('loading previous iteration data...')

    #reading total output#####################################################################################
    lp_output_df = pd.read_csv(os.path.join(input_path + '/Output/', 'Total_Output_' + p.DATA_ID + '.csv'),
                               dtype = p.VARIABLE_TYPE_DIC)
    #NOTE that, currently, Total_Output_ is written to cloud storage whether or not WRITE_TO_BQ is true
    
    #reading lp data##########################################################################################
    if p.UNC_OPT:
        lp_unc_input_df = pd.read_csv(os.path.join(input_path + '/Dynamic_Input/', 'lp_data_' + p.DATA_ID + '.csv'),
                                      dtype = p.VARIABLE_TYPE_DIC)
        lp_nounc_input_df = pd.read_csv(os.path.join(input_path + '/Dynamic_Input/', 'lp_data_nounc_' + p.DATA_ID + '.csv'),
                                        dtype = p.VARIABLE_TYPE_DIC)
        lp_vol_mv_agg_df_nounc = pd.read_csv(os.path.join(input_path + '/Output/', 'lp_data_nounc_' + p.DATA_ID + '.csv'),
                                             dtype = p.VARIABLE_TYPE_DIC)
    else:
        lp_input_df = pd.read_csv(os.path.join(input_path + '/Dynamic_Input/', 'lp_data_' + p.DATA_ID + '.csv'),
                                  dtype = p.VARIABLE_TYPE_DIC)
    #NOTE that, currently, lp_data_ is read from cloud storage only

    #reading spend data#######################################################################################
    if p.WRITE_TO_BQ:
        spend_data_df = uf.read_BQ_data(BQ.full_spend_data,
                                        project_id = p.BQ_OUTPUT_PROJECT_ID,
                                        dataset_id = p.BQ_OUTPUT_DATASET,
                                        table_id = 'Spend_data',
                                        run_id = p.AT_RUN_ID,
                                        client = ', '.join(sorted(p.CUSTOMER_ID)),
                                        period = p.TIMESTAMP,
                                        output = True)
    else:
        spend_data_df = pd.read_csv(os.path.join(input_path + '/Output/', 'Spend_data_' + str(p.TIMESTAMP) + str(pre_golive_date.month) + '.csv'),
                                    dtype = p.VARIABLE_TYPE_DIC)
    #reading mac list data ###################################################################################
    mac_list_df = pd.read_csv(os.path.join(input_path + '/Dynamic_Input/', 'mac_lists_' + p.DATA_ID + '.csv'),
                              dtype = p.VARIABLE_TYPE_DIC)
    #NOTE that, currently, mac_lists_ is read from cloud storage only
    
    #reading brand/generic offset data #######################################################################
    brand_gen_df = pd.read_csv(os.path.join(input_path + '/Dynamic_Input/', 'brand_surplus_' + p.DATA_ID + '.csv'))
    #NOTE that, currently brand_surplus_ data is only read from cloud storage
    print('finished loading the data.')
    
    #reading performance override data #######################################################################
    if p.YTD_OVERRIDE:
        perf_over_df = pd.read_csv(p.FILE_INPUT_PATH + p.LAG_YTD_Override_File, dtype = p.VARIABLE_TYPE_DIC)
        lag_perf_df = pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH,  'lag_surplus_{}.csv'.format(p.DATA_ID)), dtype = p.VARIABLE_TYPE_DIC)
    print('finished loading the data.')
    
    #######################################################################################################################################################
    #######################################################################################################################################################
    #transforming data for next iteration
    
    #transforming mac lists data
    print('transforming mac lists prices to lp output final prices...')
    mac_list_df_new = seq_sim_mac_list_transfer(mac_list_df, lp_output_df)
    
    #transforming brand/generic surplus data
    brand_generic_new = seq_sim_brand_generic_transfer(brand_gen_df, pre_ytd_date, pre_golive_date, new_ytd_date, new_golive_date)
    
    #transforming lp data
    if p.UNC_OPT:
        lp_nounc_input_df_new = seq_sim_unc_lp_data_transfer(lp_nounc_input_df, lp_output_df,
                                                             pre_ytd_date, pre_golive_date, new_ytd_date, new_golive_date,
                                                             iteration = iteration, unc_flag = False,
                                                             lp_vol_mv_agg_df_nounc = lp_vol_mv_agg_df_nounc,
                                                             lp_input_df_unc = lp_unc_input_df)
        lp_unc_input_df_new = seq_sim_unc_lp_data_transfer(lp_unc_input_df, lp_output_df,
                                                           pre_ytd_date, pre_golive_date, new_ytd_date, new_golive_date,
                                                           iteration = iteration, unc_flag = True)
    else:
        lp_input_df_new = seq_sim_lp_data_transfer(lp_input_df, lp_output_df,
                                                   pre_ytd_date, pre_golive_date, new_ytd_date, new_golive_date)
        
    #transforming performance override data
    if p.YTD_OVERRIDE:
        perf_over_df_new = seq_sim_perf_override_transfer(perf_over_df, lag_perf_df)

    #######################################################################################################################################################
    #######################################################################################################################################################
    ####performance qa checks
    print('finished transforming data.')
    #print('QA checks for transformed data...')
    #
    #print('All QA checks have passed.')

    #######################################################################################################################################################
    #######################################################################################################################################################
    ####write the transformed data
    print('writing transformed data ...')
    mac_list_df_new.to_csv(os.path.join(new_output_path, 'mac_lists_' + p.DATA_ID + '.csv'), index=False)
    brand_generic_new.to_csv(os.path.join(new_output_path, 'brand_surplus_' + p.DATA_ID + '.csv'),index=False)
    if p.UNC_OPT:
        lp_nounc_input_df_new.to_csv(os.path.join(new_output_path, 'lp_data_nounc_' + p.DATA_ID + '.csv'), index=False)
        lp_unc_input_df_new.to_csv(os.path.join(new_output_path, 'lp_data_' + p.DATA_ID + '.csv'), index=False)
    else:
        lp_input_df_new.to_csv(os.path.join(new_output_path, 'lp_data_' + p.DATA_ID + '.csv'), index=False)
    if p.YTD_OVERRIDE:
        perf_over_df_new.to_csv(p.FILE_INPUT_PATH + p.LAG_YTD_Override_File, index = False)
    
    return True
    
def annual_perf_report(excel_writer, perf_df, num_iter):
    '''
    Creates an excel workbook tab that has a bar graph showing the annual performance of the client across the different iterations.
    Breaks down the performance by breakouts. 
    Input: Excel Writer object, performance_df, number of succesful iterations.
    Will only display the performance of succesful iterations.
    '''
    import pandas as pd
    import GER_LP_Code.CPMO_parameters as p

    #filter performance dictionary for mail/retail breakouts and big capped pharmacies
    perf_df = perf_df.loc[(perf_df['Entity'].str.contains(p.CUSTOMER_ID[0])) |
                          (perf_df['Entity'].isin(p.BIG_CAPPED_PHARMACY_LIST))]

    #write performance dictionary to an excel sheet
    perf_df.to_excel(excel_writer,
                     sheet_name = 'Annual_Performance',
                     index=False)

    #initiate a workbook to creat excel charts in the same sheet
    workbook = excel_writer.book
    worksheet = excel_writer.sheets['Annual_Performance']

    #create a chart object
    chart = workbook.add_chart({'type': 'column'})
    num_format = workbook.add_format({'num_format': 44}) #source for num_format 44: https://xlsxwriter.readthedocs.io/format.html?highlight=set_num_format#set_num_format
    worksheet.set_column(first_col = 1, #first col is Entity names
                         last_col = num_iter,
                         width = None,
                         cell_format = num_format)

    #configure the series of the chart from the <Annual_Performance> sheet
    for i in range(num_iter + 1):
        #values/categories: [sheetname, first_row, first_col, last_row, last_col]
        #name: [sheetname, row, col]
        chart.add_series({'values': ['Annual_Performance', 1, i + 1, len(perf_df) + 1, i + 1],
                          'categories': ['Annual_Performance', 1, 0, len(perf_df) + 1, 0],
                          'name': ['Annual_Performance', 0, i + 1]
                         })

    #add title, set axis and shape
    chart.set_title({'name': 'Projected Annual Performance | Surplus (Liability)'})
    chart.set_size({'width': 900, 'height': 300})
    chart.set_legend({'position': 'bottom'})
    chart.set_x_axis({
        'major_gridlines': {
            'visible': True,
            'line': {'width': 1.25,
                     'dash_type': 'dash'
                    }
        },
    })
    chart.set_y_axis({'name': 'Performance ($)'})

    #insert the chart into the worksheet
    #insert_chart(row, col, chart, [options])
    worksheet.insert_chart(len(perf_df) + 3, 0, chart)
    return excel_writer

def monthly_perf_report(excel_writer, proj_df, sim_params, num_iter):
    ''''
    Creates an excel workbook tab that has a line graph showing the monthly performance of the client across the different iterations.
    Input: Excel Writer object, projected_performance_df, simulation_parameters, number of succesful iterations.
    Will only display the performance of succesful iterations.
    '''
    import pandas as pd
    import GER_LP_Code.CPMO_parameters as p

    #each client breakout will have its own performance chart
    breakout_list = proj_df['ENTITY'].unique()

    start_row = 0
    for breakout in breakout_list:
        breakout_df = proj_df.loc[proj_df['ENTITY'] == breakout].reset_index()
        #row and column settings for writing to excel
        end_row = start_row + breakout_df.shape[0]
        start_col = 0
        end_col = start_col + breakout_df.shape[1]

        #replace cumulutive performances by NA for months before go live month
        for index, row in breakout_df.iterrows():
            if (row['Iteration'] != 'Preexisting') & (row['Iteration'] != sim_params['GO_LIVE_LIST'][0]):
                iter_month = int(row['Iteration'].split('/')[0])
                na_fill_cols = [col for col in breakout_df.columns[breakout_df.columns.str.startswith('MONTH_')] if int(col.split('_')[1]) < iter_month - 1]
                row[na_fill_cols] = None
        breakout_df = breakout_df.drop(['index'], axis = 1)

        #write performances for that breakout in the excel sheet
        breakout_df.to_excel(excel_writer, sheet_name = 'Monthly_Performance', index = False, startrow = start_row)
        workbook = excel_writer.book
        worksheet = excel_writer.sheets['Monthly_Performance']


        start_row += 1
        #create a chart object per breakout
        chart_breakout = workbook.add_chart({'type': 'line'})
        num_format = workbook.add_format({'num_format': 44}) #source for num_format 44: https://xlsxwriter.readthedocs.io/format.html?highlight=set_num_format#set_num_format
        worksheet.set_column(first_col = 2, #first col is Entity names, second call is iteration
                             last_col = len(breakout_df.columns) - 1,
                             width = None,
                             cell_format = num_format)

        #configure the series of the chart from the <Monthly_Performance> sheet
        for i in range(start_row, end_row + 1):
            #values/categories: [sheetname, first_row, first_col, last_row, last_col]
            #name: [sheetname, row, col]
            chart_breakout.add_series({'values': ['Monthly_Performance', i, start_col + 2, i, end_col - 1],
                                       'categories': ['Monthly_Performance', start_row - 1, start_col + 2, start_row - 1, end_col - 1],
                                       'name': ['Monthly_Performance', i, start_col + 1],
                                       'marker': {'type': 'diamond'}})


        #add title, set axis and shape
        chart_breakout.set_title({'name': '{0} Performance'.format(breakout)})
        chart_breakout.set_size({'width': 650, 'height': 250})
        chart_breakout.set_legend({'position': 'bottom'})
        chart_breakout.set_y_axis({
            'major_gridlines': {
                'visible': True,
                'line': {'width': 0.75,
                         'dash_type': 'dash'
                        }
            },
        })
        chart_breakout.set_y_axis({'name':'Surplus($)'})

        #insert the chart into the worksheet
        #insert_chart(row, col, chart, [options])
        worksheet.insert_chart(start_row + 15 if breakout != breakout_list[0] else start_row - 1,
                               end_col + 1,
                               chart_breakout)

        #adjust next breakout operation's start row
        start_row = end_row + 2

    return excel_writer

#Rename
def new_exit_ger_report(excel_writer, awp_spend_dict, num_iter, sim_params, initial_last_data):
    ''''
    Creates an excel workbook tab that has a graph showing the monthly GER of the client across the different iterations.
    Input: Excel Writer object, awp_spend_dictionary, simulation_parameters, number of succesful iterations, initial_last_date.
    Will only display the GER of succesful iterations.
    '''

    breakout_list = awp_spend_dict[0].loc[awp_spend_dict[0]['ENTITY_TYPE'] == 'CLIENT', "BREAKOUT"].unique()
    start_row = 0

    for breakout in breakout_list:
        breakout_ger_df = pd.DataFrame()
        for i in range(num_iter):
            if i == 0:
                last_data = eval(initial_last_data)
            else:
                last_data = pd.to_datetime(sim_params['GO_LIVE_LIST'][i - 1]) + dt.timedelta(days = -1)

            ytd_days = (last_data - pd.to_datetime('1/1/2021')).days
            lag_days = (pd.to_datetime(sim_params['GO_LIVE_LIST'][i]) - last_data).days - 1
            eoy_days = (pd.to_datetime('12/31/2021') - pd.to_datetime(sim_params['GO_LIVE_LIST'][i])).days + 1

            #filter each iteration's spend data by breakout and aggregate AWP and price reimbursement columns
            awp_spend_dict[i].columns = awp_spend_dict[i].columns.str.upper()
            spend_awp_breakout = awp_spend_dict[i].loc[(awp_spend_dict[i]['BREAKOUT'] == breakout) &
                                                       (awp_spend_dict[i]['ENTITY_TYPE'] == 'CLIENT'),]
            #spend_awp_breakout.columns = spend_awp_breakout.columns.str.upper()
            groupby_cols = ['FULLAWP_ADJ', 'FULLAWP_ADJ_PROJ_LAG', 'FULLAWP_ADJ_PROJ_EOY',
                            'GEN_LAG_AWP', 'GEN_LAG_ING_COST', 'GEN_EOY_AWP', 'GEN_EOY_ING_COST',
                            'PRICE_REIMB', 'LAG_REIMB', 'OLD_PRICE_EFFECTIVE_REIMB_PROJ_EOY', 'PRICE_EFFECTIVE_REIMB_PROJ',
                            'GER_TARGET']
            #shouldn't be neccessary anymore
            spend_awp_agg = spend_awp_breakout.groupby(['CLIENT', 'BREAKOUT'])[groupby_cols].sum().reset_index()

            #evaluate average daily values for ytd, lag, and eoy periods
            avg_daily_val = pd.DataFrame()
            avg_daily_val['CLIENT'] = spend_awp_agg['CLIENT']
            avg_daily_val['BREAKOUT'] = spend_awp_agg['BREAKOUT']

            avg_daily_val['FULLAWP_ADJ'] = spend_awp_agg['FULLAWP_ADJ'] / ytd_days
            avg_daily_val['FULLAWP_ADJ_PROJ_LAG'] = (spend_awp_agg['FULLAWP_ADJ_PROJ_LAG'] + spend_awp_agg['GEN_LAG_AWP']) / lag_days
            avg_daily_val['FULLAWP_ADJ_PROJ_EOY'] = (spend_awp_agg['FULLAWP_ADJ_PROJ_EOY'] + spend_awp_agg['GEN_EOY_AWP']) / eoy_days

            avg_daily_val['PRICE_REIMB'] = spend_awp_agg['PRICE_REIMB'] / ytd_days
            avg_daily_val['LAG_REIMB'] = (spend_awp_agg['LAG_REIMB'] + spend_awp_agg['GEN_LAG_ING_COST']) / lag_days
            avg_daily_val['OLD_PRICE_EFFECTIVE_REIMB_PROJ_EOY'] = (spend_awp_agg['OLD_PRICE_EFFECTIVE_REIMB_PROJ_EOY'] + spend_awp_agg['GEN_EOY_ING_COST']) / eoy_days
            avg_daily_val['PRICE_EFFECTIVE_REIMB_PROJ'] = (spend_awp_agg['PRICE_EFFECTIVE_REIMB_PROJ'] + spend_awp_agg['GEN_EOY_ING_COST']) / eoy_days

            #generate a long dataframe of daily ytd values
            ytd_daily_val = avg_daily_val[['CLIENT', 'BREAKOUT', 'FULLAWP_ADJ', 'PRICE_REIMB']]
            date_range = pd.DataFrame([pd.to_datetime('1/1/2021') + dt.timedelta(day) for day in range(ytd_days)], columns = ['DATE'])
            #lag_daily_val = pd.merge(lag_daily_val, date_range, how = 'cross') #requires pandas >= 1.2
            #hacky replacement for above method
            date_range['tmp_key'] = 1
            ytd_daily_val['tmp_key'] = 1
            ytd_daily_val = pd.merge(ytd_daily_val, date_range, on = 'tmp_key')
            ytd_daily_val.drop('tmp_key', axis = 1, inplace = True)
            #end of hacky replacement
            ytd_daily_val.rename(columns = {'FULLAWP_ADJ': 'AWP',
                                            'PRICE_REIMB': 'SPEND'},
                                 inplace = True)
            ytd_daily_val['Preexisting_SPEND'] = ytd_daily_val['SPEND']

            #generate a long dataframe of daily lag values
            lag_daily_val = avg_daily_val[['CLIENT', 'BREAKOUT', 'FULLAWP_ADJ_PROJ_LAG', 'LAG_REIMB']]
            date_range = pd.DataFrame([last_data + dt.timedelta(day) for day in range(lag_days)], columns = ['DATE'])
            #lag_daily_val = pd.merge(lag_daily_val, date_range, how = 'cross') #requires pandas >= 1.2
            #hacky replacement for above method
            date_range['tmp_key'] = 1
            lag_daily_val['tmp_key'] = 1
            lag_daily_val = pd.merge(lag_daily_val, date_range, on = 'tmp_key')
            lag_daily_val.drop('tmp_key', axis = 1, inplace = True)
            #end of hacky replacement
            lag_daily_val.rename(columns = {'FULLAWP_ADJ_PROJ_LAG': 'AWP',
                                            'LAG_REIMB': 'SPEND'},
                                 inplace = True)
            lag_daily_val['Preexisting_SPEND'] = lag_daily_val['SPEND']

            #generate a long dataframe of daily eoy values
            eoy_daily_val = avg_daily_val[['CLIENT', 'BREAKOUT', 'FULLAWP_ADJ_PROJ_EOY', 'PRICE_EFFECTIVE_REIMB_PROJ', 'OLD_PRICE_EFFECTIVE_REIMB_PROJ_EOY']]
            date_range = pd.DataFrame([pd.to_datetime(sim_params['GO_LIVE_LIST'][i]) + dt.timedelta(day) for day in range(eoy_days)], columns = ['DATE'])
            #eoy_daily_val = pd.merge(eoy_daily_val, date_range, how = 'cross') #requires pandas >= 1.2
            #hacky replacement for above method
            date_range['tmp_key'] = 1
            eoy_daily_val['tmp_key'] = 1
            eoy_daily_val = pd.merge(eoy_daily_val, date_range, on = 'tmp_key')
            eoy_daily_val.drop('tmp_key', axis = 1, inplace = True)
            #end of hacky replacement
            eoy_daily_val.rename(columns = {'FULLAWP_ADJ_PROJ_EOY': 'AWP',
                                            'PRICE_EFFECTIVE_REIMB_PROJ': 'SPEND',
                                            'OLD_PRICE_EFFECTIVE_REIMB_PROJ_EOY': 'Preexisting_SPEND'},
                                 inplace = True)


            #generate monthly cumulative total spend = ytd+lag+eoy spend
            total_awp_spend_month = pd.concat([ytd_daily_val, lag_daily_val, eoy_daily_val], ignore_index = True)
            total_awp_spend_month = total_awp_spend_month.groupby(['CLIENT',
                                                                   'BREAKOUT',
                                                                   total_awp_spend_month['DATE'].dt.month.rename('MONTH')])['AWP', 'SPEND', 'Preexisting_SPEND'].sum().reset_index()
            total_awp_spend_month['AWP_CUM'] = total_awp_spend_month['AWP'].cumsum()
            total_awp_spend_month['SPEND_CUM'] = total_awp_spend_month['SPEND'].cumsum()

            #evaluate exit ger for preexisting and model performances
            cols_to_keep = ['CLIENT', 'BREAKOUT', 'MONTH']
            if i == 0:
                total_awp_spend_month['Preexisting_SPEND_CUM'] = total_awp_spend_month['Preexisting_SPEND'].cumsum()
                total_awp_spend_month['GER_CUM_Preexisting'] = 1 - (total_awp_spend_month['Preexisting_SPEND_CUM'] / total_awp_spend_month['AWP_CUM'])
                breakout_ger_df = total_awp_spend_month[cols_to_keep + ['GER_CUM_Preexisting']]
            total_awp_spend_month['GER_CUM_GO_LIVE_{0}'.format(sim_params['GO_LIVE_LIST'][i])] = 1 - (total_awp_spend_month['SPEND_CUM'] / total_awp_spend_month['AWP_CUM'])
            #fill none before go live dates
            total_awp_spend_month.loc[total_awp_spend_month.MONTH < int(sim_params['GO_LIVE_LIST'][i].split('/')[0]) - 1,
                                      'GER_CUM_GO_LIVE_{0}'.format(sim_params['GO_LIVE_LIST'][i])] = None

            breakout_ger_df = pd.merge(breakout_ger_df,
                                       total_awp_spend_month[cols_to_keep + ['GER_CUM_GO_LIVE_{0}'.format(sim_params['GO_LIVE_LIST'][i])]],
                                       on = cols_to_keep,
                                       how = 'left')
            #fill from left column any missing values
            #breakout_ger_df['GER_CUM_GO_LIVE_{0}'.format(sim_params['GO_LIVE_LIST'][i])] = breakout_ger_df.apply(lambda x:  x[x.last_valid_index()], axis = 1)

        #print breakout exit ger to <Exit_GER> sheet
        breakout_ger_df['GER_TARGET'] = awp_spend_dict[0].loc[(awp_spend_dict[0]['BREAKOUT'] == breakout) &
                                                              (awp_spend_dict[0]['ENTITY_TYPE'] == 'CLIENT'), 'GER_TARGET'].values[0]
        breakout_ger_df.to_excel(excel_writer,
                                 sheet_name = "Exit_GER",
                                 index = False,
                                 startrow = start_row,
                                 startcol = 0)

        end_row = start_row + breakout_ger_df.shape[0]
        start_col = 0
        end_col = start_col + breakout_ger_df.shape[1]

        workbook = excel_writer.book
        worksheet = excel_writer.sheets['Exit_GER']
        chart_breakout = workbook.add_chart({'type': 'line'})

        for i in range(start_col + 3, end_col): #start_col + 3 excludes CLIENT, BREAKOUT, MONTH columns
            #configure the series of the chart from the <Exit_GER> sheet
            #values/categories: [sheetname, first_row, first_col, last_row, last_col]
            #name: [sheetname, row, col]
            chart_breakout.add_series({'values': ['Exit_GER', start_row + 1, i, end_row, i],
                                       'categories': ['Exit_GER', start_row + 1, 2, end_row, 2],
                                       'name': ['Exit_GER', start_row, i],
                                       'marker': {'type': 'diamond'}})
        #add title, set axis and shape
        chart_breakout.set_title({'name': '{0} Cumulative GER'.format(breakout)})
        chart_breakout.set_size({'width': 900, 'height': 300})
        chart_breakout.set_legend({'position': 'bottom'})
        chart_breakout.set_y_axis({
            'major_gridlines': {
                'visible': True,
                'line': {'width': 0.75,
                         'dash_type': 'dash'}
            },
        })
        chart_breakout.set_y_axis({'name':'GER (%)'})
        #insert the chart into the worksheet
        #insert_chart(row, col, chart, [options])
        worksheet.insert_chart(start_row + 15 if breakout != breakout_list[0] else start_row,
                               end_col + 1,
                               chart_breakout)

        #adjust next breakout operation's start row
        start_row = end_row + 2
    return excel_writer

def create_sim_report(output_path, custom_params, sim_params, final_report_name, num_iter, initial_last_data):
    '''
    Creates an Excel workbook with tabs showing annual performance, monthly performance, and monthly GER for all iterations of a particular client.
    This workbook can be used to compare the different itertations and get an overall view of performance in the simulation.
    '''
    import os
    import re
    import numpy as np
    import pandas as pd
    import GER_LP_Code.CPMO_parameters as p
    import GER_LP_Code.util_funcs as uf
    import GER_LP_Code.BQ as BQ
    from string import ascii_uppercase

    #reading reporint data##################################################################################################
    print('reading reporting data...')

    #########################################################
    #read-in pre-existing performance
    if p.WRITE_TO_BQ:
        prex_perf_df = uf.read_BQ_data(BQ.performance_files,
                                       project_id = p.BQ_OUTPUT_PROJECT_ID,
                                       dataset_id = p.BQ_OUTPUT_DATASET,
                                       table_id = "Prexisting_Performance",
                                       run_id = sim_params['AT_RUN_ID_LIST'][0],
                                       client = ', '.join(sorted(p.CUSTOMER_ID)),
                                       period = p.TIMESTAMP,
                                       output = True)
    else:
        prex_perf_file_name = re.sub(r'_\d_', #pattern match to find the go live month substring in the filename
                                     r'_{0}_'.format(sim_params['GO_LIVE_LIST'][0].split('/')[0]), #the first go live month
                                     p.PRE_EXISTING_PERFORMANCE_OUTPUT) #preexisting performance file name of the last go live date
        prex_perf_path = os.path.join(output_path + '/GO_LIVE_{0}/Output'.format(sim_params['GO_LIVE_LIST'][0].replace('/', '-')),
                                      prex_perf_file_name)
        prex_perf_df = pd.read_csv(prex_perf_path)
    prex_perf_df.columns = ['Entity', 'Without_Changes']

    #########################################################
    #read-in model performances for all simulation iterations
    #merge each iteration's model performance into <final_perf_df>
    final_perf_df = prex_perf_df.copy(deep = True)
    for i in range(num_iter):
        if p.WRITE_TO_BQ:
            iter_perf_df = uf.read_BQ_data(BQ.performance_files, #the query is similar in Model and Pre_existing performance tables
                                           project_id = p.BQ_OUTPUT_PROJECT_ID,
                                           dataset_id = p.BQ_OUTPUT_DATASET,
                                           table_id = "Model_Performance",
                                           run_id = sim_params['AT_RUN_ID_LIST'][i],
                                           client = ', '.join(sorted(p.CUSTOMER_ID)),
                                           period = p.TIMESTAMP,
                                           output = True)
        else:
            iter_perf_file_name = re.sub(r'_\d_', #pattern match to find the go live month substring in the filename
                                         r'_{0}_'.format(sim_params['GO_LIVE_LIST'][i].split('/')[0]), #iteration i's go live month
                                         p.MODEL_PERFORMANCE_OUTPUT) #model performance file name of the last go live date
            iter_perf_path = os.path.join(output_path + '/GO_LIVE_{0}/Output'.format(sim_params['GO_LIVE_LIST'][i].replace('/', '-')),
                                          iter_perf_file_name)
            iter_perf_df = pd.read_csv(iter_perf_path)
        iter_perf_df.columns = ['Entity', 'GO_LIVE_{0}'.format(sim_params['GO_LIVE_LIST'][i].replace('/', '-'))]
        final_perf_df = pd.merge(final_perf_df, iter_perf_df, on = 'Entity')

    #########################################################
    #read-in monthly projections with old prices
    first_go_live_month = int(sim_params['GO_LIVE_LIST'][0].split('/')[0])
    if False: #p.WRITE_TO_BQ:
        old_proj_df = uf.read_BQ_data(BQ.old_prices_monthly_projections,
                                      project_id = p.BQ_OUTPUT_PROJECT_ID,
                                      dataset_id = p.BQ_OUTPUT_DATASET,
                                      table_id = "OLD_PRICES_MONTHLY_PROJECTIONS",
                                      run_id = sim_params['AT_RUN_ID_LIST'][0],
                                      client = ', '.join(sorted(p.CUSTOMER_ID)),
                                      period = p.TIMESTAMP,
                                      output = True)
    else:
        old_proj_file_name = 'OLD_PRICES_MONTHLY_PROJECTIONS_MONTHS_{0}_THROUGH_13{1}.csv'.format(str(first_go_live_month),
                                                                                                  p.TIMESTAMP)
        old_proj_path = os.path.join(output_path + '/GO_LIVE_{0}/Output'.format(sim_params['GO_LIVE_LIST'][0].replace('/', '-')),
                                     old_proj_file_name)
        old_proj_df = pd.read_csv(old_proj_path)
    
    #filter for the correct through month and projected months
    #NOTE that this is because the schema of the table in BQ is set to change dynamically in simulation mode and additional columns will appear in the corresponding BQ table
    cols_to_keep = ['ENTITY'] + ['THROUGH_MONTH_{0}'.format(str(first_go_live_month - 1))] + ['MONTH_{0}'.format(str(month)) for month in range(first_go_live_month, 12 + 1)]
    old_proj_df = old_proj_df[cols_to_keep]
    
    #this line of code might not be neccessary
    old_proj_df.columns = [col.replace('THROUGH_', '') for col in old_proj_df.columns]
    
    #<final_proj_df> is populated by preexisting performance of old prices as well as performance of new prices
    #old price projections (and later new price projections) are filtered for client breakouts
    final_proj_df = old_proj_df.loc[old_proj_df['ENTITY'].str.contains(p.CUSTOMER_ID[0])]
    final_proj_df.insert(1, 'Iteration', 'Preexisting')

    #########################################################
    #read-in monthly projections with new prices
    new_proj_dfs = {}
    for i in range(num_iter):
        iter_go_live_month = int(sim_params['GO_LIVE_LIST'][i].split('/')[0])
        if False: #p.WRITE_TO_BQ:
            new_proj_dfs[i] = uf.read_BQ_data(BQ.old_prices_monthly_projections, #this query is similar in reading old and new prices projections
                                              project_id = p.BQ_OUTPUT_PROJECT_ID,
                                              dataset_id = p.BQ_OUTPUT_DATASET,
                                              table_id = "NEW_PRICES_MONTHLY_PROJECTIONS",
                                              run_id = sim_params['AT_RUN_ID_LIST'][i],
                                              client = ', '.join(sorted(p.CUSTOMER_ID)),
                                              period = p.TIMESTAMP,
                                              output = True)
        else:
            new_proj_file_name = 'NEW_PRICES_MONTHLY_PROJECTIONS_MONTHS_{0}_THROUGH_13{1}.csv'.format(str(iter_go_live_month),
                                                                                                      p.TIMESTAMP)
            new_proj_path = os.path.join(output_path + '/GO_LIVE_{0}/Output'.format(sim_params['GO_LIVE_LIST'][i].replace('/', '-')),
                                         new_proj_file_name)
            new_proj_dfs[i] = pd.read_csv(new_proj_path)
        
        #filter for the correct through month and other months
        #NOTE that this is because the schema of the table in BQ is set to change dynamically in simulation mode and additional columns will appear in the corresponding BQ table
        cols_to_keep = ['ENTITY'] + ['THROUGH_MONTH_{0}'.format(str(iter_go_live_month - 1))] + ['MONTH_{0}'.format(str(month)) for month in range(iter_go_live_month, 12 + 1)]
        new_proj_dfs[i] = new_proj_dfs[i][cols_to_keep]
        
        new_proj_dfs[i] = new_proj_dfs[i].loc[new_proj_dfs[i]['ENTITY'].str.contains(p.CUSTOMER_ID[0])] #filter client breakouts
        new_proj_dfs[i] = new_proj_dfs[i].drop(new_proj_dfs[i].columns[1], axis=1) #drop and insert a column to rename performance rows
        new_proj_dfs[i].insert(1, 'Iteration', sim_params['GO_LIVE_LIST'][i])

        #fill in the gaps between different iterations
        #depending on the go live dates, there are gaps in projection data that is required to create the report
        #this is because each projection data starts from the go live month of that data
        #so if the first iteration's go live month and the second iteration's go live month are more than one month apaprt
        #the months in-between those go live dates should be filled
        if i == 0:
            #fill in gaps with old price projections
            missing_cols = list(set(old_proj_df.columns) - set(new_proj_dfs[i].columns))
            missing_cols.sort(key = lambda x:x.split('_')[-1])
            if missing_cols:
                for col in missing_cols[::1]:
                    new_proj_dfs[i].insert(2, col, old_proj_df[col])
        else:
            #fill in gaps with previous new price projections
            missing_cols = list(set(new_proj_dfs[i-1].columns) - set(new_proj_dfs[i].columns))
            missing_cols.sort(key = lambda x:x.split('_')[-1])
            if missing_cols:
                for col in missing_cols[::-1]:
                    new_proj_dfs[i].insert(2, col, new_proj_dfs[i-1][col])
        final_proj_df = final_proj_df.append(new_proj_dfs[i])

    #price performances are a cumulative sum
    final_proj_df[final_proj_df.columns[2::]] = final_proj_df[final_proj_df.columns[2::]].cumsum(axis=1)
    #########################################################
    #read-in AWP spend data for exit GER report
    awp_spend_dfs = {}
    for i in range(num_iter):
        if p.WRITE_TO_BQ:
            awp_spend_dfs[i] = uf.read_BQ_data(BQ.awp_spend_total,
                                               project_id = p.BQ_OUTPUT_PROJECT_ID,
                                               dataset_id = p.BQ_OUTPUT_DATASET,
                                               table_id = "awp_spend_total",
                                               run_id = sim_params['AT_RUN_ID_LIST'][i],
                                               client = ', '.join(sorted(p.CUSTOMER_ID)),
                                               period = p.TIMESTAMP,
                                               output = True)
        else:
            awp_spend_file_name = 'awp_spend_total_{0}.csv'.format(p.DATA_ID)
            awp_spend_path = os.path.join(output_path + '/GO_LIVE_{0}/Output'.format(sim_params['GO_LIVE_LIST'][i].replace('/', '-')),
                                          awp_spend_file_name)
            awp_spend_dfs[i] = pd.read_csv(awp_spend_path)

    #########################################################
    #read-in new & improved AWP spend data for exit GER report
    new_awp_spend_dfs = {}
    for i in range(num_iter):
        if p.WRITE_TO_BQ:
            new_awp_spend_dfs[i] = uf.read_BQ_data(BQ.awp_spend_perf,
                                                   project_id = p.BQ_OUTPUT_PROJECT_ID,
                                                   dataset_id = p.BQ_OUTPUT_DATASET,
                                                   table_id = "awp_spend_perf",
                                                   run_id = sim_params['AT_RUN_ID_LIST'][i],
                                                   client = ', '.join(sorted(p.CUSTOMER_ID)),
                                                   period = p.TIMESTAMP,
                                                   output = True)
        else:
            awp_spend_file_name = 'joined_awp_spend_performance.csv'
            awp_spend_path = os.path.join(output_path + '/GO_LIVE_{0}/Report'.format(sim_params['GO_LIVE_LIST'][i].replace('/', '-')),
                                          awp_spend_file_name)
            new_awp_spend_dfs[i] = pd.read_csv(awp_spend_path)
    #create report sheets and charts#################################################################################################
    #register an excel writer
    writer = pd.ExcelWriter('{0}.xlsx'.format(final_report_name), engine = 'xlsxwriter')

    #create annual performance report
    writer = annual_perf_report(writer, final_perf_df, num_iter)

    #create monthly performance report
    writer = monthly_perf_report(writer, final_proj_df, sim_params, num_iter)

    #creat exit GER report
    #writer = exit_ger_report(writer, awp_spend_dfs, num_iter, sim_params, initial_last_data)
    writer = new_exit_ger_report(writer, new_awp_spend_dfs, num_iter, sim_params, initial_last_data)

    writer.save()
    os.system('gsutil -m -q cp {0} {1}'.format(final_report_name + '.xlsx', output_path))
    os.system('rm {0}'.format(final_report_name + '.xlsx'))
    print('simulation report created at: ...')

def preexisting_qa_report(excel_writer, perf_df, sim_params, num_iter):
    '''
    Creates a tab in an Excel workbook comparing performances across iterations.
    This is done to verify that data is being transferred correctly across iterations. Generally a succesful simulation run should have <1% difference between the performance of one iteration and the preexisting performance of the next for any given breakout.
    '''

    def prex_warn(x):
        prex_count = 0
        for i in range(1, num_iter):
            if abs(float(x['%_Mismatch_{0}_to_{1}'.format(sim_params['GO_LIVE_LIST'][i], sim_params['GO_LIVE_LIST'][i-1])])) >= 1:
                prex_count += 1
        if prex_count == 0:
            return 'OK'
        else:
            return 'Warning: High Mismatches between {0} iterations. Please check.'.format(prex_count)

    for i in range(num_iter):
        if i >= 1:
            mismatch = 100 * abs(perf_df['Preexisting_Performance_{0}'.format(sim_params['GO_LIVE_LIST'][i])] - \
                                 perf_df['Model_Performance_{0}'.format(sim_params['GO_LIVE_LIST'][i - 1])]
                                ) / perf_df['Model_Performance_{}'.format(sim_params['GO_LIVE_LIST'][i - 1])]

            mismatch_col_name = '%_Mismatch_{0}_to_{1}'.format(sim_params['GO_LIVE_LIST'][i], sim_params['GO_LIVE_LIST'][i - 1])
            insert_col_index = perf_df.columns.get_loc('Preexisting_Performance_{0}'.format(sim_params['GO_LIVE_LIST'][i]))
            perf_df.insert(insert_col_index, mismatch_col_name, mismatch)

            perf_df[mismatch_col_name].fillna(0, inplace = True)
            perf_df[mismatch_col_name] = perf_df[mismatch_col_name].map("{0:.3f}".format)

    perf_df['Status'] = perf_df.apply(lambda x: prex_warn(x), axis = 1)
    perf_df.to_excel(excel_writer, sheet_name = 'Preexisting_Perf_Validation', index = False)
    return excel_writer

def perf_deteriorate_qa_report(excel_writer, perf_df, sim_params, num_iter):
    '''
    Creates a tab in an excel workbook that tracks performance in breakouts across iterations. If there is a persistent decrease in performance across iterations this is highlighted for review. This may mean that more iterations of the simulation are not improving results for that client.
    '''

    def deter_warn(x):
        det_count = 0
        for i in range(num_iter):
            if x['Performance_Deterioration_{0}'.format(sim_params['GO_LIVE_LIST'][i])] > 0:
                det_count += 1

        if det_count == 0:
            return 'OK'
        else:
            return 'Warning: Performancy Deteriorated {0} times in run'.format(det_count)

    #filter performance dataframe <perf_df> for the first preexisting performance and all iterations' model performance
    first_preexisting_col = 'Preexisting_Performance_{}'.format(sim_params['GO_LIVE_LIST'][0])
    perf_df_cols = ['ENTITY', first_preexisting_col] + perf_df.columns[perf_df.columns.str.startswith('Model_Performance_')].tolist()
    perf_df = perf_df[perf_df_cols]

    for i in range(num_iter):
        if i == 0:
            perf_delta = abs(perf_df['Model_Performance_{0}'.format(sim_params['GO_LIVE_LIST'][i])]) - \
                         abs(perf_df[first_preexisting_col])
        else:
            perf_delta = abs(perf_df['Model_Performance_{0}'.format(sim_params['GO_LIVE_LIST'][i])]) - \
                         abs(perf_df['Model_Performance_{0}'.format(sim_params['GO_LIVE_LIST'][i - 1])])

        perf_delta_col_name = 'Performance_Deterioration_{0}'.format(sim_params['GO_LIVE_LIST'][i])
        insert_col_index = perf_df.columns.get_loc('Model_Performance_{0}'.format(sim_params['GO_LIVE_LIST'][i])) + 1
        perf_df.insert(insert_col_index, perf_delta_col_name, perf_delta)

    perf_df['Status'] = perf_df.apply(lambda x: deter_warn(x), axis = 1)
    perf_df.to_excel(excel_writer, sheet_name = 'Performance_Deterioration_Val', index = False)
    return excel_writer

def obj_deteriorate_qa_report(excel_writer, perf_df, sim_params, num_iter):
    '''
    Creates a tab in an excel workbook with a table tracking the overall objective value of each iteration.
    If the objective value is increasing between iterations it means more iterations are harming the performance of the model. It also means the optimization model is unable to find any improvements.
    '''
    import GER_LP_Code.CPMO_parameters as p
    prex_col = 'Preexisting_Performance_{}'.format(sim_params['GO_LIVE_LIST'][0])
    prex_perf_df = perf_df[['ENTITY', prex_col]]

    #contribution from the pharmacies not in the big capped list to the objective function
    df = prex_perf_df.loc[prex_perf_df['ENTITY'].isin(set(p.AGREEMENT_PHARMACY_LIST) - set(p.BIG_CAPPED_PHARMACY_LIST))]
    df['OBJECTIVE_PRE_EX'] = df[prex_col]
    df.loc[df['OBJECTIVE_PRE_EX'] < 0, 'OBJECTIVE_PRE_EX'] = 0

    #contribution from the client to the objective function
    prex1 = prex_perf_df.loc[~prex_perf_df['ENTITY'].isin(p.PHARMACY_LIST)]
    prex_mail = prex1.loc[prex1['ENTITY'].str.contains("MAIL")]
    prex_retail = prex1.loc[~prex1['ENTITY'].str.contains("MAIL")]
    prex_mail['OBJECTIVE_PRE_EX'] = abs(prex_mail[prex_col])
    prex_retail['OBJECTIVE_PRE_EX'] = prex_retail[prex_col]
    prex_retail.loc[prex_retail['OBJECTIVE_PRE_EX'] > 0, 'OBJECTIVE_PRE_EX'] = 0

    prex_retail['OBJECTIVE_PRE_EX'] = -1 * prex_retail['OBJECTIVE_PRE_EX']

    #contribution from big capped pharmacies to the objective function
    prex2 = prex_perf_df.loc[prex_perf_df['ENTITY'].isin(p.BIG_CAPPED_PHARMACY_LIST)]
    prex2['OBJECTIVE_PRE_EX_1'] = prex2[prex_col]
    prex2.loc[prex2['OBJECTIVE_PRE_EX_1'] < 0, 'OBJECTIVE_PRE_EX_1'] = 0
    prex2['OBJECTIVE_PRE_EX_2'] = prex2[prex_col]
    prex2.loc[prex2['OBJECTIVE_PRE_EX_2'] > 0, 'OBJECTIVE_PRE_EX_2'] = 0

    prex2['OBJECTIVE_PRE_EX'] = prex2['OBJECTIVE_PRE_EX_1'] + 0.1 * prex2['OBJECTIVE_PRE_EX_2']

    prex_obj = pd.concat([prex_mail, prex_retail, prex2, df])
    prex_obj = prex_obj.drop(['OBJECTIVE_PRE_EX_1', 'OBJECTIVE_PRE_EX_2', prex_col], axis = 1)

    for i in range(num_iter):
        model_col = 'Model_Performance_{}'.format(sim_params['GO_LIVE_LIST'][i])
        model_df = perf_df[['ENTITY', model_col]]

        #contribution from the pharmacies not in the big capped list to the objective function
        obj_df = model_df.loc[model_df['ENTITY'].isin(set(p.AGREEMENT_PHARMACY_LIST)-set(p.BIG_CAPPED_PHARMACY_LIST))]
        obj_df['OBJECTIVE'] = obj_df[model_col]
        obj_df.loc[obj_df['OBJECTIVE']<0,'OBJECTIVE'] = 0

        #contribution from the client to the objective function
        perf1 = model_df.loc[~model_df['ENTITY'].isin(p.PHARMACY_LIST)]
        perf_mail = perf1.loc[perf1['ENTITY'].str.contains("MAIL")]
        perf_retail = perf1.loc[~perf1['ENTITY'].str.contains("MAIL")]
        perf_mail['OBJECTIVE'] = abs(perf_mail[model_col])
        perf_retail['OBJECTIVE'] = perf_retail[model_col]
        perf_retail.loc[perf_retail['OBJECTIVE'] > 0, 'OBJECTIVE'] = 0

        perf_retail['OBJECTIVE'] = -1 * perf_retail['OBJECTIVE']

        # df2 is the contribution from the BIG_CAPPED to the objective function
        perf2 = model_df.loc[model_df['ENTITY'].isin(p.BIG_CAPPED_PHARMACY_LIST)]
        perf2['OBJECTIVE_1'] = perf2[model_col]
        perf2.loc[perf2['OBJECTIVE_1'] < 0, 'OBJECTIVE_1'] = 0
        perf2['OBJECTIVE_2'] = perf2[model_col]
        perf2.loc[perf2['OBJECTIVE_2']> 0 , 'OBJECTIVE_2'] = 0

        perf2['OBJECTIVE'] = perf2['OBJECTIVE_1'] + 0.1 * perf2['OBJECTIVE_2']

        perf_obj = pd.concat([perf_mail, perf_retail, perf2, obj_df])
        perf_obj = perf_obj.drop(['OBJECTIVE_1', 'OBJECTIVE_2', model_col], axis = 1)
        perf_obj.columns = ['ENTITY', 'Obj_{0}'.format(sim_params['GO_LIVE_LIST'][i])]
        prex_obj = pd.merge(prex_obj, perf_obj, on='ENTITY')

    last_obj = prex_obj['OBJECTIVE_PRE_EX'].sum()
    cols = ['Preexisting_Obj']
    objs = [[last_obj]]
    for i in range(num_iter):
        cols.append('Obj_{0}'.format(sim_params['GO_LIVE_LIST'][i]))
        objs[0].append(prex_obj['Obj_{0}'.format(sim_params['GO_LIVE_LIST'][i])].sum())

    objective_df = pd.DataFrame(objs, columns = cols)
    objective_df.to_excel(excel_writer, sheet_name = 'Objective_Check', index = False)
    return excel_writer

def create_qa_report(output_path, custom_params, sim_params, final_report_name, num_iter):
    '''
    Creates an Excel workbook for QA reports. Contains tabs for preexisting performance matching checks, overall performance checks, and overall objective value checks. This workbook can be used to verify that the simulation is performing as expected with no bugs in its data transfers between iterations.
    '''
    import os
    import re
    import numpy as np
    import pandas as pd
    pd.options.mode.chained_assignment = None
    import GER_LP_Code.CPMO_parameters as p
    import GER_LP_Code.util_funcs as uf
    import GER_LP_Code.BQ as BQ
    from string import ascii_uppercase

    writer = pd.ExcelWriter('{0}.xlsx'.format(final_report_name), engine='xlsxwriter')

    #reading reporint data##################################################################################################
    #read-in preexisting and model performances
    final_perf_df = pd.DataFrame(columns = ['ENTITY'])
    for i in range(num_iter):
        if p.WRITE_TO_BQ:
            prex_perf_df = uf.read_BQ_data(BQ.performance_files,
                                           project_id = p.BQ_OUTPUT_PROJECT_ID,
                                           dataset_id = p.BQ_OUTPUT_DATASET,
                                           table_id = "Prexisting_Performance",
                                           run_id = sim_params['AT_RUN_ID_LIST'][i],
                                           client = ', '.join(sorted(p.CUSTOMER_ID)),
                                           period = p.TIMESTAMP,
                                           output = True)
            model_perf_df = uf.read_BQ_data(BQ.performance_files, #the query is similar in Model and Pre_existing performance tables
                                            project_id = p.BQ_OUTPUT_PROJECT_ID,
                                            dataset_id = p.BQ_OUTPUT_DATASET,
                                            table_id = "Model_Performance",
                                            run_id = sim_params['AT_RUN_ID_LIST'][i],
                                            client = uf.get_formatted_string(p.CUSTOMER_ID),
                                            period = p.TIMESTAMP,
                                            output = True)
        else:
            prex_perf_file_name = re.sub(r'_\d_', #pattern match to find the go live month substring in the filename
                                         r'_{0}_'.format(sim_params['GO_LIVE_LIST'][i].split('/')[0]), #the first go live month
                                         p.PRE_EXISTING_PERFORMANCE_OUTPUT) #preexisting performance file name of the last go live date
            model_perf_file_name = re.sub(r'_\d_', #pattern match to find the go live month substring in the filename
                                          r'_{0}_'.format(sim_params['GO_LIVE_LIST'][i].split('/')[0]), #iteration i's go live month
                                          p.MODEL_PERFORMANCE_OUTPUT) #model performance file name of the last go live date

            prex_perf_path = os.path.join(output_path + '/GO_LIVE_{0}/Output'.format(sim_params['GO_LIVE_LIST'][i].replace('/', '-')),
                                          prex_perf_file_name)
            model_perf_path = os.path.join(output_path + '/GO_LIVE_{0}/Output'.format(sim_params['GO_LIVE_LIST'][i].replace('/', '-')),
                                           model_perf_file_name)

            prex_perf_df = pd.read_csv(prex_perf_path)
            model_perf_df = pd.read_csv(model_perf_path)

        prex_perf_df = prex_perf_df.rename(columns = {'PERFORMANCE':'Preexisting_Performance_{}'.format(sim_params['GO_LIVE_LIST'][i])})
        model_perf_df = model_perf_df.rename(columns = {'PERFORMANCE':'Model_Performance_{}'.format(sim_params['GO_LIVE_LIST'][i])})

        iter_perf_df = pd.merge(prex_perf_df, model_perf_df, on = 'ENTITY', how = 'left')
        final_perf_df = pd.merge(final_perf_df, iter_perf_df, on = 'ENTITY', how = 'outer')

    #create qa report sheets#################################################################################################
    #register an excel writer
    writer = pd.ExcelWriter('{0}.xlsx'.format(final_report_name), engine = 'xlsxwriter')

    #create preexisting performance qa report
    writer = preexisting_qa_report(writer, final_perf_df, sim_params, num_iter)

    #create breakout performance deterioration report
    writer = perf_deteriorate_qa_report(writer, final_perf_df, sim_params, num_iter)

    #create objective deterioration report
    writer = obj_deteriorate_qa_report(writer, final_perf_df, sim_params, num_iter)

    writer.save()
    os.system('gsutil -m -q cp {0} {1}'.format(final_report_name+'.xlsx',output_path))
    os.system('rm {0}'.format(final_report_name+'.xlsx'))
    print('QA Report Created')