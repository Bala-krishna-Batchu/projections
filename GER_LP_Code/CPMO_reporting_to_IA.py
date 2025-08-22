'''
This code provides the information needed for both dashboards, PSOT and Tableau.

All the input files that are needed for either of the calculations are created directly from the output/input files from the code.

Further work should limit the creation of duplicate files and streamline things since it is not optimal.
The files and paths are read directly from CPMO_parameters

'''

def arguments():
    import argparse as ap
    import json
    import jinja2 as jj2
    import BQ
    from collections import namedtuple

    parser = ap.ArgumentParser()
    parser.add_argument(
        '--custom-args-json',
        help=('JSON file URL, file supplying custom values. '
              'for example: '
              '{"client_name": "client1", "TIMESTAMP": "2020-12-29", ...}'),
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
        '--loglevel', choices=['INFO', 'DEBUG', 'WARN', 'ERROR', 'FATAL', 'CRITICAL'],
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
        arguments = dict(json.load(open(args.custom_args_json)))
        template = jj2.Template(open(args.template).read())
        return outpt(params=template.render(**arguments), loglevel=args.loglevel)



def create_reporting_tables(params_in: str):
    """
    This part of code takes in the LP run result and generates awp_spend_perf, ytd_surplus_monthly,
    price_change and price_dist tables for reporting usage
    :param params_in: CPMO_parameters.py
    :return: the result of the LP run to either BQ or csv files
    """
    import pandas as pd
    import numpy as np
    import datetime as dt
    import os
    import BQ
    import util_funcs as uf
    import datetime
    from dateutil import relativedelta
    from types import ModuleType
    import sys
    if isinstance(params_in, ModuleType):
        p = params_in
    else:
        uf.write_params(params_in)
        import CPMO_parameters as p
    from CPMO_shared_functions import standardize_df, check_and_create_folder, update_run_status, add_virtual_r90, check_run_status, add_target_ingcost
    from Daily_Input_Read import merge_nadac_wac_netcostguarantee
    import CPMO_costsaver_functions as cs
    try:
        
        if check_run_status(run_status = 'Complete-BypassPerformance'): 
            if p.FULL_YEAR: 
                RUN_TYPE_TABLEAU_UPDATED = "".join([p.RUN_TYPE_TABLEAU, "-BypassPerformance_WS"])
            else: 
                RUN_TYPE_TABLEAU_UPDATED = "".join([p.RUN_TYPE_TABLEAU, "-BypassPerformance"])
        else:
            RUN_TYPE_TABLEAU_UPDATED = p.RUN_TYPE_TABLEAU
        
        
        
        ## Changes to make sure PSAOs are not being double counted
        p.SMALL_CAPPED_PHARMACY_LIST['GNRC']=list(set(p.SMALL_CAPPED_PHARMACY_LIST['GNRC']) - set(p.PSAO_LIST['GNRC']))
        p.SMALL_CAPPED_PHARMACY_LIST['BRND']=list(set(p.SMALL_CAPPED_PHARMACY_LIST['BRND']) - set(p.PSAO_LIST['BRND']))
        p.NON_CAPPED_PHARMACY_LIST['GNRC']=list(set(p.NON_CAPPED_PHARMACY_LIST['GNRC']+p.COGS_PHARMACY_LIST['GNRC']) - set(p.PSAO_LIST['GNRC']))
        p.NON_CAPPED_PHARMACY_LIST['BRND']=list(set(p.NON_CAPPED_PHARMACY_LIST['BRND']+p.COGS_PHARMACY_LIST['BRND']) - set(p.PSAO_LIST['BRND']))
        
        ################################## Breakout Mapping ##################################
        """
            This part of the code read the breakout mapping table
        """
        breakout_mapping = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.BREAKOUT_MAPPING_FILE, dtype = p.VARIABLE_TYPE_DIC))
        breakout_mapping = breakout_mapping[['CUSTOMER_ID','BREAKOUT','LABEL']].drop_duplicates().reset_index(drop = True)
        
        def breakout_label_mapping(data, breakout_mapping = breakout_mapping):
            data_upload = data.copy()
            data_upload = pd.merge(data_upload, breakout_mapping, how = 'left', on = ['CUSTOMER_ID','BREAKOUT'])
            data_upload['BREAKOUT'] = np.where(data_upload['LABEL'].isna(), data_upload['BREAKOUT'], data_upload['LABEL'])
            if 'ENTITY' in data.columns.tolist():
                data_upload['ENTITY'] = np.where(data_upload['LABEL'].isna(), data_upload['ENTITY'], data_upload['CUSTOMER_ID'] + "_" + data_upload['LABEL'])
            if 'CHAIN_GROUP' in data.columns.tolist():
                data_upload['CHAIN_GROUP'] = np.where(data_upload['LABEL'].isna(), data_upload['CHAIN_GROUP'], data_upload['LABEL'])
            if 'CHAIN_SUBGROUP' in data.columns.tolist():
                data_upload['CHAIN_SUBGROUP'] = np.where(data_upload['LABEL'].isna(), data_upload['CHAIN_SUBGROUP'], data_upload['LABEL'])
            data_upload.drop(columns = ['LABEL'], inplace = True)
            return data_upload
        
        ################################## Awp_spend_perf ##################################
        """
            This part of code takes in the LP run result and generates awp_spend_perf
        """
        check_and_create_folder(p.FILE_REPORT_PATH)
        # Read in awp_spend_total
        if p.WRITE_TO_BQ:
            awp_spend_total = uf.read_BQ_data(BQ.awp_spend_total,
                                              project_id = p.BQ_OUTPUT_PROJECT_ID,
                                              dataset_id = p.BQ_OUTPUT_DATASET,
                                              table_id = 'awp_spend_total_medd_comm_subgroup',
                                              client = ', '.join(sorted(p.CUSTOMER_ID)),
                                              period = p.TIMESTAMP,
                                              run_id=p.AT_RUN_ID,
                                              output = True)
        else:
            awp_spend_total = pd.read_csv(os.path.join(p.FILE_OUTPUT_PATH, 'awp_spend_total_' + p.DATA_ID + '.csv'),
                                          dtype=p.VARIABLE_TYPE_DIC)
        awp_spend_total = standardize_df(awp_spend_total)
            
        # Read in pharm_awp_spend_total
        if p.WRITE_TO_BQ:
            pharm_awp_spend_total = uf.read_BQ_data(BQ.pharm_awp_spend_total,
                                              project_id = p.BQ_OUTPUT_PROJECT_ID,
                                              dataset_id = p.BQ_OUTPUT_DATASET,
                                              table_id = 'pharm_awp_spend_total_medd_comm_subgroup',
                                              client = ', '.join(sorted(p.CUSTOMER_ID)),
                                              period = p.TIMESTAMP,
                                              run_id=p.AT_RUN_ID,
                                              output = True)
        else:
            pharm_awp_spend_total = pd.read_csv(os.path.join(p.FILE_OUTPUT_PATH, 'pharm_awp_spend_total_' + p.DATA_ID + '.csv'),
                                          dtype=p.VARIABLE_TYPE_DIC)
        pharm_awp_spend_total = standardize_df(pharm_awp_spend_total)[['CLIENT','REGION', 'BREAKOUT', 'CHAIN_GROUP','CHAIN_SUBGROUP','MEASUREMENT', 'BG_FLAG', 'PHARMACY_TYPE', 'IS_MAC', 'IS_SPECIALTY',
                                                                       'PHARM_TARG_INGCOST_ADJ','PHARM_TARG_INGCOST_ADJ_PROJ_LAG','PHARM_TARG_INGCOST_ADJ_PROJ_EOY']]
        
        old_len = len(awp_spend_total)
        
        awp_spend_total = awp_spend_total.merge(pharm_awp_spend_total, on = ['CLIENT','REGION', 'BREAKOUT', 'CHAIN_GROUP','CHAIN_SUBGROUP','MEASUREMENT', 'BG_FLAG',
                                                                                                 'PHARMACY_TYPE',  'IS_MAC', 'IS_SPECIALTY'], how='left')
        
        assert len(awp_spend_total) == old_len, "Duplication after merge of awp_spend and pharm_awp_spend"

        # Read in Gen_launch table
        generic_launch_df = standardize_df(pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.GENERIC_LAUNCH_FILE), dtype = p.VARIABLE_TYPE_DIC))

        # Assign month type,
        # (p.DATA_START_DAY - p.LAST_DATA.month) : YTD
        # (p.LAST_DATA.month - GO_LIVE) : LAG
        # (GO_LIVE -> 'EOY')
        # split the value of month last_day/go_live base on the proportion of days and assign the values to LAG, EOY
        # apply logic on the dataframe


        # Assign all months as EOY , fill in LAG at the end with 0.0
        if p.FULL_YEAR:
            generic_launch_df['MONTH_TYPE'] = 'EOY'
        else:
            generic_launch_df.loc[generic_launch_df['MONTH'] < p.LAST_DATA.month, 'MONTH_TYPE'] = 'YTD'
            generic_launch_df.loc[(generic_launch_df['MONTH'] > p.LAST_DATA.month) &
                                  (generic_launch_df['MONTH'] < p.GO_LIVE.month), 'MONTH_TYPE'] = 'LAG'
            generic_launch_df.loc[generic_launch_df['MONTH'] > p.GO_LIVE.month, 'MONTH_TYPE'] = 'EOY'
            # get the dates and days_in_month
            last_day = pd.to_datetime(p.LAST_DATA)
            go_live = pd.to_datetime(p.GO_LIVE)
            # split values of the month base on proportion of days
            # last day lag
            last_day_lag = generic_launch_df.loc[generic_launch_df['MONTH'] == p.LAST_DATA.month].copy()
            last_day_lag.loc[:,['QTY','FULLAWP','ING_COST']] = last_day_lag[['QTY','FULLAWP','ING_COST']] * (1-(last_day.day/last_day.days_in_month))
            last_day_lag['MONTH_TYPE'] = 'LAG'
            # go live lag
            go_live_lag = generic_launch_df.loc[generic_launch_df['MONTH'] == p.GO_LIVE.month].copy()
            go_live_lag.loc[:,['QTY','FULLAWP','ING_COST']] = go_live_lag[['QTY','FULLAWP','ING_COST']] * (go_live.day/go_live.days_in_month)
            go_live_lag['MONTH_TYPE'] = 'LAG'
            # go live eoy
            go_live_eoy = generic_launch_df.loc[generic_launch_df['MONTH'] == p.GO_LIVE.month].copy()
            go_live_eoy.loc[:,['QTY','FULLAWP','ING_COST']] = go_live_eoy[['QTY','FULLAWP','ING_COST']] * (1-(go_live.day/go_live.days_in_month))
            go_live_eoy['MONTH_TYPE'] = 'EOY'
            generic_launch_df = pd.concat([generic_launch_df, last_day_lag, go_live_lag, go_live_eoy])

        # slice EOY and LAG data only
        gen_lauch_LAG_EOY = generic_launch_df[generic_launch_df['MONTH_TYPE'].isin(['LAG','EOY'])]

        # agg AWP and ING_cost
        gen_lauch_LAG_EOY = gen_lauch_LAG_EOY.groupby(['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'BG_FLAG', 'CHAIN_GROUP', 'MONTH_TYPE']) \
            .agg({'FULLAWP': 'sum', 'ING_COST': 'sum'}).reset_index()


        gen_lauch_LAG_EOY_pivot = gen_lauch_LAG_EOY.pivot(index=['CLIENT', 'BREAKOUT', 'REGION','MEASUREMENT', 'BG_FLAG', 'CHAIN_GROUP'],
                                                          columns=['MONTH_TYPE'],
                                                          values=['FULLAWP', 'ING_COST']).reset_index()

        # reset to single level column index
        gen_lauch_LAG_EOY_pivot.columns = [' '.join(col).strip() for col in gen_lauch_LAG_EOY_pivot.columns.values]
        # rename columns to the original names
        gen_lauch_LAG_EOY_pivot.rename(columns={'FULLAWP EOY': 'GEN_EOY_AWP', 'FULLAWP LAG': 'GEN_LAG_AWP'
            , 'ING_COST EOY': 'GEN_EOY_ING_COST', 'ING_COST LAG': 'GEN_LAG_ING_COST'}
                                       , inplace=True)
        # fill in LAG with 0
        if p.FULL_YEAR:
            gen_lauch_LAG_EOY_pivot['GEN_LAG_AWP'] = 0.0
            gen_lauch_LAG_EOY_pivot['GEN_LAG_ING_COST'] = 0.0

        # Merge Awp_spend with pivoted GEN_LAUNCH_FILE
        awp_spend_total = pd.merge(awp_spend_total[['CLIENT', 'REGION','MEASUREMENT', 'BG_FLAG', 'BREAKOUT', 'CHAIN_GROUP','CHAIN_SUBGROUP','PHARMACY_TYPE', 'FULLAWP_ADJ',
                                                    'FULLAWP_ADJ_PROJ_LAG', 'FULLAWP_ADJ_PROJ_EOY', 'PRICE_REIMB',
                                                    'LAG_REIMB', 'OLD_PRICE_EFFECTIVE_REIMB_PROJ_EOY',
                                                    'PRICE_EFFECTIVE_REIMB_PROJ','TARG_INGCOST_ADJ_PROJ_EOY', 'TARG_INGCOST_ADJ',
                                                    'TARG_INGCOST_ADJ_PROJ_LAG','DISP_FEE','DISP_FEE_PROJ_LAG','DISP_FEE_PROJ_EOY',
                                                    'TARGET_DISP_FEE','TARGET_DISP_FEE_PROJ_LAG','TARGET_DISP_FEE_PROJ_EOY','PHARM_TARG_INGCOST_ADJ', 'PHARM_TARG_INGCOST_ADJ_PROJ_LAG',
                                                    'PHARM_TARG_INGCOST_ADJ_PROJ_EOY',
'COMPETITIVE_SCORE','CLAIM_PRICE_RATIO','CLAIMS']]
                                   , gen_lauch_LAG_EOY_pivot[['CLIENT', 'BREAKOUT','REGION','MEASUREMENT', 'BG_FLAG','CHAIN_GROUP', 'GEN_LAG_AWP',
                                                              'GEN_LAG_ING_COST', 'GEN_EOY_AWP', 'GEN_EOY_ING_COST']],
                                   on=['CLIENT','REGION','BREAKOUT','MEASUREMENT', 'BG_FLAG', 'CHAIN_GROUP'], how='left') #\
            #.rename(columns={'CLIENT': 'CUSTOMER_ID'})
            
        #Zero out YTD and LAG for 1/1 pricing runs as we only care about EOY
        awp_spend_total_backup=awp_spend_total[:]
        if p.FULL_YEAR:
            awp_spend_total['FULLAWP_ADJ'] = 0
            awp_spend_total['FULLAWP_ADJ_PROJ_LAG'] = 0
            awp_spend_total['TARG_INGCOST_ADJ'] = 0
            awp_spend_total['TARG_INGCOST_ADJ_PROJ_LAG'] = 0
            awp_spend_total['TARGET_DISP_FEE'] = 0
            awp_spend_total['TARGET_DISP_FEE_PROJ_LAG'] = 0
            awp_spend_total['DISP_FEE'] = 0
            awp_spend_total['DISP_FEE_PROJ_LAG'] = 0
            awp_spend_total['PRICE_REIMB'] = 0
            awp_spend_total['LAG_REIMB'] = 0
            awp_spend_total['PHARM_TARG_INGCOST_ADJ'] = 0
            awp_spend_total['PHARM_TARG_INGCOST_ADJ_PROJ_LAG'] = 0

            
        generic_report = awp_spend_total.groupby(['CLIENT','REGION','BREAKOUT','MEASUREMENT', 'BG_FLAG', 'PHARMACY_TYPE'])\
                                    [['FULLAWP_ADJ','FULLAWP_ADJ_PROJ_LAG','FULLAWP_ADJ_PROJ_EOY',\
                                      'PRICE_REIMB','LAG_REIMB','OLD_PRICE_EFFECTIVE_REIMB_PROJ_EOY',\
                                      'PRICE_EFFECTIVE_REIMB_PROJ','TARG_INGCOST_ADJ_PROJ_EOY', 'TARG_INGCOST_ADJ',
                                      'TARG_INGCOST_ADJ_PROJ_LAG','DISP_FEE','DISP_FEE_PROJ_LAG','DISP_FEE_PROJ_EOY',
                                      'TARGET_DISP_FEE','TARGET_DISP_FEE_PROJ_LAG','TARGET_DISP_FEE_PROJ_EOY','CLAIM_PRICE_RATIO','CLAIMS']].sum().reset_index()
        
        generic_report['COMPETITIVE_SCORE'] = generic_report['CLAIM_PRICE_RATIO']/ generic_report['CLAIMS']
        
        generic_report.drop(['CLAIM_PRICE_RATIO', 'CLAIMS'],axis = 1, inplace = True)

        # SPEND for Client metrics
        # TODO: how does this work for non-passthrough clients?
        # create rows for client-spend by aggregating breakout, for the purpose of joining with performance summary
        client_rows = awp_spend_total.drop(['CHAIN_GROUP', 'CHAIN_SUBGROUP','COMPETITIVE_SCORE'], axis=1).groupby(['CLIENT', 'BREAKOUT']).sum().reset_index()
                                                       
        
        # client_rows_comp_score = awp_spend_total.groupby(['CLIENT', 'BREAKOUT'])['COMPETITIVE_SCORE'].mean().reset_index()
        # client_rows = pd.merge(client_rows, client_rows_comp_score, how='left', on = ['CLIENT', 'BREAKOUT'])
        # bring back chain_group in order to concat client rows with pharmacy rows
        client_rows['CHAIN_GROUP'] = client_rows['BREAKOUT']
        client_rows['CHAIN_SUBGROUP'] = client_rows['BREAKOUT']
        # specify client OR pharmacy rows in order to join with performance output
        client_rows['CLIENT_OR_PHARM'] = 'CLIENT'

        # Agg pharmacy rows to avoid duplication of records in PSOT calculation
        pharmacy_rows = awp_spend_total.groupby(['CLIENT','CHAIN_GROUP','CHAIN_SUBGROUP']).sum().reset_index()
        pharmacy_rows['BREAKOUT'] = 'RETAIL'
        pharmacy_rows.loc[pharmacy_rows['CHAIN_GROUP']=='MAIL','BREAKOUT'] = 'MAIL'
        # SPEND for pharmacy rows
        pharmacy_rows['CLIENT_OR_PHARM'] = 'PHARMACY'
        

        # combine the two
        awp_spend_total_client_pharm = pd.concat([pharmacy_rows, client_rows], axis=0)
        
        awp_spend_total_client_pharm['COMPETITIVE_SCORE'] = awp_spend_total_client_pharm['CLAIM_PRICE_RATIO']/ awp_spend_total_client_pharm['CLAIMS']
        
        awp_spend_total_client_pharm.drop(['CLAIM_PRICE_RATIO', 'CLAIMS'],axis = 1, inplace = True)

        # Prepare performance dataframes
        # Read in performance file
        if p.WRITE_TO_BQ:
            pre_perf = uf.read_BQ_data(BQ.performance_files,
                                       project_id=p.BQ_OUTPUT_PROJECT_ID,
                                       dataset_id=p.BQ_OUTPUT_DATASET,
                                       table_id='Prexisting_Performance',
                                       client=', '.join(sorted(p.CUSTOMER_ID)),
                                       period=p.TIMESTAMP,
                                       run_id=p.AT_RUN_ID,
                                       output=True)
        else:
            pre_perf = pd.read_csv(os.path.join(p.FILE_OUTPUT_PATH, 'Pre_existing_Performance_{}_{}.csv'.format(str(p.GO_LIVE.month), p.DATA_ID)))

        # Read in model performance
        if p.WRITE_TO_BQ:
            mod_perf = uf.read_BQ_data(BQ.performance_files,
                                       project_id=p.BQ_OUTPUT_PROJECT_ID,
                                       dataset_id=p.BQ_OUTPUT_DATASET,
                                       table_id='Model_Performance',
                                       client=', '.join(sorted(p.CUSTOMER_ID)),
                                       period=p.TIMESTAMP,
                                       run_id=p.AT_RUN_ID,
                                       output=True)
        else:
            mod_perf = pd.read_csv(os.path.join(p.FILE_OUTPUT_PATH, 'Model_Performance_{}_{}.csv'.format(str(p.GO_LIVE.month), p.DATA_ID)))
        
        # The merger of the data
        performance_df = pd.merge(pre_perf.rename(columns={'PERFORMANCE': 'Pre_existing_Perf'}),
                                  mod_perf.rename(columns={'PERFORMANCE': 'Model_Perf'}),
                                  on='ENTITY')
        
        # Add column to specify client/pharmacy rows
        mask = (performance_df['ENTITY'].str.contains('[0-9]')) & ~(performance_df['ENTITY'].str.contains('R90OK'))
        performance_df.loc[mask, 'CLIENT_OR_PHARM'] = 'CLIENT'
        performance_df.loc[mask == False, 'CLIENT_OR_PHARM'] = 'PHARMACY'
        
        # Clean leading customer_id from ENTITY
        def get_client_entity(row):
            if row['CLIENT_OR_PHARM'] == 'CLIENT':
                return '_'.join(row['ENTITY'].split('_')[1:])
            else:
                return row['ENTITY']
        #June 7th- made this change to accomodate the pharmacy perf calcs on the chain subgroup level in calculate performance (CMP shared functions)
        performance_df['CHAIN_SUBGROUP'] = performance_df.apply(lambda x: get_client_entity(x), axis=1)
        
        #June 7th- made this block redundant
        '''
        def extract_subgroup(row):
            if row.CHAIN_GROUP[:4]=='CVS_':
                return row.CHAIN_GROUP[4:]
            return row.CHAIN_GROUP
        performance_df['CHAIN_SUBGROUP'] = performance_df.apply(extract_subgroup, axis=1)
        def extract_group(row):
            if row.CHAIN_GROUP[:4]=='CVS_':
                return row.CHAIN_GROUP[:3]
            return row.CHAIN_GROUP
        performance_df['CHAIN_GROUP'] = performance_df.apply(extract_group, axis=1)
        '''

            # Read in CLIENT_GUARANTEE_FILE
        client_guarantees = pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, 'Buffer_' + p.CLIENT_GUARANTEE_FILE), dtype = p.VARIABLE_TYPE_DIC)
        
        client_guarantees = standardize_df(client_guarantees)
        
        client_guarantees_meas = client_guarantees.copy(deep = True)
        
        if p.CLIENT_TYPE == 'MEDD' and p.FULL_YEAR and not p.EGWP:
            client_guarantees = standardize_df(pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, 'Buffer_' + p.CLIENT_GUARANTEE_FILE), dtype = p.VARIABLE_TYPE_DIC))
        
        # If client guarantees vary by measurement or pharmacy_type, we find a blended target rate for each breakout based on ytd_utilization 
        #if (client_guarantees.groupby(['CLIENT', 'REGION','BREAKOUT'])['RATE'].nunique().reset_index()['RATE'] == 1).all() == False:
        awp_total_client_side = awp_spend_total_backup[['CLIENT','REGION','MEASUREMENT', 'BG_FLAG','BREAKOUT','PHARMACY_TYPE','FULLAWP_ADJ']].groupby(
            ['CLIENT','REGION','MEASUREMENT','BREAKOUT','BG_FLAG', 'PHARMACY_TYPE']).sum().reset_index()
        awp_total_client_side=standardize_df(awp_total_client_side)
        temp_client_guarantee=pd.merge(client_guarantees,awp_total_client_side, how='left', on=['CLIENT','REGION','MEASUREMENT','BREAKOUT','BG_FLAG','PHARMACY_TYPE'])
        temp_client_guarantee['FULLAWP_ADJ']=temp_client_guarantee['FULLAWP_ADJ'].fillna(0)
        temp_client_guarantee['RATE_MULT_UTLIZATION']=temp_client_guarantee['RATE']*temp_client_guarantee['FULLAWP_ADJ']
        blended_rates=temp_client_guarantee[['CLIENT','REGION','BREAKOUT','FULLAWP_ADJ','RATE_MULT_UTLIZATION']].groupby(['CLIENT','REGION','BREAKOUT']).sum().reset_index()
        blended_rates['RATE']=blended_rates['RATE_MULT_UTLIZATION']/blended_rates['FULLAWP_ADJ']
        blended_rates['RATE']=round(blended_rates['RATE'],4)
        client_guarantees = blended_rates[['CLIENT','REGION','BREAKOUT','RATE']]
        if client_guarantees['RATE'].isna().any():
            breakouts = client_guarantees[client_guarantees['RATE'].isna()]['BREAKOUT'].unique()
            if not any(awp_spend_total_backup['BREAKOUT'].isin(breakouts)):
                client_guarantees = client_guarantees[~client_guarantees['RATE'].isna()]
                print('*Warning: Removing NaN guarantee from client guarantees -- no usage in awp_spend_perf')

        client_guarantees_a = client_guarantees.copy(deep = True)[['CLIENT', 'BREAKOUT', 'RATE']]

        if (client_guarantees_a.groupby(['CLIENT', 'BREAKOUT'])['RATE'].nunique().reset_index()['RATE'] == 1).all() == False:
            print('')
            print('*Warning: More than one guarantee rate per breakout present. Please investigate further.')
        
        # to join with perf and awp
        client_guarantees_a['CHAIN_GROUP'] = client_guarantees_a['BREAKOUT']
        customer_id = uf.get_formatted_client_name(p.CUSTOMER_ID)

        client_guarantees_a['CLIENT_OR_PHARM'] = 'CLIENT'
        # Read in pharmacy_guarantees_FILE
        pharmacy_guarantees = pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, 'Buffer_' + p.PHARM_GUARANTEE_FILE), dtype = p.VARIABLE_TYPE_DIC)
        pharmacy_guarantees = standardize_df(pharmacy_guarantees)
            #pharmacy_guarantees = standardize_df(pharmacy_guarantees) \
                #[['CLIENT', 'BREAKOUT', 'PHARMACY', 'RATE']].rename(columns={'PHARMACY': 'CHAIN_GROUP'})
            
            # If pharmacy guarantee varies by measurement, we find a blended target rate for each breakout based on ytd_utilization
        if (pharmacy_guarantees.groupby(['CLIENT','REGION', 'BREAKOUT', 'BG_FLAG', 'PHARMACY'])['RATE'].nunique().reset_index()['RATE'] == 1).all() == True:
            pharmacy_guarantees = standardize_df(pharmacy_guarantees) \
            [['CLIENT', 'BREAKOUT', 'BG_FLAG','PHARMACY', 'RATE']].rename(columns={'PHARMACY': 'CHAIN_GROUP'})
        else:
            awp_total_pharm_side = awp_spend_total_backup[['CLIENT','CHAIN_GROUP','REGION','MEASUREMENT','BG_FLAG','BREAKOUT','FULLAWP_ADJ']].groupby(['CLIENT','CHAIN_GROUP','REGION','MEASUREMENT','BG_FLAG','BREAKOUT']).sum().reset_index()
            awp_total_pharm_side=standardize_df(awp_total_pharm_side)
            temp_pharm_guarantee=pd.merge(pharmacy_guarantees,awp_total_pharm_side, how='left', left_on=['CLIENT','REGION','MEASUREMENT','BG_FLAG','BREAKOUT','PHARMACY'],right_on=['CLIENT','REGION','MEASUREMENT','BG_FLAG','BREAKOUT','CHAIN_GROUP'])
            temp_pharm_guarantee['FULLAWP_ADJ']=temp_pharm_guarantee['FULLAWP_ADJ'].fillna(0)
            temp_pharm_guarantee['RATE_MULT_UTLIZATION']=temp_pharm_guarantee['RATE']*temp_pharm_guarantee['FULLAWP_ADJ']
            blended_ratesp=temp_pharm_guarantee[['CLIENT','REGION','BREAKOUT','CHAIN_GROUP','FULLAWP_ADJ','RATE_MULT_UTLIZATION']].groupby(['CLIENT','REGION','BREAKOUT','CHAIN_GROUP']).sum().reset_index()
            blended_ratesp['RATE']=blended_ratesp['RATE_MULT_UTLIZATION']/blended_ratesp['FULLAWP_ADJ']
            blended_ratesp['RATE']=round(blended_ratesp['RATE'],4)
            pharmacy_guarantees = blended_ratesp[['CLIENT','BREAKOUT','CHAIN_GROUP','RATE']].copy()
            
        pharmacy_guarantees['CLIENT_OR_PHARM'] = 'PHARMACY'
        
        # combine two guarantees to join with awp
        # In MedD tables, Client column represents client name and not client id
        guarantee = pd.concat([client_guarantees_a, pharmacy_guarantees], axis=0).rename(columns={#'CLIENT': 'CUSTOMER_ID',
                                                                                               'RATE': 'GER_Target'})
        # Agg at chain_group level, drop breakout avoiding duplications
        guarantee = guarantee.drop('BREAKOUT', axis=1)
        guarantee = guarantee.drop_duplicates(subset = ['CLIENT','CHAIN_GROUP','CLIENT_OR_PHARM'], keep='first')    
    
        # Merge guarantees with awp_spend
        awp_spend_total_client_pharm = pd.merge(awp_spend_total_client_pharm, guarantee,
                                                on=['CLIENT', 'CHAIN_GROUP', 'CLIENT_OR_PHARM'],
                                                how='left')

        # Join awp_spend with performance
        #June 7th- removed Chain_Group from the join condition below
        awp_spend_perf = pd.merge(awp_spend_total_client_pharm, performance_df, on=['CLIENT_OR_PHARM', 'CHAIN_SUBGROUP'],
                              how='left')
        # assertation to make sure joins are correct
        assert abs(performance_df['Model_Perf'].sum() - awp_spend_perf['Model_Perf'].sum()) < .0001, \
            'The merging of performance output and awp_spend_total is incorrect'

        # create chain_group and pharmacy type mapping dict
        retail_major = {'GNRC': dict(
            zip(p.BIG_CAPPED_PHARMACY_LIST['GNRC'], ['Retail Major' for i in range(len(p.BIG_CAPPED_PHARMACY_LIST['GNRC']))])), 
                        'BRND': dict(
            zip(p.BIG_CAPPED_PHARMACY_LIST['BRND'], ['Retail Major' for i in range(len(p.BIG_CAPPED_PHARMACY_LIST['BRND']))]))}
        retail_minor = {'GNRC': dict(
            zip(p.SMALL_CAPPED_PHARMACY_LIST['GNRC'], ['Retail Minor' for i in range(len(p.SMALL_CAPPED_PHARMACY_LIST['GNRC']))])), 
                        'BRND': dict(
            zip(p.SMALL_CAPPED_PHARMACY_LIST['BRND'], ['Retail Minor' for i in range(len(p.SMALL_CAPPED_PHARMACY_LIST['BRND']))]))}
        NON_CAPPED_PHARMACY_LIST_gnrc = [i for i in p.NON_CAPPED_PHARMACY_LIST['GNRC'] if i not in p.PSAO_LIST['GNRC']]
        NON_CAPPED_PHARMACY_LIST_brnd = [i for i in p.NON_CAPPED_PHARMACY_LIST['BRND'] if i not in p.PSAO_LIST['BRND']]
        retail_independent = {'GNRC': dict(zip(NON_CAPPED_PHARMACY_LIST_gnrc,
                                      ['Retail Independent' for i in range(len(NON_CAPPED_PHARMACY_LIST_gnrc))])), 
                              'BRND': dict(zip(NON_CAPPED_PHARMACY_LIST_brnd,
                                      ['Retail Independent' for i in range(len(NON_CAPPED_PHARMACY_LIST_brnd))]))}
        retail_psao = {'GNRC': dict(zip(p.PSAO_LIST['GNRC'], ['Retail PSAO' for i in range(len(p.PSAO_LIST['GNRC']))])),
                       'BRND': dict(zip(p.PSAO_LIST['BRND'], ['Retail PSAO' for i in range(len(p.PSAO_LIST['BRND']))]))}
        # entity_type_dict = retail_major | retail_minor | retail_independent | retail_psao
        entity_type_dict = {}
        entity_type_dict['G'] = {**retail_major['GNRC'], **retail_minor['GNRC'], **retail_psao['GNRC'], **retail_independent['GNRC'], 
                                    **{'MCHOICE':'Mail Capped', 'MAIL': 'Mail Independent'}}
        entity_type_dict['B'] = {**retail_major['BRND'], **retail_minor['BRND'], **retail_psao['BRND'], **retail_independent['BRND'],
                                    **{'MCHOICE':'Mail Capped', 'MAIL': 'Mail Independent'}}
        # get entity type(eg. Big capped..)
        def get_entity_type(row, entity_type_dict=entity_type_dict):
            if row['CLIENT_OR_PHARM'] == 'CLIENT':
                return 'CLIENT'
            else:
                if p.GENERIC_OPT and not p.BRAND_OPT:
                    # CHANGE ONCE BG_FLAG IS INCORPORTATED THROUGHOUT REPORTING
                    return entity_type_dict['G'][row['CHAIN_GROUP']]
                elif (not p.GENERIC_OPT) and p.BRAND_OPT:
                    # CHANGE ONCE BG_FLAG IS INCORPORTATED THROUGHOUT REPORTING
                    return entity_type_dict['B'][row['CHAIN_GROUP']]
                else:
                    # temp fix - in case of running both Brands and generics, prioritize Generics
                    if row['CHAIN_GROUP'] in entity_type_dict['G']:
                        return entity_type_dict['G'][row['CHAIN_GROUP']]
                    else:
                        return entity_type_dict['B'][row['CHAIN_GROUP']]
                

        awp_spend_perf['ENTITY_TYPE'] = awp_spend_perf.apply(lambda x: get_entity_type(x, entity_type_dict), axis=1)

        # Add calculated fields
        awp_spend_perf['Proj_Spend_Do_Nothing'] = awp_spend_perf[
            ['PRICE_REIMB', 'LAG_REIMB', 'OLD_PRICE_EFFECTIVE_REIMB_PROJ_EOY',
             'GEN_LAG_ING_COST', 'GEN_EOY_ING_COST', 'DISP_FEE','DISP_FEE_PROJ_LAG','DISP_FEE_PROJ_EOY']].sum(axis=1)
        awp_spend_perf['Proj_IngCost_Spend_Do_Nothing'] = awp_spend_perf[
            ['PRICE_REIMB', 'LAG_REIMB', 'OLD_PRICE_EFFECTIVE_REIMB_PROJ_EOY',
             'GEN_LAG_ING_COST', 'GEN_EOY_ING_COST']].sum(axis=1)
        awp_spend_perf['Proj_DF_Spend_Do_Nothing'] = awp_spend_perf[
            ['DISP_FEE','DISP_FEE_PROJ_LAG','DISP_FEE_PROJ_EOY']].sum(axis=1)
        awp_spend_perf['Proj_IngCost_Spend_Model'] = awp_spend_perf[
            ['PRICE_REIMB', 'LAG_REIMB', 'PRICE_EFFECTIVE_REIMB_PROJ', 'GEN_LAG_ING_COST',
             'GEN_EOY_ING_COST']].sum(axis=1)
        awp_spend_perf['Proj_DF_Spend_Model'] = awp_spend_perf[
            ['DISP_FEE','DISP_FEE_PROJ_LAG','DISP_FEE_PROJ_EOY']].sum(axis=1)
        awp_spend_perf['Proj_Spend_Model'] = awp_spend_perf[
            ['PRICE_REIMB', 'LAG_REIMB', 'PRICE_EFFECTIVE_REIMB_PROJ', 'GEN_LAG_ING_COST',
             'GEN_EOY_ING_COST', 'DISP_FEE','DISP_FEE_PROJ_LAG','DISP_FEE_PROJ_EOY']].sum(axis=1)
        awp_spend_perf['Increase_in_Spend'] = awp_spend_perf['Proj_Spend_Model'] - awp_spend_perf['Proj_Spend_Do_Nothing']
        awp_spend_perf['Increase_in_Reimb'] = (-awp_spend_perf['Model_Perf']) - (-awp_spend_perf['Pre_existing_Perf'])
        awp_spend_perf['Total_Ann_AWP'] = awp_spend_perf[
            ['FULLAWP_ADJ', 'FULLAWP_ADJ_PROJ_LAG', 'FULLAWP_ADJ_PROJ_EOY']].sum(axis=1)
        awp_spend_perf['Total_PHARM_TARG'] = awp_spend_perf[
            ['PHARM_TARG_INGCOST_ADJ', 'PHARM_TARG_INGCOST_ADJ_PROJ_LAG', 'PHARM_TARG_INGCOST_ADJ_PROJ_EOY']].sum(axis=1)
        awp_spend_perf['Total_TARG_ING_COST'] = awp_spend_perf[
            ['TARG_INGCOST_ADJ', 'TARG_INGCOST_ADJ_PROJ_LAG', 'TARG_INGCOST_ADJ_PROJ_EOY']].sum(axis=1)
        awp_spend_perf['Total_TARG_DISP_FEE'] = awp_spend_perf[
                ['TARGET_DISP_FEE','TARGET_DISP_FEE_PROJ_LAG','TARGET_DISP_FEE_PROJ_EOY']].sum(axis=1)
        
        awp_spend_perf['Total_TARG'] = awp_spend_perf['Total_TARG_ING_COST'] + awp_spend_perf['Total_TARG_DISP_FEE']
        awp_spend_perf['GER_Target'] = round(1 - (awp_spend_perf['Total_TARG'] / awp_spend_perf['Total_Ann_AWP']), 6)
        awp_spend_perf['GER_Do_Nothing'] = round(1 - (awp_spend_perf['Proj_Spend_Do_Nothing'] / awp_spend_perf['Total_Ann_AWP']), 6)
        awp_spend_perf['GER_Model'] = round(1 - (awp_spend_perf['Proj_Spend_Model'] / awp_spend_perf['Total_Ann_AWP']), 6)
        awp_spend_perf['GER_Target'] = np.where((awp_spend_perf['CLIENT_OR_PHARM'] == 'PHARMACY') & (~awp_spend_perf['GER_Target'].isna()) & (awp_spend_perf['Total_Ann_AWP'] > 0), 
                                                round(1 - (awp_spend_perf['Total_PHARM_TARG'] / awp_spend_perf['Total_Ann_AWP']), 6), 
                                                awp_spend_perf['GER_Target'])

        #Adding in surplus calculations without brand performance
        awp_spend_perf['Pre_existing_Perf_Generic'] = awp_spend_perf['Total_TARG'] - awp_spend_perf['Proj_Spend_Do_Nothing']
        awp_spend_perf['Model_Perf_Generic'] = awp_spend_perf['Total_TARG'] - awp_spend_perf['Proj_Spend_Model']
        awp_spend_perf['YTD_Perf_Generic'] = awp_spend_perf['Total_TARG'] - awp_spend_perf['PRICE_REIMB']
        
        # As psot requested using GER_OPT_TAXONOMY_FINAL and for calculating run_rate
        GER_OPT_TAXONOMY_FINAL = uf.read_BQ_data(BQ.GER_OPT_TAXONOMY_FINAL,
                      project_id = p.BQ_INPUT_PROJECT_ID,
                      dataset_id = p.BQ_INPUT_DATASET_ENT_ENRV_PROD, 
                      table_id = 'GER_OPT_TAXONOMY_FINAL' + p.WS_SUFFIX, 
                      customer = ', '.join(sorted(p.CUSTOMER_ID)),
                      output = False)
        assert len(GER_OPT_TAXONOMY_FINAL) == 1, \
            "The number entries in GER_OPT_TAXONOMY_FINAL for this client doesn't equal to one"
        # should be one row for a client in GER_OPT_TAXONOMY_FINAL, if not cannot perform what is requested by DE/PSOT
        
        # add parameters from CPMO_parameters.py
        # TODO: customer_id will not work for multiple customer_ids. should not be in the dataset
        awp_spend_perf['CUSTOMER_ID'] = uf.get_formatted_client_name(p.CUSTOMER_ID) # add functionality to handle clients with multiple Ids
        awp_spend_perf['CLIENT'] = p.CLIENT_NAME_TABLEAU
        awp_spend_perf['ALGO_RUN_DATE'] = dt.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        awp_spend_perf['REC_CURR_IND'] = 'Y'
        awp_spend_perf['CLIENT_TYPE'] = p.CLIENT_TYPE
        awp_spend_perf['UNC_OPT'] = p.UNC_OPT
        awp_spend_perf['GOODRX_OPT'] = p.RMS_OPT
        awp_spend_perf['GO_LIVE'] = p.GO_LIVE.strftime("%Y-%m-%d")
        awp_spend_perf['DATA_ID'] = p.DATA_ID
        awp_spend_perf['TIERED_PRICE_LIM'] = p.TIERED_PRICE_LIM
        awp_spend_perf['RUN_TYPE'] = RUN_TYPE_TABLEAU_UPDATED
        awp_spend_perf['AT_RUN_ID'] = p.AT_RUN_ID

        #Calc run rates
        eoc = datetime.datetime.strptime(GER_OPT_TAXONOMY_FINAL['contract_expiry'].astype('str')[0], "%Y-%m-%d")
        golive = datetime.datetime.strptime(awp_spend_perf['GO_LIVE'][0], "%Y-%m-%d")
        if p.FULL_YEAR:
            contract_length = relativedelta.relativedelta(eoc, GER_OPT_TAXONOMY_FINAL['contract_eff_dt'][0]).months + 1
            eoc = eoc + relativedelta.relativedelta(months = contract_length)
            
        daygap = (eoc - golive).total_seconds()/(3600*24)
        months_left=daygap/30.5

        #run rates calculation logic per PSOT request  
        #if else to handle cases where the denominator is 0
        if months_left > 0:
            awp_spend_perf['Run_rate_do_nothing'] = (awp_spend_perf['Pre_existing_Perf_Generic'] - awp_spend_perf['YTD_Perf_Generic'])/months_left
            awp_spend_perf['Run_rate_w_changes'] = (awp_spend_perf['Model_Perf_Generic'] - awp_spend_perf['YTD_Perf_Generic'])/months_left
        else:
            awp_spend_perf['Run_rate_do_nothing'] = np.where(awp_spend_perf['Pre_existing_Perf_Generic'].notnull(), 0.00, np.nan)
            awp_spend_perf['Run_rate_w_changes'] = np.where(awp_spend_perf['Model_Perf_Generic'].notnull(), 0.00, np.nan)


        ################################## Mapping from breakouts to ia_codename ################################################
        unique_breakouts = awp_spend_perf.loc[awp_spend_perf.ENTITY_TYPE=='CLIENT','BREAKOUT'].unique()
        breakout_codename_map=pd.DataFrame(data=unique_breakouts,columns=['BREAKOUT'])
        breakout_codename_map['IA_CODENAME']=""
        
         ## Read in guarantees ##
        
        # ia_codenames = uf.read_BQ_data(
        #     BQ.ia_codenames_custom.format(_customer_id = uf.get_formatted_string(p.CUSTOMER_ID),
        #     _project=p.BQ_INPUT_ANBC_PROD_PROJECT_ID,
        #     _landing_dataset=p.BQ_INPUT_DATASET_ANBC_PROD,
        #     _table_id = 'gms_gms_taxonomy'+ p.WS_SUFFIX),
        #     project_id=p.BQ_INPUT_PROJECT_ID,
        #     dataset_id=p.BQ_INPUT_DATASET,
        #     table_id='gms_gms_taxonomy'+ p.WS_SUFFIX,
        #     client = ', '.join(sorted(p.CUSTOMER_ID)),
        #     custom = True
        # )
        
        # if p.NO_MAIL:
        #     ia_codenames = ia_codenames[ia_codenames.mailind != 'M']
        
        # assert len(unique_breakouts)==len(ia_codenames.IA_CODENAME), "Number of unique breakouts does not match number of unique IA_CODENAMES"

        offsetting_guarantee_category=['Offsetting Complex','Offsetting R30/R90','MedD/EGWP Offsetting R30/R90/LTC','MedD/EGWP Offsetting Complex','Aetna Offsetting','Aetna MR Offsetting','Aetna BG Offsetting', 'Aetna NA Offsetting']
        GER_OPT_TAXONOMY_FINAL.loc[0,'guarantee_category'] in offsetting_guarantee_category
        
        # if  GER_OPT_TAXONOMY_FINAL.loc[0,'guarantee_category'] in ['Pure Vanilla','MedD/EGWP Vanilla']:
            
        #     if 'M' in ia_codenames["mailind"].unique():
        #         breakout_codename_map.loc[breakout_codename_map.BREAKOUT.str.contains('_MAIL')==True,'IA_CODENAME'] = ia_codenames.loc[ia_codenames.mailind=='M','IA_CODENAME'].reset_index().IA_CODENAME.iloc[0]
        #     breakout_codename_map.loc[breakout_codename_map.BREAKOUT.str.contains('_RETAIL'),'IA_CODENAME'] =ia_codenames.loc[ia_codenames.mailind=='R','IA_CODENAME'].reset_index().IA_CODENAME.iloc[0]
            
        # elif GER_OPT_TAXONOMY_FINAL.loc[0,'guarantee_category'] in ['Offsetting R30/R90','MedD/EGWP Offsetting R30/R90/LTC','Offsetting Complex','MedD/EGWP Offsetting Complex']:
            
        #     if 'M' in ia_codenames["mailind"].unique():
        #         breakout_codename_map.loc[breakout_codename_map.BREAKOUT.str.contains('_MAIL')==True,'IA_CODENAME'] = ia_codenames.loc[ia_codenames.mailind=='M','IA_CODENAME'].reset_index().IA_CODENAME.iloc[0]
        #     breakout_codename_map.loc[breakout_codename_map.BREAKOUT.str.contains('_RETAIL'),'IA_CODENAME'] =ia_codenames.loc[ia_codenames.mailind=='R','IA_CODENAME'].reset_index().IA_CODENAME.iloc[0]
            
        # elif GER_OPT_TAXONOMY_FINAL.loc[0,'guarantee_category'] in ['NonOffsetting R30/R90','MedD/EGWP NonOffsetting R30/R90/LTC','MedD/EGWP NonOffsetting Complex','NonOffsetting Complex']:
        #     if 'M' in ia_codenames["mailind"].unique():
        #         breakout_codename_map.loc[breakout_codename_map.BREAKOUT.str.contains('_MAIL')==True,'IA_CODENAME'] = ia_codenames.loc[ia_codenames.mailind=='M','IA_CODENAME'].reset_index().IA_CODENAME.iloc[0]
        #     breakout_codename_map.loc[breakout_codename_map.BREAKOUT.str.contains('_R30'),'IA_CODENAME'] =ia_codenames.loc[(ia_codenames.mailind=='R') & ~(ia_codenames.IA_CODENAME.str.contains('90')),'IA_CODENAME'].reset_index().IA_CODENAME.iloc[0]
        #     breakout_codename_map.loc[breakout_codename_map.BREAKOUT.str.contains('_R90'),'IA_CODENAME'] =ia_codenames.loc[(ia_codenames.mailind=='R') & (ia_codenames.IA_CODENAME.str.contains('90')),'IA_CODENAME'].reset_index().IA_CODENAME.iloc[0]
        
        awp_spend_perf=awp_spend_perf.merge(breakout_codename_map, how='left', on=['BREAKOUT'])
        
        ################################## Bring in Leakage Report ####################################################################
        if p.LOCKED_CLIENT and p.GENERIC_OPT:
            leakage_report_temp = standardize_df(pd.read_csv(os.path.join(p.FILE_OUTPUT_PATH, 'LEAKAGE_REPORT_{}.csv'.format(p.DATA_ID))))
            # Aggregate up to Breakout level
            leakage_report = leakage_report_temp.groupby(['CLIENT','BREAKOUT']).agg({'LEAKAGE_PRE':'sum',
                                                                                     'LEAKAGE_POST':'sum',
                                                                                     'LEAKAGE_AVOID':'sum',
                                                                                     'PLAN_DESIGN_AVAIL_AWP':'sum',
                                                                                     'FULLAWP_ADJ_PROJ_EOY':'sum'}).reset_index()
            
            leakage_report['PLAN_DESIGN_AVAIL_PROP_AWP'] = leakage_report['PLAN_DESIGN_AVAIL_AWP'] / leakage_report['FULLAWP_ADJ_PROJ_EOY']
            leakage_report['ENTITY_TYPE'] = 'CLIENT'
            leakage_report['CLIENT_OR_PHARM'] = 'CLIENT'
            leakage_report.rename(columns={'CLIENT': 'CUSTOMER_ID'}, inplace=True)
            leakage_report.drop(columns=['PLAN_DESIGN_AVAIL_AWP','FULLAWP_ADJ_PROJ_EOY'], inplace=True)
            awp_spend_perf = awp_spend_perf.merge(leakage_report, how = 'left', on = ['CUSTOMER_ID','ENTITY_TYPE','CLIENT_OR_PHARM','BREAKOUT'])
        else:
            awp_spend_perf['LEAKAGE_PRE'] = np.nan
            awp_spend_perf['LEAKAGE_POST'] = np.nan
            awp_spend_perf['LEAKAGE_AVOID'] = np.nan
            awp_spend_perf['PLAN_DESIGN_AVAIL_PROP_AWP'] = np.nan
        
        ###############################################################################################################################
             
        # REORDER COLUMNS
        #if p.CLIENT_TYPE == 'MEDD':
        awp_spend_perf = awp_spend_perf[
            ['CUSTOMER_ID', 'CLIENT', 'ENTITY', 'ENTITY_TYPE', 'BREAKOUT', 'CHAIN_GROUP','CHAIN_SUBGROUP', 'CLIENT_OR_PHARM',
             'FULLAWP_ADJ', 'FULLAWP_ADJ_PROJ_LAG', 'FULLAWP_ADJ_PROJ_EOY',
             'PRICE_REIMB', 'LAG_REIMB', 'OLD_PRICE_EFFECTIVE_REIMB_PROJ_EOY', 'PRICE_EFFECTIVE_REIMB_PROJ',
             'TARG_INGCOST_ADJ_PROJ_EOY', 'TARG_INGCOST_ADJ','TARG_INGCOST_ADJ_PROJ_LAG','DISP_FEE','DISP_FEE_PROJ_LAG',
             'DISP_FEE_PROJ_EOY','TARGET_DISP_FEE','TARGET_DISP_FEE_PROJ_LAG','TARGET_DISP_FEE_PROJ_EOY','COMPETITIVE_SCORE',
             'GEN_LAG_AWP', 'GEN_LAG_ING_COST', 'GEN_EOY_AWP', 'GEN_EOY_ING_COST',
             'Pre_existing_Perf', 'Model_Perf', 'Pre_existing_Perf_Generic', 'Model_Perf_Generic', 'YTD_Perf_Generic',
             'Proj_Spend_Do_Nothing', 'Proj_IngCost_Spend_Do_Nothing', 'Proj_DF_Spend_Do_Nothing', 
             'Proj_IngCost_Spend_Model','Proj_DF_Spend_Model','Proj_Spend_Model', 'Increase_in_Spend',
             'Increase_in_Reimb', 'Total_Ann_AWP', 'Total_TARG', 'Total_TARG_ING_COST','Total_TARG_DISP_FEE','GER_Do_Nothing', 'GER_Model', 'GER_Target',
             'ALGO_RUN_DATE', #'REC_CURR_IND',
             'CLIENT_TYPE', 'UNC_OPT',
             'GOODRX_OPT', 'GO_LIVE', 'DATA_ID', 'TIERED_PRICE_LIM',
             'RUN_TYPE','AT_RUN_ID','Run_rate_do_nothing','Run_rate_w_changes','IA_CODENAME',
             'LEAKAGE_PRE','LEAKAGE_POST','LEAKAGE_AVOID','PLAN_DESIGN_AVAIL_PROP_AWP'
            ]]
        
        #Update the label of breakout for Mail & Retail offsetting clients 
        if GER_OPT_TAXONOMY_FINAL.loc[0,'guarantee_category'] == 'Aetna MR Offsetting':
            pharm_perf= awp_spend_perf[awp_spend_perf['CLIENT_OR_PHARM']!='CLIENT'].copy()
            client_perf = awp_spend_perf[awp_spend_perf['CLIENT_OR_PHARM']=='CLIENT'].copy()

            client_perf.loc[:,'BREAKOUT'] = client_perf['CUSTOMER_ID'] + "_MR"
            client_perf.loc[:,'ENTITY'] = client_perf.ENTITY.replace({'_MAIL': "_MR",'_RETAIL':"_MR"}, regex=True) 
            client_perf.loc[:,'CHAIN_SUBGROUP'] = client_perf['CUSTOMER_ID'] + "_MR"
            client_perf.loc[:,'CHAIN_GROUP'] = client_perf['CUSTOMER_ID'] + "_MR"

            client_perf.loc[:,'GER_DO_NOTHING_SPEND'] = client_perf['GER_Do_Nothing']*client_perf['Total_Ann_AWP']
            client_perf.loc[:,'GER_TARGET_SPEND'] = client_perf['GER_Target']*client_perf['Total_Ann_AWP']
            client_perf.loc[:,'GER_MODEL_SPEND'] = client_perf['GER_Model']*client_perf['Total_Ann_AWP']
            client_perf.loc[:,'GER_TARGET_ISNA'] = client_perf['GER_Target'].isna()

            client_perf = client_perf.groupby(['CUSTOMER_ID', 'CLIENT', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'ENTITY', 
                                               'CLIENT_OR_PHARM',  'ENTITY_TYPE', 'IA_CODENAME', 'BREAKOUT', 
                                               'ALGO_RUN_DATE', 'CLIENT_TYPE', 'UNC_OPT', 'GOODRX_OPT', 
                                               'GO_LIVE', 'DATA_ID', 'TIERED_PRICE_LIM', 'RUN_TYPE', 
                                               'AT_RUN_ID'], as_index=False, dropna=False).sum()

            client_perf.loc[:,'GER_Do_Nothing'] = client_perf['GER_DO_NOTHING_SPEND']/client_perf['Total_Ann_AWP']
            client_perf.loc[:,'GER_Model'] = client_perf['GER_MODEL_SPEND']/client_perf['Total_Ann_AWP']
            client_perf.loc[:,'GER_Target'] = client_perf['GER_TARGET_SPEND']/client_perf['Total_Ann_AWP']
            client_perf.loc[client_perf['GER_TARGET_ISNA']>0, 'GER_Target'] = np.nan

            client_perf.drop(columns=['GER_DO_NOTHING_SPEND', 'GER_MODEL_SPEND', 'GER_TARGET_SPEND', 'GER_TARGET_ISNA'], inplace=True)

            awp_spend_perf = pd.concat([client_perf, pharm_perf])
        
        # Save a copy for psot
        awp_spend_perf_subgroup = awp_spend_perf.copy()
        psot_awp_spend_perf = awp_spend_perf.copy()
        
        # Combine R90OK rows with non-R90OK for PSOT
        if awp_spend_perf['CHAIN_SUBGROUP'].str.contains('R90OK').any():
            awp_spend_perf['ENTITY'] = awp_spend_perf['ENTITY'].str.replace('_R90OK', '')
            awp_spend_perf['CHAIN_SUBGROUP'] = awp_spend_perf['CHAIN_SUBGROUP'].str.replace('_R90OK', '')
            awp_spend_perf['GER_Do_Nothing_Spend'] = awp_spend_perf['GER_Do_Nothing']*awp_spend_perf['Total_Ann_AWP']
            awp_spend_perf['GER_Target_Spend'] = awp_spend_perf['GER_Target']*awp_spend_perf['Total_Ann_AWP']
            awp_spend_perf['GER_Model_Spend'] = awp_spend_perf['GER_Model']*awp_spend_perf['Total_Ann_AWP']
            awp_spend_perf['GER_Target_IsNa'] = awp_spend_perf['GER_Target'].isna()
            awp_spend_perf = awp_spend_perf.groupby(['CUSTOMER_ID', 'CLIENT', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'ENTITY', 'CLIENT_OR_PHARM',  'ENTITY_TYPE', 'IA_CODENAME', 'BREAKOUT', 'GER_Target', 'ALGO_RUN_DATE', 'CLIENT_TYPE', 'UNC_OPT', 
                                        'GOODRX_OPT', 'GO_LIVE', 'DATA_ID', 'TIERED_PRICE_LIM', 'RUN_TYPE', 'AT_RUN_ID'], as_index=False, dropna=False).sum()
            awp_spend_perf['GER_Do_Nothing'] = awp_spend_perf['GER_Do_Nothing_Spend']/awp_spend_perf['Total_Ann_AWP']
            awp_spend_perf['GER_Model'] = awp_spend_perf['GER_Model_Spend']/awp_spend_perf['Total_Ann_AWP']
            awp_spend_perf['GER_Target'] = (awp_spend_perf['GER_Target_Spend']/awp_spend_perf['Total_Ann_AWP']).round(4)
            awp_spend_perf.loc[awp_spend_perf['GER_Target_IsNa']>0, 'GER_Target'] = np.nan
            awp_spend_perf.drop(columns=['GER_Do_Nothing_Spend', 'GER_Model_Spend', 'GER_Target_Spend', 'GER_Target_IsNa'], inplace=True)

        
        # Add extra SP rows as per PSOT request, and drop CHAIN_SUBGROUP
        if 'CVSSP' in awp_spend_perf['CHAIN_SUBGROUP'].values:
            awp_spend_perf.loc[awp_spend_perf['CHAIN_GROUP']=='CVS', 'ENTITY_TYPE'] = 'SP'
            awp_spend_perf.loc[awp_spend_perf['CHAIN_GROUP']=='CVS', 'CLIENT_OR_PHARM'] = 'SP'
            cvs_rows = awp_spend_perf[awp_spend_perf['CHAIN_GROUP']=='CVS']
            new_row = cvs_rows.groupby(['CUSTOMER_ID', 'CLIENT', 'CHAIN_GROUP', 'BREAKOUT', 'GER_Target', 'ALGO_RUN_DATE', 'CLIENT_TYPE', 'UNC_OPT', 
                                        'GOODRX_OPT', 'GO_LIVE', 'DATA_ID', 'TIERED_PRICE_LIM', 'RUN_TYPE', 'AT_RUN_ID'], as_index=False).sum()
            new_row['ENTITY'] = 'CVS'
            new_row['CLIENT_OR_PHARM'] = 'CVS'
            new_row['ENTITY_TYPE'] = 'Retail Major'
            new_row['IA_CODENAME'] = np.nan
            new_row['LEAKAGE_PRE'] = np.nan
            new_row['LEAKAGE_POST'] = np.nan
            new_row['LEAKAGE_AVOID'] = np.nan
            new_row['GER_Do_Nothing'] = (cvs_rows['GER_Do_Nothing']*cvs_rows['Total_Ann_AWP']).sum()/cvs_rows['Total_Ann_AWP'].sum()
            new_row['GER_Model'] = (cvs_rows['GER_Model']*cvs_rows['Total_Ann_AWP']).sum()/cvs_rows['Total_Ann_AWP'].sum()
            new_row['GER_Target'] = (cvs_rows['GER_Target']*cvs_rows['Total_Ann_AWP']).sum()/cvs_rows['Total_Ann_AWP'].sum()
            new_row['CHAIN_SUBGROUP'] = 'CVS'
            new_row = new_row[awp_spend_perf.columns]
            awp_spend_perf = pd.concat([awp_spend_perf, new_row])
            awp_spend_perf.loc[awp_spend_perf['ENTITY']=='CVSSP', 'ENTITY'] = 'CVS_SP'
            sp_mask = awp_spend_perf['ENTITY_TYPE']=='SP'
            awp_spend_perf.loc[sp_mask, 'BREAKOUT'] = awp_spend_perf.loc[sp_mask, 'ENTITY']
            awp_spend_perf.loc[sp_mask, 'CHAIN_GROUP'] = awp_spend_perf.loc[sp_mask, 'ENTITY']
        
        # Save a copy for psot
        awp_spend_perf.drop(columns=['CHAIN_SUBGROUP'], inplace=True)
        
        # Mapping breakout label -- for both awp_perf and awp_perf_subgroup
        awp_spend_perf_subgroup_upload = breakout_label_mapping(awp_spend_perf_subgroup)
        awp_spend_perf_upload = breakout_label_mapping(awp_spend_perf) 

        if p.WRITE_TO_BQ or p.UPLOAD_TO_DASH:
            uf.write_to_bq(
                awp_spend_perf_subgroup_upload,
                project_output = p.BQ_OUTPUT_PROJECT_ID,
                dataset_output = p.BQ_OUTPUT_DATASET,
                table_id = "awp_spend_perf_subgroup",
                client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                timestamp_param = p.TIMESTAMP,
                run_id=p.AT_RUN_ID,
                schema = None
            )
        else:
            awp_spend_perf_subgroup_upload.to_csv(p.FILE_REPORT_PATH + p.DATA_ID +'_awp_spend_performance_subgroup.csv', index=False)

        if p.WRITE_TO_BQ or p.UPLOAD_TO_DASH:
            awp_spend_perf.TIERED_PRICE_LIM = awp_spend_perf.TIERED_PRICE_LIM.astype("boolean")
            awp_spend_perf.UNC_OPT = awp_spend_perf.UNC_OPT.astype("boolean")
            awp_spend_perf.GOODRX_OPT = awp_spend_perf.GOODRX_OPT.astype("boolean")
            uf.write_to_bq(
                awp_spend_perf_upload,
                project_output = p.BQ_OUTPUT_PROJECT_ID,
                dataset_output = p.BQ_OUTPUT_DATASET,
                table_id = "awp_spend_perf",
                client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                timestamp_param = p.TIMESTAMP,
                run_id=p.AT_RUN_ID,
                schema = None
            )
        else:
            awp_spend_perf_upload.to_csv(p.FILE_REPORT_PATH + p.DATA_ID +'_awp_spend_performance.csv', index=False)
            
            
################################## YTD_surplus_monthly ##################################
        """
        This part of code takes in the LP run result and generates YTD_Surplus_monthly table for reporting usage
        """
        # Read in contract dates
        contract_date_df = pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.CONTRACT_DATE_FILE))
        contract_date_df = standardize_df(contract_date_df)
        assert len(contract_date_df) == 1,\
            "The number of contract eff and exprn date tied with this client doesn't equal to one"
        # the reporting granularity is defined at year-month level extract year and month below
        CONTRACT_EFF_DT = str(contract_date_df.loc[0, 'CONTRACT_EFF_DT'])[:-3]
        CONTRACT_EXPRN_DT = str(contract_date_df.loc[0, 'CONTRACT_EXPRN_DT'])[:-3]

        # READ IN DAILY TOTALS
        if p.READ_FROM_BQ:
            Daily_Totals = uf.read_BQ_data(
                BQ.daily_totals_pharm,
                project_id = p.BQ_INPUT_PROJECT_ID,
                dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP,
                table_id = 'combined_daily_totals' + p.WS_SUFFIX + p.CCP_SUFFIX,
                client = ', '.join(sorted(p.CUSTOMER_ID)),
                claim_start = p.DATA_START_DAY,
                claim_date = p.LAST_DATA.date().strftime('%m/%d/%Y')
            )
        else:
            Daily_Totals = pd.read_csv(p.FILE_INPUT_PATH + p.DAILY_TOTALS_FILE, dtype = p.VARIABLE_TYPE_DIC)
        Daily_Totals = standardize_df(Daily_Totals)
        Daily_Totals.columns = [x.upper() for x in Daily_Totals.columns]
        # Daily_Totals['MONTH'] = pd.DatetimeIndex(Daily_Totals.CLAIM_DATE).month
        Daily_Totals['MONTH'] = Daily_Totals['CLAIM_DATE'].astype('str').str[:-3]
        
        Daily_Totals.rename(columns={'AWP':'FULLAWP_ADJ','PHARMACY_QTY':'PHARM_QTY'}, inplace=True)

        Daily_Totals = merge_nadac_wac_netcostguarantee(Daily_Totals)
        
        # CLIENT_GUARANTEE_FILE being read in previous reporting table generation script
        # aggregate to region/breakout
        client_breakout_guarantees = client_guarantees.groupby(['CLIENT', 'REGION', 'BREAKOUT'])['RATE'].agg(
            ['max', 'nunique']).reset_index()

        # Each breakout should have a unique rate.
        assert (client_breakout_guarantees['nunique'] == 1).all(), 'multiple client guarantees per breakout'
        client_breakout_guarantees = client_breakout_guarantees.rename(columns={'max': 'RATE'})[
            ['CLIENT', 'REGION', 'BREAKOUT', 'RATE']]
        
        # MERGE
        Daily_Totals = pd.merge(Daily_Totals,
                                       client_guarantees[['CLIENT', 'REGION', 'BREAKOUT', 'RATE']],
                                       on=['CLIENT', 'REGION', 'BREAKOUT'], how='left')
        
        Daily_Totals = add_target_ingcost(Daily_Totals.rename(columns={'FULLAWP_ADJ':'FULLAWP'}), None, client_rate_col = 'RATE', target_cols=['TARG_INGCOST'])
        
        Daily_Totals.rename(columns={'FULLAWP':'AWP'}, inplace=True)

        # aggregate to region/breakout
        YTD_surplus_monthly = Daily_Totals.groupby(['CLIENT', 'REGION', 'BREAKOUT', 'MONTH'])[
            ['AWP', 'SPEND', 'TARG_INGCOST', 'DISP_FEE', 'TARGET_DISP_FEE']].sum().reset_index()
        # CALCUCATE SURPLUS
        YTD_surplus_monthly.rename(columns={'CLIENT': 'CUSTOMER_ID', 'SPEND':'SPEND_IC', 'DISP_FEE':'SPEND_DF', 'TARG_INGCOST':'TARGET_IC', 'TARGET_DISP_FEE':'TARGET_DF'}, inplace=True)
        YTD_surplus_monthly['SPEND'] = YTD_surplus_monthly['SPEND_IC'] + YTD_surplus_monthly['SPEND_DF']
        YTD_surplus_monthly['TARGET'] = YTD_surplus_monthly['TARGET_IC'] + YTD_surplus_monthly['TARGET_DF'] 
        YTD_surplus_monthly['SURPLUS'] = YTD_surplus_monthly.TARGET - YTD_surplus_monthly.SPEND
        YTD_surplus_monthly['SURPLUS_IC'] = YTD_surplus_monthly.TARGET_IC - YTD_surplus_monthly.SPEND_IC
        YTD_surplus_monthly['SURPLUS_DF'] = YTD_surplus_monthly.TARGET_DF - YTD_surplus_monthly.SPEND_DF

        # Assign month type
        #YTD_surplus_monthly['MONTH'] = [pd.to_datetime(i,format='%m/%d/%Y') for i in YTD_surplus_monthly['MONTH']]
        YTD_surplus_monthly.loc[YTD_surplus_monthly['MONTH'] == CONTRACT_EFF_DT, 'MONTH_TYPE'] = 'CONTRACT_EFF'
        YTD_surplus_monthly.loc[YTD_surplus_monthly['MONTH'] == CONTRACT_EXPRN_DT, 'MONTH_TYPE'] = 'CONTRACT_EXPRN'
        YTD_surplus_monthly.loc[(YTD_surplus_monthly['MONTH'] > CONTRACT_EFF_DT)
                                & (YTD_surplus_monthly['MONTH'] < CONTRACT_EXPRN_DT), 'MONTH_TYPE'] = 'YTD'

        # Add projected AWP, spend, and surplus
        # this df is created in create_performance_spend_agg_table()
        # Use the result from awp_spend_perf
        awp_spend_perf = standardize_df(awp_spend_perf)
        awp_spend_perf = awp_spend_perf[awp_spend_perf['CLIENT_OR_PHARM'] == 'CLIENT'][['CUSTOMER_ID', 'BREAKOUT',
                                                                                         'TOTAL_ANN_AWP',
                                                                                         'PROJ_SPEND_DO_NOTHING',
                                                                                         'PROJ_INGCOST_SPEND_DO_NOTHING',
                                                                                         'PROJ_DF_SPEND_DO_NOTHING',
                                                                                         'PRE_EXISTING_PERF',
                                                                                         'TOTAL_TARG',
                                                                                         'TOTAL_TARG_ING_COST',
                                                                                         'TOTAL_TARG_DISP_FEE',]]

        awp_spend_perf['REGION'] = awp_spend_perf['CUSTOMER_ID']
        awp_spend_perf.rename(
            columns={'TOTAL_ANN_AWP': 'AWP', 
                     'PROJ_SPEND_DO_NOTHING': 'SPEND', 
                     'PROJ_INGCOST_SPEND_DO_NOTHING':'SPEND_IC',
                     'PROJ_DF_SPEND_DO_NOTHING':'SPEND_DF',
                     'PRE_EXISTING_PERF': 'SURPLUS', 
                     'TOTAL_TARG':'TARGET',
                     'TOTAL_TARG_ING_COST':'TARGET_IC',
                     'TOTAL_TARG_DISP_FEE':'TARGET_DF'},
            inplace=True)
        
        awp_spend_perf['SURPLUS_IC'] = awp_spend_perf['TARGET_IC'] - awp_spend_perf['SPEND_IC']
        awp_spend_perf['SURPLUS_DF'] = awp_spend_perf['TARGET_DF'] - awp_spend_perf['SPEND_DF']

        # since the dashboard will calculate the cumulative, we need to subtract out the YTD
        YTD_surplus = YTD_surplus_monthly.groupby(['CUSTOMER_ID', 'REGION', 'BREAKOUT'])[
            ['AWP', 'TARGET', 'TARGET_IC', 'TARGET_DF', 'SPEND', 'SPEND_IC', 'SPEND_DF', 'SURPLUS','SURPLUS_IC', 'SURPLUS_DF']].sum().reset_index()

        #In 1/1 pricing YTD is not added to awp_spend_perf, so it should not be subtracted here.
        if not p.FULL_YEAR:
            awp_spend_perf = awp_spend_perf.merge(YTD_surplus, on=['CUSTOMER_ID', 'REGION', 'BREAKOUT'], how='left')
            awp_spend_perf['AWP'] = awp_spend_perf['AWP_x'] - awp_spend_perf['AWP_y']
            awp_spend_perf['SPEND'] = awp_spend_perf['SPEND_x'] - awp_spend_perf['SPEND_y']
            awp_spend_perf['SPEND_IC'] = awp_spend_perf['SPEND_IC_x'] - awp_spend_perf['SPEND_IC_y']
            awp_spend_perf['SPEND_DF'] = awp_spend_perf['SPEND_DF_x'] - awp_spend_perf['SPEND_DF_y']
            awp_spend_perf['SURPLUS'] = awp_spend_perf['SURPLUS_x'] - awp_spend_perf['SURPLUS_y']
            awp_spend_perf['SURPLUS_IC'] = awp_spend_perf['SURPLUS_IC_x'] - awp_spend_perf['SURPLUS_IC_y']
            awp_spend_perf['SURPLUS_DF'] = awp_spend_perf['SURPLUS_DF_x'] - awp_spend_perf['SURPLUS_DF_y']
            awp_spend_perf['TARGET'] = awp_spend_perf['TARGET_x'] - awp_spend_perf['TARGET_y']
            awp_spend_perf['TARGET_IC'] = awp_spend_perf['TARGET_IC_x'] - awp_spend_perf['TARGET_IC_y']
            awp_spend_perf['TARGET_DF'] = awp_spend_perf['TARGET_DF_x'] - awp_spend_perf['TARGET_DF_y']
            

        awp_spend_perf['MONTH'] = CONTRACT_EXPRN_DT
        awp_spend_perf['MONTH_TYPE'] = 'EOY_PROJ'
        awp_spend_perf = awp_spend_perf[
            ['CUSTOMER_ID', 'REGION', 'BREAKOUT', 'MONTH', 'MONTH_TYPE', 'AWP', 'TARGET', 'TARGET_IC', 'TARGET_DF', 'SPEND', 'SPEND_IC', 'SPEND_DF', 'SURPLUS', 'SURPLUS_IC', 'SURPLUS_DF']]

        final_monthly_data = pd.concat([YTD_surplus_monthly, awp_spend_perf], axis=0)

        # YTD_SURPLUS_MONTH_out['ALGO_RUN_OWNER'] = USER_NAME
        final_monthly_data['CLIENT'] = p.CLIENT_NAME_TABLEAU
        final_monthly_data['CUSTOMER_ID'] = uf.get_formatted_client_name(p.CUSTOMER_ID)
        final_monthly_data['ALGO_RUN_DATE'] = dt.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        final_monthly_data['REC_CURR_IND'] = 'Y'
        final_monthly_data['CLIENT_TYPE'] = p.CLIENT_TYPE
        final_monthly_data['UNC_OPT'] = p.UNC_OPT
        final_monthly_data['GOODRX_OPT'] = p.RMS_OPT
        final_monthly_data['GO_LIVE'] = p.GO_LIVE.strftime("%Y-%m-%d")
        final_monthly_data['DATA_ID'] = p.DATA_ID
        final_monthly_data['TIERED_PRICE_LIM'] = p.TIERED_PRICE_LIM
        final_monthly_data['RUN_TYPE'] = RUN_TYPE_TABLEAU_UPDATED
        final_monthly_data['AT_RUN_ID'] = p.AT_RUN_ID

        # REORDER COLUMNS
        final_monthly_data = final_monthly_data[
            ['CUSTOMER_ID', 'CLIENT', 'BREAKOUT', 'MONTH', 'MONTH_TYPE', 'AWP', 'TARGET', 'TARGET_IC', 'TARGET_DF', 'SPEND', 'SPEND_IC', 'SPEND_DF', 'SURPLUS', 'SURPLUS_IC', 'SURPLUS_DF', 'ALGO_RUN_DATE'
                , 'CLIENT_TYPE', 'TIERED_PRICE_LIM'
                , 'UNC_OPT', 'GOODRX_OPT', 'GO_LIVE'
                , 'DATA_ID'#, 'REC_CURR_IND'
                , 'RUN_TYPE','AT_RUN_ID']]  # ,'ALGO_RUN_OWNER'
        
        # Mapping breakout label 
        final_monthly_data_upload = breakout_label_mapping(final_monthly_data)
        
        if p.WRITE_TO_BQ or p.UPLOAD_TO_DASH:
            uf.write_to_bq(
                final_monthly_data_upload,
                project_output = p.BQ_OUTPUT_PROJECT_ID,
                dataset_output = p.BQ_OUTPUT_DATASET,
                table_id = "ytd_surplus_monthly",
                client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                timestamp_param = p.TIMESTAMP,
                run_id=p.AT_RUN_ID,
                schema = None#BQ.ytd_surplus_monthly_schema
            )
        else:
            final_monthly_data_upload.to_csv(p.FILE_REPORT_PATH + p.DATA_ID +'_YTD_SURPLUS_MONTHLY.csv', index=False)


         ################################## price_change ##################################
        """
        This part reads in the LP run result and creates price_change table for reporting
        """

        # File Read in
        price_change = pd.read_excel(
            p.FILE_OUTPUT_PATH + p.PRICE_CHANGES_OUTPUT_FILE, sheet_name='RXC_MACLISTS', dtype=p.VARIABLE_TYPE_DIC)
        price_change = standardize_df(price_change).rename(columns={'MACLIST': 'MAC_LIST'})
        price_change['MAC_LIST'] = price_change['MAC_LIST'].str[3:]
        if p.GOODRX_OPT or p.RMS_OPT:
            goodrx = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.GOODRX_FILE, dtype=p.VARIABLE_TYPE_DIC)).rename(columns={'GOODRX_UPPER_LIMIT': 'GOODRX_CHAIN_PRICE'})

        mac_1026 = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.MAC1026_FILE, dtype=p.VARIABLE_TYPE_DIC))
        gpi_class = standardize_df(pd.read_excel(p.FILE_INPUT_PATH + p.GPI_CLASSES, dtype=p.VARIABLE_TYPE_DIC))
        mac_mapping = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.MAC_MAPPING_FILE, dtype=p.VARIABLE_TYPE_DIC))
        if p.CLIENT_TYPE == 'MEDD' and p.FULL_YEAR and not p.EGWP:
            mac_mapping = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.CUSTOM_MAC_MAPPING_FILE, dtype=p.VARIABLE_TYPE_DIC))
            #Remove the new mac lists from price change file that were mapped for 2024
            price_change = price_change[price_change.MAC_LIST.isin(list(mac_mapping.MAC_LIST.astype(str).unique()))]
        
        p.VARIABLE_TYPE_DIC['GPI_12'] = str

        if False:
            total_output = uf.read_BQ_data(
                BQ.lp_total_output_df,
                project_id = p.BQ_OUTPUT_PROJECT_ID,
                dataset_id = p.BQ_OUTPUT_DATASET,
                table_id = "Total_Output_subgroup",
                run_id = p.AT_RUN_ID,
                client = ', '.join(sorted(p.CUSTOMER_ID)),
                period = p.TIMESTAMP,
                output = True)
            total_output = standardize_df(total_output)
        else:
            total_output = standardize_df(
                pd.read_csv(p.FILE_OUTPUT_PATH + p.TOTAL_OUTPUT, dtype=p.VARIABLE_TYPE_DIC))
        
        # get QUANTITY column
        total_output_agg = total_output.groupby(['MAC_LIST', 'GPI', 'NDC','BG_FLAG']).sum().reset_index() \
            [['GPI', 'NDC', 'QTY', 'QTY_PROJ_LAG', 'QTY_PROJ_EOY', 'MAC_LIST', 'BG_FLAG']]
        total_output_agg = total_output_agg.pivot(
            index = ['GPI', 'NDC', 'MAC_LIST'], 
            values = ['QTY', 'QTY_PROJ_LAG', 'QTY_PROJ_EOY'], 
            columns = ['BG_FLAG']).reset_index()
        total_output_agg.columns = [''.join(col) for col in total_output_agg.columns.to_flat_index()]
        total_output_agg.rename(columns = {
            'QTYG': 'GNRC_QTY', 
            'QTYB': 'BRND_QTY',
            'QTY_PROJ_LAGG': 'GNRC_QTY_PROJ_LAG', 
            'QTY_PROJ_LAGB': 'BRND_QTY_PROJ_LAG',
            'QTY_PROJ_EOYG': 'GNRC_QTY_PROJ_EOY', 
            'QTY_PROJ_EOYB': 'BRND_QTY_PROJ_EOY'
            }, inplace=True)
        for x in ['GNRC_QTY','BRND_QTY','GNRC_QTY_PROJ_LAG','BRND_QTY_PROJ_LAG','GNRC_QTY_PROJ_EOY','BRND_QTY_PROJ_EOY']:
            if x not in total_output_agg.columns:
                total_output_agg[x] = np.nan
        price_change = pd.merge(price_change, total_output_agg, on=['GPI', 'NDC', 'MAC_LIST'], how='left')

        #For 1/1 pricing we should disregard YTD and LAG as we only care about EOY
        if p.FULL_YEAR:
            price_change['GNRC_QUANTITY'] = price_change['GNRC_QTY_PROJ_EOY']
            price_change['BRND_QUANTITY'] = price_change['BRND_QTY_PROJ_EOY']
        else:
            price_change['GNRC_QUANTITY'] = price_change[['GNRC_QTY', 'GNRC_QTY_PROJ_EOY', 'GNRC_QTY_PROJ_LAG']].sum(axis=1)
            price_change['BRND_QUANTITY'] = price_change[['BRND_QTY', 'BRND_QTY_PROJ_EOY', 'BRND_QTY_PROJ_LAG']].sum(axis=1)

        # get mac1026 price
        # Notice current MAC1026 only have generic price, if in future brand price is included, we need to make GNRC_MAC1026_PRICE, BRND_MAC1026_PRICE
        price_change = pd.merge(price_change, mac_1026[['PRICE', 'MAC_EFF_DT', 'GPI', 'NDC']],
                                on=['GPI', 'NDC'], how='left').rename(columns={'PRICE': 'MAC1026_PRICE',
                                                                               'MAC_EFF_DT': 'MAC1026_EFF_DT'})

        # what is goodrx chain_group ind
        # get goodrx prices if goodrx_opt == True or RMS_OPT == True
        if p.GOODRX_OPT or p.RMS_OPT:
            # get 'MEASUREMENT' 'Chain_group'
            if 'CHAIN_SUBGROUP' not in mac_mapping:
                mac_mapping['CHAIN_SUBGROUP'] = mac_mapping['CHAIN_GROUP']
            price_change = pd.merge(price_change,
                                    mac_mapping[['MEASUREMENT', 'MAC_LIST', 'CHAIN_SUBGROUP', 'REGION']].drop_duplicates(),
                                    on='MAC_LIST', how='left')
            price_change['CHAIN_GROUP'] = price_change['CHAIN_SUBGROUP'].copy() #copy() prevents weird NaN error below
            price_change.loc[price_change['CHAIN_GROUP'].str.contains('CVS'), 'CHAIN_GROUP'] = 'CVS'

            # Notice goodrx price only have generic price, if in future brand price is included, we need to make GNRC_GOODRX_PRICE, BRND_GOODRX_PRICE
            price_change = pd.merge(price_change,
                                goodrx[['GPI', 'CHAIN_GROUP', 'GOODRX_CHAIN_PRICE']],
                                on=['GPI', 'CHAIN_GROUP'],
                                how='left').rename(columns={'GOODRX_CHAIN_PRICE': 'GOODRX_PRICE'})

            # Modify mac mapping after goodrx prices get stored
            price_change.loc[price_change.MAC_LIST == (price_change.REGION + '1'), 'CHAIN_GROUP'] = 'R30'
            price_change.loc[price_change.MAC_LIST == (price_change.REGION + '2'), 'CHAIN_GROUP'] = 'M30'
            price_change.loc[price_change.MAC_LIST == (price_change.REGION + '3'), 'CHAIN_GROUP'] = 'R90'
            price_change.loc[price_change.MAC_LIST == (price_change.REGION + '9'), 'CHAIN_GROUP'] = 'R90'

            # keep just the lowest goodrx price from the duplicates
            cols = list(price_change.columns)
            cols.remove('GOODRX_PRICE')
            price_change = price_change.groupby(cols)['GOODRX_PRICE'].min().reset_index()
            price_change = price_change.drop(columns=['REGION'])
        else:
            # Modify mac_mapping, method used from legacy code
            mac_mapping.loc[mac_mapping.MAC_LIST == (mac_mapping.REGION + '1'), 'CHAIN_SUBGROUP'] = 'R30'
            mac_mapping.loc[mac_mapping.MAC_LIST == (mac_mapping.REGION + '2'), 'CHAIN_SUBGROUP'] = 'M30'
            mac_mapping.loc[mac_mapping.MAC_LIST == (mac_mapping.REGION + '3'), 'CHAIN_SUBGROUP'] = 'R90'
            mac_mapping.loc[mac_mapping.MAC_LIST == (mac_mapping.REGION + '9'), 'CHAIN_SUBGROUP'] = 'R90'
            
            price_change = pd.merge(price_change,
                                    mac_mapping[['MEASUREMENT', 'MAC_LIST', 'CHAIN_SUBGROUP']].drop_duplicates(),
                                    on='MAC_LIST', how='left')
            price_change['CHAIN_GROUP'] = price_change['CHAIN_SUBGROUP'].copy() #copy() prevents weird NaN error below
            price_change.loc[price_change['CHAIN_GROUP'].str.contains('CVS',na=False), 'CHAIN_GROUP'] = 'CVS'

            price_change['GOODRX_PRICE'] = np.nan
        # GPI class name
        price_change = pd.merge(price_change,
                                gpi_class.drop_duplicates(subset=['GPI_CD']).rename(columns={'GPI_CD': 'GPI'}),
                                on='GPI', how='left')

        # add columns for reporting
        price_change['CUSTOMER_ID'] = uf.get_formatted_client_name(p.CUSTOMER_ID) # add functionality to handle clients with multiple Ids
        price_change['CLIENT'] = p.CLIENT_NAME_TABLEAU
        price_change['ALGO_RUN_DATE'] = dt.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        # price_change['REC_CURR_IND'] = 'Y'
        price_change['CLIENT_TYPE'] = p.CLIENT_TYPE
        price_change['UNC_OPT'] = p.UNC_OPT
        price_change['GOODRX_OPT'] = p.RMS_OPT
        price_change['GO_LIVE'] = p.GO_LIVE.strftime("%Y-%m-%d")
        price_change['DATA_ID'] = p.DATA_ID
        price_change['TIERED_PRICE_LIM'] = p.TIERED_PRICE_LIM
        price_change['RUN_TYPE'] = RUN_TYPE_TABLEAU_UPDATED
        price_change['AT_RUN_ID'] = p.AT_RUN_ID

        # Re-order columns
        if p.TRUECOST_CLIENT or p.UCL_CLIENT:
            price_change = price_change[['MAC_LIST', 'GPI', 'GPPC', 'NDC', 'EFFDATE', 'TERMDATE', 'GNRC_MACPRC','BRND_MACPRC',
           'GNRC_CURRENTMAC','BRND_CURRENTMAC', #'GPI_NDC', 'QTY', 'QTY_PROJ_LAG', 'QTY_PROJ_EOY',
           'GNRC_QUANTITY', 'BRND_QUANTITY',
           'CUSTOMER_ID', 'MEASUREMENT','CHAIN_GROUP', 'MAC1026_PRICE', 'MAC1026_EFF_DT',
           'GOODRX_PRICE', 'GPI_CLS_NM', 'GPI_CTGRY_NM',
           'CLIENT', 'ALGO_RUN_DATE', 'CLIENT_TYPE', 'UNC_OPT',
           'GOODRX_OPT', 'GO_LIVE', 'DATA_ID', 'TIERED_PRICE_LIM', 'RUN_TYPE', 'AT_RUN_ID']]
        else:
            price_change = price_change[['MAC_LIST', 'GPI', 'GPPC', 'NDC', 'EFFDATE', 'TERMDATE', 'MACPRC',
               'CURRENT MAC', #'GPI_NDC', 'QTY', 'QTY_PROJ_LAG', 'QTY_PROJ_EOY',
               'QUANTITY', 'CUSTOMER_ID', 'MEASUREMENT','CHAIN_GROUP', 'MAC1026_PRICE', 'MAC1026_EFF_DT',
               'GOODRX_PRICE', 'GPI_CLS_NM', 'GPI_CTGRY_NM',
               'CLIENT', 'ALGO_RUN_DATE', 'CLIENT_TYPE', 'UNC_OPT',
               'GOODRX_OPT', 'GO_LIVE', 'DATA_ID', 'TIERED_PRICE_LIM', 'RUN_TYPE', 'AT_RUN_ID']]
        # export to BQ or as CSV
        if p.WRITE_TO_BQ or p.UPLOAD_TO_DASH:
            uf.write_to_bq(
                price_change,
                project_output = p.BQ_OUTPUT_PROJECT_ID,
                dataset_output = p.BQ_OUTPUT_DATASET,
                table_id = "price_change",
                client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                timestamp_param = p.TIMESTAMP,
                run_id = p.AT_RUN_ID,
                schema = None  # TODO: create schema
            )
        else:
            price_change.to_csv(p.FILE_REPORT_PATH + p.DATA_ID +'_price_change.csv', index=False)

        ################################## price_dist ##################################
        """
        This function reads in LP run result and create price_dist table for reporting
        """

        # Price_dist is calculate using total output, total_output loaded in previous reporting table generation code
        price_dist = total_output.loc[(total_output['QTY_PROJ_EOY'] > 0) & (total_output['OLD_MAC_PRICE'] != 0)].copy(
            deep=True)
        # calculate script QTY, price

        price_dist['AVG_QTY_PER_RXS'] = np.where(price_dist['CLAIMS_PROJ_EOY'] == 0,
                                                 0,
                                                 price_dist['QTY_PROJ_EOY'] / price_dist['CLAIMS_PROJ_EOY'])
        price_dist['ORIG_DRUG_PRICE'] = (price_dist['OLD_MAC_PRICE'] * price_dist['AVG_QTY_PER_RXS']) + price_dist['AVG_DISP_FEE']
        price_dist['PRICE_CHANGE'] = price_dist['NEW_PRICE'] - price_dist['OLD_MAC_PRICE']
        price_dist['PERCENT_CHANGE'] = price_dist['PRICE_CHANGE'] / price_dist['OLD_MAC_PRICE']
        price_dist["DOLLAR_INCREASE_PER_SCRIPT"] = price_dist['PRICE_CHANGE'] * price_dist['AVG_QTY_PER_RXS']

        # Assign script bucket
        bins = [-float("inf"), -20 , -5, -0.0001, 0.0001, 3, 6, 10, 20, 50, float("inf")]
        labels = ['01. $20+ decrease',
                  '02. $5-20 decrease',
                  '03. $0-5 decrease',
                  '04. No Change',
                  '05. $0-3 Increase',
                  '06. $3-6 Increase',
                  '07. $6-10 Increase',
                  '08. $10-20 Increase',
                  '09. $20-50 Increase',
                  '10. $50+ Increase'
        ]
        price_dist['SCRIPT_BUCKET'] = pd.cut(price_dist['DOLLAR_INCREASE_PER_SCRIPT'], bins=bins, labels=labels, right=False)

       # Assign script bucket category
        bins = [-float("inf"), 3, 6, 15, 25, 50, 100, float("inf")]
        labels = ['01. $0-3 avg script',
                  '02. $3-6 avg script',
                  '03. $6-15 avg script',
                  '04. $15-25 avg script',
                  '05. $25-50 avg script',
                  '06. $50-100 avg script',
                  '07. $100+ avg script'
                  ]
        price_dist['SCRIPT_BUCKET_CAT'] = pd.cut(price_dist['ORIG_DRUG_PRICE'], bins=bins, labels=labels, right=False)
        price_dist['NO_FUTURE_CLAIMS'] = np.where(price_dist['QTY_PROJ_EOY'] == 0, 'Y', 'N')

        # Save a copy for psot before group by
        psot_price_dist = price_dist.copy()

        # aggregate price distribution
        price_dist = price_dist.groupby(['CLIENT', 'REGION', 'BREAKOUT', 'SCRIPT_BUCKET', 'SCRIPT_BUCKET_CAT']) \
            ['CLAIMS_PROJ_EOY'].agg('sum').reset_index()
        
        # add columns for reporting
        price_dist['CUSTOMER_ID'] = uf.get_formatted_client_name(p.CUSTOMER_ID) # add functionality to handle clients with multiple Ids
        price_dist['CLIENT'] = p.CLIENT_NAME_TABLEAU
        price_dist['ALGO_RUN_DATE'] = dt.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        # price_dist['REC_CURR_IND'] = 'Y'
        price_dist['CLIENT_TYPE'] = p.CLIENT_TYPE
        price_dist['UNC_OPT'] = p.UNC_OPT
        price_dist['GOODRX_OPT'] = p.RMS_OPT
        price_dist['GO_LIVE'] = p.GO_LIVE.strftime("%Y-%m-%d")
        price_dist['DATA_ID'] = p.DATA_ID
        price_dist['TIERED_PRICE_LIM'] = p.TIERED_PRICE_LIM
        price_dist['RUN_TYPE'] = RUN_TYPE_TABLEAU_UPDATED
        price_dist['AT_RUN_ID'] = p.AT_RUN_ID
        #Re-order columns
        price_dist = price_dist[['CUSTOMER_ID', 'CLIENT', 'REGION', 'BREAKOUT', 'SCRIPT_BUCKET', 'SCRIPT_BUCKET_CAT',
           'CLAIMS_PROJ_EOY', 'ALGO_RUN_DATE', 'CLIENT_TYPE',
           'UNC_OPT', 'GOODRX_OPT', 'GO_LIVE', 'DATA_ID', 'TIERED_PRICE_LIM',
           'RUN_TYPE', 'AT_RUN_ID']]
        # Mapping breakout label
        price_dist_upload = breakout_label_mapping(price_dist)
        # export to BQ or as CSV
        if p.WRITE_TO_BQ or p.UPLOAD_TO_DASH:
            uf.write_to_bq(
                price_dist_upload,
                project_output = p.BQ_OUTPUT_PROJECT_ID,
                dataset_output = p.BQ_OUTPUT_DATASET,
                table_id = "price_dist",
                client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                timestamp_param = p.TIMESTAMP,
                run_id = p.AT_RUN_ID,
                schema = None  # TODO: create schema
            )
        else:
            price_dist_upload.to_csv(p.FILE_REPORT_PATH + p.DATA_ID +'_price_dist.csv', index=False)
            
            
        ################################## price_competitiveness ##################################
        """
        This part reads in the LP run result and creates price_competitiveness table for reporting
        """

        if False:
            total_output = uf.read_BQ_data(
                BQ.lp_total_output_df,
                project_id = p.BQ_OUTPUT_PROJECT_ID,
                dataset_id = p.BQ_OUTPUT_DATASET,
                table_id = "Total_Output_subgroup",
                run_id = p.AT_RUN_ID,
                client = ', '.join(sorted(p.CUSTOMER_ID)),
                period = p.TIMESTAMP,
                output = True)
            total_output = standardize_df(total_output)
        else:
            total_output = standardize_df(
                pd.read_csv(p.FILE_OUTPUT_PATH + p.TOTAL_OUTPUT, dtype=p.VARIABLE_TYPE_DIC))
        total_output_agg_comp = total_output.groupby(['GPI', 'MAC', 'MEASUREMENT', 'BG_FLAG', 'IS_MAC', 'IS_SPECIALTY']).agg({'CHAIN_GROUP':', '.join, 'OLD_MAC_PRICE':'mean', 'FINAL_PRICE':'mean', 'SOFT_CONST_BENCHMARK_PRICE':'mean', 'FULLAWP_ADJ':'sum', 'CLAIMS':'sum', 'QTY':'sum', 'DISP_FEE':'sum', 'TARGET_DISP_FEE':'sum'}).reset_index()
        total_output_agg_comp.rename(columns={'OLD_MAC_PRICE':'OLD_PRICE', 'SOFT_CONST_BENCHMARK_PRICE':'BENCHMARK_PRICE'}, inplace=True)
        
        total_output_agg_comp['NDC'] = '***********'
        total_output_agg_comp['OLD_SCRIPT_PRICE'] = ((total_output_agg_comp['OLD_PRICE'] * total_output_agg_comp['QTY']) + 
                                                     total_output_agg_comp['DISP_FEE']) / total_output_agg_comp['CLAIMS']
        total_output_agg_comp['NEW_SCRIPT_PRICE'] = ((total_output_agg_comp['FINAL_PRICE'] * total_output_agg_comp['QTY']) + 
                                                     total_output_agg_comp['DISP_FEE']) / total_output_agg_comp['CLAIMS']
        total_output_agg_comp['BENCHMARK_SCRIPT_PRICE'] = ((total_output_agg_comp['BENCHMARK_PRICE'] * total_output_agg_comp['QTY']) + 
                                                           total_output_agg_comp['TARGET_DISP_FEE']) / total_output_agg_comp['CLAIMS']

        # add columns for reporting
        total_output_agg_comp['CUSTOMER_ID'] = uf.get_formatted_client_name(p.CUSTOMER_ID) # add functionality to handle clients with multiple Ids
        total_output_agg_comp['CLIENT'] = p.CLIENT_NAME_TABLEAU
        total_output_agg_comp['ALGO_RUN_DATE'] = dt.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        total_output_agg_comp['CLIENT_TYPE'] = p.CLIENT_TYPE
        total_output_agg_comp['GO_LIVE'] = p.GO_LIVE.strftime("%Y-%m-%d")
        total_output_agg_comp['DATA_ID'] = p.DATA_ID
        total_output_agg_comp['RUN_TYPE'] = RUN_TYPE_TABLEAU_UPDATED
        total_output_agg_comp['AT_RUN_ID'] = p.AT_RUN_ID

        # Re-order columns
        total_output_agg_comp = total_output_agg_comp[['GPI', 'NDC', 'MAC', 'MEASUREMENT', 'BG_FLAG', 'IS_MAC', 'IS_SPECIALTY', 
            'CHAIN_GROUP', 'OLD_PRICE', 'FINAL_PRICE', 'BENCHMARK_PRICE', 'OLD_SCRIPT_PRICE', 'NEW_SCRIPT_PRICE', 
            'BENCHMARK_SCRIPT_PRICE', 'FULLAWP_ADJ', 'CLAIMS', 'QTY', 'CUSTOMER_ID', 'CLIENT', 'ALGO_RUN_DATE', 
            'CLIENT_TYPE', 'GO_LIVE', 'DATA_ID', 'RUN_TYPE', 'AT_RUN_ID']]
        # export to BQ or as CSV
        if p.WRITE_TO_BQ or p.UPLOAD_TO_DASH:
            uf.write_to_bq(
                total_output_agg_comp,
                project_output = p.BQ_OUTPUT_PROJECT_ID,
                dataset_output = p.BQ_OUTPUT_DATASET,
                table_id = "price_competitiveness_TC",
                client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                timestamp_param = p.TIMESTAMP,
                run_id = p.AT_RUN_ID,
                schema = None  # TODO: create schema
            )
        else:
            total_output_agg_comp.to_csv(p.FILE_REPORT_PATH + p.DATA_ID +'_price_competitiveness.csv', index=False)


        ################################## pm_cnp_client_run_specific_info ##################################
        """
            This part of code takes in the LP run result and generates pm_cnp_client_run_specific_info for PSOT
        """
        # Change in Capped, Non-capped, and PSAO: increase in spend
        # awp_spend_perf copy saved as psot_awp_spend_perf:
        psot_change_in_pharm_spend = psot_awp_spend_perf.loc[psot_awp_spend_perf['ENTITY_TYPE'].isin(['Retail Major', 'Retail Independent', 'Retail PSAO'])]
        # Row to column Change in Capped Psao
        psot_change_in_pharm_spend = psot_change_in_pharm_spend.groupby(['AT_RUN_ID','ENTITY_TYPE'])['Increase_in_Spend'].sum().reset_index()
        psot_change_in_pharm_spend = psot_change_in_pharm_spend.pivot(index='AT_RUN_ID',
                                                                      columns='ENTITY_TYPE',
                                                                      values='Increase_in_Spend').reset_index()
        # CPI CVS, WMT..: increase in reimb
        psot_cpi_capped = psot_awp_spend_perf.loc[psot_awp_spend_perf['ENTITY_TYPE']=='Retail Major']
        psot_cpi_capped = psot_cpi_capped.groupby(['AT_RUN_ID','CHAIN_SUBGROUP'])['Increase_in_Reimb'].mean().reset_index()
        # Row to column CPI CVS, MWT
        psot_cpi_capped = psot_cpi_capped.pivot(index='AT_RUN_ID',
                                                          columns='CHAIN_SUBGROUP',
                                                          values='Increase_in_Reimb').reset_index(drop=True)
        #concate the result
        psot_perf = pd.concat([psot_change_in_pharm_spend, psot_cpi_capped] ,axis=1)
        
        # Rename the columns
        psot_perf.rename(columns={'AT_RUN_ID':'RUN_ID',
                                 'Retail Major':'CHANGE_IN_CAPPED',
                                 'Retail Independent':'CHANGE_IN_NON_CAPPED',
                                 'Retail PSAO':'CHANGE_IN_PSAO',
                                 'CVS':'CPI_CVS',
                                 'KRG':'CPI_KRG',
                                 'RAD':'CPI_RAD',
                                 'WAG':'CPI_WAG',
                                 'WMT':'CPI_WMT'
                                }
                         , inplace =True)
        if 'CVSSP' in psot_perf.columns:
            psot_perf.rename(columns={'CVSSP': 'CPI_CVS_SP'}, inplace=True)
        # Add cpi total: sum of big5 CPI
        psot_perf['CPI_TOTAL'] = psot_perf[[i for i in psot_perf.columns.tolist() if 'CPI' in i]].sum(axis=1)


        # Add offsetting T/F indicator using the logic shared by DE
        offsetting_guarantee_category=['Offsetting Complex','Offsetting R30/R90','MedD/EGWP Offsetting R30/R90/LTC','MedD/EGWP Offsetting Complex','Aetna Offsetting','Aetna MR Offsetting','Aetna BG Offsetting', 'Aetna NA Offsetting']
        psot_perf['OFFSET'] = GER_OPT_TAXONOMY_FINAL.loc[0,'guarantee_category'] in offsetting_guarantee_category
        psot_perf['BG_Offset']= GER_OPT_TAXONOMY_FINAL.loc[0,'BG_Offset']

        if p.CLIENT_LOB == 'AETNA':
            psot_perf['CLIENT_GUARANTEE_STRUCTURE'] = GER_OPT_TAXONOMY_FINAL.loc[0,'guarantee_category'] + '_'  + GER_OPT_TAXONOMY_FINAL.loc[0,'recon_basis'] 
        else:
            psot_perf['CLIENT_GUARANTEE_STRUCTURE'] = GER_OPT_TAXONOMY_FINAL.loc[0,'guarantee_category']
        psot_perf['CONTRACT_EFF_DT'] = contract_date_df.loc[0, 'CONTRACT_EFF_DT']
        psot_perf['CONTRACT_EXPRN_DT'] = contract_date_df.loc[0, 'CONTRACT_EXPRN_DT']
                
        # If the guarantee_category within the four offsetting categories or BG_Offset = 1 then sum surplus of retail, else null
        if (psot_perf.loc[0,'OFFSET'] == True) or (psot_perf.loc[0,'BG_Offset'] == 1):
            psot_perf['RETAIL_BLENDED_SURPLUS_AMT'] = psot_awp_spend_perf[~(psot_awp_spend_perf['BREAKOUT'].str.contains('M')) & \
                                                                          (psot_awp_spend_perf['CLIENT_OR_PHARM']=='CLIENT')]['Model_Perf'].sum()
        else:
            psot_perf['RETAIL_BLENDED_SURPLUS_AMT'] = np.nan
            
        # If BG_Offset = 1 then sum surplus of mail, else null
        if  psot_perf.loc[0,'BG_Offset'] == 1:
            psot_perf['MAIL_BLENDED_SURPLUS_AMT'] = psot_awp_spend_perf[(psot_awp_spend_perf['BREAKOUT'].str.contains('M')) & \
                                                                        (psot_awp_spend_perf['CLIENT_OR_PHARM']=='CLIENT')]['Model_Perf'].sum()
        else:
            psot_perf['MAIL_BLENDED_SURPLUS_AMT'] = np.nan
            
        # If the guarantee_category within the four offsetting categories or BG_Offset = 1 then sum pre_existing_perf of retail, else null  
        if (psot_perf.loc[0,'OFFSET'] == True) or (psot_perf.loc[0,'BG_Offset'] == 1):
            psot_perf['RETAIL_BLENDED_SURPLUS_DO_NOTHING'] = psot_awp_spend_perf[~(psot_awp_spend_perf['BREAKOUT'].str.contains('M')) & \
                                                                                 (psot_awp_spend_perf['CLIENT_OR_PHARM']=='CLIENT')]['Pre_existing_Perf'].sum()
        else:
            psot_perf['RETAIL_BLENDED_SURPLUS_DO_NOTHING'] = np.nan
            
        # If BG_Offset = 1 then sum pre_existing_perf of mail, else null
        if  psot_perf.loc[0,'BG_Offset'] == 1:
            psot_perf['MAIL_BLENDED_SURPLUS_DO_NOTHING'] = psot_awp_spend_perf[(psot_awp_spend_perf['BREAKOUT'].str.contains('M')) & \
                                                                               (psot_awp_spend_perf['CLIENT_OR_PHARM']=='CLIENT')]['Pre_existing_Perf'].sum()
        else:
            psot_perf['MAIL_BLENDED_SURPLUS_DO_NOTHING'] = np.nan    
        
        # psot dist: copy saved above of price_dist gpi level
        psot_price_dist['AT_RUN_ID'] = p.AT_RUN_ID
        # As requested by PSOT, attribute buckets using OLD_MAC_PRICE
        # Assign old_mac_price bucket category
        bins = [-float("inf"), 3, 6, 15, 25, 50, 100, float("inf")]
        labels = ['01. $0-3 avg script',
                  '02. $3-6 avg script',
                  'SIX_TO_FIFTEEN_DOLLARS',
                  'FIFTEEN_TO_TWENTYFIVE_DOLLARS',
                  'TWENTYFIVE_TO_FIFTY_DOLLARS',
                  'FIFTY_TO_HUNDRED_DOLLARS',
                  'HUNDRED_DOLLARS_AND_ABOVE'
                  ]
        psot_price_dist['OLD_MAC_PRICE_BUCKET'] = pd.cut(psot_price_dist['ORIG_DRUG_PRICE'], bins=bins, labels=labels, right=False)
        # Calculate the max percent change = price_change/price , group by script_bucket_Cat and agg using max
        psot_price_dist = psot_price_dist.groupby(['AT_RUN_ID','OLD_MAC_PRICE_BUCKET'])['PERCENT_CHANGE'].max().reset_index()
        # Row to columns
        psot_price_dist_pivot = psot_price_dist.pivot(index='AT_RUN_ID',
                                                      columns='OLD_MAC_PRICE_BUCKET',
                                                      values='PERCENT_CHANGE').rename(columns=str).reset_index(drop=True)
        # Reorganize columns
        psot_price_dist_pivot.columns.name = None
        psot_price_dist_pivot.drop(['01. $0-3 avg script','02. $3-6 avg script'],axis = 1,inplace=True)
    #     psot_price_dist_pivot.rename(columns = {
    #           '03. $6-15 avg script':'SIX_TO_FIFTEEN_DOLLARS',
    #           '04. $15-25 avg script':'FIFTEEN_TO_TWENTYFIVE_DOLLARS',
    #           '05. $25-50 avg script':'TWENTYFIVE_TO_FIFTY_DOLLARS',
    #           '06. $50-100 avg script':'FIFTY_TO_HUNDRED_DOLLARS',
    #           '07. $100+ avg script':'HUNDRED_DOLLARS_AND_ABOVE'}, inplace=True)
        psot = pd.concat([psot_perf, psot_price_dist_pivot], axis = 1)

        # All columns in schema
        psot_all_columns = ['RUN_ID', 'SIX_TO_FIFTEEN_DOLLARS',
           'FIFTEEN_TO_TWENTYFIVE_DOLLARS', 'TWENTYFIVE_TO_FIFTY_DOLLARS',
           'FIFTY_TO_HUNDRED_DOLLARS', 'HUNDRED_DOLLARS_AND_ABOVE',
            'CHANGE_IN_CAPPED', 'CHANGE_IN_NON_CAPPED', 'CHANGE_IN_PSAO', 'CPI_CVS', 'CPI_CVS_SP', 'CPI_KRG',
           'CPI_RAD', 'CPI_WAG', 'CPI_WMT', 'CPI_TOTAL','OFFSET',
           'RETAIL_BLENDED_SURPLUS_AMT','MAIL_BLENDED_SURPLUS_AMT','MAIL_BLENDED_SURPLUS_DO_NOTHING','RETAIL_BLENDED_SURPLUS_DO_NOTHING',
           'CLIENT_GUARANTEE_STRUCTURE','CONTRACT_EFF_DT','CONTRACT_EXPRN_DT']
        psot_columns = psot.columns.values
        # Add null if not exist in the table (required if schema being specified)
        for col in psot_all_columns:
            if col not in psot_columns:
                psot[col] = np.nan
        #Re-order columns to the same as requested by psot DE
        psot = psot[psot_all_columns]
        
        assert psot[['RUN_ID','OFFSET','CHANGE_IN_CAPPED','CPI_TOTAL']].notnull().any().min(), "Null data found in non-nullable field."

        # export to BQ or as CSV
        if p.WRITE_TO_BQ or p.UPLOAD_TO_DASH:
            uf.write_to_bq(
                psot,
                project_output = p.BQ_OUTPUT_PROJECT_ID,
                dataset_output = "pricing_management",
                table_id = "pm_cnp_client_run_specific_info",
                client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                timestamp_param = p.TIMESTAMP,
                run_id = p.AT_RUN_ID,
                schema = BQ.psot_schema
            )
        else:
            psot.to_csv(p.FILE_REPORT_PATH + p.DATA_ID +'_psot.csv', index=False)
         
        ################################## rms_report ##################################
        """
        This part reads in the LP run result and GoodRx prices for reporting
        """

        # File Read in
        if p.RMS_OPT or p.APPLY_GENERAL_MULTIPLIER:
            grx_calculated = standardize_df(pd.read_csv(os.path.join(p.FILE_LOG_PATH, 'GOODRX_CALCS_LOG_{}.csv'.format(p.DATA_ID)),dtype=p.VARIABLE_TYPE_DIC))
            cols = ['CUSTOMER_ID','REGION','BREAKOUT','MEASUREMENT','CHAIN_GROUP','GPI','BG_FLAG',
                    'SPEND_CLAIM','MBR_COST_CLAIM','CLNT_COST_CLAIM','MIN_GRX_PRC_QTY','MAX_GRX_PRC_QTY',
                    'GOODRX_UNIT_PRICE_SAME','GOODRX_CHAIN_PRICE']
            grx_raw = grx_calculated[cols]
            grx_raw = grx_raw.rename(columns={"GOODRX_UNIT_PRICE_SAME": "GOODRX_RAW_PRICE", "CUSTOMER_ID":"CLIENT"})
            
            if p.READ_FROM_BQ == False:
                total_output = uf.read_BQ_data(
                    BQ.lp_total_output_df,
                    project_id = p.BQ_OUTPUT_PROJECT_ID,
                    dataset_id = p.BQ_OUTPUT_DATASET,
                    table_id = "Total_Output",
                    run_id = p.AT_RUN_ID,
                    client = ', '.join(sorted(p.CUSTOMER_ID)),
                    period = p.TIMESTAMP,
                    output = True)
                total_output = standardize_df(total_output)
            else:
                total_output = standardize_df(pd.read_csv(p.FILE_OUTPUT_PATH + p.TOTAL_OUTPUT, dtype=p.VARIABLE_TYPE_DIC))

            cols = ['CLIENT','REGION','BREAKOUT','MEASUREMENT','BG_FLAG','CHAIN_GROUP','GPI','MAC_LIST','CURRENT_MAC_PRICE','GOODRX_UPPER_LIMIT','FINAL_PRICE']
            lp_output = total_output[cols].copy()
            ## Assume GRX is only applying to generic drug prices
            lp_output = lp_output.loc[lp_output.BG_FLAG == 'G']
            lp_output.loc[lp_output.CHAIN_GROUP == 'MCHOICE' ,'CHAIN_GROUP'] = 'MAIL'
            lp_output.loc[~(lp_output.CHAIN_GROUP.isin(['CVS','KRG','RAD','WAG','WMT'])) & (lp_output.MEASUREMENT != 'M30') ,'CHAIN_GROUP'] = 'OTH'
            lp_output = lp_output.drop_duplicates()

            final_output = pd.merge(lp_output, grx_raw, how = 'left', on =  ['CLIENT','REGION','BREAKOUT','MEASUREMENT','CHAIN_GROUP', 'GPI','BG_FLAG'])

            costplus_df=pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.COSTPLUS_FILE)
            costplus_df['MCCP_UNIT_PRICE'] = costplus_df['MAIL_CEILING']/p.MAIL_MULTIPLIER[0]
            len1=final_output.shape[0]
            final_output = pd.merge(final_output, costplus_df, how = 'left', on =  ['MEASUREMENT', 'GPI','BG_FLAG'])
            len2=final_output.shape[0]
            assert len1 == len2 , 'Cost plus file potentially has duplicates'
                       
            # add columns for reporting
            final_output['CUSTOMER_ID'] = uf.get_formatted_client_name(p.CUSTOMER_ID)
            final_output['CLIENT'] = p.CLIENT_NAME_TABLEAU
            final_output['CLIENT_TYPE'] = p.CLIENT_TYPE
            final_output['GO_LIVE'] = p.GO_LIVE.strftime("%Y-%m-%d")
            final_output['DATA_ID'] = p.DATA_ID
            final_output['RUN_TYPE'] = RUN_TYPE_TABLEAU_UPDATED
            final_output['RETAIL_CEILING']=3*final_output['GOODRX_RAW_PRICE']

            final_output = final_output[['CUSTOMER_ID','CLIENT','REGION','BREAKOUT','MEASUREMENT','BG_FLAG','CHAIN_GROUP','GPI','MAC_LIST',
                            'SPEND_CLAIM','MBR_COST_CLAIM','CLNT_COST_CLAIM','GOODRX_RAW_PRICE','GOODRX_UPPER_LIMIT',
                            'CURRENT_MAC_PRICE','FINAL_PRICE','CLIENT_TYPE','GO_LIVE','DATA_ID','RUN_TYPE','MAIL_CEILING','MCCP_UNIT_PRICE','RETAIL_CEILING','GOODRX_CHAIN_PRICE']]

            # Mapping breakout label
            final_output_upload = breakout_label_mapping(final_output)

            if p.WRITE_TO_BQ or p.UPLOAD_TO_DASH:
                uf.write_to_bq(
                    final_output_upload,
                    project_output = p.BQ_OUTPUT_PROJECT_ID,
                    dataset_output = p.BQ_OUTPUT_DATASET,
                    table_id = "RMS_Report",
                    client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                    timestamp_param = p.TIMESTAMP,
                    run_id = p.AT_RUN_ID,
                    schema = None  # TODO: create schema
                )
            else:
                final_output_upload.to_csv(p.FILE_REPORT_PATH + p.DATA_ID +'_RMS_Report.csv', index=False)
                
        ################################## Client_report ##################################
        ## Notice for this report we are allowing the functionality to report BER, but the column name stays the same at this point with couple of column start with pre fix of 'GER_'
        ## The intention is to leave the col/structure the same as it is for now, while we discuss more with cnp/enhancement team for the reporting needs for brand
        generic_report['BUCKET'] = np.where(generic_report.BG_FLAG == 'G','GER','BER')
        generic_report['MAILIND'] = generic_report['MEASUREMENT'].str[0]
        generic_report['CHANNEL'] = 'MAIL'
        generic_report.loc[generic_report.MAILIND == 'R', 'CHANNEL'] = 'RETAIL'
        generic_report['CLIENT'] =  generic_report['CLIENT'].astype(str)
        generic_report['GUARANTEE_CATEGORY'] = pd.merge(generic_report, GER_OPT_TAXONOMY_FINAL, left_on='CLIENT', right_on='Customer_ID', how='left')['guarantee_category']

        generic_report = generic_report.merge(client_guarantees_meas, how = 'left', on = ['CLIENT','REGION','BREAKOUT','MEASUREMENT','BG_FLAG','PHARMACY_TYPE'])\
                                       .drop(columns = 'PHARMACY_TYPE')
        client_report = generic_report.copy()
        
        if p.CLIENT_LOB == "AETNA":
            brand_report = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.BRAND_SURPLUS_REPORT, dtype = p.VARIABLE_TYPE_DIC))
            brand_report['GUARANTEE_CATEGORY'] = generic_report['GUARANTEE_CATEGORY'].iloc[0]
            client_report = pd.concat([brand_report, client_report]) 
            
        client_report = client_report.rename(columns = {'RATE':'GER_TARGET'})
        client_report['CUSTOMER_ID'] = uf.get_formatted_client_name(p.CUSTOMER_ID)
        client_report['CLIENT'] = p.CLIENT_NAME_TABLEAU

        #Calculations
        client_report['ANNUALIZED_AWP'] = client_report[['FULLAWP_ADJ','FULLAWP_ADJ_PROJ_LAG','FULLAWP_ADJ_PROJ_EOY']].sum(axis = 1)
        client_report['ANNUALIZED_TARGET'] = client_report[['TARG_INGCOST_ADJ_PROJ_EOY', 'TARG_INGCOST_ADJ', 'TARG_INGCOST_ADJ_PROJ_LAG', 
                                                            'TARGET_DISP_FEE','TARGET_DISP_FEE_PROJ_LAG','TARGET_DISP_FEE_PROJ_EOY']].sum(axis = 1)
        client_report['ANNUALIZED_DN_SPEND'] = client_report[['PRICE_REIMB','LAG_REIMB','OLD_PRICE_EFFECTIVE_REIMB_PROJ_EOY','DISP_FEE','DISP_FEE_PROJ_LAG','DISP_FEE_PROJ_EOY']].sum(axis = 1)
        client_report['ANNUALIZED_MODEL_SPEND'] = client_report[['PRICE_REIMB','LAG_REIMB','PRICE_EFFECTIVE_REIMB_PROJ','DISP_FEE','DISP_FEE_PROJ_LAG','DISP_FEE_PROJ_EOY']].sum(axis = 1)
        client_report['GER_TARGET'] = round(1 - (client_report['ANNUALIZED_TARGET'] / client_report['ANNUALIZED_AWP']), 6)

        client_report['YTD_ACTUAL_SURPLUS'] = (client_report['TARG_INGCOST_ADJ'] + client_report['DISP_FEE']) - client_report['PRICE_REIMB']
        client_report['ANNUALILZED_DN_SURPLUS'] = client_report['ANNUALIZED_TARGET'] - client_report['ANNUALIZED_DN_SPEND']
        client_report['ANNUALILZED_MODEL_SURPLUS'] = client_report['ANNUALIZED_TARGET'] - client_report['ANNUALIZED_MODEL_SPEND']

        client_report['GER_ACTUAL'] = 1 - (client_report['PRICE_REIMB'] + client_report['DISP_FEE'])/client_report['FULLAWP_ADJ']
        client_report['GER_DO_NOTHING'] = 1 - client_report['ANNUALIZED_DN_SPEND']/client_report['ANNUALIZED_AWP']
        client_report['GER_MODEL'] = 1 - client_report['ANNUALIZED_MODEL_SPEND']/client_report['ANNUALIZED_AWP']

        client_report['BPS_ACTUAL'] = (client_report['GER_ACTUAL'] - client_report['GER_TARGET'])*10000
        client_report['BPS_DN'] = (client_report['GER_DO_NOTHING'] - client_report['GER_TARGET'])*10000
        client_report['BPS_MODEL'] = (client_report['GER_MODEL'] - client_report['GER_TARGET'])*10000

        ## Calculate ytd_run_rate
        if p.READ_FROM_BQ:
            run_rate = uf.read_BQ_data(
                BQ.run_rate_custom.format(
                        _project_id = p.BQ_INPUT_PROJECT_ID,
                        _dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP,
                        _table_id = uf.get_formatted_table_id('combined_daily_totals') + p.WS_SUFFIX,
                        # _table_id='combined_daily_totals' + p.WS_SUFFIX,
                        _customer_id = uf.get_formatted_string(p.CUSTOMER_ID),
                        _last_data = "'" + p.LAST_DATA.strftime('%Y-%m-%d') +"'"),
                project_id=p.BQ_INPUT_PROJECT_ID,
                dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                table_id='combined_daily_totals' + p.WS_SUFFIX,
                custom = True)
            if p.GENERIC_OPT and not p.BRAND_OPT:
                run_rate = run_rate.loc[run_rate.BG_FLAG == 'G']
            elif not p.GENERIC_OPT and p.BRAND_OPT:
                run_rate = run_rate.loc[run_rate.BG_FLAG == 'B']
        else:
            assert False, "Get data from BQ"

        run_rate = standardize_df(run_rate)
        run_rate['BUCKET'] = np.where(run_rate.BG_FLAG == 'G','GER','BER')
      
        client_report = pd.merge(client_report, run_rate, how = 'left', on = ['CUSTOMER_ID','BUCKET','MEASUREMENT', 'BG_FLAG'])

        client_report['YTD_RUN_RATE'] = (1-client_report['GER_TARGET'])*client_report['RR_AWP'] - client_report['RR_SPEND']

        if months_left > 0:
            client_report['RUN_RATE_DN'] = (client_report['ANNUALILZED_DN_SURPLUS'] - client_report['YTD_ACTUAL_SURPLUS'])/months_left
            client_report['RUN_RATE_MODEL'] = (client_report['ANNUALILZED_MODEL_SURPLUS'] - client_report['YTD_ACTUAL_SURPLUS'])/months_left
        else:
            client_report['RUN_RATE_DN'] = np.where(client_report['ANNUALILZED_DN_SURPLUS'].notnull(), 0.00, np.nan)
            client_report['RUN_RATE_MODEL'] = np.where(client_report['ANNUALILZED_MODEL_SURPLUS'].notnull(), 0.00, np.nan)

        client_report.drop(columns = ['RR_AWP','RR_SPEND'], inplace = True)

        client_report['ALGO_RUN_DATE'] = dt.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
        client_report['CLIENT_TYPE'] = p.CLIENT_TYPE
        client_report['GO_LIVE'] = p.GO_LIVE.strftime("%Y-%m-%d")
        client_report['DATA_ID'] = p.DATA_ID
        client_report['RUN_TYPE'] = RUN_TYPE_TABLEAU_UPDATED
        client_report['AT_RUN_ID'] = p.AT_RUN_ID


        cols = ['CUSTOMER_ID','CLIENT','REGION','BREAKOUT','MEASUREMENT','BG_FLAG','MAILIND','BUCKET','CHANNEL', 'GUARANTEE_CATEGORY',
                'FULLAWP_ADJ','FULLAWP_ADJ_PROJ_LAG','FULLAWP_ADJ_PROJ_EOY','PRICE_REIMB',
                'LAG_REIMB','OLD_PRICE_EFFECTIVE_REIMB_PROJ_EOY','PRICE_EFFECTIVE_REIMB_PROJ','ANNUALIZED_AWP', 'ANNUALIZED_TARGET',
                'ANNUALIZED_DN_SPEND','ANNUALIZED_MODEL_SPEND','YTD_ACTUAL_SURPLUS','ANNUALILZED_DN_SURPLUS',
                'ANNUALILZED_MODEL_SURPLUS','GER_TARGET','GER_ACTUAL','GER_DO_NOTHING','GER_MODEL','BPS_ACTUAL',
                'BPS_DN','BPS_MODEL','YTD_RUN_RATE','RUN_RATE_DN','RUN_RATE_MODEL','ALGO_RUN_DATE','CLIENT_TYPE',
                'GO_LIVE','DATA_ID','RUN_TYPE','AT_RUN_ID']  
        
        # Mapping breakout label
        client_report_upload = breakout_label_mapping(client_report)

        if GER_OPT_TAXONOMY_FINAL.loc[0,'guarantee_category'] == 'Aetna MR Offsetting':
                client_report_upload['BREAKOUT'] = client_report['CUSTOMER_ID'] + "_MR"

        if p.WRITE_TO_BQ or p.UPLOAD_TO_DASH:
            uf.write_to_bq(
                client_report_upload[cols],
                project_output = p.BQ_OUTPUT_PROJECT_ID,
                dataset_output = p.BQ_OUTPUT_DATASET,
                table_id = "client_performance_report",
                client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                timestamp_param = p.TIMESTAMP,
                run_id = p.AT_RUN_ID,
                schema = None  # TODO: create schema
            )
        else:
            client_report_upload.to_csv(p.FILE_REPORT_PATH + p.DATA_ID +'_client_performance_report.csv', index=False)
                            
        ################################## Interceptor_report ##################################
        """
        This part reads in the LP run result and Interceptor data for reporting
        The report is generated irrespective of whether we run on INTERCEPTOR logic or not
        """
        if p.CLIENT_TYPE == 'COMMERCIAL' and p.GENERIC_OPT:     
            cs.interceptor_reporting()


    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'Reporting Tables', repr(e), error_loc)
        raise e

    if not check_run_status(run_status='Complete-BypassPerformance'):
        # Only if LP code has not with 'Complete-BypassPerformance' status during the LP run, the status should be updated to 'Complete'
        update_run_status('Complete')


# Upload to teradata
#TODO add uploading process of price_dist and price_change
def upload_to_teradata():
    """
    This function works only on prem for the purpose of uploading reporting tables to teradata

    :return: None
    """
    import pandas as pd
    import numpy as np
    import datetime as dt
    import CPMO_parameters as p
    import sql_config
    import td_reporting_sql as tsql
    from CPMO_shared_functions import check_run_status

    def create_table_if_exists(TABLE, teradata_conn):
        cursor = teradata_conn.cursor()
        query = tsql.query_check_table_exist.format(_table=TABLE)
        cursor.execute(query)
        print(query)
        row_count = cursor.fetchone()[0]
        # print(row_count)
        return row_count
        
    if check_run_status(run_status = 'Complete-BypassPerformance'): 
        if p.FULL_YEAR:
            RUN_TYPE_TABLEAU_UPDATED = "".join([p.RUN_TYPE_TABLEAU, "-BypassPerformance_WS"])
        else:
            RUN_TYPE_TABLEAU_UPDATED = "".join([p.RUN_TYPE_TABLEAU, "-BypassPerformance"])
        
    else:
        RUN_TYPE_TABLEAU_UPDATED = p.RUN_TYPE_TABLEAU

    awp_spend_perf = pd.read_csv(p.FILE_REPORT_PATH + 'joined_awp_spend_performance.csv')
    awp_spend_perf = awp_spend_perf.where(awp_spend_perf.notnull(), None)
    YTD_surplus_monthly = pd.read_csv(p.FILE_REPORT_PATH + 'YTD_SURPLUS_MONTHLY.csv')
    YTD_surplus_monthly = YTD_surplus_monthly.where(YTD_surplus_monthly.notnull(), None)
    conn = sql_config.get_teradata_connection()
    with conn:

        # Upload awp_spend_perf
        row_count = create_table_if_exists('GER_LP_OUT_AWP_SPEND_PERF', conn)
        if row_count > 0:
            print('Updating Rec_curr_ind')
            conn.execute(tsql.update_rec_curr_ind_query_run_type \
                         .format(_table_name='SB_Finance_G2_GER_OPT.GER_LP_OUT_AWP_SPEND_PERF',
                                 _customer_id=uf.get_formatted_client_name(p.CUSTOMER_ID),# add functionality to handle clients with multiple Ids 
                                 _tiered_price_lim=p.TIERED_PRICE_LIM,
                                 _go_live=p.GO_LIVE.strftime("%Y-%m-%d"),
                                 _run_type=RUN_TYPE_TABLEAU_UPDATED
                                 )
                         )
            print('Uploading to teradata')

            awp_spend_perf['REC_CURR_IND'] = 'Y'
            
            awp_spend_perf = awp_spend_perf[
                ['CUSTOMER_ID', 'CLIENT', 'ENTITY', 'ENTITY_TYPE', 'BREAKOUT', 'CHAIN_GROUP', 'CLIENT_OR_PHARM',
                 'FULLAWP_ADJ', 'FULLAWP_ADJ_PROJ_LAG', 'FULLAWP_ADJ_PROJ_EOY',
                 'PRICE_REIMB', 'LAG_REIMB', 'OLD_PRICE_EFFECTIVE_REIMB_PROJ_EOY', 'PRICE_EFFECTIVE_REIMB_PROJ',
                 'GEN_LAG_AWP', 'GEN_LAG_ING_COST', 'GEN_EOY_AWP', 'GEN_EOY_ING_COST',
                 'Pre_existing_Perf', 'Model_Perf', 'Pre_existing_Perf_Generic', 'Model_Perf_Generic', 'YTD_Perf_Generic',
                 'Proj_Spend_Do_Nothing', 'Proj_Spend_Model', 'Increase_in_Spend',
                 'Increase_in_Reimb', 'Total_Ann_AWP', 'GER_Do_Nothing', 'GER_Model', 'GER_Target',
                 'ALGO_RUN_DATE', #'REC_CURR_IND',
                 'CLIENT_TYPE', 'UNC_OPT',
                 'GOODRX_OPT', 'GO_LIVE', 'DATA_ID', 'TIERED_PRICE_LIM', 'RUN_TYPE','AT_RUN_ID','Run_rate_do_nothing','Run_rate_w_changes','IA_CODENAME',
                 'LEAKAGE_PRE','LEAKAGE_POST','LEAKAGE_AVOID']]


            data = [tuple(x) for x in awp_spend_perf.to_records(index=False)]
            # print (data)
            conn.executemany(
                "INSERT INTO SB_Finance_G2_GER_OPT.GER_LP_OUT_AWP_SPEND_PERF VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data, batch=True)
        else:
            print('Creating Table AWP Spend Performance')
            conn.execute(tsql.create_table_awp_spend_perf)
            data = [tuple(x) for x in awp_spend_perf.to_records(index=False)]
            print('Uploading to teradata')
            conn.executemany(
                "INSERT INTO SB_Finance_G2_GER_OPT.GER_LP_OUT_AWP_SPEND_PERF VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data, batch=True)

        # Upload YTD_surplus_monthly
        row_count = create_table_if_exists('GER_LP_OUT_YTD_SURPLUS_MONTHLY', conn)
        if row_count > 0:
            print('Updating Rec_curr_ind')
            conn.execute(tsql.update_rec_curr_ind_query_run_type \
                         .format(_table_name='SB_Finance_G2_GER_OPT.GER_LP_OUT_YTD_SURPLUS_MONTHLY',
                                 _customer_id=uf.get_formatted_client_name(p.CUSTOMER_ID),
                                 _tiered_price_lim=p.TIERED_PRICE_LIM,
                                 _go_live=p.GO_LIVE.strftime("%Y-%m-%d"),
                                 _run_type=RUN_TYPE_TABLEAU_UPDATED
                                 )
                         )
            print('Uploading to teradata')

            YTD_surplus_monthly['REC_CURR_IND'] = 'Y'
            YTD_surplus_monthly = YTD_surplus_monthly[['CUSTOMER_ID', 'CLIENT', 'BREAKOUT', 'MONTH', 'AWP', 'SPEND',
                                                       'SURPLUS', 'ALGO_RUN_DATE', 'CLIENT_TYPE', 'TIERED_PRICE_LIM',
                                                       'UNC_OPT', 'GOODRX_OPT', 'GO_LIVE', 'DATA_ID', 'REC_CURR_IND',
                                                       'RUN_TYPE']]

            data = [tuple(x) for x in YTD_surplus_monthly.to_records(index=False)]
            # print (data)
            conn.executemany(
                "INSERT INTO SB_Finance_G2_GER_OPT.GER_LP_OUT_YTD_SURPLUS_MONTHLY VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data, batch=True)
        else:
            print('Creating Table YTD_surplus_monthly')
            conn.execute(tsql.create_table_YTD_surplus_monthly)
            data = [tuple(x) for x in YTD_surplus_monthly.to_records(index=False)]
            print('Uploading to teradata')
            conn.executemany(
                "INSERT INTO SB_Finance_G2_GER_OPT.GER_LP_OUT_YTD_SURPLUS_MONTHLY VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                data, batch=True)


def collect_statistic():
    """
    This function is added as requested to update the table statistics and avoid Teradata Spool space Issue

    """
    import sql_config
    conn = sql_config.get_teradata_connection()
    with conn:
        # old version
        conn.execute("COLLECT STATISTICS ON SB_Finance_G2_GER_OPT.GER_LP_OUT_AWP_SPEND;")
        conn.execute("COLLECT STATISTICS ON SB_Finance_G2_GER_OPT.GER_LP_OUT_PERFORMANCE_SUM;")
        conn.execute("COLLECT STATISTICS ON SB_Finance_G2_GER_OPT.GER_LP_OUT_PRE_MOD_PERF;")
        conn.execute("COLLECT STATISTICS ON SB_Finance_G2_GER_OPT.GER_LP_OUT_PRICE_CHANGE;")
        conn.execute("COLLECT STATISTICS ON SB_Finance_G2_GER_OPT.GER_LP_OUT_PRICE_DIST;")
        conn.execute("COLLECT STATISTICS ON SB_Finance_G2_GER_OPT.GER_LP_OUT_YTD_SURPLUS;")
        # new version
        conn.execute("COLLECT STATISTICS ON SB_Finance_G2_GER_OPT.GER_LP_OUT_AWP_SPEND_PERF;")
        conn.execute("COLLECT STATISTICS ON SB_Finance_G2_GER_OPT.GER_LP_OUT_YTD_SURPLUS_MONTHLY;")  
        
if __name__ == '__main__':
    import os
    import sys
    import datetime as dt
    import util_funcs as uf
    import importlib
    import BQ

    args = arguments()
    # write parameters
    pfile = f'CPMO_parameters.py'
    ppath = os.path.join(os.path.dirname(__file__), pfile)
    with open(ppath, 'w') as pf:
        pf.write(args.params)
    p = importlib.import_module(pfile.replace('.py', ''), package='PBM_Opt_Code')
    from CPMO_shared_functions import check_and_create_folder

    check_and_create_folder(p.FILE_REPORT_PATH)
    # update_rec_curr_ind_BQ() # not needed if using run_id within a certain window calculation
    if p.PROGRAM_INPUT_PATH[:3] == 'gs:':
        # On GCP
        create_reporting_tables(p)
    else:
        # On prem
        create_reporting_tables(p)
        upload_to_teradata()
        collect_statistic()
