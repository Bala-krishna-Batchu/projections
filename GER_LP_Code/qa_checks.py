# -*- coding: utf-8 -*-
"""

"""

import CPMO_parameters as p
import pandas as pd
import logging 
from datetime import date, datetime
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# =============================================================================
#     Monthly AWP within difference threshold",
#    "Monthly ingredient cost within difference threshold",
#    "Claims appear on every date",
#    "No duplicate claim numbers",
#    "IGNORE - not a check",
#    "All AWP values non-negative and not NA",
#    "All NORMING values non-negative and not NA",
#    "All NORMING values less than or equal to UCAMT",
#    "All QTY values non-negative and not NA",
#    "All GPI values are 14 characters",
#    "All NDC values are 11 characters" 
# =============================================================================

#MARCEL: some files have BQ support and some don't. We need to be clear about
#which file will always be csv since the read functions all use thesame 
#if READ_FROM_BQ flag. 

"""
1. generate_GPI_report():
    -reading and processing daily total is also done in Daily_Input_Read.py (same read function?)
    -MAC_PRICE_OVERRIDE_FILE currently has no support for BQ, only reading from file
    -SPECIALTY_EXCLUSION_FILE currently has no BQ support
    -Should we write to csv instead of excel (change csv)
    
2. generate_measurement_mapping_report():
    -MEASUREMENT_MAPPING currently has no BQ support, only read from file
    -MAC_MAPPING_FILE: processing this file is also done in Daily_Input_Read.py
    -MAC_CONSTRAINT_FILE: no big query support (if we want BQ support, we need to create the BQ datasets)
    -CLIENT_GUARANTEE_FILE: no BQ support
    -PHARM_GUARANTEE_FILE: no BQ support
    -PHARMACY_YTD: no BQ support
    -GENERIC_LAUNCH_FILE : no BQ support
    -PHARMACY_APPROXIMATIONS: no BQ support
    -should we write to csv instead of excel?
"""


def qa_dataframe(df_,duplist=None,nanlist=None,dataset=None):
    df=df_.copy(deep=True)
    passed = True
    print('\n************************************************** \nQA CHECK for {}'.format(dataset))
    df.columns = map(str.upper, df.columns)
    qa_report_file_path = p.FILE_LOG_PATH + '_QA_dataset_{}.csv'.format(dataset.replace('.csv',''))
    qa_report_file=pd.DataFrame()
    
    if duplist == None:
        duplist = df.columns
    if nanlist == None:
        nanlist = df.columns
    
    # empty file check
    n = len(df)
    if n == 0:
        print('\n ***** ALERT: {} is empty'.format(dataset))
        passed = False        
    
    #duplicated rows check
    dups = df.duplicated(subset = duplist, keep=False)        
    if dups.sum() > 0:
        print('\n ***** ALERT: {} duplicate rows in {}'.format(dups,dataset))
        qa_report_file = append_qa_file('Duplicate rows', df[dups],qa_report_file )
        passed = False

    nans = df[nanlist].isna().any(axis=1)
    if nans.any() == True:
        print('\n ***** ALERT: Missing data in {}'.format(dataset))
        qa_report_file = append_qa_file('Missing data', df[nans],qa_report_file )
        passed = False
        
    if 'GPI' in df.columns:
        fltrd_df = df[df.GPI.astype(str).str.len() != 14]
        qa_report_file = append_qa_file('GPI Length not equal to 14', fltrd_df,qa_report_file )   
        if len(fltrd_df) > 0:
            print('\n ALERT:Some GPIs Length not equal to 14')
            passed = False
        
        #Check to see if GPI consists of only numbers and digits
        fltrd_df = df[~df.GPI.astype(str).str.isalnum()]
        qa_report_file = append_qa_file('GPI consists of characters other than numbers and digit', fltrd_df,qa_report_file )
        if len(fltrd_df) > 0:
            print('\n ALERT:Some GPIs consists of characters other than numbers and digits')
            passed = False
        
        #check to see if last four digits are not all zeroes
        if p.READ_FROM_BQ == False:
            fltrd_df = df[df.GPI.astype(str).str[-4:]=='0000']
            qa_report_file = append_qa_file('GPI last for 4 digits are all zero', fltrd_df,qa_report_file )
            if len(fltrd_df) > 0:
                print('\n ALERT:Some GPIs with last for 4 digits that are all zeroes. Double check that this is not true for every GPI; you may have overwritten them when opening a CSV. Otherwise some GPIs with this issue is OK.')
                passed = False

        #check to see if GPI in scientific notation
        fltrd_df = df[df.GPI.str.contains('.[a-zA-Z0-9]*E\+')]
        qa_report_file = append_qa_file('GPI in scientific notation',fltrd_df,qa_report_file)
        if len(fltrd_df) > 0:
            print('\n ALERT:Some GPIs in scientific notation')
            passed = False
            

    if 'NDC' in df.columns:
        #length of NDC should be 11
        fltrd_df = df[df.NDC.astype(str).str.len() != 11]
        qa_report_file = append_qa_file('NDC Length not equal to 11', fltrd_df,qa_report_file )
        if len(fltrd_df) > 0:
            print('\n ALERT:NDC Lengths not equal to 11')
            passed = False

        #The NDCs are all a series of digits or a string of 11 “*”s
        fltrd_df = df[ ~(df.NDC.astype(str).str.isdigit()) & ~(df.NDC.astype(str).str.match(pat = '\*{11}')) ]
        qa_report_file = append_qa_file('NDC not a series of digits or a string of 11 “*”s', fltrd_df,qa_report_file )
        if len(fltrd_df) > 0:
            print('\n ALERT:NDCs not a series of digits or a string of 11 “*”s')
            passed = False

    
#TODO: The following three checks should be generalized to a bunch of quantitative fields and prices:       
#TODO: updating field types .astype() should be moved to CPMO_shared_functions.standardize_df()
    if 'AWP' in df.columns:
        if df.AWP.dtype != float:
            df.AWP = df.AWP.astype('float')
        fltrd_df = df[(df.AWP.isna()) | (df.AWP < 0)]
        qa_report_file = append_qa_file('Negative or missing-data in AWP value', fltrd_df,qa_report_file ) 
        if len(fltrd_df) > 0:
            print('\n ALERT:Negative or missing-data in AWP values')
            passed = False
    
    if 'QTY' in df.columns:
        if df.QTY.dtype != np.int64:
                df.QTY = df.QTY.astype('int64')
        fltrd_df = df[(df.QTY.isna()) | (df.QTY < 0) ]
        qa_report_file = append_qa_file('Negative or missing-data in QTY value', fltrd_df,qa_report_file ) 
        if len(fltrd_df) > 0:
            print('\n ALERT:Negative or missing-data in QTY values')
            passed = False
    
    if 'SPEND' in df.columns:
        if df.SPEND.dtype != float:
                df.SPEND = df.SPEND.astype('float')
        fltrd_df = df[(df.SPEND.isna()) | (df.SPEND < 0) ]
        qa_report_file = append_qa_file('Negative or missing-data in SPEND value', fltrd_df,qa_report_file )   
        if len(fltrd_df) > 0:
            print('\n ALERT:Negative or missing-data in SPEND value')
            passed = False
  

    if not passed:
        print('\n ALERT {} failed QA checks. Check the logfile:\n {}.'.format(dataset,qa_report_file_path))
        qa_report_file.to_csv(qa_report_file_path)
        assert passed, 'ALERT {} failed QA checks. Check the logfile:\n {}.'.format(dataset,qa_report_file_path)
    elif passed:
        print('\nPassed')
    
    print('**************************************')
    
    return passed

def append_qa_file(measure, fltrd_df,  qa_report_file):
    if not fltrd_df.empty:
            fltrd_df=fltrd_df.assign(QA_FAILED_MEASURE=measure)
            qa_report_file = qa_report_file.append(fltrd_df)
    return qa_report_file

'''
#TODO: move to database checks:
    if 'CLAIM_ID'in df.columns:
        df.drop_duplicates(subset=['CLAIM_ID'], inplace=True)
    
   
    # this is not going to work:
    # start_dt = df.CLAIM_DATE.min()
    # end_dt = df.CLAIM_DATE.max()
    # qa_report_file = check_claim_date(start_dt,end_dt,df,qa_report_file)
        
'''


def check_claim_date(start_dt,end_dt,df,qa_report_file):
    dt_range = pd.date_range(start_dt, end_dt)
    df['CLAIM_DATE']=df['CLAIM_DATE'].astype('Datetime64')
    fltrd_df = df[~df['CLAIM_DATE'].isin(dt_range)]
    if not fltrd_df.empty:
            fltrd_df=fltrd_df.assign(QA_FAILED_MEASURE=measure)
            qa_report_file = qa_report_file.append(fltrd_df)
    return qa_report_file    


def check_contractual_rates():

    '''
    Check the client follows its contractual price restrictions and has TIERED_PRICE_LIM parameter set to FALSE
    
    '''
    import CPMO_parameters as p
    
    if any(ext in p.CLIENT_NAME_TABLEAU.upper() for ext in ['AON', 'WTW','MVP','HEALTHFIRST']) or any(cid == p.CUSTOMER_ID for cid in ['183C','185C']):
        if p.FULL_YEAR == False: 
            assert p.TIERED_PRICE_LIM == False and p.REMOVE_WTW_RESTRICTION == False, "TIERED_PRICE_LIM should be false for the client."           
        elif p.REMOVE_WTW_RESTRICTION == False:
            if any(ext in p.CLIENT_NAME_TABLEAU.upper() for ext in ['AON', 'WTW']):
                assert p.TIERED_PRICE_LIM == False and p.GPI_UP_FAC < 0.25, "GPI_UP_FAC should be <0.25 for AON and WTW client "
            if any(ext in p.CLIENT_NAME_TABLEAU.upper() for ext in ['MVP','HEALTHFIRST']) or any(cid == p.CUSTOMER_ID for cid in ['183C','185C']):
                assert p.TIERED_PRICE_LIM == False and p.GPI_UP_FAC <= 0.10, "GPI_UP_FAC should be <= 0.10 for MVP/HEALTHFIRST/STRS OHIO client"
        
def generate_GPI_report():
    '''
    This report merges together all data sets that include GPI as a merge-key. 
    
    '''
    import pandas as pd
    import numpy as np
    import CPMO_parameters as p
    from CPMO_shared_functions import standardize_df,read_tru_mac_list_prices
    import util_funcs as uf
    import BQ
    print("Generating GPI report")
    
    # check daily totals MARCEL: this read function is also in Daily_Input_Read.py
    if p.READ_FROM_BQ:
        gpi_vol_awp_df = uf.read_BQ_data(
            BQ.daily_totals_pharm,
            project_id = p.BQ_INPUT_PROJECT_ID,
            dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP,
            table_id = 'combined_daily_totals' + p.WS_SUFFIX + p.CCP_SUFFIX,            
            client = ', '.join(sorted(p.CUSTOMER_ID)),
            claim_start = p.DATA_START_DAY,
            claim_date = p.LAST_DATA.strftime('%m/%d/%Y')
        )
    else:
        gpi_vol_awp_df = pd.read_csv(p.FILE_INPUT_PATH + p.DAILY_TOTALS_FILE)
    gpi_vol_awp_df = standardize_df(gpi_vol_awp_df) 
    daily_totals = gpi_vol_awp_df.copy(deep=True)
    del gpi_vol_awp_df
        
    #daily_totals = standardize_df(pd.read_csv(p.FILE_INPUT_PATH + p.DAILY_TOTALS_FILE)) 
    unique_gpi_df = daily_totals.groupby(['CLIENT','GPI','BG_FLAG']).agg({ 'CLAIMS': 'sum' , 'AWP':'sum'}).reset_index()\
                .rename(columns={'AWP':'Total claims AWP','CLAIMS':'Number of CLAIMS'})
    del daily_totals


    # check mac1026 MARCEL: this read function is also in Daily_Input_Read.py
    if p.READ_FROM_BQ:
        mac1026_df = uf.read_BQ_data(
            BQ.mac_1026, 
            project_id = p.BQ_INPUT_PROJECT_ID, 
            dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP, 
            table_id = 'mac_1026' + p.WS_SUFFIX)
    else:
        mac1026_df = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.MAC1026_FILE)
        mac1026_df = mac1026_df.rename(
            index=str,
            columns = {'mac_cost_amt': 'MAC1026_UNIT_PRICE',
                       'gpi':'GPI',
                       'ndc':'NDC'})
    mac1026_df = standardize_df(mac1026_df) 
    
    #mac1026_df = standardize_df(pd.read_csv(p.FILE_INPUT_PATH + p.MAC1026_FILE))
    mac_1026_gpi = mac1026_df.loc[mac1026_df.NDC == '***********']\
        .rename(columns = {'PRICE': 'MAC1026 GPI PRICE'})\
        .drop(columns=['GPI_NDC','MAC_LIST','NDC'])
    mac_1026_ndc = mac1026_df.loc[mac1026_df.NDC != '***********']
    mac1026_ndcagg = mac_1026_ndc.groupby(['GPI','BG_FLAG'])\
        .agg({'PRICE':'mean'}).reset_index()\
        .rename(columns={'PRICE':'MAC1026 AVG NDC PRICE'})
    unique_gpi_df = unique_gpi_df.merge(mac_1026_gpi,on=['GPI','BG_FLAG'],how='left')\
        .merge(mac1026_ndcagg,on=['GPI','BG_FLAG'],how='left')
    del mac1026_df,mac_1026_gpi, mac_1026_ndc, mac1026_ndcagg

    # check mac list
    if p.READ_FROM_BQ:
        ##the distinction between MEDD and COMMERCIAL might be temporary:
        ##if DE updates <mac_list> table in <ds_pro_lp> data set according to the new <query_client_mac_list> in <sql_queries.py>,
        ##the <BQ.mac_list> query would work for both COMMERCIAL and MEDD clients
        if (p.CLIENT_TYPE == "COMMERCIAL" or p.CLIENT_TYPE == "MEDICAID"):
            if p.TRUECOST_CLIENT or p.UCL_CLIENT:
                """Read TRUECOST mac list from only Big Query"""
                print('Reading in TRUECOST mac list.........')
                assert p.READ_FROM_BQ == True, "Use p.READ_FROM_BQ=True to read table"
                mac_list_df = read_tru_mac_list_prices()
                mac_list_df = mac_list_df[mac_list_df['PRICE'] > 0]
            else:
                mac_list_df = uf.read_BQ_data(
                    BQ.mac_list,
                    project_id = p.BQ_INPUT_PROJECT_ID,
                    dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP,
                    table_id = "mac_list" + p.WS_SUFFIX,
                    mac = True,
                    vcml_ref_table_id='vcml_reference',
                    customer = ', '.join(sorted(p.CUSTOMER_ID))
                )
        elif p.CLIENT_TYPE == "MEDD":
            #This was added to update vcml_ref table as per DE, later changes can be moved directly to CPMO Params
            #Updated _landing_dataset and _table_id_vcml to accomodate new table
            mac_list_df = uf.read_BQ_data(
                BQ.mac_list_for_medd_custom.format(_customer_id = uf.get_formatted_string(p.CUSTOMER_ID),
                                                   _project = p.BQ_INPUT_PROJECT_ID,
                                                   _landing_dataset= p.BQ_INPUT_DATASET_ENT_CNFV_PROD,
                                                   _landing_dataset_vcml= p.BQ_INPUT_DATASET_ENT_ENRV_PROD,
                                                   _table_id_base = 'gms_ger_opt_base_mac_lists',
                                                   _table_id_vcml = 'v_cmk_vcml_reference'),
                project_id = p.BQ_INPUT_PROJECT_ID,
                dataset_id = p.BQ_INPUT_DATASET_ENT_CNFV_PROD,
                table_id = 'gms_ger_opt_base_mac_lists',
                custom = True
            )
        else:
            assert False, "CLIENT_TYPE is not of type COMMERCIAL or MEDD or MEDICAID."
    else:
        mac_list_df = pd.read_csv(p.FILE_INPUT_PATH + p.MAC_LIST_FILE, dtype = p.VARIABLE_TYPE_DIC)
    mac_list_df = standardize_df(mac_list_df)
    mac_list = mac_list_df.copy()
    del mac_list_df
    
    #mac_list = standardize_df(pd.read_csv(p.FILE_INPUT_PATH + p.MAC_LIST_FILE))
    mac_list['CLIENT']=uf.get_formatted_client_name(p.CUSTOMER_ID)
    #mac_list['CLIENT'] = mac_list['MAC'].str[3:7]
    mac_list_gpi = mac_list.loc[mac_list.NDC == '***********']
    mac_list_gpi_agg = mac_list_gpi.groupby(['CLIENT','GPI','BG_FLAG'])\
        .agg({'MAC':'nunique','PRICE':'mean'}).reset_index()\
        .rename(columns={
            'MAC':'Unique VCMLs with Mac List Prices',
            'PRICE':'MAC LIST AVG GPI PRICE'
            })

    mac_list_ndc = mac_list.loc[mac_list.NDC != '***********']            
    mac_list_ndc_agg = mac_list_ndc.groupby(['CLIENT','GPI','BG_FLAG'])\
        .agg({'PRICE':'mean'}).reset_index()\
        .rename(columns={'PRICE':'MAC LIST AVG NDC PRICE'})
    unique_gpi_df = unique_gpi_df.merge(mac_list_gpi_agg,on=['CLIENT','GPI','BG_FLAG'],how='left')\
        .merge(mac_list_ndc_agg,on=['CLIENT','GPI','BG_FLAG'],how='left')
    del mac_list, mac_list_gpi, mac_list_gpi_agg, mac_list_ndc, mac_list_ndc_agg
    
    # check AWP history file
    #see Daily_Input_Read.py 
    if p.READ_FROM_BQ:
        
        curr_awp_df = uf.read_BQ_data(
            BQ.awp_history_table, 
            project_id = p.BQ_INPUT_PROJECT_ID, 
            dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP,
            table_id = 'awp_history_table' + p.WS_SUFFIX)
        
    else:
        curr_awp_df = pd.read_csv(p.FILE_INPUT_PATH + p.AWP_FILE)
    curr_awp_df = standardize_df(curr_awp_df)

    #print("MARCEL: ", curr_awp_df.dtypes) #all types are object, even DRUG_PRICE_AT which should be float
    #curr_awp_df = standardize_df(pd.read_csv(p.FILE_INPUT_PATH + p.AWP_FILE))
    curr_awp_df = curr_awp_df.sort_values(
        by=['GPI_NDC','SRC_CD', 'EFF_DATE'], ascending=[True,True,False])
    curr_awp_df = curr_awp_df.drop_duplicates(subset=['GPI_NDC'], keep='first')
    curr_app_agg = curr_awp_df.groupby(['GPI','BG_FLAG'])\
        .agg({'DRUG_PRICE_AT':'mean','EFF_DATE':'max'}).reset_index()\
        .rename(columns={
            'DRUG_PRICE_AT': 'AVG current AWP price',
            'EFF_DATE':'Maximum AWP price eff date'})
    unique_gpi_df = unique_gpi_df.merge(curr_app_agg,on=['GPI','BG_FLAG'],how='left')
    del curr_awp_df, curr_app_agg
    
    # check mac_price_overrides
    # only at the GPI x VCML_ID grain
    #TODO: this will break if VCML_ID = 'ALL'
    #MARCEL: MAC_PRICE_OVERRIDE_FILE currently has no BQ support
    mac_price_overrides = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.MAC_PRICE_OVERRIDE_FILE))
    if p.CLIENT_LOB != 'AETNA' and not p.TRUECOST_CLIENT:
        mac_price_overrides_agg = mac_price_overrides.groupby(['CLIENT','GPI','BG_FLAG'])\
            .agg({'VCML_ID':'nunique','PRICE_OVRD_AMT':'mean'})\
            .reset_index().rename(columns={
                'VCML_ID':'UNIQUE price override VCML IDs',
                'PRICE_OVRD_AMT':'Price override average price'})
        unique_gpi_df = unique_gpi_df.merge(mac_price_overrides_agg,on=['CLIENT','GPI','BG_FLAG'],how='left')
        del mac_price_overrides, mac_price_overrides_agg
    
    # check specialty_exclusions
    #MARCEL: SPECIALTY_EXCLUSION_FILE currently has no BQ support
    specialty_exclusions = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.SPECIALTY_EXCLUSION_FILE))\
                .rename(columns={'GPI_CD':'GPI','DRUG_ID':'NDC'})
    specialty_exclusions['SPECIALTY_EXCLUDED'] = True
    specialty_exclusions_all = specialty_exclusions[specialty_exclusions['CLIENT'] == 'ALL'][['GPI']]
    specialty_exclusions_all['Specialty excluded All'] = True
    specialty_exclusions_clients = specialty_exclusions[specialty_exclusions['CLIENT'] != 'ALL'][['CLIENT','GPI']]
    specialty_exclusions_clients['Specialty excluded Client'] = True
    unique_gpi_df = unique_gpi_df.merge(specialty_exclusions_all,on=['GPI'],how='left')
    if len(specialty_exclusions_clients) > 0:
        unique_gpi_df = unique_gpi_df.merge(specialty_exclusions_clients,on=['CLIENT','GPI'],how='left')
    del specialty_exclusions, specialty_exclusions_all ,specialty_exclusions_clients 
     
    # create the report:
    # get the MAC1026 missing GPIs and high claim count to top
    unique_gpi_df.sort_values(['MAC1026 GPI PRICE','MAC LIST AVG GPI PRICE','AVG current AWP price','Number of CLAIMS'],
                              inplace=True,na_position='first',ascending=False)
    
    nas = unique_gpi_df[['MAC1026 GPI PRICE','MAC LIST AVG GPI PRICE','AVG current AWP price']].isna().sum()
    if nas.any() > 0:
        print('\nALERT: Data Failed GPI missing data checks. Check GPI_COVERAGE_REPORT_{}.csv in the \Logs directory.'.format(p.DATA_ID))
        print('\nNo Mac1026, MAC LIst, or Average AWP prices should have missing prices.' )
    elif nas.all() == 0:
        print('\nPassed GPI check')
    
    #writer = pd.ExcelWriter(p.FILE_LOG_PATH + 'GPI_COVERAGE_REPORT_{}.xlsx'.format(p.DATA_ID), engine='xlsxwriter')
    #unique_gpi_df.to_excel(writer, sheet_name='GPI list', index=False)
    #writer.save()
    unique_gpi_df.to_csv(p.FILE_LOG_PATH + 'GPI_COVERAGE_REPORT_{}.csv'.format(p.DATA_ID),  index=False)
    
def generate_measurement_mapping_report():
    '''
    This report merges all the data that includes the measurement mapping merge-keys into one report.
    With this report data scientists can identify areas where key data are missing and where joins between
    data sets may fail.  

    '''
    import pandas as pd
    import numpy as np
    import CPMO_parameters as p
    from CPMO_shared_functions import standardize_df, add_virtual_r90
    import util_funcs as uf
    import BQ
    print("\n\nGenerating Measurement mapping report")

    # read-in measurement mapping
    if p.READ_FROM_BQ:
        measurement_mapping = uf.read_BQ_data(
                BQ.ger_opt_msrmnt_map.format(_customer_id=uf.get_formatted_string(p.CUSTOMER_ID)),
                project_id=p.BQ_INPUT_PROJECT_ID,
                dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                table_id="combined_measurement_mapping" + p.WS_SUFFIX + p.CCP_SUFFIX,
                customer=', '.join(sorted(p.CUSTOMER_ID)))
    else:
        measurement_mapping = pd.read_csv(p.FILE_INPUT_PATH + p.MEASUREMENT_MAPPING)
    measurement_mapping = standardize_df(measurement_mapping)
    mm_report = measurement_mapping.groupby(['CLIENT','BREAKOUT','REGION','MEASUREMENT','CHAIN_GROUP','CHAIN_SUBGROUP','BG_FLAG','PREFERRED'])\
                                   .agg({'FULLAWP': "sum", "CLAIMS": "sum"})\
                                   .reset_index()
    # create pharmacy lists
    big = pd.DataFrame({'CHAIN_GROUP':p.BIG_CAPPED_PHARMACY_LIST['BRND']+p.BIG_CAPPED_PHARMACY_LIST['GNRC'],
                        'Pharmacy type':['BRND_BIG_CAPPED_PHARMACY'] * len(p.BIG_CAPPED_PHARMACY_LIST['BRND']) + 
                                        ['GNRC_BIG_CAPPED_PHARMACY'] * len(p.BIG_CAPPED_PHARMACY_LIST['GNRC']),
                        'BG_FLAG':['B'] * len(p.BIG_CAPPED_PHARMACY_LIST['BRND']) + ['G'] * len(p.BIG_CAPPED_PHARMACY_LIST['GNRC'])
                       })
    small = pd.DataFrame({'CHAIN_GROUP':p.SMALL_CAPPED_PHARMACY_LIST['BRND']+p.SMALL_CAPPED_PHARMACY_LIST['GNRC'],
                        'Pharmacy type':['BRND_SMALL_CAPPED_PHARMACY'] * len(p.SMALL_CAPPED_PHARMACY_LIST['BRND']) + 
                                        ['GNRC_SMALL_CAPPED_PHARMACY'] * len(p.SMALL_CAPPED_PHARMACY_LIST['GNRC']),
                        'BG_FLAG':['B'] * len(p.SMALL_CAPPED_PHARMACY_LIST['BRND']) + ['G'] * len(p.SMALL_CAPPED_PHARMACY_LIST['GNRC'])
                       })
    non = pd.DataFrame({'CHAIN_GROUP':p.NON_CAPPED_PHARMACY_LIST['BRND']+p.NON_CAPPED_PHARMACY_LIST['GNRC'],
                        'Pharmacy type':['BRND_NON_CAPPED_PHARMACY'] * len(p.NON_CAPPED_PHARMACY_LIST['BRND']) + 
                                        ['GNRC_NON_CAPPED_PHARMACY'] * len(p.NON_CAPPED_PHARMACY_LIST['GNRC']),
                        'BG_FLAG':['B'] * len(p.NON_CAPPED_PHARMACY_LIST['BRND']) + ['G'] * len(p.NON_CAPPED_PHARMACY_LIST['GNRC'])
                       })
    cogs = pd.DataFrame({'CHAIN_GROUP':p.COGS_PHARMACY_LIST['BRND']+p.COGS_PHARMACY_LIST['GNRC'],
                        'Pharmacy type':['BRND_COGS_PHARMACY'] * len(p.COGS_PHARMACY_LIST['BRND']) + 
                                        ['GNRC_COGS_PHARMACY'] * len(p.COGS_PHARMACY_LIST['GNRC']),
                        'BG_FLAG':['B'] * len(p.COGS_PHARMACY_LIST['BRND']) + ['G'] * len(p.COGS_PHARMACY_LIST['GNRC'])
                       })
    mail = pd.DataFrame({'CHAIN_GROUP':['MAIL']*2,'BG_FLAG':['B','G'],'Pharmacy type':['MAIL']*2},index=[0,1])
    pharms = pd.concat([big,small,non,cogs,mail])
    mm_report = mm_report.merge(pharms,on = ['CHAIN_GROUP','BG_FLAG'], how = 'left')
    # read-in mac mapping
    # MAC MAPPING MARCEL: MAC_MAPPING_FILE is not written to BQ 
    if False: # p.READ_FROM_BQ: or p.WRITE_TO_BQ:? CAN WE DELETE?
        chain_region_mac_mapping =  uf.read_BQ_data(
            BQ.Mac_Mapping, 
            project_id = p.BQ_OUTPUT_PROJECT_ID,
            dataset_id = p.BQ_OUTPUT_DATASET,
            table_id = 'Mac_Mapping' + p.WS_SUFFIX,
            client = ', '.join(sorted(p.CUSTOMER_ID)),
            period = p.TIMESTAMP,
            output = True)
    else:
        chain_region_mac_mapping = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.MAC_MAPPING_FILE)
        
    chain_region_mac_mapping = standardize_df(chain_region_mac_mapping)
    mac_mapping = chain_region_mac_mapping.copy()
    del chain_region_mac_mapping

    mm_report = mm_report.merge(mac_mapping,
                                left_on = ['CLIENT', 'REGION', 'MEASUREMENT', 'CHAIN_SUBGROUP'],
                                right_on = ['CUSTOMER_ID', 'REGION', 'MEASUREMENT', 'CHAIN_SUBGROUP'],
                                how = 'outer')\
                         .drop(columns = ['CUSTOMER_ID'])\
                         .rename(columns = {'MAC_LIST': 'MAC MAPPING LIST'})
    # read-in mac constraints
    mac_constraints = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.MAC_CONSTRAINT_FILE))
    mac_constraints_melt = pd.melt(mac_constraints,
                                   id_vars = ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT'],
                                   var_name = 'CHAIN_SUBGROUP',
                                   value_name = 'Mac Constraint Set')
    mm_report = mm_report.merge(mac_constraints_melt,
                                left_on = ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'CHAIN_SUBGROUP'],
                                right_on = ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'CHAIN_SUBGROUP'],
                                how = 'left')
    # read-in client guarantees
    client_guarantees = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.CLIENT_GUARANTEE_FILE, dtype = p.VARIABLE_TYPE_DIC)
    client_guarantees = standardize_df(client_guarantees)

    # read-in client guarantees
    if p.CLIENT_TYPE == 'MEDD' and p.FULL_YEAR and not p.EGWP:
        client_guarantees = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.CLIENT_GUARANTEE_FILE, dtype = p.VARIABLE_TYPE_DIC))
        
    # NOTE: NONPREF, Non_Preferred, Non-Preferred, PREF, Preferred should be standardized
    # TEMPORARY work around: standardize NONPREF, Non_Preferred, Non-Preferred --> Non_Preferred
    #                        standardize PREF, Preferred --> Preferred
    mm_report.loc[mm_report.PREFERRED.str.lower().str.startswith('pref', na = False), 'PREFERRED'] = 'Preferred'
    mm_report.loc[mm_report.PREFERRED.str.lower().str.startswith('non', na = False), 'PREFERRED'] = 'Non_Preferred'
    client_guarantees.loc[client_guarantees.PHARMACY_TYPE.str.lower().str.startswith('pref', na = False), 'PHARMACY_TYPE'] = 'Preferred'
    client_guarantees.loc[client_guarantees.PHARMACY_TYPE.str.lower().str.startswith('non', na = False), 'PHARMACY_TYPE'] = 'Non_Preferred'
    mm_report = mm_report.merge(client_guarantees,
                                left_on = ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT','BG_FLAG','PREFERRED'],
                                right_on = ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT','BG_FLAG','PHARMACY_TYPE'],
                                how = 'left')\
                         .rename(columns = {'RATE': 'Client Guarantee Rate'})

    # read-in pharmacy guarantees
    pharmacy_guarantees = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.PHARM_GUARANTEE_FILE))
    # find common dimensions between pharmacy_guarantees and measurement_mapping because
    # for now, pharmacy_guarantees for medd clients have more dimensions (e.g., measurement) than commercial clients
    common_merge_col = list(set(['BREAKOUT', 'REGION', 'MEASUREMENT','BG_FLAG']).intersection(list(pharmacy_guarantees.columns)))

    mm_report = mm_report.merge(pharmacy_guarantees,
                                left_on = ['CLIENT', 'CHAIN_GROUP', 'CHAIN_SUBGROUP'] + common_merge_col,
                                right_on = ['CLIENT', 'PHARMACY', 'PHARMACY_SUB'] + common_merge_col,
                                how = 'left')\
                         .drop(columns = ['PHARMACY'])\
                         .rename(columns = {'RATE': 'Pharmacy Guarantee Rate'})

    if p.READ_FROM_BQ:
        pharmacy_claims = uf.read_BQ_data(
            BQ.daily_totals_pharm,
            project_id = p.BQ_INPUT_PROJECT_ID,
            dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP,
            table_id = 'combined_daily_totals' + p.WS_SUFFIX + p.CCP_SUFFIX,
            client = ', '.join(sorted(p.CUSTOMER_ID)),
            claim_start = p.DATA_START_DAY,
            claim_date = p.LAST_DATA.strftime('%m/%d/%Y')
        )
    else:
        pharmacy_claims = pd.read_csv(p.FILE_INPUT_PATH + p.DAILY_TOTALS_FILE, dtype = p.VARIABLE_TYPE_DIC)

    pharmacy_claims.loc[:, 'BREAKOUT'] = pharmacy_claims["BREAKOUT"].str.upper()
    
    pharmacy_claims_agg = pharmacy_claims.groupby(['client', 'Region', 'BREAKOUT', 'MEASUREMENT', 'CHAIN_GROUP','BG_FLAG'])\
                                         .agg({'PHARMACY_AWP':"sum",'PHARMACY_NADAC':"sum",'PHARMACY_ACC':"sum", 'PHARMACY_TARGET_IC_COST':"sum",'PHARMACY_CLAIMS':"sum", 'AWP':"sum", 'CLAIMS':"sum"})\
                                         .reset_index()\
                                         .rename(columns = {'AWP':'FULLAWP'})

    mm_report = mm_report.drop(['FULLAWP', 'CLAIMS'], axis = 1).merge(pharmacy_claims_agg,
                                left_on = ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'CHAIN_GROUP','BG_FLAG'],
                                right_on = ['client', 'BREAKOUT', 'Region', 'MEASUREMENT', 'CHAIN_GROUP','BG_FLAG'],
                                how = 'left')

    # read-in generic launches
    gen_launch = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.GENERIC_LAUNCH_FILE))
    gen_launch_agg = gen_launch.groupby(['CLIENT', 'REGION', 'BREAKOUT', 'MEASUREMENT', 'CHAIN_GROUP'])['FULLAWP'].sum()\
                               .reset_index()\
                               .rename(columns = {'FULLAWP': 'Generic launch AWP'})
    mm_report = mm_report.merge(gen_launch_agg,
                                left_on = ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'CHAIN_GROUP'],
                                right_on = ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'CHAIN_GROUP'],
                                how = 'left')
    
    incl_groups = p.AGREEMENT_PHARMACY_LIST
    report_errors = []   
     
    if p.NO_MAIL:
        mm_report = mm_report[mm_report['MEASUREMENT'] != "M30"]    
    # find missing information for any rows with non-zero claims only in inclusion groups
    nas = mm_report.loc[(mm_report['CLAIMS'] > 0) &
                        (mm_report['CHAIN_GROUP'].isin(incl_groups))].isna().sum()
    if nas.any() > 0:
        report_errors.append('Measurement Mapping report includes missing values. Check MEASUREMENT_MAPPING_REPORT in the \Logs directory for more details.'.format(p.DATA_ID))

    # find pharmacy missing information for any rows which have non-zero claims only in inclusion groups
    pharm_nas = (mm_report['CLAIMS'] > 0) & \
                (mm_report['CHAIN_GROUP'].isin(incl_groups)) & \
                pd.isnull(mm_report['Pharmacy Guarantee Rate']) & \
                pd.isnull(mm_report['PHARMACY_AWP']) & \
                pd.isnull(mm_report['PHARMACY_NADAC']) & \
                pd.isnull(mm_report['PHARMACY_ACC']) & \
                pd.isnull(mm_report['PHARMACY_TARGET_IC_COST']) & \
                pd.isnull(mm_report['PHARMACY_CLAIMS'])
    if pharm_nas.sum() > 0:
        no_pharm_info_groups = ', '.join(mm_report[pharm_nas]['CHAIN_GROUP'].values)
        report_errors.append('{} chain groups are missing some pharmacy information'.format(pharm_nas))

    # check for duplicates, usually from multiple client guarantees
    dup_mm_report = mm_report[pd.notna(mm_report['MEASUREMENT']) & pd.notna(mm_report['FULLAWP'])]
    if 'CHAIN_SUBGROUP' not in dup_mm_report.columns:
        dup_mm_report['CHAIN_SUBGROUP'] = dup_mm_report['CHAIN_GROUP']
    if len(dup_mm_report.groupby(['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'CHAIN_SUBGROUP','BG_FLAG'])) != len(dup_mm_report):
        report_errors.append('Duplicate chain groups found')

    # only allow empty pharmacy claims if client claims is very low
    no_claims = (mm_report['CLAIMS'] > 50) & \
                pd.isnull(mm_report['PHARMACY_CLAIMS']) & \
                (mm_report['CHAIN_GROUP'].isin(incl_groups))
    if sum(no_claims) > 0:
        no_claim_groups = ', '.join(mm_report[no_claims]['CHAIN_GROUP'].values)
        report_errors.append('{} chain groups do not have pharmacy claims'.format(no_claim_groups))
        
    mm_report = mm_report.rename(columns={'CLAIMS':'CLIENT_CLAIMS', 'FULLAWP':'CLIENT_AWP'})

    mm_report.to_csv(p.FILE_LOG_PATH + 'MEASUREMENT_MAPPING_REPORT_{}.csv'.format(p.DATA_ID), index = False)
    if report_errors:
        raise ValueError('\n\n'.join(report_errors))

def check_price_bound_parameters():
    # Test that the price bound parameters have been set in increasing/decreasing order
    upper_bound = list(p.PRICE_BOUNDS_DF['upper_bound'])
    max_percent_increase = list(p.PRICE_BOUNDS_DF['max_percent_increase'])
    max_dollar_increase = list(p.PRICE_BOUNDS_DF['max_dollar_increase'])
    assert sorted(upper_bound) == upper_bound and sorted(max_dollar_increase) == max_dollar_increase \
           and sorted(max_percent_increase, reverse=True) == max_percent_increase, "Doublecheck p.PRICE_BOUNDS input"

    # Check that the biggest upper bound is above any reasonable drug price
    assert upper_bound[-1] > 99999, "upper_bound[-1] > 99999"

    # Check for correct shape
    assert len(upper_bound) == len(max_percent_increase) == len(max_dollar_increase), "len(upper_bound) == len(max_percent_increase) == len(max_dollar_increase)"

    # Check for nan/0
    assert p.PRICE_BOUNDS_DF.eq(0).any().any() == False, "p.PRICE_BOUNDS_DF.eq(0).any().any() == False"
    assert p.PRICE_BOUNDS_DF.isna().any().any() == False, "p.PRICE_BOUNDS_DF.isna().any().any() == False"

def cross_contract_utilization_check(cross_contract_data, contract_date_df, date_offset = 3, rho_cutoff = .700, above_below = 'above', percentile = .30):
    # This check is now being done as part of the client parameters table. Clients failing this check will have CROSS_CONTRACT_PROJ turned off.
    
    # Tests that when using cross-contract claims data that the utilization mix between both contract periods are roughly similar. 
    # By default compares all GPIs from the last three months of the old contract (date_offset) versus all GPIs and months in the current contract. 
    # Uses Spearman's Rho of .700 as the failure cutoff by default. The analysis can be adjusted to only consider GPIs either above or below a percentile
    # cutoff based on AWP of each GPI.
    
    print('\n************************************************** \nCross Contract Utilization Check\n')
    # Separate data into old an new contract dfs
    df_old = cross_contract_data.loc[(cross_contract_data['DOF'] >= (contract_date_df['CONTRACT_EFF_DT'] - pd.DateOffset(months = date_offset)).astype('str').values[0])
                                     & (cross_contract_data['DOF'] < contract_date_df['CONTRACT_EFF_DT'].astype('str').values[0])]
    df_new = cross_contract_data.loc[(cross_contract_data['DOF'] >= contract_date_df['CONTRACT_EFF_DT'].astype('str').values[0])]
    
    # Sum quantity at BREAKOUT_GPI level for analysis
    df_old = df_old.groupby(['CLIENT','GPI','BREAKOUT']).agg({'QTY':'sum','FULLAWP_ADJ':'sum'})
    df_new = df_new.groupby(['CLIENT','GPI','BREAKOUT']).agg({'QTY':'sum'})
    
    # Merge old and new sums together, filling in missing GPIs with 0
    df_chk = df_old.merge(df_new, how = 'outer', on = ['CLIENT','BREAKOUT','GPI']).fillna(0).reset_index()
    
    #Get percentile ranks for the percentile filter. Using sums from prior contract for calculating percentiles
    for i in df_chk['BREAKOUT'].unique():
        df_chk.loc[(df_chk['BREAKOUT'] == i)] = df_chk.loc[(df_chk['BREAKOUT'] == i)].sort_values(by=['QTY_x'], ascending=False)
        df_chk.loc[(df_chk['BREAKOUT'] == i), 'PCTL'] = df_chk.loc[(df_chk['BREAKOUT'] == i), 'FULLAWP_ADJ'].rank(pct=True)

    # Filter based on percentile
    if above_below == 'above':
        df_chk = df_chk.loc[(df_chk['PCTL'] > percentile)]
    elif above_below == 'below':
        df_chk = df_chk.loc[(df_chk['PCTL'] < percentile)]
        
    df_chk.drop(columns = ['FULLAWP_ADJ','PCTL'], inplace = True)
 
    # Using Spearman's Rho as it doesn't assume normality, isn't sensitive to outliers, and compares relative ranks of GPIs. If rankings change
    # between time periods, Rho becomes weaker. If they stay similar, Rho becomes stronger.
    df_corr = df_chk.groupby(['CLIENT','BREAKOUT']).corr(method = 'spearman')

    #Simplify output to remove redundant cells
    df_corr = df_corr.groupby(['CLIENT','BREAKOUT']).agg({'QTY_x':'min'}).reset_index().rename(columns={'QTY_x':'RHO'})

    # Outputting data to evaluate the test
    df_chk.rename(columns = {'QTY_x':'Old_Contract_QTY','QTY_y':'New_Contract_QTY'}, inplace = True)
    df_chk.to_csv(p.FILE_LOG_PATH + 'CROSS_CONTRACT_PROJ_CORR_{}.csv'.format(p.DATA_ID), index = False)

    for i in df_chk['BREAKOUT'].unique():
        assert not np.isnan(df_corr['RHO'].loc[df_corr['BREAKOUT'] == i].values.min()), "This Cross Contract Utilization check is a NaN. This indicates that there is a change in breakouts between the old contract period and the new contract period. Please reconsider using CCP for this client"
        print("{} Rho: {}".format(i, df_corr['RHO'].loc[df_corr['BREAKOUT'] == i].values.min().round(3)))

    assert df_corr['RHO'].min() >= rho_cutoff, "Cross contract utilization check failed with a Rho of {} at {}. Old contract data may not be appropriate for new contract projections. See CROSS_CONTRACT_PROJ_CORR_{}.csv in Logs folder to create scatterplots.".format(df_corr['RHO'].values.min().round(3), df_corr.loc[(df_corr['RHO'] == df_corr['RHO'].values.min()), 'BREAKOUT'].values[0], p.DATA_ID)
    
    print("Passed")
    print('**************************************')  
    
def check_pstcosttype():
    import datetime as dt
    import pandas as pd
    import numpy as np
    import os
    import CPMO_parameters as p
    from CPMO_shared_functions import standardize_df
    import util_funcs as uf
    import BQ
    
    print('\n************************************************** \nPSTCOSTTYPE Check\n')
    
    assert p.READ_FROM_BQ, "PSTCOSTTYPE check cannot be performed when READ_FROM_BQ is False. Please manually check whether R90 claims are adjudicating on the mail VCML."
    current_contract_data = standardize_df(pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.CONTRACT_DATE_FILE), dtype = p.VARIABLE_TYPE_DIC))
    pstcosttype_df = uf.read_BQ_data(
        BQ.pstcosttype_check_custom.format(
                                            _project_id = p.BQ_INPUT_PROJECT_ID,
                                            _dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP,
                                            _table_id = 'FULL_OPT_CLIENT_PHARM_CLAIMS_STANDARD',
                                            _customer_id = uf.get_formatted_string(p.CUSTOMER_ID),
                                            _claim_start = current_contract_data.CONTRACT_EFF_DT[0],
                                            _claim_date = p.LAST_DATA.strftime('%Y-%m-%d')
                                            ),
        project_id = p.BQ_INPUT_PROJECT_ID,
        dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP,
        table_id = 'FULL_OPT_CLIENT_PHARM_CLAIMS_STANDARD',
        client = ', '.join(sorted(p.CUSTOMER_ID)),
        custom=True
    )    
    # Filtering out rows in vcml_df for truecost_clients # to be used when we have the data for TRUECOST coming in 
    if p.TRUECOST_CLIENT:
        pstcosttype_df = pstcosttype_df[pstcosttype_df['PSTCOSTTYPE'].str.contains('TRU')]
    elif p.UCL_CLIENT:
        pstcosttype_df = pstcosttype_df[pstcosttype_df['PSTCOSTTYPE'].str.contains('UCL')]
    else:
        pstcosttype_df = pstcosttype_df[~((pstcosttype_df['PSTCOSTTYPE'].str.contains('TRU')) | (pstcosttype_df['PSTCOSTTYPE'].str.contains('UCL')))]
    print(pstcosttype_df)
    if 'R90' not in pstcosttype_df['MEASUREMENT_CLEAN'].unique():
        return
    for cid in p.CUSTOMER_ID:
        mail_mac = 'MAC'+cid+'2'
        mail_mac_claims = pstcosttype_df[pstcosttype_df.PSTCOSTTYPE == mail_mac].rename(columns={'NUM_CLAIMS': 'NUM_MAIL_CLAIMS'})
        all_claims = pstcosttype_df[pstcosttype_df['CUSTOMER_ID'] == cid].groupby(['CUSTOMER_ID', 'MEASUREMENT_CLEAN'], as_index = False).sum()
        all_claims = all_claims.merge(mail_mac_claims, on=['CUSTOMER_ID', 'MEASUREMENT_CLEAN'])
        all_claims['FRAC_MAIL'] = all_claims['NUM_MAIL_CLAIMS']/all_claims['NUM_CLAIMS']
        if p.R90_AS_MAIL:
            assert (all_claims[all_claims['MEASUREMENT_CLEAN']=='R90']['FRAC_MAIL'] > 0.5).all(), \
            "R90_AS_MAIL parameter is set, but less than 50% of R90 claims adjudicate on the MAIL VCML"
        else:
            assert (all_claims[all_claims['MEASUREMENT_CLEAN']=='R90']['FRAC_MAIL'] < 0.5).all(), \
            "R90_AS_MAIL parameter is not set, but R90 claims are adjudicating on the MAIL MAC." + \
            "Confirm this is correct and then rerun with p.R90_AS_MAIL = True"
    print("Passed")
    print('**************************************')        
    

def check_vcml():
    # Tests that the VCMLs in MAC MAPPING are same as the VCMLs in the claims data
    import datetime as dt
    import pandas as pd
    import numpy as np
    import CPMO_parameters as p
    from CPMO_shared_functions import standardize_df
    import util_funcs as uf
    import BQ
    
    print('\n************************************************** \nVCML Check\n')
    
    vcml_ref_df = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.VCML_REFERENCE_FILE, dtype = p.VARIABLE_TYPE_DIC, parse_dates = ['REC_EFFECTIVE_DATE','REC_EXPIRATION_DATE']))
    vcml_format = p.APPLY_VCML_PREFIX + f"{uf.get_formatted_string(p.CUSTOMER_ID)[1:-1]}"

    flagged_vcmls = p.IGNORED_VCMLS 
        
    if vcml_format[3:] in p.XA_XR_LIST:
        flagged_vcmls.append('XR')
        flagged_vcmls.append('XA')
        
    if len(set(['34', '35', '36', '37', '38']) - set(vcml_ref_df.VCML_ID.astype(str).str[len(p.CUSTOMER_ID[0])+3:])) == 0:
        flagged_vcmls.append('9')

    if len(set(['177', '199']) - set(vcml_ref_df.VCML_ID.astype(str).str[len(p.CUSTOMER_ID[0])+3:])) == 0:
        flagged_vcmls.append('1')
        
    if (len(set(['377', '399']) - set(vcml_ref_df.VCML_ID.astype(str).str[len(p.CUSTOMER_ID[0])+3:])) == 0):
        flagged_vcmls.append('3')
        
    custom_mask = vcml_ref_df['VCML_ID'].str[len(vcml_format):].isin(flagged_vcmls)
    date_mask = (vcml_ref_df['REC_EFFECTIVE_DATE'] <= p.GO_LIVE) & (vcml_ref_df['REC_EXPIRATION_DATE'] >= p.GO_LIVE)

    # Extract the suffix from VCML_ID by removing both the three-letter prefix and the CUSTOMER_ID.
    # The VCML_ID column contains values like 'MAC1234AB', where:
    #   - The first three characters (e.g., 'MAC', 'TRU', 'UCL') are the prefix.
    #   - The next set of characters represent the CUSTOMER_ID (e.g., '1234').
    #   - The remaining characters are the suffix we want to extract (e.g., 'AB').
    # For example:
    #   - If VCML_ID is 'MAC1234AB' and CUSTOMER_ID is '1234', the suffix will be 'AB'.
    #   - If VCML_ID is 'TRU6789XY' and CUSTOMER_ID is '6789', the suffix will be 'XY'.
    # This approach ensures we always extract the portion of VCML_ID that comes after both the prefix and the CUSTOMER_ID.
    vcml_ref_df['VCML_SUFFIX'] = [vcml_id[3 + len(str(customer_id)):] 
                                  for vcml_id, customer_id in zip(vcml_ref_df['VCML_ID'], vcml_ref_df['CUSTOMER_ID'])]

    vcml_ref_list = set(vcml_ref_df.loc[vcml_ref_df.VCML_ID.str.startswith(vcml_format) & date_mask & ~custom_mask,'VCML_SUFFIX'].unique())   
    
    mac_mapping_df = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.MAC_MAPPING_FILE, dtype = p.VARIABLE_TYPE_DIC))
    if vcml_format[3:] in p.XA_XR_LIST:
        mac_mapping_df = mac_mapping_df[~mac_mapping_df['MAC_LIST'].isin([vcml_format[3:]+'XA', vcml_format[3:]+'XR'])]
    
    # Remove customer_id from the MAC_LIST column value 
    mac_mapping_df['MAC_SUFFIX'] = mac_mapping_df.apply(lambda row: row['MAC_LIST'][len(row['CUSTOMER_ID']):], axis=1)
    
    vcml_mapping_list = set(mac_mapping_df.MAC_SUFFIX.values)
    
    print(vcml_ref_list)
    print(vcml_mapping_list)
    
    assert vcml_ref_list == vcml_mapping_list, "*Mismatch list: {} Error: VCMLs in MAC MAPPING FILE do not match the active VCMLs in the VCML REFERENCE table.".format(vcml_mapping_list^vcml_ref_list)                         
    print("Passed")
    print('**************************************') 

    
def check_runid():
##### This function checks the run ids. If the runid for Truecost client starts with UCL or not.
    import CPMO_parameters as p
    print('\n************************************************** \nRunID check\n')
    run_id = p.AT_RUN_ID
    print(run_id)
    if p.TRUECOST_CLIENT or p.UCL_CLIENT:
        assert run_id.startswith("UCL"), f" Invalid runid : {run_id}. For Truecost clients or UCL clients, run id must start with 'UCL'."
    
        print("Runid is valid for Truecost client or UCL clients.")
    else:
        assert not run_id.startswith("UCL"), f" Invalid runid : {run_id}. For clients other then Truecost client or UCL client, run id must not start with 'UCL'."
        
        print("Runid is valid for non-Truecost, non-UCL client.")
                        
def main():
    from CPMO_shared_functions import check_and_create_folder
    import CPMO_parameters as p
    import util_funcs as uf
    check_and_create_folder(p.FILE_LOG_PATH)

    generate_GPI_report()
    generate_measurement_mapping_report()
    
if __name__ == '__main__':
    from CPMO_shared_functions import update_run_status
    try:
        from CPMO_shared_functions import check_and_create_folder
        import CPMO_parameters as p
        check_and_create_folder(p.FILE_LOG_PATH)

        generate_GPI_report()
        generate_measurement_mapping_report()
        
        check_pstcosttype()
        check_vcml()
        check_contractual_rates()
        check_runid()
        
    except Exception as e:
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'QA Checks', repr(e), error_loc)
        raise e
